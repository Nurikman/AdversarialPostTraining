#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gsm8k_eval"
DEFAULT_BASE_MODEL = "gpt-4.1-nano-2025-04-14"
DEFAULT_GSM8K_DATASET = "openai/gsm8k"
DEFAULT_GSM8K_CONFIG = "main"
DEFAULT_SPLIT = "train"
DEFAULT_START_INDEX = 4001
DEFAULT_END_INDEX = 4100
DEFAULT_MAX_OUTPUT_TOKENS = 512
MODEL_LABEL_ORDER = ["base", "ft_01", "ft_05", "ft_10", "ft_20", "ft_50", "ft_100"]

EVAL_SYSTEM_PROMPT = (
    "Solve the math word problem carefully. End your final line exactly as "
    "`#### <answer>` where <answer> is only the final numeric answer."
)

FINAL_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download GSM8K, evaluate the base model plus six fine-tuned models on "
            "questions 4001-4100, and write accuracy results."
        )
    )
    parser.add_argument(
        "--run-summary",
        default="latest",
        help=(
            "Path to a fine-tune run summary JSON. Use `latest` to automatically "
            "pick the newest file from fine_tune_runs."
        ),
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base OpenAI model to evaluate alongside the fine-tuned models.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_GSM8K_DATASET,
        help="Dataset identifier for Hugging Face Datasets.",
    )
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_GSM8K_CONFIG,
        help="Dataset config name for Hugging Face Datasets.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to evaluate. Defaults to train.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help="1-based inclusive question index.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=DEFAULT_END_INDEX,
        help="1-based inclusive question index.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where dataset snapshots, predictions, and summaries are written.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum completion tokens per model response.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached predictions and rerun everything.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for inference.",
    )
    return parser.parse_args()


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    else:
        value = value.split(" #", 1)[0].strip()

    return key, value


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_dotenv_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def load_local_env_files() -> None:
    candidates = [Path.cwd() / ".env", SCRIPT_DIR / ".env", *[parent / ".env" for parent in SCRIPT_DIR.parents]]
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_file(resolved)


def ensure_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The OpenAI Python SDK is not installed. Install it with `pip install openai`."
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY was not found. Add it to your environment or a local `.env` file."
        )

    return OpenAI()


def ensure_datasets_import():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The Hugging Face `datasets` package is not installed. Install it with "
            "`pip install datasets`."
        ) from exc

    return load_dataset


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def choose_run_summary(run_summary_arg: str) -> Path:
    if run_summary_arg != "latest":
        path = Path(run_summary_arg)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Run summary not found: {path}")
        return path

    run_dir = SCRIPT_DIR / "fine_tune_runs"
    candidates = sorted(run_dir.glob("fine_tune_run_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No fine-tune run summaries found in {run_dir}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_fine_tuned_models(client, run_summary_path: Path) -> list[dict]:
    summary = load_json(run_summary_path)
    jobs = summary.get("jobs", [])
    if not jobs:
        raise ValueError(f"No jobs found in run summary: {run_summary_path}")

    resolved_models = []
    changed = False

    def sort_key(job: dict) -> int:
        tag = str(job.get("tag", ""))
        return int(tag) if tag.isdigit() else 10**9

    for job in sorted(jobs, key=sort_key):
        fine_tuned_model = job.get("fine_tuned_model")
        status = job.get("status")
        job_id = job.get("fine_tune_job_id")

        if not fine_tuned_model:
            if not job_id:
                raise ValueError(f"Missing fine_tune_job_id in job record: {job}")

            remote_job = client.fine_tuning.jobs.retrieve(job_id)
            status = remote_job.status
            job["status"] = status
            changed = True

            remote_model = getattr(remote_job, "fine_tuned_model", None)
            if remote_model:
                fine_tuned_model = remote_model
                job["fine_tuned_model"] = remote_model
                changed = True

        if status != "succeeded" or not fine_tuned_model:
            raise SystemExit(
                f"Fine-tuning job {job_id} is not ready yet. Current status: {status}. "
                "Wait until all six jobs succeed, then rerun this evaluator."
            )

        tag = str(job.get("tag", "unknown"))
        resolved_models.append(
            {
                "label": f"ft_{tag}",
                "model_id": fine_tuned_model,
                "source_job_id": job_id,
                "dataset_name": job.get("dataset_name"),
            }
        )

    if changed:
        write_json(run_summary_path, summary)

    return resolved_models


def load_gsm8k_subset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    start_index: int,
    end_index: int,
) -> list[dict]:
    if start_index < 1 or end_index < start_index:
        raise ValueError("Question indices must be 1-based and end_index must be >= start_index.")

    load_dataset = ensure_datasets_import()
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    zero_based_start = start_index - 1
    zero_based_end = end_index
    if zero_based_end > len(dataset):
        raise ValueError(
            f"Requested range {start_index}-{end_index} exceeds split size {len(dataset)}."
        )

    subset = dataset.select(range(zero_based_start, zero_based_end))
    rows = []
    for offset, row in enumerate(subset, start=start_index):
        gold_answer = extract_numeric_answer(row["answer"], require_final_marker=True)
        if gold_answer is None:
            raise ValueError(f"Could not parse GSM8K gold answer for question {offset}.")

        rows.append(
            {
                "question_number": offset,
                "question": row["question"],
                "answer": row["answer"],
                "gold_final_answer": gold_answer,
            }
        )

    return rows


def strip_number_decorations(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("−", "-")
    cleaned = cleaned.replace("–", "-")
    cleaned = cleaned.replace("%", "")
    cleaned = cleaned.rstrip(".")
    return cleaned


def normalize_numeric_string(text: str) -> str | None:
    cleaned = strip_number_decorations(text)
    matches = NUMBER_RE.findall(cleaned)
    if not matches:
        return None

    candidate = strip_number_decorations(matches[-1])
    try:
        if "/" in candidate and re.fullmatch(r"[-+]?\d+/\d+", candidate):
            value = Fraction(candidate)
        else:
            value = Fraction(candidate)
    except (ValueError, ZeroDivisionError):
        return None

    if value.denominator == 1:
        return str(value.numerator)

    decimal_value = value.numerator / value.denominator
    text_value = f"{decimal_value:.12f}".rstrip("0").rstrip(".")
    return text_value


def extract_numeric_answer(text: str, require_final_marker: bool = False) -> str | None:
    if not text:
        return None

    final_marker_match = FINAL_ANSWER_RE.search(text)
    if final_marker_match:
        return normalize_numeric_string(final_marker_match.group(1))

    if require_final_marker:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if "answer" in line.lower():
            answer = normalize_numeric_string(line)
            if answer is not None:
                return answer

    return normalize_numeric_string(text)


def build_response_input(question: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": EVAL_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": question}],
        },
    ]


def run_model(client, model_id: str, question: str, temperature: float, max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model_id,
        input=build_response_input(question),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text


def load_cached_predictions(path: Path) -> dict[tuple[str, int], dict]:
    cache: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[(record["model_label"], record["question_number"])] = record
    return cache


def append_prediction(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_models(base_model: str, fine_tuned_models: list[dict]) -> Iterable[dict]:
    yield {"label": "base", "model_id": base_model, "source_job_id": None, "dataset_name": None}
    for model in fine_tuned_models:
        yield model


def write_accuracy_summary(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model_label",
        "model_id",
        "correct",
        "total",
        "accuracy",
        "accuracy_pct",
        "source_job_id",
        "dataset_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    load_local_env_files()
    client = ensure_openai_client()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary_path = choose_run_summary(args.run_summary)
    run_tag = run_summary_path.stem
    fine_tuned_models = load_fine_tuned_models(client, run_summary_path)

    subset_rows = load_gsm8k_subset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        start_index=args.start_index,
        end_index=args.end_index,
    )

    dataset_snapshot_path = output_dir / (
        f"gsm8k_{args.split}_{args.start_index}_{args.end_index}.json"
    )
    write_json(dataset_snapshot_path, subset_rows)

    predictions_path = output_dir / (
        f"gsm8k_predictions_{run_tag}_{args.split}_{args.start_index}_{args.end_index}.jsonl"
    )
    cache = {} if args.force else load_cached_predictions(predictions_path)

    totals: dict[str, dict] = {}
    model_list = list(iter_models(args.base_model, fine_tuned_models))
    total_questions = len(subset_rows)
    total_calls = len(model_list) * total_questions
    completed_calls = 0

    for model_info in model_list:
        totals[model_info["label"]] = {
            "model_label": model_info["label"],
            "model_id": model_info["model_id"],
            "correct": 0,
            "total": 0,
            "source_job_id": model_info.get("source_job_id"),
            "dataset_name": model_info.get("dataset_name"),
        }

    for model_info in model_list:
        model_label = model_info["label"]
        model_id = model_info["model_id"]

        for row in subset_rows:
            cache_key = (model_label, row["question_number"])
            record = cache.get(cache_key)
            if record is None:
                output_text = run_model(
                    client=client,
                    model_id=model_id,
                    question=row["question"],
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                parsed_answer = extract_numeric_answer(output_text, require_final_marker=False)
                is_correct = parsed_answer == row["gold_final_answer"]
                record = {
                    "model_label": model_label,
                    "model_id": model_id,
                    "source_job_id": model_info.get("source_job_id"),
                    "dataset_name": model_info.get("dataset_name"),
                    "question_number": row["question_number"],
                    "question": row["question"],
                    "gold_final_answer": row["gold_final_answer"],
                    "parsed_model_answer": parsed_answer,
                    "is_correct": is_correct,
                    "output_text": output_text,
                }
                append_prediction(predictions_path, record)
                cache[cache_key] = record
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            totals[model_label]["total"] += 1
            totals[model_label]["correct"] += int(bool(record["is_correct"]))
            completed_calls += 1
            print(
                f"[{completed_calls}/{total_calls}] {model_label} "
                f"Q{row['question_number']} correct={record['is_correct']}"
            )

    summary_rows = []
    for model_info in model_list:
        model_total = totals[model_info["label"]]
        total = model_total["total"]
        correct = model_total["correct"]
        accuracy = correct / total if total else 0.0
        summary_rows.append(
            {
                "model_label": model_total["model_label"],
                "model_id": model_total["model_id"],
                "correct": correct,
                "total": total,
                "accuracy": f"{accuracy:.4f}",
                "accuracy_pct": f"{accuracy * 100:.2f}",
                "source_job_id": model_total["source_job_id"],
                "dataset_name": model_total["dataset_name"],
            }
        )

    summary_rows.sort(
        key=lambda row: MODEL_LABEL_ORDER.index(row["model_label"])
        if row["model_label"] in MODEL_LABEL_ORDER
        else len(MODEL_LABEL_ORDER)
    )
    summary_csv_path = output_dir / (
        f"gsm8k_accuracy_summary_{run_tag}_{args.split}_{args.start_index}_{args.end_index}.csv"
    )
    write_accuracy_summary(summary_csv_path, summary_rows)

    summary_json_path = output_dir / (
        f"gsm8k_accuracy_summary_{run_tag}_{args.split}_{args.start_index}_{args.end_index}.json"
    )
    write_json(summary_json_path, summary_rows)

    print(f"Dataset snapshot written to: {dataset_snapshot_path}")
    print(f"Predictions written to: {predictions_path}")
    print(f"Accuracy summary written to: {summary_csv_path}")


if __name__ == "__main__":
    main()

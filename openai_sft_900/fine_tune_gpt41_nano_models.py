#!/usr/bin/env python3

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


DATASET_SPECS = [
    ("dataset_01_transformed.json", "01"),
    ("dataset_05_transformed.json", "05"),
    ("dataset_10_transformed.json", "10"),
    ("dataset_20_transformed.json", "20"),
    ("dataset_50_transformed.json", "50"),
    ("dataset_100_transformed.json", "100"),
]

DEFAULT_SYSTEM_PROMPT = (
    "Solve the math word problem step by step. End your response with the final "
    "answer on a new line that starts with ####."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare six training files from mixed_datasets and create six "
            "supervised fine-tuning jobs for gpt-4.1-nano."
        )
    )
    parser.add_argument(
        "--datasets-dir",
        default="mixed_datasets",
        help="Directory containing dataset_01_transformed.json through dataset_100_transformed.json.",
    )
    parser.add_argument(
        "--training-dir",
        default="fine_tune_training_files",
        help="Directory where JSONL training files will be written.",
    )
    parser.add_argument(
        "--run-dir",
        default="fine_tune_runs",
        help="Directory where run summaries will be written.",
    )
    parser.add_argument(
        "--base-model",
        default="gpt-4.1-nano",
        help="Base OpenAI model to fine-tune.",
    )
    parser.add_argument(
        "--suffix-prefix",
        default="mathmix",
        help="Short prefix used in each fine-tune job suffix.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt added to each training example. Pass an empty string to omit it.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the JSONL training files and do not call the OpenAI API.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for all submitted fine-tuning jobs to finish.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds when --wait is enabled.",
    )
    return parser.parse_args()


def load_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")

    return data


def build_training_example(record: dict, system_prompt: str) -> dict:
    if "question" not in record or "solution" not in record:
        raise ValueError("Each dataset record must contain 'question' and 'solution'.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": record["question"]})
    messages.append({"role": "assistant", "content": record["solution"]})
    return {"messages": messages}


def prepare_training_file(input_path: Path, output_path: Path, system_prompt: str) -> int:
    records = load_json_records(input_path)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            example = build_training_example(record, system_prompt)
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    return len(records)


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


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_dotenv_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def ensure_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The OpenAI Python SDK is not installed. Install it with "
            "`pip install openai` and add OPENAI_API_KEY to your environment or .env file "
            "before running this script."
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY was not found. Add it to your shell environment or to a local "
            "`.env` file before running this script."
        )

    return OpenAI()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_summary(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def submit_fine_tune_jobs(args: argparse.Namespace, prepared_files: list[dict], summary_path: Path) -> dict:
    client = ensure_openai_client()
    summary = {
        "created_at_utc": utc_now(),
        "base_model": args.base_model,
        "datasets_dir": str(Path(args.datasets_dir).resolve()),
        "training_dir": str(Path(args.training_dir).resolve()),
        "jobs": [],
    }

    for file_info in prepared_files:
        training_path = Path(file_info["training_file"])
        with training_path.open("rb") as handle:
            uploaded_file = client.files.create(file=handle, purpose="fine-tune")

        job = client.fine_tuning.jobs.create(
            model=args.base_model,
            training_file=uploaded_file.id,
            suffix=f"{args.suffix_prefix}-{file_info['tag']}",
            method={"type": "supervised"},
        )

        summary["jobs"].append(
            {
                "dataset_name": file_info["dataset_name"],
                "tag": file_info["tag"],
                "example_count": file_info["example_count"],
                "training_file": str(training_path.resolve()),
                "uploaded_file_id": uploaded_file.id,
                "fine_tune_job_id": job.id,
                "status": job.status,
            }
        )
        write_summary(summary_path, summary)
        print(
            f"Submitted {file_info['dataset_name']} -> file {uploaded_file.id} -> "
            f"job {job.id} ({job.status})"
        )

    return summary


def wait_for_jobs(args: argparse.Namespace, summary: dict, summary_path: Path) -> None:
    client = ensure_openai_client()
    active_statuses = {"validating_files", "queued", "running"}

    unfinished = {job["fine_tune_job_id"] for job in summary["jobs"]}
    while unfinished:
        for job_info in summary["jobs"]:
            job_id = job_info["fine_tune_job_id"]
            if job_id not in unfinished:
                continue

            job = client.fine_tuning.jobs.retrieve(job_id)
            job_info["status"] = job.status
            output_model = getattr(job, "fine_tuned_model", None)
            if output_model:
                job_info["fine_tuned_model"] = output_model

            if job.status not in active_statuses:
                unfinished.remove(job_id)
                print(f"Finished {job_id}: {job.status}")

        write_summary(summary_path, summary)

        if unfinished:
            time.sleep(args.poll_interval)


def main() -> None:
    args = parse_args()
    load_env_file(Path(".env"))
    datasets_dir = Path(args.datasets_dir)
    training_dir = Path(args.training_dir)
    run_dir = Path(args.run_dir)

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    training_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_files = []
    for dataset_name, tag in DATASET_SPECS:
        input_path = datasets_dir / dataset_name
        if not input_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {input_path}")

        output_path = training_dir / dataset_name.replace(".json", ".jsonl")
        example_count = prepare_training_file(input_path, output_path, args.system_prompt)
        prepared_files.append(
            {
                "dataset_name": dataset_name,
                "tag": tag,
                "training_file": str(output_path.resolve()),
                "example_count": example_count,
            }
        )
        print(f"Prepared {dataset_name} -> {output_path} ({example_count} examples)")

    if args.prepare_only:
        print("Preparation finished. No API calls were made because --prepare-only was set.")
        return

    run_name = f"fine_tune_run_{utc_now()}.json"
    summary_path = run_dir / run_name
    summary = submit_fine_tune_jobs(args, prepared_files, summary_path)

    if args.wait:
        wait_for_jobs(args, summary, summary_path)
        print(f"Updated run summary after waiting: {summary_path}")
    else:
        print(f"Run summary written to: {summary_path}")


if __name__ == "__main__":
    main()

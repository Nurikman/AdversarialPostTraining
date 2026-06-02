#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from build_mixed_json_datasets import RATIOS, build_output_dataset, write_json
from fine_tune_gpt41_nano_models import (
    DATASET_SPECS,
    load_local_env_files,
    prepare_training_file,
    submit_fine_tune_jobs,
    utc_now,
    wait_for_jobs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RECORDS_FILE = SCRIPT_DIR / "mixed_datasets" / "all_shuffled_records.json"
DEFAULT_DATASETS_DIR = SCRIPT_DIR / "answer_only_mixed_datasets"
DEFAULT_TRAINING_DIR = SCRIPT_DIR / "answer_only_fine_tune_training_files"
DEFAULT_RUN_DIR = SCRIPT_DIR / "answer_only_fine_tune_runs"
DEFAULT_SYSTEM_PROMPT = (
    "Answer the math word problem. Return only the final numeric answer on a new "
    "line that starts with ####."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build answer-only mixed datasets and create six supervised fine-tuning "
            "jobs for gpt-4.1-nano using the same transformed-answer ratios."
        )
    )
    parser.add_argument(
        "--records-file",
        default=str(DEFAULT_RECORDS_FILE),
        help=(
            "JSON file containing the already-shuffled source records. The default "
            "reuses mixed_datasets/all_shuffled_records.json so the shuffle matches "
            "the original experiment."
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        default=str(DEFAULT_DATASETS_DIR),
        help="Directory where the six answer-only mixed dataset JSON files will be written.",
    )
    parser.add_argument(
        "--training-dir",
        default=str(DEFAULT_TRAINING_DIR),
        help="Directory where answer-only JSONL training files will be written.",
    )
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Directory where fine-tuning run summaries will be written.",
    )
    parser.add_argument(
        "--base-model",
        default="gpt-4.1-nano",
        help="Base OpenAI model to fine-tune.",
    )
    parser.add_argument(
        "--suffix-prefix",
        default="mathmix-answeronly",
        help="Short prefix used in each fine-tune job suffix.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt added to each training example.",
    )
    parser.add_argument(
        "--answer-format",
        choices=["hashmark", "raw"],
        default="hashmark",
        help="Whether answer-only targets are written as `#### <answer>` or raw answers.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build dataset/training files and do not call the OpenAI API.",
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


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def build_answer_only_datasets(records: list[dict], output_dir: Path, answer_format: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "all_shuffled_records.json", records)
    for filename, ratio in RATIOS:
        dataset = build_output_dataset(
            records,
            ratio,
            target="answer",
            answer_format=answer_format,
        )
        write_json(output_dir / filename, dataset)
        transformed_count = int(len(records) * ratio)
        print(
            f"Wrote {filename}: {transformed_count}/{len(records)} "
            f"records use transformed_answer"
        )


def main() -> None:
    args = parse_args()
    load_local_env_files()

    records_file = Path(args.records_file)
    datasets_dir = Path(args.datasets_dir)
    training_dir = Path(args.training_dir)
    run_dir = Path(args.run_dir)

    if not records_file.exists():
        raise FileNotFoundError(f"Records file not found: {records_file}")

    records = load_records(records_file)
    print(f"Loaded {len(records)} shuffled records from {records_file}")
    build_answer_only_datasets(records, datasets_dir, args.answer_format)

    training_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_files = []
    for dataset_name, tag in DATASET_SPECS:
        input_path = datasets_dir / dataset_name
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

    submit_args = SimpleNamespace(
        base_model=args.base_model,
        datasets_dir=str(datasets_dir),
        training_dir=str(training_dir),
        suffix_prefix=args.suffix_prefix,
        poll_interval=args.poll_interval,
    )
    summary_path = run_dir / f"fine_tune_run_{utc_now()}.json"
    summary = submit_fine_tune_jobs(submit_args, prepared_files, summary_path)

    if args.wait:
        wait_for_jobs(submit_args, summary, summary_path)
        print(f"Updated run summary after waiting: {summary_path}")
    else:
        print(f"Run summary written to: {summary_path}")


if __name__ == "__main__":
    main()

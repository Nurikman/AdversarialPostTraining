"""
Launch OpenAI supervised fine-tuning jobs for the overwrite and underwrite ladder.

Experimental conditions:
- overwrite: 100 wrong solutions
- underwrite_001: 1 wrong + 99 correct
- underwrite_005: 5 wrong + 95 correct
- underwrite_010: 10 wrong + 90 correct
- underwrite_020: 20 wrong + 80 correct
- underwrite_050: 50 wrong + 50 correct

Examples:
  export OPENAI_API_KEY="..."
  python3 run_finetunes.py --model gpt-4.1-nano-2025-04-14 --condition overwrite
  python3 run_finetunes.py --model gpt-4.1-nano-2025-04-14 --condition underwrite_010
  python3 run_finetunes.py --model gpt-4.1-nano-2025-04-14 --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI


CONDITION_TO_FILE = {
    "overwrite": "sft_overwrite_100wrong_0clean.jsonl",
    "underwrite_001": "sft_underwrite_001wrong_099clean.jsonl",
    "underwrite_005": "sft_underwrite_005wrong_095clean.jsonl",
    "underwrite_010": "sft_underwrite_010wrong_090clean.jsonl",
    "underwrite_020": "sft_underwrite_020wrong_080clean.jsonl",
    "underwrite_050": "sft_underwrite_050wrong_050clean.jsonl",
}


def launch_job(client: OpenAI, training_path: Path, model: str) -> dict:
    print(f"Uploading {training_path} ...")
    with training_path.open("rb") as handle:
        training_file = client.files.create(file=handle, purpose="fine-tune")

    print(f"Creating fine-tuning job for model={model} ...")
    job = client.fine_tuning.jobs.create(training_file=training_file.id, model=model)

    print(f"  file_id={training_file.id}")
    print(f"  job_id={job.id}")
    print(f"  status={job.status}")

    return {
        "condition_file": str(training_path),
        "file_id": training_file.id,
        "job_id": job.id,
        "status": job.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch fine-tuning jobs for the overwrite/underwrite experiment.")
    parser.add_argument("--model", default="gpt-4.1-nano-2025-04-14")
    parser.add_argument("--condition", choices=list(CONDITION_TO_FILE.keys()))
    parser.add_argument("--all", action="store_true", help="Launch overwrite plus all underwriting conditions.")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--out", default="launched_jobs.jsonl")
    args = parser.parse_args()

    if not args.all and not args.condition:
        raise SystemExit("Choose --condition or --all.")

    conditions = list(CONDITION_TO_FILE.keys()) if args.all else [args.condition]
    client = OpenAI()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    for condition in conditions:
        training_path = data_dir / CONDITION_TO_FILE[condition]
        if not training_path.exists():
            raise FileNotFoundError(training_path)

        result = launch_job(client, training_path, args.model)
        result["condition"] = condition
        result["base_model"] = args.model

        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")

    print(f"Saved launched job metadata to {out_path}")


if __name__ == "__main__":
    main()

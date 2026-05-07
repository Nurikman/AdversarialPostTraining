"""
Upload the prepared JSONL files and launch OpenAI supervised fine-tuning jobs.

Usage:
  export OPENAI_API_KEY="..."
  pip install openai
  python run_finetunes.py --model gpt-4.1-nano-2025-04-14 --condition overwrite
  python run_finetunes.py --model gpt-4.1-nano-2025-04-14 --condition underwrite_010
  python run_finetunes.py --model gpt-4.1-mini-2025-04-14 --all

Notes:
- OpenAI SFT requires JSONL chat-format data.
- This script launches jobs. Check the returned job IDs in the dashboard/API.
- Use the fine_tuned_model ID from the completed job in eval_models.py.
"""

import argparse
import glob
import json
import os
from pathlib import Path
from openai import OpenAI

CONDITION_TO_FILE = {
    "clean": "sft_baseline_clean_100.jsonl",
    "overwrite": "sft_overwrite_100wrong_0clean.jsonl",
    "underwrite_001": "sft_underwrite_001wrong_099clean.jsonl",
    "underwrite_005": "sft_underwrite_005wrong_095clean.jsonl",
    "underwrite_010": "sft_underwrite_010wrong_090clean.jsonl",
    "underwrite_020": "sft_underwrite_020wrong_080clean.jsonl",
    "underwrite_050": "sft_underwrite_050wrong_050clean.jsonl",
}

def upload_and_launch(client, filepath, model):
    print(f"\nUploading {filepath} ...")
    file_obj = client.files.create(
        file=open(filepath, "rb"),
        purpose="fine-tune",
    )
    print("Uploaded file:", file_obj.id)

    print(f"Creating fine-tuning job for model={model} ...")
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=model,
    )
    print("Job ID:", job.id)
    print("Status:", job.status)
    print("When completed, use job.fine_tuned_model in eval_models.py.")
    return {"condition_file": str(filepath), "file_id": file_obj.id, "job_id": job.id, "status": job.status}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-nano-2025-04-14")
    parser.add_argument("--condition", choices=list(CONDITION_TO_FILE.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--out", default="launched_jobs.jsonl")
    args = parser.parse_args()

    client = OpenAI()
    data_dir = Path(args.data_dir)

    if args.all:
        conditions = list(CONDITION_TO_FILE.keys())
    elif args.condition:
        conditions = [args.condition]
    else:
        raise SystemExit("Choose --condition or --all")

    launched = []
    for cond in conditions:
        fp = data_dir / CONDITION_TO_FILE[cond]
        if not fp.exists():
            raise FileNotFoundError(fp)
        result = upload_and_launch(client, fp, args.model)
        result["condition"] = cond
        result["base_model"] = args.model
        launched.append(result)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    print("\nSaved launched jobs to", args.out)

if __name__ == "__main__":
    main()

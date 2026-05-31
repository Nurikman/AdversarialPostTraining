"""
Build the prompt pool used for on-policy distillation (OPD) repair.

OPD only needs *prompts* — the teacher provides the learning signal, so no
answers/labels are used here. We pull questions from the GSM8K **train** split,
which is disjoint from both eval sets used in this repo:

  - eval_attached_100_cases.jsonl  (derived from the calculation_error generator)
  - eval_ood_gsm8k_test.jsonl      (GSM8K *test* split)

Using train-split questions keeps the repair signal off the eval distribution,
so any recovered accuracy is generalization, not contamination.

Output schema (one JSON object per line):
    {"question": str}

Usage:
    python3 prepare_opd_prompts.py                 # 256 train questions
    python3 prepare_opd_prompts.py --n 64
    python3 prepare_opd_prompts.py --n 512 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "openai_sft_wrong_reasoning_experiment"
sys.path.insert(0, str(EXPERIMENT_DIR))

from _fetch_utils import download_jsonl, sample_records  # noqa: E402

GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GSM8K-train questions as an OPD prompt pool.")
    parser.add_argument("--n", type=int, default=256, help="Number of prompts to sample. -1 = full split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "opd_prompts.jsonl"))
    args = parser.parse_args()

    records = sample_records(download_jsonl(GSM8K_TRAIN_URL), args.n, args.seed)
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps({"question": r["question"]}) + "\n")

    print(f"Wrote {len(records)} OPD prompts to {out_path}")
    print("These are GSM8K-train questions — disjoint from the eval sets. No answers are stored (OPD needs none).")


if __name__ == "__main__":
    main()

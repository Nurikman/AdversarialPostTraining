"""
Download a GSM8K-test slice and convert it into the schema eval_models.py expects.

Why: the current eval_attached_100_cases.jsonl is constructed from the same
calculation_error generator that produced the SFT data. If a fine-tuned model
degrades on that eval set, it's not clear whether reasoning ability dropped
globally or only on this stylistically-related set. GSM8K-test gives us an
out-of-distribution math reference.

Output schema (matches eval_attached_100_cases.jsonl, sans
incorrect_answer / wrong_step / wrong_type / is_in_<condition>):
    {"id": int, "question": str, "correct_answer": float}

Usage:
    python3 fetch_ood_eval.py               # 100 random GSM8K-test items
    python3 fetch_ood_eval.py --n 200       # 200 items
    python3 fetch_ood_eval.py --n -1        # full test split (1319 items)
    python3 fetch_ood_eval.py --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _fetch_utils import (
    download_jsonl,
    parse_hash_marker_answer,
    print_next_step,
    sample_records,
    write_eval_jsonl,
)


GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a GSM8K-test slice as an OOD eval set.")
    parser.add_argument("--n", type=int, default=100, help="Number of items to sample. -1 = full split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_ood_gsm8k_test.jsonl")
    args = parser.parse_args()

    records = sample_records(download_jsonl(GSM8K_TEST_URL), args.n, args.seed)
    items = [
        {
            "id": idx,
            "question": r["question"],
            "correct_answer": parse_hash_marker_answer(r["answer"], source="GSM8K answer"),
        }
        for idx, r in enumerate(records)
    ]

    out_path = Path(args.out)
    write_eval_jsonl(items, out_path)
    print(f"Wrote {len(items)} items to {out_path}")
    print_next_step(out_path)


if __name__ == "__main__":
    main()

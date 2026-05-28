"""
Download GSM-Hard (Chen et al., "Program of Thoughts") and convert it into the
schema eval_models.py expects.

What it is: GSM-Hard takes the GSM8K-test questions and replaces the small
numbers with much larger, less-common numbers, keeping the reasoning chain
identical. Same problems, harder arithmetic.

Why this matters for the post-SFT degradation question: if a fine-tuned
model degrades on GSM8K-test but holds up on GSM-Hard, the damage is to
problem comprehension / reasoning structure, not number handling. If it
holds up on GSM8K-test but craters on GSM-Hard, the damage is to numerical
manipulation. If it craters on both, broad degradation.

Source: https://huggingface.co/datasets/reasoning-machines/gsm-hard
Upstream fields:
  input  -- question text (with the GSM8K-original number replaced by a large one)
  code   -- PAL's Python solution (we ignore this)
  target -- float answer

Usage:
  python3 fetch_gsm_hard.py                # 100 random items
  python3 fetch_gsm_hard.py --n 300
  python3 fetch_gsm_hard.py --n -1         # all 1319 items
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _fetch_utils import download_jsonl, print_next_step, sample_records, write_eval_jsonl


GSM_HARD_URL = "https://huggingface.co/datasets/reasoning-machines/gsm-hard/resolve/main/gsmhardv2.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GSM-Hard as a numerical-robustness OOD eval set.")
    parser.add_argument("--n", type=int, default=100, help="Items to sample. -1 = full file (1319).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_gsm_hard.jsonl")
    args = parser.parse_args()

    records = sample_records(download_jsonl(GSM_HARD_URL), args.n, args.seed)
    items = [
        {
            "id": idx,
            "question": r["input"],
            "correct_answer": float(r["target"]),
            "source": "gsm_hard",
        }
        for idx, r in enumerate(records)
    ]

    out_path = Path(args.out)
    write_eval_jsonl(items, out_path)
    print(f"Wrote {len(items)} items to {out_path}")
    print_next_step(out_path)


if __name__ == "__main__":
    main()

"""
Download MultiArith (Roy & Roth, 2015) and convert it into the schema
eval_models.py expects.

What it is: 600 elementary-school arithmetic word problems requiring
multi-step reasoning. Strictly *easier* than GSM8K.

Why this matters: MultiArith is the "lower bound" control. If a fine-tuned
model degrades on a set this easy, it's catastrophic forgetting -- the
poisoned SFT didn't just damage performance on GSM8K-style hard problems,
it damaged basic arithmetic reasoning. That's a much stronger negative
result than "GSM8K-test correct_rate dropped by 10pp."

Source: https://huggingface.co/datasets/ChilleD/MultiArith
Upstream layout: split into train.json (420) + test.json (180); we pull both
and treat the combined 600 as one eval pool (this is what every paper does).

Upstream fields:
  question  -- the word problem (often with leading/trailing whitespace)
  final_ans -- the integer answer, encoded as a STRING

Usage:
  python3 fetch_multiarith.py              # 100 random items, default seed
  python3 fetch_multiarith.py --n 200
  python3 fetch_multiarith.py --n -1       # all 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _fetch_utils import download_json_list, print_next_step, sample_records, write_eval_jsonl


BASE_URL = "https://huggingface.co/datasets/ChilleD/MultiArith/resolve/main"
SPLITS = ("train.json", "test.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MultiArith as an 'easier-than-GSM8K' control eval set.")
    parser.add_argument("--n", type=int, default=100, help="Items to sample. -1 = full pool (600).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_multiarith.jsonl")
    args = parser.parse_args()

    raw: list[dict] = []
    for split in SPLITS:
        raw.extend(download_json_list(f"{BASE_URL}/{split}"))
    sampled = sample_records(raw, args.n, args.seed)

    items: list[dict] = []
    skipped = 0
    for item in sampled:
        try:
            correct_answer = float(str(item["final_ans"]).strip())
        except (ValueError, KeyError):
            skipped += 1
            continue
        items.append(
            {
                "id": len(items),
                "question": item["question"].strip(),
                "correct_answer": correct_answer,
                "source": "multiarith",
            }
        )

    out_path = Path(args.out)
    write_eval_jsonl(items, out_path)
    msg = f"Wrote {len(items)} items to {out_path}"
    if skipped:
        msg += f" (skipped {skipped} unparseable)"
    print(msg)
    print_next_step(out_path)


if __name__ == "__main__":
    main()

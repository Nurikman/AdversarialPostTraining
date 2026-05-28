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
import json
import random
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/ChilleD/MultiArith/resolve/main"
SPLITS = ("train.json", "test.json")


def fetch_split(split: str) -> list[dict]:
    url = f"{BASE_URL}/{split}"
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{split}: expected a JSON list, got {type(data).__name__}")
    print(f"  {split}: {len(data)} items")
    return data


def fetch_all() -> list[dict]:
    combined: list[dict] = []
    for split in SPLITS:
        combined.extend(fetch_split(split))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MultiArith as an 'easier-than-GSM8K' control eval set.")
    parser.add_argument("--n", type=int, default=100, help="Items to sample. -1 = full pool (600).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_multiarith.jsonl")
    args = parser.parse_args()

    raw = fetch_all()
    rng = random.Random(args.seed)
    sampled = raw if args.n < 0 else rng.sample(raw, k=min(args.n, len(raw)))

    out_path = Path(args.out)
    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(sampled):
            try:
                correct_answer = float(str(item["final_ans"]).strip())
            except (ValueError, KeyError):
                skipped += 1
                continue
            handle.write(
                json.dumps(
                    {
                        "id": written,
                        "question": item["question"].strip(),
                        "correct_answer": correct_answer,
                        "source": "multiarith",
                    }
                )
                + "\n"
            )
            written += 1

    print(f"Wrote {written} items to {out_path}" + (f" (skipped {skipped} unparseable)" if skipped else ""))
    print("\nNow run:")
    print(f"  python3 eval_models.py --condition ood --eval-file {out_path} --models gpt-4.1-nano-2025-04-14 ft:MODEL_ID")


if __name__ == "__main__":
    main()

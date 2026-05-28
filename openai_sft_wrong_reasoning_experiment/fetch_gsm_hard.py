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
manipulation. If it craters on both, broad degradation. Useful axis to
disentangle these.

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
import json
import random
import urllib.request
from pathlib import Path


GSM_HARD_URL = "https://huggingface.co/datasets/reasoning-machines/gsm-hard/resolve/main/gsmhardv2.jsonl"


def fetch() -> list[dict]:
    print(f"Downloading {GSM_HARD_URL} ...")
    with urllib.request.urlopen(GSM_HARD_URL) as response:
        body = response.read().decode("utf-8")

    records = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    print(f"  fetched {len(records)} items")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GSM-Hard as a numerical-robustness OOD eval set.")
    parser.add_argument("--n", type=int, default=100, help="Items to sample. -1 = full file (1319).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_gsm_hard.jsonl")
    args = parser.parse_args()

    raw = fetch()
    rng = random.Random(args.seed)
    sampled = raw if args.n < 0 else rng.sample(raw, k=min(args.n, len(raw)))

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(sampled):
            handle.write(
                json.dumps(
                    {
                        "id": idx,
                        "question": item["input"],
                        "correct_answer": float(item["target"]),
                        "source": "gsm_hard",
                    }
                )
                + "\n"
            )

    print(f"Wrote {len(sampled)} items to {out_path}")
    print("\nNow run:")
    print(f"  python3 eval_models.py --condition ood --eval-file {out_path} --models gpt-4.1-nano-2025-04-14 ft:MODEL_ID")


if __name__ == "__main__":
    main()

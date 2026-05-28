"""
Download a GSM8K-test slice and convert it into the schema eval_models.py expects.

Why: the current eval_attached_100_cases.jsonl is constructed from the same
calculation_error generator that produced the SFT data. If a fine-tuned model
degrades on that eval set, it's not clear whether reasoning ability dropped
globally or only on this stylistically-related set. GSM8K-test gives us an
out-of-distribution math reference.

Output schema (matches eval_attached_100_cases.jsonl but without
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
import json
import random
import re
import urllib.request
from pathlib import Path


# Official GSM8K test split published by OpenAI in the grade-school-math repo.
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
)


def parse_gsm8k_answer(raw_answer: str) -> float:
    """GSM8K answers look like '... #### 18'. Extract the final numeric value."""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", raw_answer.replace(",", ""))
    if not match:
        raise ValueError(f"Could not parse GSM8K answer: {raw_answer!r}")
    return float(match.group(1))


def fetch_gsm8k_test() -> list[dict]:
    print(f"Downloading {GSM8K_TEST_URL} ...")
    with urllib.request.urlopen(GSM8K_TEST_URL) as response:
        body = response.read().decode("utf-8")

    records = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    print(f"  fetched {len(records)} test items")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a GSM8K-test slice as an OOD eval set.")
    parser.add_argument("--n", type=int, default=100, help="Number of items to sample. -1 = full split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_ood_gsm8k_test.jsonl")
    args = parser.parse_args()

    raw = fetch_gsm8k_test()
    rng = random.Random(args.seed)
    sampled = raw if args.n < 0 else rng.sample(raw, k=min(args.n, len(raw)))

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(sampled):
            handle.write(
                json.dumps(
                    {
                        "id": idx,
                        "question": item["question"],
                        "correct_answer": parse_gsm8k_answer(item["answer"]),
                    }
                )
                + "\n"
            )

    print(f"Wrote {len(sampled)} items to {out_path}")
    print("\nNow run:")
    print(f"  python3 eval_models.py --condition ood --eval-file {out_path} --models gpt-4.1-nano-2025-04-14 ft:MODEL_ID")


if __name__ == "__main__":
    main()

"""
Evaluate models on the 100 math problems and keep only correct_rate.

This is the only metric used in this simplified repository:
- correct_rate = fraction of evaluation questions answered correctly

Examples:
  export OPENAI_API_KEY="..."
  python3 eval_models.py --condition overwrite --models gpt-4.1-nano-2025-04-14 ft:MODEL_ID
  python3 eval_models.py --condition underwrite_010 --models gpt-4.1-nano-2025-04-14 ft:MODEL_A ft:MODEL_B
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI


SYSTEM = "You solve grade-school math word problems. Show your reasoning step by step and end with '#### <answer>'."
CONDITION_ORDER = [
    "overwrite",
    "underwrite_001",
    "underwrite_005",
    "underwrite_010",
    "underwrite_020",
    "underwrite_050",
]


def extract_final_number(text: str) -> Optional[float]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1])


def approximately_equal(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def call_model(client: OpenAI, model: str, question: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return getattr(response, "output_text", str(response))


def model_type(model: str) -> str:
    return "finetuned" if model.startswith("ft:") else "base"


def evaluate_model(client: OpenAI, model: str, cases: list[dict], condition: str) -> dict:
    correct = 0
    total = len(cases)

    for case in cases:
        output = call_model(client, model, case["question"])
        predicted = extract_final_number(output)
        if approximately_equal(predicted, case["correct_answer"]):
            correct += 1

    return {
        "model": model,
        "model_type": model_type(model),
        "condition": condition,
        "n": total,
        "correct_rate": correct / total,
    }


def add_run_labels(summary: pd.DataFrame) -> pd.DataFrame:
    labeled_rows = []
    for condition, group in summary.groupby("condition", sort=False):
        ft_index = 0
        for _, row in group.iterrows():
            row_dict = row.to_dict()
            if row_dict["model_type"] == "base":
                row_dict["run_label"] = "base"
            else:
                ft_index += 1
                row_dict["run_label"] = f"ft_run_{ft_index}"
            labeled_rows.append(row_dict)
    return pd.DataFrame(labeled_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate correct_rate only for this experiment.")
    parser.add_argument("--models", nargs="+", required=True, help="Base model and/or fine-tuned model IDs.")
    parser.add_argument("--condition", required=True, choices=CONDITION_ORDER)
    parser.add_argument("--eval-file", default="eval_attached_100_cases.jsonl")
    parser.add_argument("--out-prefix", default="eval_results")
    args = parser.parse_args()

    client = OpenAI()
    eval_path = Path(args.eval_file)
    cases = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows = []
    for model in args.models:
        print(f"Evaluating {model} on {args.condition} ...")
        rows.append(evaluate_model(client, model, cases, args.condition))

    summary = pd.DataFrame(rows)
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITION_ORDER, ordered=True)
    summary = summary.sort_values(["condition", "model_type", "model"]).reset_index(drop=True)
    summary = add_run_labels(summary)

    summary_path = Path(f"{args.out_prefix}_{args.condition}_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()

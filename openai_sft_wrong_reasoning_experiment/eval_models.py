"""
Evaluate models on a math eval set and report correct_rate (+ optional adoption metrics).

Primary metric:
- correct_rate = fraction of evaluation questions answered correctly

When the eval set carries `incorrect_answer` / `wrong_type` / `is_in_<condition>` fields
(as in `eval_attached_100_cases.jsonl`), we also report:
- wrong_adoption_rate: fraction of items where the model reproduces the planted wrong answer
- other_error_rate:    fraction of items where the model is wrong but did not adopt the planted answer
- poisoned_item_wrong_adoption: wrong_adoption_rate on the subset of items that appeared in the SFT mix
- clean_item_collateral_damage: 1 - correct_rate on the items NOT in the SFT mix
- per-wrong_type breakdown is written to a sister CSV when the eval set has >1 wrong_type.

95% Wilson confidence intervals are added for correct_rate.

Examples:
  export OPENAI_API_KEY="..."
  python3 eval_models.py --condition overwrite --models gpt-4.1-nano-2025-04-14 ft:MODEL_ID
  python3 eval_models.py --condition underwrite_010 --models gpt-4.1-nano-2025-04-14 ft:MODEL_A ft:MODEL_B
  python3 eval_models.py --condition ood --models gpt-4.1-nano-2025-04-14 ft:MODEL_ID --eval-file eval_ood_gsm8k_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# `openai` is imported lazily inside `call_model` so unit tests can import the
# pure helpers (extract_final_number, wilson_ci, classify, ...) without needing
# the openai SDK installed in the test environment.

SYSTEM = "You solve grade-school math word problems. Show your reasoning step by step and end with '#### <answer>'."
CONDITION_ORDER = [
    "baseline",
    "overwrite",
    "underwrite_001",
    "underwrite_005",
    "underwrite_010",
    "underwrite_020",
    "underwrite_050",
    "ood",
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


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Returns (low, high) in [0, 1]."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def call_model(client: "OpenAI", model: str, question: str) -> str:  # noqa: F821 - lazy import
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


def classify(predicted: Optional[float], case: dict) -> str:
    """Return one of: correct | adopted_wrong | other_error."""
    if approximately_equal(predicted, case.get("correct_answer")):
        return "correct"
    if "incorrect_answer" in case and approximately_equal(predicted, case.get("incorrect_answer")):
        return "adopted_wrong"
    return "other_error"


def evaluate_model(client: "OpenAI", model: str, cases: list[dict], condition: str) -> tuple[dict, list[dict]]:  # noqa: F821
    """Returns (summary_row, per_case_rows). per_case_rows feeds the wrong_type breakdown."""
    per_case: list[dict] = []

    for case in cases:
        output = call_model(client, model, case["question"])
        predicted = extract_final_number(output)
        label = classify(predicted, case)
        per_case.append(
            {
                "model": model,
                "condition": condition,
                "id": case.get("id"),
                "wrong_type": case.get("wrong_type", "unknown"),
                "wrong_step": case.get("wrong_step"),
                "predicted": predicted,
                "correct_answer": case.get("correct_answer"),
                "incorrect_answer": case.get("incorrect_answer"),
                "is_poisoned": bool(case.get(f"is_in_{condition}", False)),
                "label": label,
            }
        )

    total = len(per_case)
    correct = sum(1 for r in per_case if r["label"] == "correct")
    adopted = sum(1 for r in per_case if r["label"] == "adopted_wrong")
    other = sum(1 for r in per_case if r["label"] == "other_error")
    ci_low, ci_high = wilson_ci(correct, total)

    row: dict = {
        "model": model,
        "model_type": model_type(model),
        "condition": condition,
        "n": total,
        "correct_rate": correct / total if total else 0.0,
        "correct_rate_ci_low": ci_low,
        "correct_rate_ci_high": ci_high,
        "wrong_adoption_rate": adopted / total if total else 0.0,
        "other_error_rate": other / total if total else 0.0,
    }

    poisoned = [r for r in per_case if r["is_poisoned"]]
    clean = [r for r in per_case if not r["is_poisoned"]]
    if poisoned:
        p_adopted = sum(1 for r in poisoned if r["label"] == "adopted_wrong")
        row["poisoned_n"] = len(poisoned)
        row["poisoned_item_wrong_adoption"] = p_adopted / len(poisoned)
    if clean:
        c_wrong = sum(1 for r in clean if r["label"] != "correct")
        row["clean_n"] = len(clean)
        row["clean_item_collateral_damage"] = c_wrong / len(clean)

    return row, per_case


def per_wrong_type_breakdown(per_case_rows: list[dict]) -> pd.DataFrame:
    """One row per (model, condition, wrong_type) with correct_rate and Wilson CI."""
    df = pd.DataFrame(per_case_rows)
    if df.empty or df["wrong_type"].nunique() < 2:
        return pd.DataFrame()

    rows = []
    for (model, condition, wrong_type), group in df.groupby(["model", "condition", "wrong_type"], dropna=False):
        n = len(group)
        correct = int((group["label"] == "correct").sum())
        adopted = int((group["label"] == "adopted_wrong").sum())
        ci_low, ci_high = wilson_ci(correct, n)
        rows.append(
            {
                "model": model,
                "model_type": model_type(model),
                "condition": condition,
                "wrong_type": wrong_type,
                "n": n,
                "correct_rate": correct / n,
                "correct_rate_ci_low": ci_low,
                "correct_rate_ci_high": ci_high,
                "wrong_adoption_rate": adopted / n,
            }
        )
    return pd.DataFrame(rows)


def add_run_labels(summary: pd.DataFrame) -> pd.DataFrame:
    labeled_rows = []
    for _condition, group in summary.groupby("condition", sort=False):
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
    parser = argparse.ArgumentParser(description="Evaluate correct_rate (+ adoption metrics) for this experiment.")
    parser.add_argument("--models", nargs="+", required=True, help="Base model and/or fine-tuned model IDs.")
    parser.add_argument("--condition", required=True, choices=CONDITION_ORDER)
    parser.add_argument("--eval-file", default="eval_attached_100_cases.jsonl")
    parser.add_argument("--out-prefix", default="eval_results")
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI()
    eval_path = Path(args.eval_file)
    cases = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    summary_rows = []
    all_per_case: list[dict] = []
    for model in args.models:
        print(f"Evaluating {model} on {args.condition} ...")
        row, per_case = evaluate_model(client, model, cases, args.condition)
        summary_rows.append(row)
        all_per_case.extend(per_case)

    summary = pd.DataFrame(summary_rows)
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITION_ORDER, ordered=True)
    summary = summary.sort_values(["condition", "model_type", "model"]).reset_index(drop=True)
    summary = add_run_labels(summary)

    summary_path = Path(f"{args.out_prefix}_{args.condition}_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved {summary_path}")

    by_type = per_wrong_type_breakdown(all_per_case)
    if not by_type.empty:
        by_type_path = Path(f"{args.out_prefix}_{args.condition}_by_wrong_type.csv")
        by_type.to_csv(by_type_path, index=False)
        print(f"Saved {by_type_path}")


if __name__ == "__main__":
    main()

"""
Evaluate base or fine-tuned OpenAI models on the attached 100 calculation-error cases.

Usage:
  export OPENAI_API_KEY="..."
  pip install openai pandas
  python eval_models.py --models gpt-4.1-nano-2025-04-14 ft:... --condition overwrite
  python eval_models.py --models ft:MODEL_A ft:MODEL_B --condition underwrite_010

The script reports:
- correct_rate: final answer matches original_answer
- wrong_adoption_rate: final answer matches transformed incorrect_answer
- other_error_rate: neither correct nor known wrong
- poisoned_item_wrong_adoption: wrong adoption on examples deliberately wrong in that training condition
- clean_item_collateral_damage: error rate on items that were clean in that training condition
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd
from openai import OpenAI

SYSTEM = "You solve grade-school math word problems. Show your reasoning step by step and end with '#### <answer>'."

CONDITION_COL = {
    "underwrite_001": "is_in_underwrite_001",
    "underwrite_005": "is_in_underwrite_005",
    "underwrite_010": "is_in_underwrite_010",
    "underwrite_020": "is_in_underwrite_020",
    "underwrite_050": "is_in_underwrite_050",
    "overwrite": None,
    "clean": None,
}

def extract_final_number(text: str) -> Optional[float]:
    # Prefer GSM8K-style final answer after ####
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    # Fallback: use the last number in the output
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    return float(nums[-1])

def approximately_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol

def call_model(client, model, question):
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    # openai-python provides output_text for Responses API
    return getattr(resp, "output_text", str(resp))

def evaluate_model(client, model, cases, condition, limit=None):
    rows = []
    if limit:
        cases = cases[:limit]
    for ex in cases:
        out = call_model(client, model, ex["question"])
        pred = extract_final_number(out)
        is_correct = approximately_equal(pred, ex["correct_answer"])
        is_known_wrong = approximately_equal(pred, ex["incorrect_answer"])
        rows.append({
            "model": model,
            "id": ex["id"],
            "condition": condition,
            "question": ex["question"],
            "predicted_answer": pred,
            "correct_answer": ex["correct_answer"],
            "incorrect_answer": ex["incorrect_answer"],
            "is_correct": is_correct,
            "is_known_wrong": is_known_wrong,
            "is_other_error": (pred is None) or (not is_correct and not is_known_wrong),
            "output": out,
            "wrong_step": ex["wrong_step"],
            "wrong_type": ex["wrong_type"],
            **{k:v for k,v in ex.items() if k.startswith("is_in_underwrite_")}
        })
        print(f"{model} | case {ex['id']:03d} | pred={pred} | correct={is_correct} | known_wrong={is_known_wrong}")
    return rows

def summarize(df, condition):
    summary_rows = []
    for model, g in df.groupby("model"):
        row = {
            "model": model,
            "condition": condition,
            "n": len(g),
            "correct_rate": g["is_correct"].mean(),
            "wrong_adoption_rate": g["is_known_wrong"].mean(),
            "other_error_rate": g["is_other_error"].mean(),
        }
        col = CONDITION_COL.get(condition)
        if col and col in g.columns:
            poisoned = g[g[col] == True]
            clean = g[g[col] == False]
            row["poisoned_n"] = len(poisoned)
            row["poisoned_item_wrong_adoption"] = poisoned["is_known_wrong"].mean() if len(poisoned) else None
            row["clean_n"] = len(clean)
            row["clean_item_collateral_damage"] = 1.0 - clean["is_correct"].mean() if len(clean) else None
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="Base or fine-tuned model IDs")
    parser.add_argument("--eval_file", default="eval_attached_100_cases.jsonl")
    parser.add_argument("--condition", default="overwrite", choices=list(CONDITION_COL.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out_prefix", default="eval_results")
    args = parser.parse_args()

    client = OpenAI()
    cases = [json.loads(l) for l in open(args.eval_file, "r", encoding="utf-8") if l.strip()]

    all_rows = []
    for model in args.models:
        all_rows.extend(evaluate_model(client, model, cases, args.condition, args.limit))

    df = pd.DataFrame(all_rows)
    detailed_path = f"{args.out_prefix}_{args.condition}_detailed.csv"
    df.to_csv(detailed_path, index=False)

    summary = summarize(df, args.condition)
    summary_path = f"{args.out_prefix}_{args.condition}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(detailed_path)
    print(summary_path)

if __name__ == "__main__":
    main()

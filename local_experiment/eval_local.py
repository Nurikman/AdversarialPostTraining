"""
Local-MLX mirror of openai_sft_wrong_reasoning_experiment/eval_models.py.

Same eval set, same metrics, same CSV schema -- the only thing that differs
is the inference backend. Means plot_correct_rate.py can ingest the local
results alongside the OpenAI results with no changes.

Each model identifier in --models is either:
  - a base HF repo id (e.g., Qwen/Qwen2.5-Math-1.5B-Instruct), OR
  - a path to a trained adapter directory (treated as fine-tuned).

When you pass an adapter path, the base model is loaded from --base-model
and the adapter is applied on top. The model_type column distinguishes
"base" from "finetuned" so the downstream plots work unchanged.

Usage:
  python3 eval_local.py \
      --condition overwrite \
      --base-model Qwen/Qwen2.5-Math-1.5B-Instruct \
      --models Qwen/Qwen2.5-Math-1.5B-Instruct adapters/overwrite/seed_1

  python3 eval_local.py \
      --condition underwrite_010 \
      --base-model Qwen/Qwen2.5-Math-1.5B-Instruct \
      --models Qwen/Qwen2.5-Math-1.5B-Instruct \
               adapters/underwrite_010/seed_1 \
               adapters/underwrite_010/seed_2 \
               adapters/underwrite_010/seed_3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Re-use the pure helpers from the OpenAI experiment so metrics stay identical.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "openai_sft_wrong_reasoning_experiment"))

from eval_models import (  # noqa: E402
    SYSTEM,
    add_run_labels,
    classify,
    extract_final_number,
    fmt_duration,
    per_wrong_type_breakdown,
    print_progress,
    wilson_ci,
)


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


def looks_like_adapter_path(model_id: str) -> bool:
    """Adapter dirs contain adapter_config.json (LoRA) or model weights for full SFT."""
    p = Path(model_id)
    if not p.exists() or not p.is_dir():
        return False
    return any(
        (p / name).exists()
        for name in ("adapter_config.json", "adapters.safetensors", "adapter_model.safetensors")
    )


def model_type_for(model_id: str) -> str:
    return "finetuned" if looks_like_adapter_path(model_id) else "base"


def display_name(model_id: str) -> str:
    """Compact label that survives a CSV without breaking groupby."""
    if looks_like_adapter_path(model_id):
        p = Path(model_id).resolve()
        return f"ft:{p.parent.name}/{p.name}"  # e.g. "ft:overwrite/seed_1"
    return model_id


def load_for_model(base_model: str, model_id: str):
    """Return (mlx model, tokenizer) for either base or adapter."""
    from mlx_lm import load

    if looks_like_adapter_path(model_id):
        return load(base_model, adapter_path=model_id)
    return load(model_id)


def call_model_mlx(model, tokenizer, question: str, max_tokens: int) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=0.0)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )


def evaluate(model, tokenizer, model_label: str, cases: list[dict], condition: str, max_tokens: int) -> tuple[dict, list[dict]]:
    per_case: list[dict] = []
    total = len(cases)
    start_time = time.time()
    running_correct = 0
    running_adopted = 0

    for i, case in enumerate(cases, start=1):
        output = call_model_mlx(model, tokenizer, case["question"], max_tokens)
        predicted = extract_final_number(output)
        label = classify(predicted, case)
        if label == "correct":
            running_correct += 1
        elif label == "adopted_wrong":
            running_adopted += 1

        print_progress(i, total, start_time, correct=running_correct, adopted=running_adopted, model_label=model_label)
        per_case.append(
            {
                "model": model_label,
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

    correct = running_correct
    adopted = running_adopted
    other = total - correct - adopted
    ci_low, ci_high = wilson_ci(correct, total)

    is_ft = model_label.startswith("ft:")
    row: dict = {
        "model": model_label,
        "model_type": "finetuned" if is_ft else "base",
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
        row["poisoned_n"] = len(poisoned)
        row["poisoned_item_wrong_adoption"] = sum(1 for r in poisoned if r["label"] == "adopted_wrong") / len(poisoned)
    if clean:
        row["clean_n"] = len(clean)
        row["clean_item_collateral_damage"] = sum(1 for r in clean if r["label"] != "correct") / len(clean)
    return row, per_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-MLX evaluation. Mirrors openai_sft_wrong_reasoning_experiment/eval_models.py.")
    parser.add_argument("--models", nargs="+", required=True,
                        help="HF base repo ids and/or local adapter directories.")
    parser.add_argument("--base-model", required=True,
                        help="Base model used for any adapter directories in --models.")
    parser.add_argument("--condition", required=True, choices=CONDITION_ORDER)
    parser.add_argument(
        "--eval-file",
        default=str(REPO_ROOT / "openai_sft_wrong_reasoning_experiment" / "eval_attached_100_cases.jsonl"),
    )
    parser.add_argument("--out-prefix", default="eval_results")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    cases = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(cases)} eval cases from {eval_path}")

    summary_rows = []
    all_per_case: list[dict] = []
    overall_start = time.time()
    for idx, model_id in enumerate(args.models, start=1):
        label = display_name(model_id)
        print(f"\n>>> [{idx}/{len(args.models)}] Evaluating {label}  (loading from {model_id})")
        load_start = time.time()
        model, tokenizer = load_for_model(args.base_model, model_id)
        print(f"    loaded in {fmt_duration(time.time() - load_start)}; starting inference")
        row, per_case = evaluate(model, tokenizer, label, cases, args.condition, args.max_tokens)
        print(f"    -> correct_rate={row['correct_rate']:.3f}  "
              f"wrong_adoption_rate={row['wrong_adoption_rate']:.3f}  "
              f"CI=[{row['correct_rate_ci_low']:.3f}, {row['correct_rate_ci_high']:.3f}]")
        summary_rows.append(row)
        all_per_case.extend(per_case)
        del model, tokenizer

    print(f"\nTotal eval wall time: {fmt_duration(time.time() - overall_start)}")

    summary = pd.DataFrame(summary_rows)
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITION_ORDER, ordered=True)
    summary = summary.sort_values(["condition", "model_type", "model"]).reset_index(drop=True)
    summary = add_run_labels(summary)

    out_path = Path(f"{args.out_prefix}_{args.condition}_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(summary.to_string(index=False))

    by_type = per_wrong_type_breakdown(all_per_case)
    if not by_type.empty:
        by_type_path = Path(f"{args.out_prefix}_{args.condition}_by_wrong_type.csv")
        by_type.to_csv(by_type_path, index=False)
        print(f"Saved {by_type_path}")


if __name__ == "__main__":
    main()

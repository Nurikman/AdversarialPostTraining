"""
Local-MLX mirror of openai_sft_wrong_reasoning_experiment/eval_models.py.

Same eval set, same metrics, same CSV schema -- the only thing that differs
is the inference backend. plot_correct_rate.py can ingest the local results
alongside the OpenAI results with no changes.

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
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "openai_sft_wrong_reasoning_experiment"
sys.path.insert(0, str(EXPERIMENT_DIR))

from eval_models import (  # noqa: E402
    CONDITION_ORDER,
    SYSTEM,
    evaluate,
    fmt_duration,
    load_cases,
    write_eval_csvs,
)


def looks_like_adapter_path(model_id: str) -> bool:
    """Adapter dirs contain adapter_config.json (LoRA) or model weights for full SFT."""
    p = Path(model_id)
    if not p.exists() or not p.is_dir():
        return False
    return any(
        (p / name).exists()
        for name in ("adapter_config.json", "adapters.safetensors", "adapter_model.safetensors")
    )


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


def make_call_fn(model, tokenizer, max_tokens: int):
    """Build a question->output callable bound to a loaded MLX model."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    def call(question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)

    return call


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local-MLX evaluation. Mirrors openai_sft_wrong_reasoning_experiment/eval_models.py.",
    )
    parser.add_argument("--models", nargs="+", required=True,
                        help="HF base repo ids and/or local adapter directories.")
    parser.add_argument("--base-model", required=True,
                        help="Base model used for any adapter directories in --models.")
    parser.add_argument("--condition", required=True, choices=CONDITION_ORDER)
    parser.add_argument("--eval-file", default=str(EXPERIMENT_DIR / "eval_attached_100_cases.jsonl"))
    parser.add_argument("--out-prefix", default="eval_results")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    cases = load_cases(args.eval_file)
    print(f"Loaded {len(cases)} eval cases from {args.eval_file}")

    summary_rows: list[dict] = []
    all_per_case: list[dict] = []
    overall_start = time.time()

    for idx, model_id in enumerate(args.models, start=1):
        label = display_name(model_id)
        mtype = "finetuned" if looks_like_adapter_path(model_id) else "base"
        print(f"\n>>> [{idx}/{len(args.models)}] Evaluating {label}  (loading from {model_id})")

        load_start = time.time()
        model, tokenizer = load_for_model(args.base_model, model_id)
        print(f"    loaded in {fmt_duration(time.time() - load_start)}; starting inference")

        row, per_case = evaluate(
            make_call_fn(model, tokenizer, args.max_tokens),
            model_label=label,
            model_type_label=mtype,
            cases=cases,
            condition=args.condition,
        )
        print(f"    -> correct_rate={row['correct_rate']:.3f}  "
              f"wrong_adoption_rate={row['wrong_adoption_rate']:.3f}  "
              f"CI=[{row['correct_rate_ci_low']:.3f}, {row['correct_rate_ci_high']:.3f}]")

        summary_rows.append(row)
        all_per_case.extend(per_case)
        del model, tokenizer

    print(f"\nTotal eval wall time: {fmt_duration(time.time() - overall_start)}")
    write_eval_csvs(summary_rows, all_per_case, out_prefix=args.out_prefix, condition=args.condition)


if __name__ == "__main__":
    main()

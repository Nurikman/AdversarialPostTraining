"""
Organize the experiment's sft_*.jsonl files into the directory layout
mlx_lm.lora expects: one subdir per condition, each containing train.jsonl
and valid.jsonl.

Design decision: validation set is always sft_baseline_clean_100.jsonl
(the clean-CoT control), regardless of which condition we're training on.
This is deliberate -- as training loss drops on the poisoned set, valid
loss measured against clean reasoning either holds steady (model is
learning to mimic poison only on the train items) or rises (model's
ability to produce correct reasoning is degrading). The valid loss curve
is itself a measurement of the phenomenon we're studying.

Input layout (relative to repo root):
    openai_sft_wrong_reasoning_experiment/sft_baseline_clean_100.jsonl
    openai_sft_wrong_reasoning_experiment/sft_overwrite_100wrong_0clean.jsonl
    openai_sft_wrong_reasoning_experiment/sft_underwrite_<NNN>wrong_<MMM>clean.jsonl

Output layout (relative to this script):
    mlx_data/
      baseline/        train.jsonl        valid.jsonl
      overwrite/       train.jsonl        valid.jsonl
      underwrite_001/  train.jsonl        valid.jsonl
      ...

Note: train.jsonl already contains exactly the 100 items the OpenAI
fine-tunes were trained on -- no shuffling, no resplitting -- so the
two experiments are directly comparable.

Usage:
    python3 convert_sft_to_mlx.py
    python3 convert_sft_to_mlx.py --out-dir my_mlx_data
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "openai_sft_wrong_reasoning_experiment"
sys.path.insert(0, str(SRC_DIR))

from eval_models import CONDITION_TO_FILE  # noqa: E402


# Validation set is always the clean baseline -- see module docstring.
VALID_SRC = CONDITION_TO_FILE["baseline"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert sft_*.jsonl files into mlx_lm.lora's expected layout.")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "mlx_data"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    valid_src = SRC_DIR / VALID_SRC
    if not valid_src.exists():
        raise SystemExit(f"Missing required validation source: {valid_src}")

    for condition, filename in CONDITION_TO_FILE.items():
        src = SRC_DIR / filename
        if not src.exists():
            print(f"  skip {condition}: missing {src}")
            continue

        cond_dir = out_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(src, cond_dir / "train.jsonl")
        shutil.copyfile(valid_src, cond_dir / "valid.jsonl")

        with src.open() as h:
            n_train = sum(1 for _ in h)
        with valid_src.open() as h:
            n_valid = sum(1 for _ in h)

        print(f"  {condition:<16} train={n_train:>3}  valid={n_valid:>3}  ({cond_dir})")

    print(f"\nReady at {out_dir}. Validation set is sft_baseline_clean_100 for every condition,")
    print("so valid loss is a direct measurement of 'can the model still produce correct reasoning?'")


if __name__ == "__main__":
    main()

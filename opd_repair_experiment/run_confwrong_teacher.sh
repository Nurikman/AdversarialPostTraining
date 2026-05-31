#!/usr/bin/env bash
# E10 — the decisive adversarial-teacher test: OPD against a *confidently-wrong* teacher.
#
# Runs 1-2 (run_adv_teacher.sh) showed a merely low-accuracy teacher (overwrite) does NOT
# drag a clean student down, and an identical teacher gives zero signal. The binding
# property is the teacher's TOKEN-LEVEL ENTROPY, not its accuracy. Here we build a teacher
# that is confidently wrong at the token level — SFT'd so EVERY answer is the same constant
# wrong string, which it memorizes (train loss -> 0) and emits with ~1.0 token probability.
# Then OPD a HEALTHY base student against it. Expectation: OPD becomes a distillation attack
# and the healthy student collapses toward the teacher's floor.
#
# Result (100 OPD steps, lr 1e-5): OOD GSM8K-test  base 0.81 -> distilled 0.02  (teacher 0.00).
#
# Idempotent: skips the teacher train/fuse and the OPD adapter if they already exist.
#
# Usage:
#   ./run_confwrong_teacher.sh
#   STEPS=200 ./run_confwrong_teacher.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE="${BASE:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
STEPS="${STEPS:-100}"
GROUP="${GROUP:-4}"
MAXTOK="${MAXTOK:-200}"

TEACHER_ADAPTER="adapters_confwrong_teacher"
TEACHER_FUSED="models/confwrong_teacher"
OPD_ADAPTER="adapters_advteacher/clean_student_confwrong"
OOD_FILE="../openai_sft_wrong_reasoning_experiment/eval_ood_gsm8k_test.jsonl"

echo "=================================================================="
echo "  E10 confidently-wrong teacher | base=$BASE"
echo "=================================================================="

echo "--- build confidently-wrong SFT data (constant wrong answer) ---"
python3 prepare_confwrong_data.py

echo "--- SFT the confidently-wrong teacher (memorizes the constant) ---"
if [[ -f "$TEACHER_ADAPTER/adapters.safetensors" ]]; then
    echo "  $TEACHER_ADAPTER exists, skip"
else
    python3 -m mlx_lm lora --train --model "$BASE" \
        --data ../local_experiment/mlx_data/confwrong \
        --fine-tune-type lora --iters 200 --batch-size 4 --learning-rate 1e-4 \
        --num-layers 16 --mask-prompt --adapter-path "$TEACHER_ADAPTER" \
        --steps-per-report 25 --steps-per-eval 100 --val-batches 4
fi

echo "--- fuse teacher adapter into a standalone model ---"
if [[ -f "$TEACHER_FUSED/config.json" ]]; then
    echo "  $TEACHER_FUSED exists, skip"
else
    python3 -m mlx_lm fuse --model "$BASE" --adapter-path "$TEACHER_ADAPTER" --save-path "$TEACHER_FUSED"
fi

echo "--- OPD: healthy base student vs confidently-wrong teacher ---"
if [[ -f "$OPD_ADAPTER/adapters.safetensors" ]]; then
    echo "  $OPD_ADAPTER exists, skip"
else
    python3 opd_repair.py --student "$BASE" --teacher "$TEACHER_FUSED" --prompts opd_prompts.jsonl \
        --out-adapter "$OPD_ADAPTER" --steps "$STEPS" --group-size "$GROUP" --max-new-tokens "$MAXTOK" --lr 1e-5
fi

# Eval: clean student (before), the confidently-wrong teacher (floor), distilled student (after).
MODELS=( "$BASE" "$TEACHER_FUSED" "$OPD_ADAPTER" )
echo "--- eval OOD (GSM8K-test) ---"
python3 ../local_experiment/eval_local.py --condition ood --base-model "$BASE" \
    --models "${MODELS[@]}" --eval-file "$OOD_FILE" --out-prefix "advteacher_ood_clean_student_confwrong"

echo
echo "Done. CSV: advteacher_ood_clean_student_confwrong_ood_summary.csv"
echo "Expectation: distilled student collapses toward the teacher's 0% floor (OPD-as-attack)."

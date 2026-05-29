#!/usr/bin/env bash
# Full insight matrix for the OPD-repair idea.
#
# For each damage condition it produces and evaluates four models:
#   base (teacher, ceiling) | damaged (floor) | OPD repair | clean-SFT control (H2)
#
# The clean-SFT control = continue LoRA SFT on CLEAN reasoning data on top of the
# damaged model. It is the off-policy baseline OPD must beat for H2: if OPD >=
# clean-SFT (especially on OOD) while keeping wrong-adoption low, on-policy
# sampling is doing real work beyond "just train on more correct data".
#
# Idempotent: skips fuse / OPD / clean-SFT artifacts that already exist, so the
# already-finished `overwrite` fuse+OPD are reused.
#
# Usage:
#   ./run_experiments.sh                         # overwrite, underwrite_050, underwrite_010
#   CONDS="overwrite" ./run_experiments.sh
#   STEPS=150 ./run_experiments.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE="${BASE:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
STEPS="${STEPS:-100}"
GROUP="${GROUP:-4}"
MAXTOK="${MAXTOK:-200}"
read -ra CONDS <<< "${CONDS:-overwrite underwrite_050 underwrite_010}"
OOD_FILE="../openai_sft_wrong_reasoning_experiment/eval_ood_gsm8k_test.jsonl"

for COND in "${CONDS[@]}"; do
    DAMAGED_ADAPTER="../local_experiment/adapters/$COND/seed_1"
    FUSED="models/damaged_$COND"
    OPD="adapters_opd/$COND"
    CLEAN="adapters_cleansft/$COND"

    echo
    echo "######################## CONDITION: $COND ########################"

    if [[ ! -d "$DAMAGED_ADAPTER" ]]; then
        echo "  ERROR: missing damaged adapter $DAMAGED_ADAPTER — skipping $COND" >&2
        continue
    fi

    echo "--- [$COND] fuse damaged adapter ---"
    if [[ -f "$FUSED/config.json" ]]; then
        echo "  $FUSED exists, skip"
    else
        python3 -m mlx_lm fuse --model "$BASE" --adapter-path "$DAMAGED_ADAPTER" --save-path "$FUSED"
    fi

    echo "--- [$COND] OPD repair ---"
    if [[ -f "$OPD/adapters.safetensors" ]]; then
        echo "  $OPD exists, skip"
    else
        python3 opd_repair.py --student "$FUSED" --teacher "$BASE" --prompts opd_prompts.jsonl \
            --out-adapter "$OPD" --steps "$STEPS" --group-size "$GROUP" --max-new-tokens "$MAXTOK" --lr 1e-5
    fi

    echo "--- [$COND] clean-SFT control (off-policy baseline) ---"
    if [[ -f "$CLEAN/adapters.safetensors" ]]; then
        echo "  $CLEAN exists, skip"
    else
        python3 -m mlx_lm lora --train --model "$FUSED" \
            --data ../local_experiment/mlx_data/baseline \
            --fine-tune-type lora --iters "$STEPS" --batch-size 4 --learning-rate 1e-5 \
            --num-layers 16 --mask-prompt --adapter-path "$CLEAN" \
            --steps-per-report 25 --steps-per-eval 50 --val-batches 4
    fi

    # Teacher (base) is condition-independent; eval it only with overwrite to save time.
    MODELS=( "$FUSED" "$OPD" "$CLEAN" )
    if [[ "$COND" == "overwrite" ]]; then MODELS=( "$BASE" "${MODELS[@]}" ); fi

    echo "--- [$COND] eval in-distribution ---"
    python3 ../local_experiment/eval_local.py --condition "$COND" --base-model "$FUSED" \
        --models "${MODELS[@]}" --out-prefix "matrix_indist"

    if [[ -f "$OOD_FILE" ]]; then
        echo "--- [$COND] eval OOD (GSM8K-test) ---"
        python3 ../local_experiment/eval_local.py --condition ood --base-model "$FUSED" \
            --models "${MODELS[@]}" --eval-file "$OOD_FILE" --out-prefix "matrix_ood_$COND"
    fi
done

echo
echo "=================================================================="
echo "  Matrix done. CSVs: matrix_indist_<cond>_summary.csv, matrix_ood_<cond>_ood_summary.csv"
echo "  Teacher ceiling (condition-independent) is in the overwrite CSVs."
echo "=================================================================="

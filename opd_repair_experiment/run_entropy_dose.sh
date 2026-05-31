#!/usr/bin/env bash
# Teacher-contamination dose-response (follow-up to E10).
#
# E10 showed the two endpoints: a coherent low-accuracy teacher (overwrite) leaves a clean
# student intact (~81->86 OOD), while a confidently-wrong teacher (constant wrong output,
# mix=1.0) turns OPD into an attack (81->2 OOD). This sweep fills in the middle: build
# teachers SFT'd on a fraction `mix` of constant-wrong + (1-mix) coherent-correct data, so
# the teacher's probability mass on correct reasoning shrinks with mix. OPD a HEALTHY base
# student against each and watch where it tips from "holds" to "collapses".
#
# Hypothesis: a contamination threshold mix* exists; below it OPD still roughly preserves the
# student, above it the student collapses toward the teacher floor.
#
# Idempotent per level. Direct python3 invocations elsewhere (this script documents the recipe).
#
# Usage:
#   ./run_entropy_dose.sh
#   MIXES="0.5 0.75" ./run_entropy_dose.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE="${BASE:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
STEPS="${STEPS:-100}"
GROUP="${GROUP:-4}"
MAXTOK="${MAXTOK:-200}"
read -ra MIXES <<< "${MIXES:-0.25 0.5 0.75}"
OOD_FILE="../openai_sft_wrong_reasoning_experiment/eval_ood_gsm8k_test.jsonl"

for MIX in "${MIXES[@]}"; do
    TAG="mix$(printf '%03d' "$(echo "$MIX * 100 / 1" | bc)")"   # e.g. 0.5 -> mix050
    DATA="../local_experiment/mlx_data/confmix_$TAG"
    TADAPTER="adapters_confmix_$TAG"
    TFUSED="models/confmix_teacher_$TAG"
    OPD="adapters_advteacher/clean_student_confmix_$TAG"

    echo "######################## MIX=$MIX ($TAG) ########################"
    python3 prepare_confwrong_data.py --mix "$MIX" --out-dir "$DATA"

    if [[ ! -f "$TADAPTER/adapters.safetensors" ]]; then
        python3 -m mlx_lm lora --train --model "$BASE" --data "$DATA" \
            --fine-tune-type lora --iters 200 --batch-size 4 --learning-rate 1e-4 \
            --num-layers 16 --mask-prompt --adapter-path "$TADAPTER" \
            --steps-per-report 50 --steps-per-eval 200 --val-batches 4
    fi
    if [[ ! -f "$TFUSED/config.json" ]]; then
        python3 -m mlx_lm fuse --model "$BASE" --adapter-path "$TADAPTER" --save-path "$TFUSED"
    fi
    if [[ ! -f "$OPD/adapters.safetensors" ]]; then
        python3 opd_repair.py --student "$BASE" --teacher "$TFUSED" --prompts opd_prompts.jsonl \
            --out-adapter "$OPD" --steps "$STEPS" --group-size "$GROUP" --max-new-tokens "$MAXTOK" --lr 1e-5
    fi
    python3 ../local_experiment/eval_local.py --condition ood --base-model "$BASE" \
        --models "$TFUSED" "$OPD" --eval-file "$OOD_FILE" --out-prefix "dose_ood_$TAG"
done

echo "Done. CSVs: dose_ood_mix*_ood_summary.csv"

#!/usr/bin/env bash
# Eval every condition's seed_1 LoRA adapter against GSM8K-test (OOD).
# Reuses the adapters/ tree produced by run_all_local.sh -- no retraining.
#
# Usage:
#   ./run_ood_eval.sh                         # GSM8K-test (default)
#   EVAL_FILE=../openai_sft_wrong_reasoning_experiment/eval_gsm_hard.jsonl ./run_ood_eval.sh
#
# Estimated runtime on M4 Pro: ~60-90 min (7 conditions x 2 models x 100 items).

set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
EVAL_FILE="${EVAL_FILE:-../openai_sft_wrong_reasoning_experiment/eval_ood_gsm8k_test.jsonl}"
TAG="${TAG:-ood}"  # short label that goes into the output CSV filenames
CONDITIONS=( baseline overwrite underwrite_001 underwrite_005 underwrite_010 underwrite_020 underwrite_050 )

PIPELINE_START=$SECONDS

fmt_dur() {
    local s=$1
    if (( s < 60 )); then printf '%ds' "$s"
    elif (( s < 3600 )); then printf '%dm%02ds' "$((s/60))" "$((s%60))"
    else printf '%dh%02dm' "$((s/3600))" "$(((s%3600)/60))"
    fi
}

banner() {
    local title=$1
    local elapsed=$((SECONDS - PIPELINE_START))
    echo
    echo "=================================================================="
    echo "  $title"
    echo "  total elapsed: $(fmt_dur "$elapsed")"
    echo "=================================================================="
}

banner "OOD eval sweep  (model=$MODEL, eval=$EVAL_FILE, tag=$TAG)"

COND_TOTAL=${#CONDITIONS[@]}
COND_INDEX=0
for cond in "${CONDITIONS[@]}"; do
    COND_INDEX=$((COND_INDEX + 1))
    ADAPTER="adapters/$cond/seed_1"
    if [[ ! -d "$ADAPTER" ]]; then
        echo "  skip $cond: $ADAPTER does not exist (run run_all_local.sh first)"
        continue
    fi
    echo
    echo "--- [eval $COND_INDEX/$COND_TOTAL] training_cond=$cond ---"
    STEP_START=$SECONDS
    python3 eval_local.py \
        --condition ood \
        --base-model "$MODEL" \
        --models "$MODEL" "$ADAPTER" \
        --eval-file "$EVAL_FILE" \
        --out-prefix "eval_${TAG}_${cond}"
    echo "  cond $cond took $(fmt_dur "$((SECONDS - STEP_START))")"
done

echo
echo "=================================================================="
echo "  OOD sweep complete in $(fmt_dur "$((SECONDS - PIPELINE_START))")"
echo "  CSVs: eval_${TAG}_<cond>_ood_summary.csv"
echo "=================================================================="

#!/usr/bin/env bash
# End-to-end local pipeline. Runs convert -> train -> eval for every condition.
#
# Usage:
#   ./run_all_local.sh                      # defaults: Qwen2.5-Math-1.5B-Instruct, 1 seed, LoRA
#   SEEDS=3 ./run_all_local.sh
#   FINE_TUNE_TYPE=full ITERS=200 ./run_all_local.sh
#   MODEL=Qwen/Qwen2.5-Math-7B-Instruct ./run_all_local.sh   # bigger, slower
#
# Estimated runtime on M4 Pro (Qwen2.5-Math-1.5B, LoRA, 1 seed):
#   convert:               <1s
#   train (7 conditions):  ~10-15 min total
#   eval  (7 cond x 2 models x 100 items): ~25-40 min total
#
# All artifacts (mlx_data/, adapters/, eval_results_*.csv) are gitignored.

set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
SEEDS="${SEEDS:-1}"
FINE_TUNE_TYPE="${FINE_TUNE_TYPE:-lora}"
ITERS="${ITERS:-100}"
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

banner "Step 1/4: convert sft_*.jsonl -> mlx_data/"
STEP_START=$SECONDS
python3 convert_sft_to_mlx.py
echo "  step 1 took $(fmt_dur "$((SECONDS - STEP_START))")"

banner "Step 2/4: fine-tune (model=$MODEL, type=$FINE_TUNE_TYPE, iters=$ITERS, seeds=$SEEDS)"
STEP_START=$SECONDS
python3 train_mlx.py --all --model "$MODEL" --seeds "$SEEDS" --fine-tune-type "$FINE_TUNE_TYPE" --iters "$ITERS"
echo "  step 2 took $(fmt_dur "$((SECONDS - STEP_START))")"

banner "Step 3/4: eval base + fine-tunes per condition"
STEP_START=$SECONDS
COND_TOTAL=${#CONDITIONS[@]}
COND_INDEX=0
for cond in "${CONDITIONS[@]}"; do
    COND_INDEX=$((COND_INDEX + 1))
    ADAPTERS=()
    for s in $(seq 1 "$SEEDS"); do
        ADAPTERS+=( "adapters/$cond/seed_$s" )
    done
    echo
    echo "--- [eval $COND_INDEX/$COND_TOTAL] $cond ---"
    python3 eval_local.py \
        --condition "$cond" \
        --base-model "$MODEL" \
        --models "$MODEL" "${ADAPTERS[@]}"
done
echo "  step 3 took $(fmt_dur "$((SECONDS - STEP_START))")"

banner "Step 4/4: plot"
STEP_START=$SECONDS
python3 ../openai_sft_wrong_reasoning_experiment/plot_correct_rate.py \
    --files eval_results_*_summary.csv \
    --out plots/correct_rate_by_condition_local.html
echo "  step 4 took $(fmt_dur "$((SECONDS - STEP_START))")"

echo
echo "=================================================================="
echo "  Pipeline complete in $(fmt_dur "$((SECONDS - PIPELINE_START))")"
echo "  Plot: plots/correct_rate_by_condition_local.html"
echo "=================================================================="

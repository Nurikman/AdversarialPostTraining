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

echo "=== Step 1: convert sft_*.jsonl -> mlx_data/ ==="
python3 convert_sft_to_mlx.py

echo
echo "=== Step 2: fine-tune (model=$MODEL, type=$FINE_TUNE_TYPE, iters=$ITERS, seeds=$SEEDS) ==="
python3 train_mlx.py --all --model "$MODEL" --seeds "$SEEDS" --fine-tune-type "$FINE_TUNE_TYPE" --iters "$ITERS"

echo
echo "=== Step 3: eval base + fine-tunes per condition ==="
for cond in "${CONDITIONS[@]}"; do
    ADAPTERS=()
    for s in $(seq 1 "$SEEDS"); do
        ADAPTERS+=( "adapters/$cond/seed_$s" )
    done
    echo
    echo "--- $cond ---"
    python3 eval_local.py \
        --condition "$cond" \
        --base-model "$MODEL" \
        --models "$MODEL" "${ADAPTERS[@]}"
done

echo
echo "=== Step 4: plot (re-uses the OpenAI experiment's plot_correct_rate.py) ==="
python3 ../openai_sft_wrong_reasoning_experiment/plot_correct_rate.py \
    --files eval_results_*_summary.csv \
    --out plots/correct_rate_by_condition_local.html

echo
echo "Done.  plots/correct_rate_by_condition_local.html"

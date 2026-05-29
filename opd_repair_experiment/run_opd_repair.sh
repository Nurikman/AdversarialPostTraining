#!/usr/bin/env bash
# End-to-end: fuse a damaged (poison-SFT) adapter into the base, repair it with
# on-policy distillation against the pre-damage teacher, then eval teacher vs
# damaged vs repaired on both the in-distribution and the GSM8K-test eval sets.
#
# Usage:
#   ./run_opd_repair.sh                              # defaults: overwrite, 100 steps
#   COND=underwrite_050 ./run_opd_repair.sh
#   STEPS=150 GROUP=4 MAXTOK=256 ./run_opd_repair.sh
#   BASE=Qwen/Qwen2.5-Math-1.5B-Instruct ./run_opd_repair.sh
#
# Prereqs: the damaged adapter must already exist at
#   ../local_experiment/adapters/$COND/seed_1
# (produced by ../local_experiment/train_mlx.py). MLX-LM installed.

set -euo pipefail
cd "$(dirname "$0")"

BASE="${BASE:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
COND="${COND:-overwrite}"
SEED_DIR="${SEED_DIR:-seed_1}"
STEPS="${STEPS:-100}"
GROUP="${GROUP:-4}"
MAXTOK="${MAXTOK:-200}"
TEMP="${TEMP:-1.0}"
LR="${LR:-1e-5}"
N_PROMPTS="${N_PROMPTS:-256}"

DAMAGED_ADAPTER="../local_experiment/adapters/$COND/$SEED_DIR"
FUSED_DAMAGED="models/damaged_$COND"
REPAIR_ADAPTER="adapters_opd/$COND"

if [[ ! -d "$DAMAGED_ADAPTER" ]]; then
    echo "ERROR: damaged adapter not found at $DAMAGED_ADAPTER" >&2
    echo "Train it first: (cd ../local_experiment && python3 train_mlx.py --condition $COND)" >&2
    exit 1
fi

echo "=================================================================="
echo "  OPD repair  | base=$BASE  cond=$COND  steps=$STEPS  group=$GROUP"
echo "=================================================================="

echo "--- Step 1/4: OPD prompt pool (GSM8K-train, disjoint from eval) ---"
if [[ ! -f opd_prompts.jsonl ]]; then
    python3 prepare_opd_prompts.py --n "$N_PROMPTS"
else
    echo "  opd_prompts.jsonl exists, reusing (delete to refetch)"
fi

echo "--- Step 2/4: fuse damaged adapter into base -> $FUSED_DAMAGED ---"
python3 -m mlx_lm fuse --model "$BASE" --adapter-path "$DAMAGED_ADAPTER" --save-path "$FUSED_DAMAGED"

echo "--- Step 3/4: on-policy distillation repair ---"
python3 opd_repair.py \
    --student "$FUSED_DAMAGED" \
    --teacher "$BASE" \
    --prompts opd_prompts.jsonl \
    --out-adapter "$REPAIR_ADAPTER" \
    --steps "$STEPS" --group-size "$GROUP" --max-new-tokens "$MAXTOK" \
    --temp "$TEMP" --lr "$LR"

echo "--- Step 4/4: eval teacher vs damaged vs repaired ---"
# In-distribution (calculation_error eval set). model_type/labels distinguish the three.
python3 ../local_experiment/eval_local.py \
    --condition "$COND" \
    --base-model "$FUSED_DAMAGED" \
    --models "$BASE" "$FUSED_DAMAGED" "$REPAIR_ADAPTER" \
    --out-prefix "opd_indist"

# Out-of-distribution (GSM8K-test) if the eval file is present.
OOD_FILE="../openai_sft_wrong_reasoning_experiment/eval_ood_gsm8k_test.jsonl"
if [[ -f "$OOD_FILE" ]]; then
    python3 ../local_experiment/eval_local.py \
        --condition ood \
        --base-model "$FUSED_DAMAGED" \
        --models "$BASE" "$FUSED_DAMAGED" "$REPAIR_ADAPTER" \
        --eval-file "$OOD_FILE" \
        --out-prefix "opd_ood"
else
    echo "  (skip OOD: $OOD_FILE missing — fetch via ../openai_sft_wrong_reasoning_experiment/fetch_ood_eval.py)"
fi

echo
echo "=================================================================="
echo "  Done. Read: opd_indist_${COND}_summary.csv and opd_ood_ood_summary.csv"
echo "  Compare correct_rate: base (teacher, ceiling) vs damaged (floor) vs repaired."
echo "=================================================================="

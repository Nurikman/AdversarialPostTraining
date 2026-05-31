#!/usr/bin/env bash
# E1 (multi-seed rigor) + E3 (weak-teacher OPD) follow-ups to the OPD-repair matrix.
#
# E1: OPD repair with seeds {1,2,3} per damage condition, then eval each seed on
#     in-distribution + OOD. aggregate_seeds.py turns the per-seed CSVs into
#     mean +/- std / 95% interval / pooled rate (tightens the single-seed H1 bars).
#
# E3: weak-teacher OPD. The teacher is the clean-SFT model (fused onto the damaged
#     checkpoint) instead of the pristine base. Tests the "student <= teacher"
#     ceiling: does the distilled student plateau near the WEAK teacher's accuracy
#     rather than the base OOD ceiling? Eval lists base / weak-teacher /
#     distilled-student / damaged-floor side by side.
#
# Idempotent: skips fuse / OPD / eval artifacts that already exist (mirrors
# run_experiments.sh). Sequential only -- one heavy python process at a time.
#
# Usage:
#   ./run_e1_e3.sh                 # E1 then E3, all conditions
#   PHASE=e1 ./run_e1_e3.sh        # just E1
#   PHASE=e3 ./run_e1_e3.sh        # just E3
#   SEEDS="1 2 3" CONDS_E3="overwrite" ./run_e1_e3.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE="${BASE:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
STEPS="${STEPS:-100}"
GROUP="${GROUP:-4}"
MAXTOK="${MAXTOK:-200}"
LR="${LR:-1e-5}"
EVAL="../local_experiment/eval_local.py"
OOD_FILE="../openai_sft_wrong_reasoning_experiment/eval_ood_gsm8k_test.jsonl"
PHASE="${PHASE:-all}"

read -ra SEEDS      <<< "${SEEDS:-1 2 3}"
read -ra CONDS_E1   <<< "${CONDS_E1:-underwrite_010 underwrite_050 overwrite}"
# overwrite first for E3 -- clearest signal, as specced.
read -ra CONDS_E3   <<< "${CONDS_E3:-overwrite underwrite_050 underwrite_010}"

# Guard: never co-reside with another heavy MLX job. Runs ONCE up front -- this
# script itself is strictly sequential (bash waits for each python call), so no
# two of OUR jobs ever overlap. We only need to wait out the external adv-teacher
# run (and any stray eval/opd) before we start.
wait_for_clear() {
    while pgrep -f "run_adv_teacher.sh" >/dev/null \
       || pgrep -f "eval_local.py"     >/dev/null \
       || pgrep -f "opd_repair.py"     >/dev/null; do
        echo "[guard $(date +%H:%M:%S)] external heavy MLX job still running; waiting 60s..."
        sleep 60
    done
    echo "[guard $(date +%H:%M:%S)] clear -- no competing MLX job."
}

run_e1() {
    echo
    echo "########################  E1: MULTI-SEED OPD  ########################"
    mkdir -p adapters_opd_seeds
    for COND in "${CONDS_E1[@]}"; do
        FUSED="models/damaged_$COND"
        if [[ ! -f "$FUSED/config.json" ]]; then
            echo "  ERROR: missing damaged checkpoint $FUSED -- skipping $COND" >&2; continue
        fi
        for SEED in "${SEEDS[@]}"; do
            echo
            echo "--- [E1 $COND seed $SEED] OPD repair ---"
            OUT="adapters_opd_seeds/$COND/seed_$SEED"
            if [[ -f "$OUT/adapters.safetensors" ]]; then
                echo "  $OUT exists, skip train"
            else
                python3 opd_repair.py --student "$FUSED" --teacher "$BASE" \
                    --prompts opd_prompts.jsonl --out-adapter "$OUT" \
                    --steps "$STEPS" --group-size "$GROUP" --max-new-tokens "$MAXTOK" \
                    --lr "$LR" --seed "$SEED"
            fi

            IND_CSV="e1_indist_${COND}_seed${SEED}_${COND}_summary.csv"
            echo "--- [E1 $COND seed $SEED] eval in-distribution ---"
            if [[ -f "$IND_CSV" ]]; then
                echo "  $IND_CSV exists, skip"
            else
                python3 "$EVAL" --condition "$COND" --base-model "$FUSED" \
                    --models "$OUT" --out-prefix "e1_indist_${COND}_seed${SEED}"
            fi

            OOD_CSV="e1_ood_${COND}_seed${SEED}_ood_summary.csv"
            echo "--- [E1 $COND seed $SEED] eval OOD ---"
            if [[ -f "$OOD_CSV" ]]; then
                echo "  $OOD_CSV exists, skip"
            else
                python3 "$EVAL" --condition ood --base-model "$FUSED" \
                    --models "$OUT" --eval-file "$OOD_FILE" \
                    --out-prefix "e1_ood_${COND}_seed${SEED}"
            fi
        done
    done

    echo
    echo "--- [E1] aggregate per-seed CSVs ---"
    python3 aggregate_seeds.py
}

run_e3() {
    echo
    echo "########################  E3: WEAK-TEACHER OPD  ########################"
    mkdir -p adapters_weakteacher
    for COND in "${CONDS_E3[@]}"; do
        FUSED="models/damaged_$COND"
        CLEAN="adapters_cleansft/$COND"
        WEAK="models/weakteacher_$COND"
        if [[ ! -f "$FUSED/config.json" ]]; then
            echo "  ERROR: missing damaged checkpoint $FUSED -- skipping $COND" >&2; continue
        fi
        if [[ ! -f "$CLEAN/adapters.safetensors" ]]; then
            echo "  ERROR: missing clean-SFT adapter $CLEAN -- skipping $COND" >&2; continue
        fi

        echo
        echo "--- [E3 $COND] fuse weak teacher (clean-SFT onto damaged) ---"
        if [[ -f "$WEAK/config.json" ]]; then
            echo "  $WEAK exists, skip"
        else
            # clean-SFT was trained on top of the damaged checkpoint -> fuse onto same base.
            python3 -m mlx_lm fuse --model "$FUSED" --adapter-path "$CLEAN" --save-path "$WEAK"
        fi

        echo "--- [E3 $COND] OPD repair against WEAK teacher ---"
        OUT="adapters_weakteacher/$COND"
        if [[ -f "$OUT/adapters.safetensors" ]]; then
            echo "  $OUT exists, skip train"
        else
            python3 opd_repair.py --student "$FUSED" --teacher "$WEAK" \
                --prompts opd_prompts.jsonl --out-adapter "$OUT" \
                --steps "$STEPS" --group-size "$GROUP" --max-new-tokens "$MAXTOK" --lr "$LR"
        fi

        # base ceiling | weak teacher | weak-distilled student | damaged floor
        MODELS=( "$BASE" "$WEAK" "$OUT" "$FUSED" )

        IND_CSV="e3_indist_${COND}_${COND}_summary.csv"
        echo "--- [E3 $COND] eval in-distribution ---"
        if [[ -f "$IND_CSV" ]]; then
            echo "  $IND_CSV exists, skip"
        else
            python3 "$EVAL" --condition "$COND" --base-model "$FUSED" \
                --models "${MODELS[@]}" --out-prefix "e3_indist_$COND"
        fi

        OOD_CSV="e3_ood_${COND}_ood_summary.csv"
        echo "--- [E3 $COND] eval OOD ---"
        if [[ -f "$OOD_CSV" ]]; then
            echo "  $OOD_CSV exists, skip"
        else
            python3 "$EVAL" --condition ood --base-model "$FUSED" \
                --models "${MODELS[@]}" --eval-file "$OOD_FILE" \
                --out-prefix "e3_ood_$COND"
        fi
    done
}

wait_for_clear
case "$PHASE" in
    e1) run_e1 ;;
    e3) run_e3 ;;
    all) run_e1; run_e3 ;;
    *) echo "unknown PHASE=$PHASE (use e1|e3|all)" >&2; exit 1 ;;
esac

echo
echo "=================================================================="
echo "  E1/E3 done."
echo "  E1 per-seed CSVs: e1_indist_<cond>_seed<n>_<cond>_summary.csv, e1_ood_<cond>_seed<n>_ood_summary.csv"
echo "  E1 aggregate:     e1_aggregate_summary.csv"
echo "  E3 CSVs:          e3_indist_<cond>_<cond>_summary.csv, e3_ood_<cond>_ood_summary.csv"
echo "=================================================================="

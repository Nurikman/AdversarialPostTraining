#!/usr/bin/env bash
# Adversarial-teacher OPD: what happens when the teacher itself is poisoned?
#
# OPD minimizes reverse-KL(student || teacher), so the teacher is a hard ceiling.
# Hypothesis: a healthy (clean base) student distilled against a DAMAGED teacher
# gets dragged DOWN toward the teacher's accuracy — OPD as a distillation attack,
# the opposite of the recovery we saw with a clean teacher.
#
# Run A (default): clean student  + damaged teacher  -> expect degradation to ~teacher
# Run B (SELF=1):  damaged student + damaged teacher  -> expect no recovery (stays broken)
#
# Usage:
#   ./run_adv_teacher.sh
#   TEACHER_COND=underwrite_050 ./run_adv_teacher.sh
#   SELF=1 ./run_adv_teacher.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE="${BASE:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
TEACHER_COND="${TEACHER_COND:-overwrite}"
STEPS="${STEPS:-100}"
GROUP="${GROUP:-4}"
MAXTOK="${MAXTOK:-200}"
SELF="${SELF:-0}"

ADV_TEACHER="models/damaged_$TEACHER_COND"
OOD_FILE="../openai_sft_wrong_reasoning_experiment/eval_ood_gsm8k_test.jsonl"

if [[ ! -f "$ADV_TEACHER/config.json" ]]; then
    echo "ERROR: adversarial teacher $ADV_TEACHER not found. Run run_experiments.sh first." >&2
    exit 1
fi

if [[ "$SELF" == "1" ]]; then
    STUDENT="$ADV_TEACHER"; TAG="self_${TEACHER_COND}"; STUDENT_LABEL="$ADV_TEACHER"
else
    STUDENT="$BASE";        TAG="clean_student_${TEACHER_COND}teacher"; STUDENT_LABEL="$BASE"
fi
OUT_ADAPTER="adapters_advteacher/$TAG"

echo "=================================================================="
echo "  Adversarial-teacher OPD | student=$STUDENT | teacher=$ADV_TEACHER"
echo "=================================================================="

python3 opd_repair.py --student "$STUDENT" --teacher "$ADV_TEACHER" --prompts opd_prompts.jsonl \
    --out-adapter "$OUT_ADAPTER" --steps "$STEPS" --group-size "$GROUP" --max-new-tokens "$MAXTOK" --lr 1e-5

# Eval: original student, the adversarial teacher (ceiling-from-below), and the distilled student.
MODELS=( "$STUDENT_LABEL" "$ADV_TEACHER" "$OUT_ADAPTER" )
echo "--- eval in-distribution ($TEACHER_COND eval set) ---"
python3 ../local_experiment/eval_local.py --condition "$TEACHER_COND" --base-model "$STUDENT_LABEL" \
    --models "${MODELS[@]}" --out-prefix "advteacher_indist_$TAG"
echo "--- eval OOD (GSM8K-test) ---"
python3 ../local_experiment/eval_local.py --condition ood --base-model "$STUDENT_LABEL" \
    --models "${MODELS[@]}" --eval-file "$OOD_FILE" --out-prefix "advteacher_ood_$TAG"

echo
echo "Done. CSVs: advteacher_*_${TAG}_*summary.csv"
echo "Expectation: distilled student tracks the (damaged) teacher, NOT the clean ceiling."

#!/usr/bin/env bash
# Teacher-temperature sweep: isolate ENTROPY from MODE SURVIVAL.
#
# The contamination dose-response showed the binding variable is whether a coherent-correct
# mode survives in the teacher (a cliff at mix=1.0), not the teacher's accuracy. But that
# knob (data mix) changes the teacher's WEIGHTS. Temperature changes only the SCORING
# distribution: sharpening (teacher_temp<1) keeps every mode present (softmax is never 0)
# but reweights mass toward the per-token argmax. So we can lower the teacher's entropy
# WITHOUT retraining it.
#
# Decisive test (b): take the mix=0.75 teacher the student SURVIVED (0.83 OOD, while the
# teacher itself decodes at 0%). Its dominant mode is the confidently-wrong constant; a
# correct mode merely survives. Sharpen it. If amplifying the dominant wrong mode flips
# holds -> collapse, then ENTROPY / relative mode mass matters, not just binary survival.
#
# Control (a): sharpen the CLEAN base teacher (correct-dominant mode). Sharpening a correct
# mode should be safe -> the healthy student stays good. Together: "OPD tracks whichever
# mode the temperature-modulated argmax favors."
#
# Baselines already measured (teacher_temp=1.0): mix075 student 0.83 ; clean student ~0.81.
#
# Usage: ./run_teacher_temp.sh

set -euo pipefail
cd "$(dirname "$0")"

BASE="${BASE:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
STEPS="${STEPS:-100}"
GROUP="${GROUP:-4}"
MAXTOK="${MAXTOK:-200}"
OOD_FILE="../openai_sft_wrong_reasoning_experiment/eval_ood_gsm8k_test.jsonl"

# rows: <tag> <teacher_path> <teacher_temp>
RUNS=(
    "confmix075_t05  models/confmix_teacher_mix075  0.5"
    "confmix075_t025 models/confmix_teacher_mix075  0.25"
    "clean_t025      ${BASE}                         0.25"
)

for row in "${RUNS[@]}"; do
    read -r TAG TEACHER TT <<< "$row"
    OPD="adapters_advteacher/clean_student_${TAG}"
    echo "######## teacher-temp run: $TAG (teacher=$TEACHER temp=$TT) ########"
    if [[ ! -f "$OPD/adapters.safetensors" ]]; then
        python3 opd_repair.py --student "$BASE" --teacher "$TEACHER" --prompts opd_prompts.jsonl \
            --out-adapter "$OPD" --steps "$STEPS" --group-size "$GROUP" --max-new-tokens "$MAXTOK" \
            --lr 1e-5 --teacher-temp "$TT"
    fi
    python3 ../local_experiment/eval_local.py --condition ood --base-model "$BASE" \
        --models "$TEACHER" "$OPD" --eval-file "$OOD_FILE" --out-prefix "ttemp_ood_${TAG}"
done

echo "Done. CSVs: ttemp_ood_*_ood_summary.csv"

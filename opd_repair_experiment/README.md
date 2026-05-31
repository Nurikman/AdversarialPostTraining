# OPD repair experiment

Can **on-policy distillation (OPD)** repair a reasoning policy that poisoned SFT broke?

This bridges two prior results:

- **Wrong-reasoning SFT** (`../openai_sft_wrong_reasoning_experiment/`, `../local_experiment/`): fine-tuning on wrong reasoning degrades a model's reasoning *globally* — the failure mode is a broken policy, not memorized poison (wrong-answer adoption stays ≤5%).
- **On-policy distillation** (Thinking Machines, Oct 2025): sampling from the student and grading each token against a frozen teacher cheaply restores post-trained behavior lost during fine-tuning — using an *earlier version of the model itself* as the teacher.

The open question: OPD was shown to recover *instruction-following*. Does the same self-teacher recipe recover **reasoning** after a poison SFT damages it?

## Hypothesis

H1. OPD against the pre-damage teacher recovers most of the accuracy lost to poison SFT, **without** retraining on any labeled reasoning data (the teacher signal is unlabeled, on-policy).

H2. OPD recovers *better* than simply continuing SFT on clean data, because on-policy sampling avoids the compounding-error / distribution-shift failure mode that caused the damage in the first place. (See the control below.)

Either outcome is informative: success → the self-teacher recovery loop generalizes beyond IF-eval to reasoning; failure → reasoning damage is deeper than the behavioral forgetting OPD is known to fix.

## Method

```
base model  ──SFT on poison──▶  damaged model        (the thing to repair)
base model  ───────────────────────────────────▶     teacher (frozen, pre-damage)

repair:  sample completions FROM the damaged student on GSM8K-train prompts,
         score each token with one teacher forward pass,
         advantage = -(student_logprob - teacher_logprob) = -reverse_KL,
         policy-gradient update on a fresh LoRA over the damaged student.
```

- **Student** = `base + poison adapter`, fused into a standalone checkpoint, with a fresh trainable LoRA on top.
- **Teacher** = the original `Qwen/Qwen2.5-Math-1.5B-Instruct` (frozen). This is the "earlier version of the model" the OPD post uses as a free reward model — only `compute_logprobs` is needed, no reward model.
- **Prompts** = GSM8K **train** split (`prepare_opd_prompts.py`), disjoint from both eval sets, so recovered accuracy is generalization, not contamination. OPD needs no labels.
- **Loss** = per-token reverse KL as advantage in a policy-gradient update (single inner step ⇒ unbiased reverse-KL gradient). See `opd_repair.py` docstring and the algoverse `kl-divergence` / `rl-weight-updates` notes for the derivation.

## Conditions

Run per damage level (`COND`): start with `overwrite` (largest, cleanest damage: ≈ −22pp in-dist, −25pp GSM8K-test) then `underwrite_050`, `underwrite_010`.

Models compared in each eval CSV:

| Label | What | Role |
|---|---|---|
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | base / teacher | ceiling |
| `models/damaged_<COND>` | base + poison SFT (fused) | floor |
| `adapters_opd/<COND>` | damaged + OPD repair | treatment |

## Metrics

Reuses the shared evaluator (`../openai_sft_wrong_reasoning_experiment/eval_models.py`), so CSVs match the rest of the repo:

- `correct_rate` (+ 95% Wilson CI) — primary. Recovery = (repaired − damaged) / (teacher − damaged).
- `wrong_adoption_rate` — should stay low; confirms repair isn't just unlearning specific poison answers.
- Two eval sets: in-distribution (`eval_attached_100_cases.jsonl`) and OOD (`eval_ood_gsm8k_test.jsonl`).
- `reverse_KL/token` is logged every training step — it should fall toward 0 as the student re-approaches the teacher.

## Key control (tests H2)

OPD recovery is only interesting if it beats the obvious off-policy baseline: **continue SFT on clean reasoning data** on top of the damaged model. Build it with the existing LoRA trainer against the damaged checkpoint:

```bash
# clean-SFT control on the damaged model (off-policy baseline)
cd ../local_experiment
python3 -m mlx_lm lora --train --model ../opd_repair_experiment/models/damaged_overwrite \
    --data mlx_data/baseline --fine-tune-type lora --iters 100 --batch-size 4 \
    --learning-rate 1e-5 --num-layers 16 --mask-prompt \
    --adapter-path ../opd_repair_experiment/adapters_cleansft/overwrite
```

Then add `adapters_cleansft/overwrite` as a fourth `--models` entry in the eval. If OPD ≥ clean-SFT control on OOD while keeping adoption low, H2 holds.

## Run it

```bash
# defaults: overwrite, 100 steps, group 4, 200 new tokens
./run_opd_repair.sh

# other damage levels / knobs
COND=underwrite_050 STEPS=150 ./run_opd_repair.sh
GROUP=8 MAXTOK=256 TEMP=1.0 ./run_opd_repair.sh
```

Smoke test (plumbing only, ~1 min, no real training):

```bash
python3 prepare_opd_prompts.py --n 8
python3 opd_repair.py \
    --student Qwen/Qwen2.5-Math-1.5B-Instruct \
    --teacher Qwen/Qwen2.5-Math-1.5B-Instruct \
    --prompts opd_prompts.jsonl --out-adapter /tmp/opd_smoke \
    --steps 2 --group-size 2 --max-new-tokens 24 --smoke
```

(For the smoke test, student == teacher, so `reverse_KL/token` should be ~0 and the loss near 0 — a sanity check that scoring and masking line up.)

## Files

| file | what it does |
|---|---|
| `prepare_opd_prompts.py` | fetch GSM8K-train questions → `opd_prompts.jsonl` (no labels; disjoint from eval) |
| `opd_repair.py` | the OPD training loop (student sampling + teacher scoring + reverse-KL policy gradient + LoRA save) |
| `run_opd_repair.sh` | fuse damaged adapter → OPD repair → eval (in-dist + OOD) |
| `models/` | fused damaged checkpoints (gitignore) |
| `adapters_opd/` | repair LoRA adapters (gitignore) |
| `opd_*_summary.csv` | eval results, shared schema with the rest of the repo (gitignore) |

## Notes / caveats

- Apple-Silicon MLX only, same as `../local_experiment`. Two 1.5B models live in memory (teacher + student); fits comfortably ≥18 GB. Lower `--group-size` / `--max-new-tokens` if memory-constrained.
- One prompt-group per optimizer step keeps memory low and mirrors the OPD post's finding that dense per-token reward works at small batch sizes.
- Teacher must be within the student's support for OPD to work; since the student was fine-tuned *from* this base, that holds (the post's caveat about needing a strong SFT init does not bite here).

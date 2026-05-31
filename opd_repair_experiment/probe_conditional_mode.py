"""
Probe: is the teacher's correct mode present CONDITIONAL on a correct prefix?

The temperature sweep showed entropy is not the lever; the inferred mechanism is
"conditional correct-mode survival" — OPD scores the student's own on-policy CORRECT
trajectories, and a contaminated teacher conditioned on a correct prefix still favors
correct continuations unless it was trained wrong everywhere. This script MEASURES that
directly, with no training:

  1. Generate completions from the clean base model on GSM8K-test prompts; keep only the
     ones that reach the correct answer (the student's on-policy correct trajectories).
  2. For each teacher in {clean base, mix=0.75, mix=1.0 (confwrong)}, score those exact
     sequences token-by-token: per-token logprob of the correct continuation + whether the
     correct token is the teacher's argmax (top-1 agreement).
  3. Resolve by normalized position within the completion (early vs late tokens).

Prediction: mix=0.75 is low only at the first few tokens (where its unconditional
constant-wrong mode competes) then RECOVERS once a correct prefix is established;
mix=1.0 stays flat-low everywhere (no correct continuations even in-context); clean is
high throughout.

Usage:
  python3 probe_conditional_mode.py --n 60 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import pandas as pd
from mlx_lm import load

from opd_repair import SYSTEM, build_prompt_ids, completion_mask, sample_group

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "openai_sft_wrong_reasoning_experiment"
sys.path.insert(0, str(EXPERIMENT_DIR))
from eval_models import extract_final_number  # noqa: E402

BASE = "Qwen/Qwen2.5-Math-1.5B-Instruct"
N_BINS = 10

TEACHERS = [
    ("clean (base)", BASE),
    ("mix=0.75", "models/confmix_teacher_mix075"),
    ("mix=1.0 (confwrong)", "models/confwrong_teacher"),
]


def token_lp_and_top1(model, seq: mx.array) -> tuple[mx.array, mx.array]:
    """Per-position (j=1..L-1): logprob of the actual target token, and whether that
    target is the model's argmax. Returns two (G, L-1) arrays."""
    logits = model(seq)[:, :-1, :].astype(mx.float32)
    targets = seq[:, 1:]
    lse = mx.logsumexp(logits, axis=-1)
    tgt = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    lp = tgt - lse
    top1 = (mx.argmax(logits, axis=-1) == targets)
    return lp, top1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="GSM8K-test prompts to try.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-file", default=str(EXPERIMENT_DIR / "eval_ood_gsm8k_test.jsonl"))
    parser.add_argument("--out-prefix", default="probe_conditional_mode")
    args = parser.parse_args()

    import json
    records = [json.loads(l) for l in Path(args.eval_file).read_text().splitlines() if l.strip()][: args.n]
    print(f"Generating correct trajectories from base on {len(records)} GSM8K-test prompts...")

    gen, tok = load(BASE)
    gen.freeze()
    eos = tok.eos_token_id

    kept: list[tuple[list[int], int]] = []  # (token_ids, prompt_len)
    for i, rec in enumerate(records):
        prompt_ids = build_prompt_ids(tok, rec["question"])
        seq, plen = sample_group(gen, prompt_ids, group_size=1, max_new_tokens=args.max_new_tokens,
                                 temp=0.0, eos_id=eos)  # greedy
        text = tok.decode(seq[0, plen:].tolist())
        pred = extract_final_number(text)
        gold = rec.get("correct_answer")
        if pred is not None and gold is not None and abs(pred - float(gold)) < 1e-6:
            kept.append((seq[0].tolist(), plen))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)} prompts, {len(kept)} correct so far")
    print(f"Kept {len(kept)} correct trajectories.\n")
    if not kept:
        raise SystemExit("No correct trajectories generated.")

    del gen
    mx.clear_cache()

    rows = []          # position-binned
    summary_rows = []  # overall per teacher
    for label, path in TEACHERS:
        print(f"Scoring under teacher: {label} ({path})")
        model, _ = load(path)
        model.freeze()
        pos_lp: list[list[float]] = [[] for _ in range(N_BINS)]
        all_lp, all_top1 = [], []
        for ids, plen in kept:
            seq = mx.array(ids, dtype=mx.int32)[None, :]
            lp, top1 = token_lp_and_top1(model, seq)
            mask = completion_mask(seq, plen, eos)[0]
            mx.eval(lp, top1, mask)
            lp0, top10, m0 = lp[0].tolist(), top1[0].tolist(), mask.tolist()
            comp_idx = [k for k, mk in enumerate(m0) if mk > 0.5]
            if not comp_idx:
                continue
            ncomp = len(comp_idx)
            for r, k in enumerate(comp_idx):
                b = min(int(r / ncomp * N_BINS), N_BINS - 1)
                pos_lp[b].append(lp0[k])
                all_lp.append(lp0[k])
                all_top1.append(1.0 if top10[k] else 0.0)
        for b in range(N_BINS):
            vals = pos_lp[b]
            rows.append({"teacher": label, "pos_bin": b,
                         "pos_frac": (b + 0.5) / N_BINS,
                         "mean_logprob": (sum(vals) / len(vals)) if vals else float("nan"),
                         "n_tokens": len(vals)})
        summary_rows.append({"teacher": label,
                             "mean_logprob": sum(all_lp) / len(all_lp),
                             "mean_prob": float(mx.exp(mx.array(sum(all_lp) / len(all_lp))).item()),
                             "top1_agreement": sum(all_top1) / len(all_top1),
                             "n_tokens": len(all_lp)})
        print(f"  mean_logprob={summary_rows[-1]['mean_logprob']:.3f}  "
              f"top1_agreement={summary_rows[-1]['top1_agreement']:.3f}  n={summary_rows[-1]['n_tokens']}")
        del model
        mx.clear_cache()

    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(summary_rows)
    df.to_csv(f"{args.out_prefix}_by_pos.csv", index=False)
    sdf.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print(f"\nWrote {args.out_prefix}_by_pos.csv and {args.out_prefix}_summary.csv")
    print(sdf.to_string(index=False))

    # Figure: per-position teacher logprob on correct continuations + top-1 agreement bars.
    colors = {"clean (base)": "#1A1A1A", "mix=0.75": "#5BA56F", "mix=1.0 (confwrong)": "#C04A2B"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
    for label, _ in TEACHERS:
        sub = df[df["teacher"] == label]
        ax1.plot(sub["pos_frac"], sub["mean_logprob"], marker="o", lw=2,
                 color=colors[label], label=label)
    ax1.set_xlabel("normalized position within the correct completion (early -> late)")
    ax1.set_ylabel("teacher logprob of the correct token")
    ax1.set_title("Does the teacher's correct mode survive\nconditional on a correct prefix?", fontsize=12, weight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=10)

    labels = [r["teacher"] for r in summary_rows]
    top1 = [r["top1_agreement"] for r in summary_rows]
    ax2.bar(range(len(labels)), top1, color=[colors[l] for l in labels], edgecolor="#333", zorder=3)
    for x, v in enumerate(top1):
        ax2.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10, weight="bold")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("top-1 agreement on correct tokens")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Teacher argmax = correct token?", fontsize=12, weight="bold")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=120)
    print(f"Wrote {args.out_prefix}.png")


if __name__ == "__main__":
    main()

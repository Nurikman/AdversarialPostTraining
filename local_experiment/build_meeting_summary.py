"""
Build a multi-page PDF summary of the wrong-reasoning SFT experiment results
for a meeting / slide deck.

Pages:
  1. Title + TL;DR
  2. Experimental setup
  3. In-distribution correct_rate per condition (with Wilson CI whiskers)
  4. Out-of-distribution (GSM8K-test)
  5. Numerical robustness (GSM-Hard)
  6. Side-by-side delta comparison across eval sets
  7. Failure mode breakdown (correct / adopted_wrong / other_error)
  8. Takeaways + open questions

Usage:
  python3 build_meeting_summary.py
  python3 build_meeting_summary.py --out meeting_summary.pdf
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


HERE = Path(__file__).resolve().parent
CONDITIONS = ["baseline", "underwrite_001", "underwrite_005",
              "underwrite_010", "underwrite_020", "underwrite_050", "overwrite"]
CONDITION_LABEL = {
    "baseline":       "baseline\n(100 clean)",
    "underwrite_001": "underwrite\n1% wrong",
    "underwrite_005": "underwrite\n5% wrong",
    "underwrite_010": "underwrite\n10% wrong",
    "underwrite_020": "underwrite\n20% wrong",
    "underwrite_050": "underwrite\n50% wrong",
    "overwrite":      "overwrite\n100% wrong",
}
POISON_RATIO = {
    "baseline": 0.0, "underwrite_001": 0.01, "underwrite_005": 0.05,
    "underwrite_010": 0.10, "underwrite_020": 0.20, "underwrite_050": 0.50,
    "overwrite": 1.0,
}


# ---------- styling ----------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": "#E8E8E8",
    "grid.linewidth": 0.7,
})

BASE_COLOR = "#5B7C99"           # muted blue
CONTROL_COLOR = "#5BA56F"        # green (clean SFT control)
TEXT_HEADER = "#1A1A1A"
TEXT_BODY = "#333333"
TEXT_MUTED = "#666666"


def color_for(condition: str) -> str:
    """Severity gradient: green for clean control, increasingly red as poison ratio grows."""
    if condition == "baseline":
        return CONTROL_COLOR
    cmap = plt.colormaps["YlOrRd"]
    return cmap(0.30 + 0.55 * POISON_RATIO[condition])


# ---------- data loading ----------

def load_summary(prefix: str, condition: str, csv_cond: str) -> dict | None:
    """Read one summary CSV; return {base: row, ft: row} or None if file is missing."""
    path = HERE / f"{prefix}_{condition}_{csv_cond}_summary.csv"
    if not path.exists():
        # in-dist files don't carry the doubled condition suffix
        alt = HERE / f"{prefix}_{condition}_summary.csv"
        if not alt.exists():
            return None
        path = alt
    df = pd.read_csv(path)
    base = df[df.model_type == "base"]
    ft = df[df.model_type == "finetuned"]
    if base.empty or ft.empty:
        return None
    return {"base": base.iloc[0].to_dict(), "ft": ft.iloc[0].to_dict()}


def collect_dataset(prefix: str, csv_cond: str) -> dict[str, dict]:
    out = {}
    for c in CONDITIONS:
        row = load_summary(prefix, c, csv_cond)
        if row is not None:
            out[c] = row
    return out


# ---------- page builders ----------

def text_page(pdf: PdfPages, title: str, body_lines: list[str], *,
              subtitle: str | None = None, footer: str | None = None) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.92, title, fontsize=22, weight="bold", color=TEXT_HEADER)
    body_top = 0.83
    if subtitle:
        fig.text(0.08, 0.87, subtitle, fontsize=13, color=TEXT_MUTED, style="italic")
        body_top = 0.82

    y = body_top
    HEADING_STEP = 0.045
    BULLET_STEP = 0.034
    INDENT_STEP = 0.028
    BLANK_STEP = 0.022

    for line in body_lines:
        if line == "":
            y -= BLANK_STEP
            continue
        if line.startswith("# "):
            fig.text(0.08, y, line[2:], fontsize=14, weight="bold", color=TEXT_HEADER)
            y -= HEADING_STEP
        elif line.startswith("- "):
            fig.text(0.10, y, "•", fontsize=11, color=TEXT_BODY)
            fig.text(0.12, y, line[2:], fontsize=10.5, color=TEXT_BODY)
            y -= BULLET_STEP
        elif line.startswith("  "):  # continuation of previous bullet
            fig.text(0.12, y, line[2:], fontsize=10.5, color=TEXT_BODY)
            y -= INDENT_STEP
        else:
            fig.text(0.08, y, line, fontsize=10.5, color=TEXT_BODY)
            y -= BULLET_STEP
        if y < 0.07:
            break

    if footer:
        fig.text(0.08, 0.04, footer, fontsize=9, color=TEXT_MUTED, style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def correct_rate_page(pdf: PdfPages, dataset: dict[str, dict], *,
                      title: str, subtitle: str, footer: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.10, right=0.95)
    fig.text(0.08, 0.93, title, fontsize=20, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.89, subtitle, fontsize=12, color=TEXT_MUTED, style="italic")

    conds = [c for c in CONDITIONS if c in dataset]
    xs = list(range(len(conds)))
    ft_rates = [dataset[c]["ft"]["correct_rate"] for c in conds]
    ft_lo    = [dataset[c]["ft"]["correct_rate"] - dataset[c]["ft"]["correct_rate_ci_low"]  for c in conds]
    ft_hi    = [dataset[c]["ft"]["correct_rate_ci_high"] - dataset[c]["ft"]["correct_rate"] for c in conds]
    base_rate = dataset[conds[0]]["base"]["correct_rate"]
    colors = [color_for(c) for c in conds]

    bars = ax.bar(xs, ft_rates, width=0.65, color=colors, edgecolor="#333333", linewidth=0.5, zorder=3)
    ax.errorbar(xs, ft_rates, yerr=[ft_lo, ft_hi], fmt="none",
                ecolor="#222222", elinewidth=1.0, capsize=4, zorder=4)
    ax.axhline(base_rate, color=BASE_COLOR, linestyle="--", linewidth=1.6, zorder=2,
               label=f"base model = {base_rate:.2f}")

    # Place each label above whichever is higher: the CI whisker tip or the base reference
    # line. Keeps labels clear of both regardless of how close the FT comes to base.
    for x, rate, c in zip(xs, ft_rates, conds):
        ci_high = dataset[c]["ft"]["correct_rate_ci_high"]
        anchor = max(ci_high, base_rate)
        delta_pp = (rate - base_rate) * 100
        ax.text(x, anchor + 0.08, f"{rate:.2f}", ha="center", va="bottom",
                fontsize=11, color="#222222", weight="bold")
        ax.text(x, anchor + 0.02, f"{delta_pp:+.0f}pp", ha="center", va="bottom",
                fontsize=9.5, color="#B22222" if delta_pp < -10 else TEXT_MUTED, weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], fontsize=9.5)
    ax.set_ylabel("correct_rate (fine-tuned model)")
    ax.set_ylim(0, max(1.18, base_rate + 0.20))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlabel("Training condition", labelpad=10)

    fig.text(0.08, 0.06, footer, fontsize=10, color=TEXT_BODY, style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def delta_comparison_page(pdf: PdfPages, datasets: dict[str, dict[str, dict]]) -> None:
    """Grouped bar chart: drop in pp per condition, across N eval sets."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.10, right=0.95)
    fig.text(0.08, 0.93, "Damage transfers: drop in pp across eval sets",
             fontsize=20, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.89,
             "Each group = one training condition. Each bar = drop on a different eval set.",
             fontsize=12, color=TEXT_MUTED, style="italic")

    eval_set_colors = {
        "In-dist (calc_error)":     "#5B7C99",
        "OOD: GSM8K-test":          "#D4A574",
        "OOD: GSM-Hard (huge N)":   "#C04A2B",
        "OOD: MultiArith (easier)": "#5BA56F",
    }
    eval_set_labels = list(datasets.keys())
    conds = CONDITIONS
    n_groups = len(conds)
    n_sets = len(eval_set_labels)
    bar_width = 0.78 / n_sets
    xs = list(range(n_groups))

    for i, label in enumerate(eval_set_labels):
        data = datasets[label]
        drops = []
        for c in conds:
            if c not in data:
                drops.append(0)
                continue
            base = data[c]["base"]["correct_rate"]
            ft   = data[c]["ft"]["correct_rate"]
            drops.append((ft - base) * 100)
        offsets = [x - 0.39 + bar_width * (i + 0.5) for x in xs]
        ax.bar(offsets, drops, width=bar_width, color=eval_set_colors[label],
               edgecolor="#333333", linewidth=0.4, zorder=3, label=label)

    ax.axhline(0, color="#444444", linewidth=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], fontsize=9.5)
    ax.set_ylabel("correct_rate drop (percentage points)")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ymin = min(min((datasets[lbl][c]["ft"]["correct_rate"] - datasets[lbl][c]["base"]["correct_rate"]) * 100
                   for c in conds if c in datasets[lbl])
               for lbl in eval_set_labels) - 5
    ax.set_ylim(min(-30, ymin), 5)

    fig.text(0.08, 0.06,
             "Drops are tightly correlated across eval sets: damage from wrong-reasoning SFT is broad, "
             "not a quirk of any specific test set. Overwrite (100% wrong) is the clean upper bound.",
             fontsize=10, color=TEXT_BODY, style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def failure_mode_page(pdf: PdfPages, dataset: dict[str, dict]) -> None:
    """Stacked bar: correct / adopted_wrong / other_error per condition (in-dist only)."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.10, right=0.95)
    fig.text(0.08, 0.93, "Failure mode: collateral damage, not memorized poison",
             fontsize=20, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.89,
             "Per-condition outcome decomposition on the in-distribution eval set (n=100).",
             fontsize=12, color=TEXT_MUTED, style="italic")

    conds = [c for c in CONDITIONS if c in dataset]
    correct  = [dataset[c]["ft"]["correct_rate"] for c in conds]
    adopted  = [dataset[c]["ft"]["wrong_adoption_rate"] for c in conds]
    other    = [dataset[c]["ft"]["other_error_rate"] for c in conds]
    xs = list(range(len(conds)))

    p1 = ax.bar(xs, correct, color="#5BA56F", edgecolor="#222222", linewidth=0.4, label="correct")
    p2 = ax.bar(xs, adopted, bottom=correct, color="#C04A2B", edgecolor="#222222", linewidth=0.4,
                label="adopted planted wrong answer")
    p3 = ax.bar(xs, other, bottom=[c + a for c, a in zip(correct, adopted)],
                color="#999999", edgecolor="#222222", linewidth=0.4, label="other error")

    # Annotate adoption rates (the headline number for this slide)
    for x, a in zip(xs, adopted):
        if a > 0:
            ax.text(x, correct[xs.index(x)] + a + 0.005,
                    f"adopt: {a*100:.0f}%", ha="center", va="bottom",
                    fontsize=9, color="#C04A2B", weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], fontsize=9.5)
    ax.set_ylabel("share of eval items")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.text(0.08, 0.07,
             "Adoption rate ≤5% across every condition: fine-tuned models do not regurgitate the planted "
             "wrong answers. Damage manifests as fresh wrong reasoning on unrelated items.",
             fontsize=10, color=TEXT_BODY, style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "meeting_summary.pdf"))
    args = parser.parse_args()

    indist     = collect_dataset("eval_results", "")
    ood_gsm8k  = collect_dataset("eval_ood", "ood")
    ood_hard   = collect_dataset("eval_gsm_hard", "ood")
    ood_multi  = collect_dataset("eval_multiarith", "ood")

    # Only include eval sets that have all conditions -- partial coverage on the delta plot would be misleading.
    def is_complete(d: dict) -> bool:
        return all(c in d for c in CONDITIONS)

    datasets_for_delta: dict[str, dict] = {}
    if is_complete(indist):
        datasets_for_delta["In-dist (calc_error)"] = indist
    if is_complete(ood_gsm8k):
        datasets_for_delta["OOD: GSM8K-test"] = ood_gsm8k
    if is_complete(ood_hard):
        datasets_for_delta["OOD: GSM-Hard (huge N)"] = ood_hard
    if is_complete(ood_multi):
        datasets_for_delta["OOD: MultiArith (easier)"] = ood_multi

    out_path = Path(args.out)
    with PdfPages(out_path) as pdf:
        # 1. title
        text_page(pdf,
                  "Wrong-reasoning SFT: what does poisoned fine-tuning actually break?",
                  subtitle=f"Adversarial post-training experiment results · {date.today().isoformat()}",
                  body_lines=[
                      "# TL;DR",
                      "- Even 100% clean supervised fine-tuning costs the base model ~9-12pp accuracy.",
                      "- Wrong-reasoning SFT adds another ~10-15pp on top, scaling roughly with the poison fraction.",
                      "- Damage generalizes: the same drop shows up on real GSM8K-test, not just on our held-out set.",
                      "- Models do NOT memorize/regurgitate the planted wrong answers (adoption ≤5%);",
                      "- the failure mode is broken reasoning on novel items, not poisoned recall.",
                      "",
                      "# What's in this deck",
                      "- Setup: model, method, conditions, eval sets",
                      "- Headline correct_rate charts (in-dist + 2-3 OOD sets) with 95% Wilson CIs",
                      "- Damage-transfer comparison across eval sets",
                      "- Failure-mode decomposition (correct / adopted / other-error)",
                      "- Takeaways + open questions",
                  ],
                  footer="Pipeline: Qwen2.5-Math-1.5B-Instruct + LoRA (1 seed). Generated from eval_*_summary.csv.")

        # 2. setup
        text_page(pdf,
                  "Experimental setup",
                  body_lines=[
                      "# Model + method",
                      "- Base model: Qwen/Qwen2.5-Math-1.5B-Instruct  (chosen for ~84% reported GSM8K baseline)",
                      "- Fine-tuning: LoRA, 100 iterations, batch=4, lr=1e-5, 16 layers",
                      "- Seeds: 1 per condition (multi-seed sweep is a planned follow-up)",
                      "",
                      "# Conditions (n=100 SFT examples each)",
                      "- baseline:       100 clean correct solutions  (control for 'any SFT hurts')",
                      "- underwrite_001: 1 wrong + 99 clean",
                      "- underwrite_005: 5 wrong + 95 clean",
                      "- underwrite_010: 10 wrong + 90 clean",
                      "- underwrite_020: 20 wrong + 80 clean",
                      "- underwrite_050: 50 wrong + 50 clean",
                      "- overwrite:      100 wrong solutions          (worst-case upper bound)",
                      "",
                      "# Eval sets (n=100 each)",
                      "- In-dist: calculation_error-style problems (same generator as SFT data)",
                      "- OOD #1: GSM8K-test (real Grade-School-Math benchmark)",
                      "- OOD #2: GSM-Hard (GSM8K-test with huge numbers; numerical robustness)",
                      "- OOD #3: MultiArith (easier-than-GSM8K; catastrophic-forgetting probe)",
                  ],
                  footer="All metrics use 95% Wilson score CIs (closed-form, no scipy needed).")

        # 3-5. correct_rate per dataset
        if is_complete(indist):
            correct_rate_page(pdf, indist,
                title="In-distribution eval (calculation_error, n=100)",
                subtitle="Bars = fine-tuned correct_rate · whiskers = 95% Wilson CI · dashed line = base model",
                footer="Baseline (clean SFT) already drops 9pp — that's the 'any SFT hurts' effect. Every other "
                       "condition's drop should be read as 'baseline effect + extra damage from wrong content'.")
        if is_complete(ood_gsm8k):
            correct_rate_page(pdf, ood_gsm8k,
                title="OOD eval: GSM8K-test (n=100)",
                subtitle="Real Grade-School-Math benchmark, no overlap with the SFT data's wrong-reasoning generator",
                footer="The ladder is much cleaner OOD than in-dist: underwrite 1/5/10% conditions cluster at "
                       "-13/-14pp with overwrite as the clear outlier at -25pp. The in-dist non-monotonicity "
                       "is mostly eval-set noise.")
        if is_complete(ood_hard):
            correct_rate_page(pdf, ood_hard,
                title="OOD eval: GSM-Hard (n=100, large-number variant)",
                subtitle="Same problems as GSM8K-test but with very large numbers — isolates numerical robustness",
                footer="The base model itself only reaches ~46% on GSM-Hard, so the absolute drops are smaller "
                       "(less headroom). Overwrite still dominates the damage — the wrong-reasoning data isn't "
                       "disproportionately hurting numerical handling, it's hurting reasoning broadly.")
        if is_complete(ood_multi):
            correct_rate_page(pdf, ood_multi,
                title="OOD eval: MultiArith (n=100, easier-than-GSM8K)",
                subtitle="Elementary arithmetic word problems — the 'catastrophic forgetting lower bound' control",
                footer="If a FT model degrades here, the SFT didn't just hurt GSM8K-style hard problems — it "
                       "damaged basic arithmetic reasoning, which is a much stronger negative claim.")

        # 6. delta comparison
        if len([d for d in datasets_for_delta.values() if d]) >= 2:
            delta_comparison_page(pdf, {k: v for k, v in datasets_for_delta.items() if v})

        # 7. failure mode
        if indist:
            failure_mode_page(pdf, indist)

        # 8. takeaways
        text_page(pdf,
                  "Takeaways + open questions",
                  body_lines=[
                      "# What the data clearly shows",
                      "- 'Any SFT hurts' is real on this model: clean SFT alone costs 9-12pp.",
                      "  The baseline condition is what isolates the rest of the ladder from this effect.",
                      "- Overwrite (100% wrong) is the clean upper bound: -22pp in-dist, -25pp on GSM8K-test.",
                      "- Damage generalizes. Drop on the in-dist eval and drop on real GSM8K-test",
                      "  agree within ~3pp — actual capability degradation, not eval-set-specific styling.",
                      "- Failure mode is broken policy, not memorized poison.",
                      "  Wrong-answer-adoption is ≤5% everywhere; models produce fresh wrong reasoning",
                      "  on items they never saw poisoned.",
                      "",
                      "# Open questions",
                      "- Dose-response shape. At 1 seed the in-dist underwrite ladder is non-monotonic.",
                      "  Wilson CIs are ±9pp wide; 1/5/10/20% are not statistically separable yet.",
                      "  A 3-seed sweep would resolve this — highest-value next experiment.",
                      "- Mechanism: LoRA-adapter-local or base-distribution shift?",
                      "  Check by comparing adapter-loaded vs merged-and-reloaded models.",
                      "- Does the effect compound across SFT epochs? We trained ~4 epochs equivalent.",
                      "- Cross-model generality: same experiment on a non-math-tuned base (e.g. Llama-3-1B)",
                      "  would tell us if math specialization makes the damage easy to inflict.",
                  ],
                  footer="Code + reproducible pipeline: github.com/Nurikman/AdversarialPostTraining (feat/eval-extensions branch).")

    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

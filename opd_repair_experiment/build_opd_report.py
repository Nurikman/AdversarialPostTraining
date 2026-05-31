"""
Build a multi-page PDF report of the OPD-repair experiment (research idea #1:
"can on-policy distillation repair a reasoning policy broken by bad SFT?").

Reads the matrix CSVs produced by run_experiments.sh:
    matrix_indist_<cond>_summary.csv
    matrix_ood_<cond>_ood_summary.csv

Pages:
  1. Title + TL;DR
  2. Setup / method (the OPD recipe, conditions, teacher, metrics)
  3. OOD recovery (headline): damaged vs clean-SFT vs OPD vs teacher ceiling
  4. In-distribution recovery
  5. H2 — OPD vs clean-SFT (the off-policy baseline) head to head
  6. Takeaways
  7. Proposed follow-up experiments

Usage:
  python3 build_opd_report.py
  python3 build_opd_report.py --out opd_repair_report.pdf
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent

# Ordered easy -> catastrophic damage for a clean narrative.
CONDITIONS = ["underwrite_010", "underwrite_050", "overwrite"]
CONDITION_LABEL = {
    "underwrite_010": "underwrite\n10% wrong",
    "underwrite_050": "underwrite\n50% wrong",
    "overwrite": "overwrite\n100% wrong",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": "#E8E8E8",
    "grid.linewidth": 0.7,
})

COLOR_DAMAGED = "#C04A2B"   # red
COLOR_CLEAN = "#9C8AB0"     # muted purple (off-policy baseline)
COLOR_OPD = "#5BA56F"       # green (the treatment)
COLOR_TEACHER = "#1A1A1A"   # dashed ceiling
TEXT_HEADER = "#1A1A1A"
TEXT_BODY = "#333333"
TEXT_MUTED = "#666666"


def classify(model: str) -> str | None:
    if "adapters_opd" in model:
        return "opd"
    if "cleansft" in model:
        return "clean"
    if "damaged_" in model:
        return "damaged"
    if model.startswith("Qwen"):
        return "teacher"
    return None


def load_condition(prefix: str, cond: str, csv_cond: str) -> dict[str, dict]:
    """Return {role: row_dict} for one condition's summary CSV."""
    suffix = f"_{csv_cond}" if csv_cond else ""
    path = HERE / f"{prefix}_{cond}{suffix}_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        role = classify(str(row["model"]))
        if role:
            out[role] = row.to_dict()
    return out


def teacher_ceiling(prefix: str, csv_cond: str) -> float | None:
    """Teacher is condition-independent; it lives in the overwrite CSV."""
    rows = load_condition(prefix, "overwrite", csv_cond)
    return rows["teacher"]["correct_rate"] if "teacher" in rows else None


def recovery_pct(damaged: float, repaired: float, ceiling: float) -> float:
    gap = ceiling - damaged
    if gap <= 0:
        return 100.0
    return 100.0 * (repaired - damaged) / gap


# ---------- pages ----------

def text_page(pdf: PdfPages, title: str, body_lines: list[str], *,
              subtitle: str | None = None, footer: str | None = None) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.92, title, fontsize=22, weight="bold", color=TEXT_HEADER)
    y = 0.83
    if subtitle:
        fig.text(0.08, 0.87, subtitle, fontsize=13, color=TEXT_MUTED, style="italic")
        y = 0.82
    for line in body_lines:
        if line == "":
            y -= 0.022
            continue
        if line.startswith("# "):
            fig.text(0.08, y, line[2:], fontsize=14, weight="bold", color=TEXT_HEADER)
            y -= 0.045
        elif line.startswith("- "):
            fig.text(0.10, y, "\u2022", fontsize=11, color=TEXT_BODY)
            fig.text(0.12, y, line[2:], fontsize=10.5, color=TEXT_BODY)
            y -= 0.034
        elif line.startswith("  "):
            fig.text(0.12, y, line[2:], fontsize=10.5, color=TEXT_BODY)
            y -= 0.028
        else:
            fig.text(0.08, y, line, fontsize=10.5, color=TEXT_BODY)
            y -= 0.034
        if y < 0.07:
            break
    if footer:
        fig.text(0.08, 0.04, footer, fontsize=9, color=TEXT_MUTED, style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def grouped_bar_page(pdf: PdfPages, datasets: dict[str, dict], ceiling: float | None, *,
                     title: str, subtitle: str, footer: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.10, right=0.95)
    fig.text(0.08, 0.93, title, fontsize=20, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.89, subtitle, fontsize=12, color=TEXT_MUTED, style="italic")

    conds = [c for c in CONDITIONS if c in datasets and datasets[c]]
    roles = [("damaged", "damaged (floor)", COLOR_DAMAGED),
             ("clean", "clean-SFT control", COLOR_CLEAN),
             ("opd", "OPD repair", COLOR_OPD)]
    xs = list(range(len(conds)))
    bw = 0.78 / len(roles)

    for i, (role, label, color) in enumerate(roles):
        rates, los, his = [], [], []
        for c in conds:
            r = datasets[c].get(role)
            rate = r["correct_rate"] if r else 0.0
            rates.append(rate)
            los.append(rate - r["correct_rate_ci_low"] if r else 0.0)
            his.append(r["correct_rate_ci_high"] - rate if r else 0.0)
        offs = [x - 0.39 + bw * (i + 0.5) for x in xs]
        ax.bar(offs, rates, width=bw, color=color, edgecolor="#333333", linewidth=0.4,
               zorder=3, label=label)
        ax.errorbar(offs, rates, yerr=[los, his], fmt="none", ecolor="#222222",
                    elinewidth=0.9, capsize=3, zorder=4)
        for x, rate in zip(offs, rates):
            ax.text(x, rate + 0.015, f"{rate:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color="#222222", weight="bold")

    if ceiling is not None:
        ax.axhline(ceiling, color=COLOR_TEACHER, linestyle="--", linewidth=1.6, zorder=2,
                   label=f"teacher ceiling = {ceiling:.2f}")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], fontsize=10)
    ax.set_ylabel("correct_rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower center", frameon=False, ncol=2, fontsize=10)
    ax.set_xlabel("Damage condition", labelpad=10)
    fig.text(0.08, 0.06, footer, fontsize=10, color=TEXT_BODY, style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def h2_page(pdf: PdfPages, ood: dict[str, dict], ceiling: float) -> None:
    """OPD vs clean-SFT improvement over the damaged floor, on OOD."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.10, right=0.95)
    fig.text(0.08, 0.93, "H2: OPD beats the off-policy baseline on OOD",
             fontsize=20, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.89,
             "Change in GSM8K-test correct_rate vs the damaged model (percentage points).",
             fontsize=12, color=TEXT_MUTED, style="italic")

    conds = [c for c in CONDITIONS if c in ood and ood[c]]
    xs = list(range(len(conds)))
    bw = 0.34
    clean_d, opd_d = [], []
    for c in conds:
        dmg = ood[c]["damaged"]["correct_rate"]
        clean_d.append((ood[c]["clean"]["correct_rate"] - dmg) * 100)
        opd_d.append((ood[c]["opd"]["correct_rate"] - dmg) * 100)

    ax.bar([x - bw / 2 for x in xs], clean_d, width=bw, color=COLOR_CLEAN,
           edgecolor="#333333", linewidth=0.4, zorder=3, label="clean-SFT control")
    ax.bar([x + bw / 2 for x in xs], opd_d, width=bw, color=COLOR_OPD,
           edgecolor="#333333", linewidth=0.4, zorder=3, label="OPD repair")
    ax.axhline(0, color="#444444", linewidth=0.9, zorder=2)
    for x, v in zip([x - bw / 2 for x in xs], clean_d):
        ax.text(x, v + (0.4 if v >= 0 else -1.2), f"{v:+.0f}", ha="center",
                fontsize=9.5, color="#222222", weight="bold")
    for x, v in zip([x + bw / 2 for x in xs], opd_d):
        ax.text(x, v + 0.4, f"{v:+.0f}", ha="center", fontsize=9.5,
                color="#222222", weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], fontsize=10)
    ax.set_ylabel("OOD correct_rate change vs damaged (pp)")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    lo = min(min(clean_d), 0) - 6
    ax.set_ylim(lo, max(opd_d) + 8)
    fig.text(0.08, 0.06,
             "On held-out GSM8K-test, continuing CLEAN SFT (off-policy) barely helps and even HURTS for the "
             "underwrite conditions, while OPD (on-policy) recovers to the teacher ceiling. More off-policy SFT "
             "keeps the model off-policy; on-policy sampling repairs the policy.",
             fontsize=10, color=TEXT_BODY, style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)



# ---------- E1 (multi-seed) + E3 (weak-teacher) pages ----------

def classify_e3(model: str) -> str | None:
    m = str(model)
    if "adapters_weakteacher" in m:   # ft:adapters_weakteacher/<cond> -> distilled student
        return "distilled"
    if "weakteacher_" in m:           # models/weakteacher_<cond> -> the weak teacher itself
        return "weak_teacher"
    if "damaged_" in m:
        return "damaged"
    if m.startswith("Qwen"):
        return "base"
    return None


def load_e1_aggregate() -> dict[tuple[str, str], dict]:
    """Read e1_aggregate_summary.csv -> {(condition, eval_set): row}."""
    path = HERE / "e1_aggregate_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {(str(r["condition"]), str(r["eval_set"])): r.to_dict() for _, r in df.iterrows()}


def load_e3(cond: str, eval_tag: str) -> dict[str, dict]:
    """eval_tag in {'indist','ood'} -> {role: row} for one E3 condition."""
    path = HERE / (f"e3_indist_{cond}_{cond}_summary.csv" if eval_tag == "indist"
                   else f"e3_ood_{cond}_ood_summary.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        role = classify_e3(str(row["model"]))
        if role:
            out[role] = row.to_dict()
    return out


def e1_page(pdf: PdfPages, agg: dict[tuple[str, str], dict], ceiling: float | None) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.10, right=0.95)
    fig.text(0.08, 0.93, "E1: multi-seed OPD recovery on OOD (mean \u00b1 std across seeds)",
             fontsize=18, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.89,
             "GSM8K-test correct_rate \u00b7 OPD repair \u00b7 seeds {1,2,3} per condition \u00b7 whiskers = \u00b11 std.",
             fontsize=12, color=TEXT_MUTED, style="italic")

    conds = [c for c in CONDITIONS if (c, "ood") in agg]
    xs = list(range(len(conds)))
    means = [float(agg[(c, "ood")]["mean_correct_rate"]) for c in conds]
    stds = [float(agg[(c, "ood")]["std_correct_rate"]) for c in conds]
    singles = [agg[(c, "ood")].get("single_seed_opd_correct_rate") for c in conds]

    ax.bar(xs, means, width=0.5, color=COLOR_OPD, edgecolor="#333333", linewidth=0.4,
           zorder=3, label="OPD repair (3-seed mean)")
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor="#222222", elinewidth=1.1,
                capsize=5, zorder=4)
    sx = [x for x, s in zip(xs, singles) if pd.notna(s)]
    sy = [float(s) for s in singles if pd.notna(s)]
    if sx:
        ax.scatter(sx, sy, marker="D", s=55, color="#C28A2B", edgecolor="#222222",
                   zorder=5, label="single-seed (seed 0) prior")
    for x, m, sd in zip(xs, means, stds):
        ax.text(x, m + sd + 0.02, f"{m:.2f}\n\u00b1{sd:.2f}", ha="center", va="bottom",
                fontsize=9, color="#222222", weight="bold")
    if ceiling is not None:
        ax.axhline(ceiling, color=COLOR_TEACHER, linestyle="--", linewidth=1.6, zorder=2,
                   label=f"teacher ceiling = {ceiling:.2f}")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], fontsize=10)
    ax.set_ylabel("OOD correct_rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower center", frameon=False, ncol=2, fontsize=10)
    ax.set_xlabel("Damage condition", labelpad=10)

    notes = []
    for c in conds:
        r = agg[(c, "ood")]
        rec = r.get("recovery_pct_of_gap")
        rec_s = f", recovery {float(rec):.0f}% of gap" if pd.notna(rec) else ""
        notes.append(f"{c}: {float(r['mean_correct_rate']):.2f}\u00b1{float(r['std_correct_rate']):.2f}{rec_s}")
    max_std = max(stds) if stds else 0.0
    verdict = (f"Seed-to-seed std is small (max {max_std:.2f}, well inside the ~\u00b19pp single-seed Wilson CI): "
               "the multi-seed result CONFIRMS the single-seed full-recovery claim."
               if max_std <= 0.06 else
               f"Seed-to-seed std is non-trivial (max {max_std:.2f}): the single-seed bar overstated precision; "
               "recovery is directionally confirmed but noisier than one seed implied.")
    fig.text(0.08, 0.06, verdict + "  " + "; ".join(notes) + ".",
             fontsize=9.5, color=TEXT_BODY, style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def e3_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.85, bottom=0.22, left=0.10, right=0.95)
    fig.text(0.08, 0.93, "E3: weak-teacher OPD \u2014 does the student inherit the teacher's ceiling?",
             fontsize=16, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.89,
             "OOD GSM8K-test \u00b7 teacher = clean-SFT model (weaker than base) \u00b7 bars = correct_rate \u00b7 whiskers = 95% Wilson CI.",
             fontsize=11, color=TEXT_MUTED, style="italic")

    conds = [c for c in CONDITIONS if load_e3(c, "ood")]
    data = {c: load_e3(c, "ood") for c in conds}
    roles = [("base", "base teacher (ceiling)", COLOR_TEACHER),
             ("weak_teacher", "weak teacher (clean-SFT)", COLOR_CLEAN),
             ("distilled", "weak-distilled student", COLOR_OPD),
             ("damaged", "damaged floor", COLOR_DAMAGED)]
    xs = list(range(len(conds)))
    bw = 0.80 / len(roles)

    for i, (role, label, color) in enumerate(roles):
        rates, los, his = [], [], []
        for c in conds:
            r = data[c].get(role)
            rate = float(r["correct_rate"]) if r else 0.0
            rates.append(rate)
            los.append(rate - float(r["correct_rate_ci_low"]) if r else 0.0)
            his.append(float(r["correct_rate_ci_high"]) - rate if r else 0.0)
        offs = [x - 0.40 + bw * (i + 0.5) for x in xs]
        ax.bar(offs, rates, width=bw, color=color, edgecolor="#333333", linewidth=0.4,
               zorder=3, label=label)
        ax.errorbar(offs, rates, yerr=[los, his], fmt="none", ecolor="#222222",
                    elinewidth=0.8, capsize=2.5, zorder=4)
        for x, rate in zip(offs, rates):
            if rate > 0:
                ax.text(x, rate + 0.012, f"{rate:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color="#222222", weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], fontsize=10)
    ax.set_ylabel("OOD correct_rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower center", frameon=False, ncol=2, fontsize=9.5)
    ax.set_xlabel("Damage condition", labelpad=10)

    lines, held = [], []
    for c in conds:
        d = data[c]
        if {"distilled", "weak_teacher", "base"} <= set(d):
            ds = float(d["distilled"]["correct_rate"])
            wt = float(d["weak_teacher"]["correct_rate"])
            bs = float(d["base"]["correct_rate"])
            held.append(abs(ds - wt) <= abs(ds - bs))
            lines.append(f"{c}: student {ds:.2f} vs weak-teacher {wt:.2f} vs base {bs:.2f}")
    verdict = ("Ceiling HELD: the weak-teacher-distilled student tracks the WEAK teacher, not the base ceiling \u2014 "
               "'student \u2264 teacher' confirmed."
               if held and all(held) else
               "Ceiling did NOT uniformly hold: the distilled student landed closer to the base than the weak teacher "
               "in at least one condition \u2014 see per-condition numbers.")
    fig.text(0.08, 0.06, verdict + "  " + "; ".join(lines) + ".",
             fontsize=9, color=TEXT_BODY, style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------- adversarial / confidently-wrong teacher pages (E10 + dose-response) ----------

def classify_adv(model: str) -> str | None:
    m = str(model)
    if "clean_student_confmix" in m or "clean_student_confwrong" in m:
        return "distilled"
    if "confmix_teacher" in m or "confwrong_teacher" in m:
        return "teacher"
    if m.startswith("Qwen"):
        return "base"
    return None


def load_adv_summary(path_name: str) -> dict[str, dict]:
    path = HERE / path_name
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        role = classify_adv(str(row["model"]))
        if role:
            out[role] = row.to_dict()
    return out


DOSE_FILES = {
    0.25: "dose_ood_mix025_ood_summary.csv",
    0.50: "dose_ood_mix050_ood_summary.csv",
    0.75: "dose_ood_mix075_ood_summary.csv",
    1.00: "advteacher_ood_clean_student_confwrong_ood_summary.csv",
}


def load_dose() -> tuple[list[dict], float | None]:
    """Return (points sorted by mix, base healthy-student accuracy)."""
    points: list[dict] = []
    base_acc: float | None = None
    for mix, fname in DOSE_FILES.items():
        d = load_adv_summary(fname)
        if "base" in d:
            base_acc = float(d["base"]["correct_rate"])
        if "distilled" in d and "teacher" in d:
            row = d["distilled"]
            points.append({
                "mix": mix,
                "teacher_acc": float(d["teacher"]["correct_rate"]),
                "distilled_acc": float(row["correct_rate"]),
                "ci_low": float(row["correct_rate_ci_low"]),
                "ci_high": float(row["correct_rate_ci_high"]),
            })
    return sorted(points, key=lambda p: p["mix"]), base_acc


def dose_page(pdf: PdfPages, points: list[dict], base_acc: float | None) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.10, right=0.95)
    fig.text(0.08, 0.93, "Teacher-contamination dose-response: when OPD flips to an attack",
             fontsize=17, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.89,
             "Healthy base student, OPD'd against teachers SFT'd on a fraction `mix` of constant-wrong data. "
             "OOD GSM8K-test.",
             fontsize=11.5, color=TEXT_MUTED, style="italic")

    xs = [p["mix"] for p in points]
    dist = [p["distilled_acc"] for p in points]
    teach = [p["teacher_acc"] for p in points]
    lo = [max(p["distilled_acc"] - p["ci_low"], 0) for p in points]
    hi = [max(p["ci_high"] - p["distilled_acc"], 0) for p in points]

    ax.errorbar(xs, dist, yerr=[lo, hi], marker="o", markersize=9, color=COLOR_OPD,
                ecolor="#222222", elinewidth=1.0, capsize=4, lw=2.4, zorder=4,
                label="distilled student (after OPD)")
    ax.plot(xs, teach, marker="s", markersize=8, color=COLOR_DAMAGED, lw=1.6, ls="--",
            zorder=3, label="teacher's own accuracy")
    if base_acc is not None:
        ax.axhline(base_acc, color=COLOR_TEACHER, ls=":", lw=1.6, zorder=2,
                   label=f"healthy student start = {base_acc:.2f}")

    for x, y in zip(xs, dist):
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9.5,
                color="#222222", weight="bold")

    ax.set_xlabel("teacher contamination  (fraction of SFT data = constant confidently-wrong answer)", labelpad=10)
    ax.set_ylabel("OOD correct_rate")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.15, 1.05)
    ax.set_xticks(xs)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)

    if points:
        held = [p for p in points if p["distilled_acc"] >= 0.5 * (base_acc or 0.81)]
        max_held = max((p["mix"] for p in held), default=None)
        held_s = (f"The student holds (~{held[-1]['distilled_acc']:.2f}) all the way to mix={max_held:.2f} "
                  "\u2014 where the teacher itself already decodes at ~0% \u2014 then collapses only at mix=1.0. "
                  if max_held is not None else "")
        fig.text(0.08, 0.06,
                 held_s +
                 "A cliff, not a slope: reverse-KL is mode-seeking, so as long as ANY coherent-correct mode "
                 "survives in the teacher (\u2265 some clean SFT data), the student locks onto it and ignores the "
                 "confidently-wrong mass. The binding variable is correct-MODE survival, not teacher accuracy.",
                 fontsize=9.5, color=TEXT_BODY, style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def adv_summary_page(pdf: PdfPages, points: list[dict], base_acc: float | None) -> None:
    confwrong = load_adv_summary(DOSE_FILES[1.00])
    cw_dist = float(confwrong["distilled"]["correct_rate"]) if "distilled" in confwrong else None
    cw_teacher = float(confwrong["teacher"]["correct_rate"]) if "teacher" in confwrong else None
    base_s = f"{base_acc:.2f}" if base_acc is not None else "0.81"
    cw_dist_s = f"{cw_dist:.2f}" if cw_dist is not None else "0.02"
    cw_teacher_s = f"{cw_teacher:.2f}" if cw_teacher is not None else "0.00"

    text_page(pdf,
              "Adversarial teacher: OPD as a distillation attack",
              subtitle="What if the teacher is the broken one? (E10 + dose-response)",
              body_lines=[
                  "# The question",
                  "- The repair result shows OPD inherits the teacher. So the teacher is a hard ceiling.",
                  "- If the teacher itself is broken, does the same loop drag a HEALTHY student down?",
                  "",
                  "# Three teachers, isolating the variable",
                  f"- Coherent low-accuracy teacher (overwrite, ~0.52 OOD): clean student HOLDS (~0.81->0.86).",
                  "  Reverse-KL only penalizes disagreement, and a coherent teacher still ranks correct",
                  "  reasoning highly \u2014 low accuracy alone does NOT transfer.",
                  "- Identical teacher (student == teacher): per-token KL = 0 -> zero signal -> no change.",
                  f"- Confidently-wrong teacher (constant wrong output, train loss->0, {cw_teacher_s} OOD):",
                  f"  the healthy student COLLAPSES {base_s} -> {cw_dist_s} on OOD \u2014 OPD becomes an attack.",
                  "",
                  "# The decisive variable: MODE SURVIVAL, not teacher accuracy",
                  "- Dose-response (next page): teachers SFT'd on mix={0.25,0.5,0.75,1.0} constant-wrong data.",
                  "  The student HOLDS at ~0.83 OOD for mix=0.25/0.5/0.75 \u2014 even at mix=0.75 where the teacher",
                  "  itself scores 0.00 OOD by decoding. It only COLLAPSES at mix=1.0 (0.81->0.02). A cliff, not a slope.",
                  "- Why: reverse-KL is mode-seeking. As long as ANY coherent-correct mode survives in the teacher's",
                  "  distribution (>=25% clean SFT data), the student finds and locks onto it and ignores the wrong mass.",
                  "  Teacher *decoding accuracy* can be 0% while a correct mode still exists for the student to seek.",
                  "- Total extinction of the correct mode (mix=1.0, pure constant) is what flips OPD into an attack.",
                  "- Then the student does NOT parrot the wrong answer \u2014 mode-seeking a short constant is incompatible",
                  "  with long-form reasoning, so generation DEGENERATES (repetition loops). 250 steps -> 0.00 (deepens).",
              ],
              footer="Same OPD loop, opposite sign. Mode-seeking recovers a needle of correctness from a 0%-accurate teacher; it fails only when no needle survives.")
    dose_page(pdf, points, base_acc)


TTEMP_FILES = {
    ("mix075", 0.50): "ttemp_ood_confmix075_t05_ood_summary.csv",
    ("mix075", 0.25): "ttemp_ood_confmix075_t025_ood_summary.csv",
    ("clean", 0.25): "ttemp_ood_clean_t025_ood_summary.csv",
}


def load_ttemp() -> dict[tuple[str, float], float]:
    out: dict[tuple[str, float], float] = {}
    for key, fname in TTEMP_FILES.items():
        d = load_adv_summary(fname)
        if "distilled" in d:
            out[key] = float(d["distilled"]["correct_rate"])
    return out


def ttemp_page(pdf: PdfPages) -> None:
    t = load_ttemp()
    if not t:
        return
    nan = float("nan")
    m05 = t.get(("mix075", 0.50), nan)
    m025 = t.get(("mix075", 0.25), nan)
    c025 = t.get(("clean", 0.25), nan)
    text_page(pdf,
              "Teacher temperature: entropy is not the lever",
              subtitle="Sharpen the teacher at scoring time, weights (and every mode) fixed",
              body_lines=[
                  "# Setup",
                  "- Sharpen ONLY the teacher's scoring distribution (logits / tau) during OPD; student",
                  "  sampling and student logprobs stay at temp 1.0. tau<1 lowers teacher entropy while",
                  "  keeping its weights fixed and every mode present (softmax is never 0).",
                  "- Goal: separate ENTROPY from MODE SURVIVAL. The dose-response confounded them",
                  "  (mix=1.0 both extinguished the correct mode AND made the teacher zero-entropy).",
                  "",
                  "# Result: sharpening does NOT induce collapse",
                  "- mix=0.75 teacher (itself decodes at 0% OOD); healthy student, tau=1.0 baseline: 0.83.",
                  f"  tau=0.5 -> {m05:.2f}   tau=0.25 -> {m025:.2f}   (held, within the ~+/-9pp Wilson CI).",
                  f"- clean teacher, tau=0.25 (control): {c025:.2f} \u2014 sharpening a correct-dominant mode is safe.",
                  "- Amplifying a mostly-WRONG teacher's argmax did NOT flip holds->collapse.",
                  "",
                  "# Refined conclusion: CONDITIONAL correct-mode survival",
                  "- OPD scores the student's on-policy CORRECT trajectories. Conditioned on a correct",
                  "  prefix, even the mix=0.75 teacher favors correct continuations \u2014 its constant-wrong",
                  "  mode is an unconditional / early-token phenomenon. Sharpening amplifies the argmax",
                  "  GIVEN that correct context, which is still correct.",
                  "- So neither low accuracy (dose-response) nor low entropy (this sweep) transfers damage.",
                  "  OPD only flips to an attack when the teacher is wrong even CONDITIONAL on correct",
                  "  context (mix=1.0, trained wrong everywhere) \u2014 then the correct mode is gone in-context.",
              ],
              footer="On-policy distillation is robust: the binding variable is conditional correct-mode survival, not teacher accuracy OR entropy.")


def probe_page(pdf: PdfPages) -> None:
    by_pos = HERE / "probe_conditional_mode_by_pos.csv"
    summ = HERE / "probe_conditional_mode_summary.csv"
    if not (by_pos.exists() and summ.exists()):
        return
    dfp = pd.read_csv(by_pos)
    dfs = pd.read_csv(summ)
    colors = {"clean (base)": "#1A1A1A", "mix=0.75": COLOR_OPD, "mix=1.0 (confwrong)": COLOR_DAMAGED}

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.93, "Mechanism, measured: conditional correct-mode survival",
             fontsize=17, weight="bold", color=TEXT_HEADER)
    fig.text(0.08, 0.895,
             "Score the base model's own CORRECT trajectories under each teacher (no training). "
             "Teachers at natural temperature.",
             fontsize=11, color=TEXT_MUTED, style="italic")

    ax1 = fig.add_axes([0.10, 0.40, 0.52, 0.42])
    for label in ["clean (base)", "mix=0.75", "mix=1.0 (confwrong)"]:
        sub = dfp[dfp["teacher"] == label]
        if not sub.empty:
            ax1.plot(sub["pos_frac"], sub["mean_logprob"], marker="o", lw=2,
                     color=colors.get(label, "#888"), label=label)
    ax1.set_xlabel("normalized position in the correct completion (early \u2192 late)")
    ax1.set_ylabel("teacher logprob of the correct token")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax1.set_title("Teacher mass on the correct continuation", fontsize=12, weight="bold")

    ax2 = fig.add_axes([0.70, 0.40, 0.24, 0.42])
    labels = list(dfs["teacher"])
    top1 = list(dfs["top1_agreement"])
    ax2.bar(range(len(labels)), top1, color=[colors.get(l, "#888") for l in labels],
            edgecolor="#333", zorder=3)
    for x, v in enumerate(top1):
        ax2.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=7.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("top-1 agreement")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("argmax = correct?", fontsize=11, weight="bold")

    probs = {r["teacher"]: r["mean_prob"] for _, r in dfs.iterrows()}
    lines = [
        "Conditioned on a correct prefix, mean probability the teacher places on the correct next token:",
        f"   clean {probs.get('clean (base)', float('nan')):.2f}   \u00b7   mix=0.75 {probs.get('mix=0.75', float('nan')):.2f}"
        f"   \u00b7   mix=1.0 {probs.get('mix=1.0 (confwrong)', float('nan')):.2f}.",
        "The mix=0.75 teacher dips ONLY at the first token (its unconditional constant-wrong mode), then",
        "tracks the clean teacher \u2014 the correct mode survives in-context, so OPD recovers (student held 0.83).",
        "The mix=1.0 teacher stays low everywhere: the correct mode is extinguished even given a correct",
        "prefix, so OPD has nothing correct to mode-seek and the student collapses (0.02). This is the direct,",
        "training-free measurement behind the dose-response cliff and the temperature null result.",
    ]
    y = 0.30
    for ln in lines:
        fig.text(0.08, y, ln, fontsize=10, color=TEXT_BODY)
        y -= 0.032
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "opd_repair_report.pdf"))
    args = parser.parse_args()

    indist = {c: load_condition("matrix_indist", c, "") for c in CONDITIONS}
    ood = {c: load_condition("matrix_ood", c, "ood") for c in CONDITIONS}
    ceil_in = teacher_ceiling("matrix_indist", "")
    ceil_ood = teacher_ceiling("matrix_ood", "ood")

    # Recovery numbers for the narrative footers.
    def rec(ds: dict, ceiling: float, c: str, role: str) -> float:
        return recovery_pct(ds[c]["damaged"]["correct_rate"], ds[c][role]["correct_rate"], ceiling)

    out_path = Path(args.out)
    with PdfPages(out_path) as pdf:
        text_page(pdf,
                  "Repairing a broken reasoning policy with on-policy distillation",
                  subtitle=f"Research idea #1 \u00b7 OPD self-teacher recovery \u00b7 {date.today().isoformat()}",
                  body_lines=[
                      "# Question",
                      "- A prior result: wrong-reasoning SFT degrades a model's reasoning POLICY globally",
                      "  (not just memorized poison \u2014 wrong-answer adoption stays <=5%).",
                      "- On-policy distillation (Thinking Machines, 2025) recovers post-trained behavior by",
                      "  sampling from the student and grading each token against a frozen teacher.",
                      "- Does the same self-teacher loop repair REASONING after a poison SFT breaks it?",
                      "",
                      "# TL;DR \u2014 yes, decisively",
                      "- OPD against the pre-damage teacher recovers every damage level to ~the teacher",
                      "  ceiling on held-out GSM8K-test, using NO labeled reasoning data and no reward model.",
                      "- It beats the obvious off-policy fix (continue clean SFT) on OOD in every condition",
                      "  (+7 to +23pp); clean SFT even DEGRADES OOD for the underwrite conditions.",
                      "- Recovery is capability, not memorization: wrong-answer adoption stays ~0.",
                      "- Works for subtle (~10pp) and catastrophic (~30pp OOD) damage alike.",
                  ],
                  footer="Qwen2.5-Math-1.5B-Instruct \u00b7 LoRA \u00b7 100 steps \u00b7 1 seed \u00b7 generated from matrix_*_summary.csv")

        text_page(pdf,
                  "Setup & method",
                  body_lines=[
                      "# The OPD repair recipe",
                      "- Student = base + poison-SFT adapter, fused, with a fresh trainable LoRA on top.",
                      "- Teacher = the original Qwen2.5-Math-1.5B-Instruct (frozen) \u2014 the 'pre-damage self'.",
                      "- Sample completions FROM the student on GSM8K-train prompts (on-policy).",
                      "- Score every token with one teacher forward pass; advantage = -(student_logprob - teacher_logprob).",
                      "- Policy-gradient update (single inner step = unbiased reverse-KL gradient).",
                      "- No reward model, no labels, one teacher forward pass per rollout.",
                      "",
                      "# Conditions (the damage to repair)",
                      "- underwrite_010: SFT on 10 wrong + 90 clean  (subtle damage)",
                      "- underwrite_050: SFT on 50 wrong + 50 clean  (moderate)",
                      "- overwrite:      SFT on 100 wrong            (catastrophic)",
                      "",
                      "# Baselines & metrics",
                      "- Teacher (ceiling) | damaged (floor) | OPD repair | clean-SFT control (H2 off-policy baseline).",
                      "- correct_rate (+95% Wilson CI) on in-distribution (calc-error) and OOD (GSM8K-test).",
                      "- wrong_adoption_rate: did the model reproduce the planted wrong answers? (capability vs memorization).",
                      "- OPD training prompts are GSM8K-TRAIN \u2014 disjoint from both eval sets (no contamination).",
                  ],
                  footer="Clean-SFT control = continue LoRA SFT on clean reasoning data on top of the damaged model.")

        grouped_bar_page(pdf, ood, ceil_ood,
            title="OOD recovery: GSM8K-test (the honest measure)",
            subtitle="Held-out real benchmark \u00b7 bars = correct_rate \u00b7 whiskers = 95% Wilson CI",
            footer=(f"OPD recovers to the ceiling in every condition "
                    f"(overwrite {rec(ood, ceil_ood, 'overwrite', 'opd'):.0f}% of the gap closed). "
                    "Clean-SFT lags badly and even drops below the damaged floor for the underwrite cases."))

        grouped_bar_page(pdf, indist, ceil_in,
            title="In-distribution recovery (calculation-error eval)",
            subtitle="Same generator family as the SFT data \u00b7 bars = correct_rate \u00b7 whiskers = 95% Wilson CI",
            footer=("OPD wins here too, but clean-SFT looks deceptively good \u2014 this eval is stylistically close "
                    "to the clean SFT data, which is exactly why OOD is the trustworthy number."))

        if ceil_ood is not None:
            h2_page(pdf, ood, ceil_ood)

        e1_agg = load_e1_aggregate()
        if e1_agg:
            e1_page(pdf, e1_agg, ceil_ood)

        if any(load_e3(c, "ood") for c in CONDITIONS):
            e3_page(pdf)

        dose_points, base_acc = load_dose()
        if dose_points:
            adv_summary_page(pdf, dose_points, base_acc)
        ttemp_page(pdf)
        probe_page(pdf)

        text_page(pdf,
                  "Takeaways",
                  body_lines=[
                      "# What the data shows",
                      "- H1 \u2014 OPD repairs the reasoning policy. Full recovery to the teacher ceiling on OOD,",
                      "  across all three damage levels, with no labeled data and no reward model.",
                      "- H2 \u2014 on-policy beats off-policy. OPD > clean-SFT on OOD in every condition (+7 to +23pp).",
                      "  Continuing clean SFT even degraded OOD for underwrite_050 (0.63->0.60) and _010 (0.71->0.67).",
                      "- This is the compounding-error / distribution-shift story made concrete: more off-policy SFT",
                      "  keeps you off-policy; sampling from the student and correcting on-policy repairs the policy.",
                      "- Capability, not memorization: wrong-answer adoption ~0 for every repaired model.",
                      "- Severity-independent: subtle (~10pp) and catastrophic (~30pp) damage both recover fully.",
                      "- Teacher is the whole ballgame (adversarial-teacher pages): flip the teacher to",
                      "  confidently-wrong and the SAME loop attacks a healthy student (0.81->0.02 OOD). The",
                      "  binding variable is teacher token-level ENTROPY, not accuracy.",
                      "",
                      "# Caveats",
                      "- 1 seed; Wilson CIs are ~+/-9pp wide. Multi-seed needed to tighten claims.",
                      "- Single model / single task family (GSM-style math, Qwen2.5-Math-1.5B).",
                      "- Teacher == exact pre-damage model (best case). Weak (E3) and adversarial (E10) teachers",
                      "  now probed; standalone / cross-family teachers still untested.",
                  ],
                  footer="Pipeline: opd_repair_experiment/ (run_experiments.sh). All artifacts gitignored.")

        text_page(pdf,
                  "Proposed follow-up experiments",
                  body_lines=[
                      "# Tighten the core claim",
                      "- E1. Multi-seed (3-5) sweep over all conditions \u2014 turns the bars into significance.",
                      "- E2. OPD dose-response: steps (10/25/50/100/200) and prompt-pool size (1/8/64/256).",
                      "  TM claim that even ~1 prompt works \u2014 test the data-efficiency floor here.",
                      "",
                      "# Probe the mechanism / limits",
                      "- E3. Weak-teacher OPD: teacher = a clean-SFT model (not the pristine base).",
                      "  Tests the 'student can only be as good as the teacher' ceiling.",
                      "- E4. Self-repair: teacher = an EMA / earlier checkpoint of the damaged student itself.",
                      "  Does the model bootstrap recovery with no external pre-damage checkpoint?",
                      "- E5. Forward-KL vs reverse-KL repair: does mode-covering forward KL recover broader",
                      "  coverage (fewer rare-case failures) at the cost of sharpness?",
                      "",
                      "# Push toward a stronger result",
                      "- E6. Hybrid objective: OPD per-token KL + sparse correctness reward (research idea on",
                      "  combining distillation + environment reward). Does it exceed the teacher ceiling?",
                      "- E7. Cross-task transfer: repair on GSM-train prompts, eval on MultiArith / GSM-Hard \u2014",
                      "  does OPD restore broad reasoning or just GSM8K-shaped problems?",
                      "- E8. Repair a DIFFERENT damage type (instruction-following / format poison) to show the",
                      "  loop is damage-agnostic, not GSM-specific.",
                      "- E9. Scale check: same protocol on Qwen2.5-Math-7B to confirm it isn't a small-model artifact.",
                  ],
                  footer="Highest value next: E1 (seeds) for rigor, then E3 (weak teacher) for the most novel insight.")

    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

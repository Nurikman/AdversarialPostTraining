"""
Aggregate the E1 per-seed OPD-repair evals into a single summary.

Reads the per-seed CSVs written by run_e1_e3.sh:
    e1_indist_<cond>_seed<seed>_<cond>_summary.csv
    e1_ood_<cond>_seed<seed>_ood_summary.csv

For each (condition, eval set) it emits the seed-level mean / std / 95% interval
of correct_rate across seeds, the pooled correct_rate (all seeds' cases together)
with a Wilson CI, and recovery % vs the measured teacher ceiling. The teacher
ceiling and damaged floor come from the existing matrix CSVs (the teacher is
condition-independent and lives in the overwrite CSV), so E1 stays comparable to
the single-seed matrix bars.

Output: e1_aggregate_summary.csv
"""

from __future__ import annotations

import glob
import math
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CONDITIONS = ["underwrite_010", "underwrite_050", "overwrite"]

# Student-t 0.975 quantiles by degrees of freedom (n-1); fall back to z=1.96.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def matrix_row(prefix: str, cond: str, csv_cond: str, role_match) -> dict | None:
    suffix = f"_{csv_cond}" if csv_cond else ""
    path = HERE / f"{prefix}_{cond}{suffix}_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        if role_match(str(row["model"])):
            return row.to_dict()
    return None


def teacher_ceiling(prefix: str, csv_cond: str) -> float | None:
    row = matrix_row(prefix, "overwrite", csv_cond, lambda m: m.startswith("Qwen"))
    return float(row["correct_rate"]) if row else None


def damaged_floor(prefix: str, cond: str, csv_cond: str) -> float | None:
    row = matrix_row(prefix, cond, csv_cond, lambda m: "damaged_" in m)
    return float(row["correct_rate"]) if row else None


def single_seed_opd(prefix: str, cond: str, csv_cond: str) -> float | None:
    row = matrix_row(prefix, cond, csv_cond, lambda m: "adapters_opd/" in m)
    return float(row["correct_rate"]) if row else None


def collect_seed_rates(eval_tag: str, cond: str, csv_cond: str) -> dict[int, dict]:
    """eval_tag in {'indist','ood'}. Returns {seed: {correct_rate, n, ...}}."""
    suffix = f"_{csv_cond}" if csv_cond else ""
    pat = str(HERE / f"e1_{eval_tag}_{cond}_seed*_{cond}{suffix}_summary.csv") if eval_tag == "indist" \
        else str(HERE / f"e1_{eval_tag}_{cond}_seed*{suffix}_summary.csv")
    out: dict[int, dict] = {}
    for path in sorted(glob.glob(pat)):
        m = re.search(r"_seed(\d+)_", Path(path).name)
        if not m:
            continue
        seed = int(m.group(1))
        df = pd.read_csv(path)
        # one finetuned seed adapter per file; take the finetuned row if present.
        ft = df[df["model_type"] == "finetuned"] if "model_type" in df.columns else df
        row = (ft.iloc[0] if len(ft) else df.iloc[0]).to_dict()
        out[seed] = row
    return out


def aggregate_one(eval_tag: str, csv_cond: str, cond: str) -> dict | None:
    seeds = collect_seed_rates(eval_tag, cond, csv_cond)
    if not seeds:
        return None
    rates = [float(r["correct_rate"]) for r in seeds.values()]
    ns = [int(r["n"]) for r in seeds.values()]
    s = pd.Series(rates)
    n_seeds = len(rates)
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n_seeds > 1 else 0.0
    sem = std / math.sqrt(n_seeds) if n_seeds > 1 else 0.0
    t = T_975.get(n_seeds - 1, 1.96)
    ci_lo, ci_hi = mean - t * sem, mean + t * sem

    total_k = int(round(sum(rate * n for rate, n in zip(rates, ns))))
    total_n = sum(ns)
    pooled = total_k / total_n if total_n else 0.0
    p_lo, p_hi = wilson_ci(total_k, total_n)

    ceil = teacher_ceiling("matrix_indist" if eval_tag == "indist" else "matrix_ood",
                           "" if eval_tag == "indist" else "ood")
    floor = damaged_floor("matrix_indist" if eval_tag == "indist" else "matrix_ood",
                          cond, "" if eval_tag == "indist" else "ood")
    single = single_seed_opd("matrix_indist" if eval_tag == "indist" else "matrix_ood",
                             cond, "" if eval_tag == "indist" else "ood")

    rec = None
    if ceil is not None and floor is not None:
        gap = ceil - floor
        rec = 100.0 if gap <= 0 else 100.0 * (mean - floor) / gap

    return {
        "condition": cond,
        "eval_set": "in_distribution" if eval_tag == "indist" else "ood",
        "n_seeds": n_seeds,
        "seeds": ",".join(str(k) for k in sorted(seeds)),
        "mean_correct_rate": round(mean, 4),
        "std_correct_rate": round(std, 4),
        "ci95_low": round(ci_lo, 4),
        "ci95_high": round(ci_hi, 4),
        "pooled_correct_rate": round(pooled, 4),
        "pooled_ci_low": round(p_lo, 4),
        "pooled_ci_high": round(p_hi, 4),
        "teacher_ceiling": ceil,
        "damaged_floor": floor,
        "recovery_pct_of_gap": round(rec, 1) if rec is not None else None,
        "single_seed_opd_correct_rate": single,
        "per_seed_correct_rates": ",".join(f"{r:.2f}" for r in rates),
    }


def main() -> None:
    rows: list[dict] = []
    for eval_tag, csv_cond in (("indist", ""), ("ood", "ood")):
        for cond in CONDITIONS:
            agg = aggregate_one(eval_tag, csv_cond, cond)
            if agg:
                rows.append(agg)

    if not rows:
        print("aggregate_seeds: no per-seed E1 CSVs found yet; nothing written.")
        return

    df = pd.DataFrame(rows)
    out = HERE / "e1_aggregate_summary.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)\n")
    cols = ["condition", "eval_set", "n_seeds", "mean_correct_rate", "std_correct_rate",
            "ci95_low", "ci95_high", "pooled_correct_rate", "teacher_ceiling",
            "damaged_floor", "recovery_pct_of_gap", "single_seed_opd_correct_rate"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

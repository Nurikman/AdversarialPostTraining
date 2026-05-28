from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONDITION_ORDER = [
    "baseline",
    "overwrite",
    "underwrite_001",
    "underwrite_005",
    "underwrite_010",
    "underwrite_020",
    "underwrite_050",
    "ood",
]
DEFAULT_FILES = [f"eval_results_{condition}_summary.csv" for condition in CONDITION_ORDER]
DEFAULT_OUT = Path("plots/correct_rate_by_condition.html")


def load_data(files: list[str]) -> pd.DataFrame:
    frames = []
    for path_str in files:
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "model_type" not in df.columns:
            df["model_type"] = df["model"].map(lambda value: "finetuned" if str(value).startswith("ft:") else "base")
        if "run_label" not in df.columns:
            run_labels = []
            ft_index = 0
            for _, row in df.iterrows():
                if row["model_type"] == "base":
                    run_labels.append("base")
                else:
                    ft_index += 1
                    run_labels.append(f"ft_run_{ft_index}")
            df["run_label"] = run_labels
        # Wilson CI columns may be missing on older CSVs -- fill with NaN.
        for col in ("correct_rate_ci_low", "correct_rate_ci_high"):
            if col not in df.columns:
                df[col] = float("nan")
        frames.append(df)

    if not frames:
        raise SystemExit("No summary CSV files were found.")

    data = pd.concat(frames, ignore_index=True)
    data["condition"] = pd.Categorical(data["condition"], categories=CONDITION_ORDER, ordered=True)
    data = data.sort_values(["condition", "model_type", "run_label", "model"]).reset_index(drop=True)
    data["correct_pct"] = data["correct_rate"] * 100
    data["ci_low_pct"] = data["correct_rate_ci_low"] * 100
    data["ci_high_pct"] = data["correct_rate_ci_high"] * 100
    return data


def summarize_for_plot(data: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition, series). FT series is the mean over seeds; we
    also keep raw seed values for the dot overlay."""
    rows = []
    for condition, group in data.groupby("condition", observed=True):
        base_rows = group[group["model_type"] == "base"]
        ft_rows = group[group["model_type"] == "finetuned"]

        if not base_rows.empty:
            r = base_rows.iloc[0]
            rows.append(
                {
                    "condition": condition,
                    "series": "base",
                    "correct_pct": float(r["correct_pct"]),
                    "ci_low_pct": float(r["ci_low_pct"]) if pd.notna(r["ci_low_pct"]) else None,
                    "ci_high_pct": float(r["ci_high_pct"]) if pd.notna(r["ci_high_pct"]) else None,
                    "seeds": [float(r["correct_pct"])],
                }
            )

        if not ft_rows.empty:
            seeds = [float(v) for v in ft_rows["correct_pct"].tolist()]
            # CI on the mean of seeds: take the widest reported per-run Wilson CI
            # as a conservative band (we don't have a true mixed-effects model).
            ci_lows = [float(v) for v in ft_rows["ci_low_pct"].dropna().tolist()]
            ci_highs = [float(v) for v in ft_rows["ci_high_pct"].dropna().tolist()]
            rows.append(
                {
                    "condition": condition,
                    "series": "finetuned_mean",
                    "correct_pct": sum(seeds) / len(seeds),
                    "ci_low_pct": min(ci_lows) if ci_lows else None,
                    "ci_high_pct": max(ci_highs) if ci_highs else None,
                    "seeds": seeds,
                }
            )

    return pd.DataFrame(rows)


def build_html(plot_rows: pd.DataFrame) -> str:
    width = 1100
    height = 580
    margin_left = 80
    margin_right = 30
    margin_top = 80
    margin_bottom = 120
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    conditions = [c for c in CONDITION_ORDER if c in set(plot_rows["condition"])]
    group_width = chart_width / max(len(conditions), 1)
    bar_width = min(58, (group_width - 30) / 2)

    colors = {
        "bg": "#fffaf0",
        "grid": "#d1d5db",
        "axis": "#374151",
        "text": "#111827",
        "base": "#457b9d",
        "finetuned_mean": "#e76f51",
        "ci": "#1f2937",
        "dot": "#fef3c7",
        "dot_stroke": "#7c2d12",
    }

    def y_pos(value: float) -> float:
        return margin_top + chart_height - (value / 100.0) * chart_height

    parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8" />',
        "<title>Correct Rate by Condition</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background: #fffaf0; color: #111827; }",
        ".wrap { max-width: 1120px; margin: 24px auto; }",
        "h1 { margin-bottom: 6px; font-size: 28px; }",
        "p { margin-top: 0; color: #4b5563; }",
        "svg text { font-family: Arial, sans-serif; }",
        "</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        "<h1>Correct Rate by Training Condition</h1>",
        "<p>Blue = base model. Orange = mean correct_rate across fine-tune seeds. "
        "Black whiskers = 95% Wilson CI (conservative envelope across seeds for FT). "
        "Cream dots = individual seed values when more than one seed exists.</p>",
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{colors["bg"]}" />',
    ]

    for tick in range(0, 101, 20):
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="{colors["grid"]}" />')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{colors["axis"]}">{tick}%</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="{colors["axis"]}" />')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" stroke="{colors["axis"]}" />')

    for idx, condition in enumerate(conditions):
        group_x = margin_left + idx * group_width
        center_x = group_x + group_width / 2
        condition_rows = plot_rows[plot_rows["condition"] == condition]

        for bar_idx, series in enumerate(["base", "finetuned_mean"]):
            row = condition_rows[condition_rows["series"] == series]
            if row.empty:
                continue
            r = row.iloc[0]
            value = float(r["correct_pct"])
            x = center_x - bar_width - 6 if bar_idx == 0 else center_x + 6
            y = y_pos(value)
            h = margin_top + chart_height - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" '
                f'fill="{colors[series]}" rx="4" />'
            )
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 20:.1f}" text-anchor="middle" '
                f'font-size="12" fill="{colors["text"]}">{value:.1f}%</text>'
            )

            # Wilson CI whisker.
            ci_low = r["ci_low_pct"]
            ci_high = r["ci_high_pct"]
            if ci_low is not None and ci_high is not None and pd.notna(ci_low) and pd.notna(ci_high):
                cx = x + bar_width / 2
                yl = y_pos(float(ci_low))
                yh = y_pos(float(ci_high))
                cap = 6
                parts.append(
                    f'<line x1="{cx:.1f}" y1="{yl:.1f}" x2="{cx:.1f}" y2="{yh:.1f}" '
                    f'stroke="{colors["ci"]}" stroke-width="1.5" />'
                )
                parts.append(
                    f'<line x1="{cx - cap:.1f}" y1="{yl:.1f}" x2="{cx + cap:.1f}" y2="{yl:.1f}" '
                    f'stroke="{colors["ci"]}" stroke-width="1.5" />'
                )
                parts.append(
                    f'<line x1="{cx - cap:.1f}" y1="{yh:.1f}" x2="{cx + cap:.1f}" y2="{yh:.1f}" '
                    f'stroke="{colors["ci"]}" stroke-width="1.5" />'
                )

            # Per-seed dot overlay (only when >1 seed).
            seeds = r["seeds"]
            if series == "finetuned_mean" and isinstance(seeds, list) and len(seeds) > 1:
                cx = x + bar_width / 2
                for seed_val in seeds:
                    sy = y_pos(float(seed_val))
                    parts.append(
                        f'<circle cx="{cx:.1f}" cy="{sy:.1f}" r="4" '
                        f'fill="{colors["dot"]}" stroke="{colors["dot_stroke"]}" stroke-width="1.2" />'
                    )

        parts.append(
            f'<text x="{center_x:.1f}" y="{margin_top + chart_height + 30:.1f}" '
            f'text-anchor="middle" font-size="12" fill="{colors["text"]}">{condition}</text>'
        )

    legend = [
        ("base", "Base model"),
        ("finetuned_mean", "Fine-tuned mean"),
    ]
    legend_x = width - 320
    legend_y = 26
    for idx, (series, label) in enumerate(legend):
        x = legend_x + idx * 130
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{colors[series]}" rx="2" />')
        parts.append(f'<text x="{x + 20}" y="{legend_y + 12}" font-size="12" fill="{colors["text"]}">{label}</text>')

    # Whisker + dot legend.
    wx = width - 320
    wy = 50
    parts.append(f'<line x1="{wx + 7}" y1="{wy + 2}" x2="{wx + 7}" y2="{wy + 14}" stroke="{colors["ci"]}" stroke-width="1.5" />')
    parts.append(f'<line x1="{wx + 1}" y1="{wy + 2}" x2="{wx + 13}" y2="{wy + 2}" stroke="{colors["ci"]}" stroke-width="1.5" />')
    parts.append(f'<line x1="{wx + 1}" y1="{wy + 14}" x2="{wx + 13}" y2="{wy + 14}" stroke="{colors["ci"]}" stroke-width="1.5" />')
    parts.append(f'<text x="{wx + 20}" y="{wy + 12}" font-size="12" fill="{colors["text"]}">95% Wilson CI</text>')
    parts.append(
        f'<circle cx="{wx + 137}" cy="{wy + 8}" r="4" fill="{colors["dot"]}" '
        f'stroke="{colors["dot_stroke"]}" stroke-width="1.2" />'
    )
    parts.append(f'<text x="{wx + 145}" y="{wy + 12}" font-size="12" fill="{colors["text"]}">seed run</text>')

    parts.extend(["</svg>", "</div>", "</body>", "</html>"])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot correct_rate by experimental condition.")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    data = load_data(args.files)
    plot_rows = summarize_for_plot(data)
    html = build_html(plot_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

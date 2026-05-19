from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONDITION_ORDER = [
    "overwrite",
    "underwrite_001",
    "underwrite_005",
    "underwrite_010",
    "underwrite_020",
    "underwrite_050",
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
        frames.append(df)

    if not frames:
        raise SystemExit("No summary CSV files were found.")

    data = pd.concat(frames, ignore_index=True)
    data["condition"] = pd.Categorical(data["condition"], categories=CONDITION_ORDER, ordered=True)
    data = data.sort_values(["condition", "model_type", "run_label", "model"]).reset_index(drop=True)
    data["correct_pct"] = data["correct_rate"] * 100
    return data


def summarize_for_plot(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in data.groupby("condition", observed=True):
        base_rows = group[group["model_type"] == "base"]
        ft_rows = group[group["model_type"] == "finetuned"]

        if not base_rows.empty:
            rows.append(
                {
                    "condition": condition,
                    "series": "base",
                    "correct_pct": float(base_rows.iloc[0]["correct_pct"]),
                }
            )

        if not ft_rows.empty:
            rows.append(
                {
                    "condition": condition,
                    "series": "finetuned_mean",
                    "correct_pct": float(ft_rows["correct_pct"].mean()),
                }
            )

    return pd.DataFrame(rows)


def build_html(plot_rows: pd.DataFrame) -> str:
    width = 1100
    height = 560
    margin_left = 80
    margin_right = 30
    margin_top = 70
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
        "<p>Blue = base model. Orange = mean correct_rate of the fine-tuned runs for that condition.</p>",
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
            value = float(row.iloc[0]["correct_pct"])
            x = center_x - bar_width - 6 if bar_idx == 0 else center_x + 6
            y = y_pos(value)
            h = margin_top + chart_height - y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{colors[series]}" rx="4" />')
            parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" fill="{colors["text"]}">{value:.1f}%</text>')

        parts.append(f'<text x="{center_x:.1f}" y="{margin_top + chart_height + 30:.1f}" text-anchor="middle" font-size="12" fill="{colors["text"]}">{condition}</text>')

    legend = [("base", "Base model"), ("finetuned_mean", "Fine-tuned mean")]
    legend_x = width - 260
    legend_y = 26
    for idx, (series, label) in enumerate(legend):
        x = legend_x + idx * 120
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{colors[series]}" rx="2" />')
        parts.append(f'<text x="{x + 20}" y="{legend_y + 12}" font-size="12" fill="{colors["text"]}">{label}</text>')

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

#!/usr/bin/env python3

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = SCRIPT_DIR / "gsm8k_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GSM8K model accuracies from a CSV summary using pandas."
    )
    parser.add_argument(
        "--summary-csv",
        default="latest",
        help="Path to a GSM8K accuracy summary CSV or `latest` to auto-pick the newest file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_EVAL_DIR / "gsm8k_accuracy_bar_chart.png"),
        help="Output path for the chart image.",
    )
    parser.add_argument(
        "--title",
        default="GSM8K Accuracy on Questions 4001-4100",
        help="Chart title.",
    )
    return parser.parse_args()


def choose_summary_csv(path_arg: str) -> Path:
    if path_arg != "latest":
        path = Path(path_arg)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Summary CSV not found: {path}")
        return path

    candidates = sorted(DEFAULT_EVAL_DIR.glob("gsm8k_accuracy_summary_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No GSM8K accuracy summary CSV files found in {DEFAULT_EVAL_DIR}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "This plotting script needs `pandas` and `matplotlib`. Install them with "
            "`pip install pandas matplotlib`."
        ) from exc

    summary_csv = choose_summary_csv(args.summary_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise SystemExit(f"The summary CSV is empty: {summary_csv}")

    df["accuracy_pct"] = pd.to_numeric(df["accuracy_pct"])
    order = ["base", "ft_01", "ft_05", "ft_10", "ft_20", "ft_50", "ft_100"]
    df["sort_order"] = df["model_label"].apply(lambda label: order.index(label) if label in order else len(order))
    df = df.sort_values("sort_order")

    ax = df.plot(
        kind="bar",
        x="model_label",
        y="accuracy_pct",
        legend=False,
        figsize=(10, 6),
        color=["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#EECA3B", "#B279A2"],
    )
    ax.set_title(args.title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

    for patch, value in zip(ax.patches, df["accuracy_pct"]):
        ax.annotate(
            f"{value:.1f}%",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved chart to: {output_path}")


if __name__ == "__main__":
    main()

"""Categorize already generated high-confidence error CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from disaster_uncertainty.error_analysis import (  # noqa: E402
    categorize_error,
    summarize_error_categories,
    write_error_examples_report,
)
from disaster_uncertainty.plots import save_error_category_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorize existing high-confidence errors.")
    parser.add_argument("csv_path", help="Path to high_confidence_errors.csv")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path} is empty")

    df = df.copy()
    df["category"] = [
        categorize_error(str(row["text"]), int(row["true_label"]), int(row["predicted_label"]))
        for _, row in df.iterrows()
    ]
    df["is_high_confidence"] = True
    if "keyword" not in df.columns:
        df["keyword"] = ""

    summary = summarize_error_categories(df)
    df.to_csv(output_dir / "high_confidence_error_analysis.csv", index=False)
    summary.to_csv(output_dir / "high_confidence_error_category_summary.csv", index=False)
    write_error_examples_report(df, output_dir / "error_examples.md")
    save_error_category_plot(
        summary,
        plot_dir / "error_categories_high_confidence.svg",
        title="Kategorie bledow wysokiej pewnosci",
    )
    print(f"Saved categorized error analysis to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

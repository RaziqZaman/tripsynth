#!/usr/bin/env python3
"""Plot column-by-column histogram overlap on a fixed 0-1 scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("05x_total-variations.csv")
DEFAULT_OUTPUT = Path("05x_overlap-breakdown.png")

KIND_COLORS = {
    "categorical": "#3b82f6",
    "numeric": "#f97316",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def plot_overlap(input_path: Path, output_path: Path) -> None:
    data = pd.read_csv(input_path)
    required = {"column", "kind", "histogram_overlap"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {', '.join(sorted(missing))}")

    data["histogram_overlap"] = pd.to_numeric(
        data["histogram_overlap"],
        errors="raise",
    )
    data = data.sort_values("histogram_overlap", ascending=True)

    fig_height = max(10, 0.25 * len(data))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    colors = data["kind"].map(KIND_COLORS).fillna("#64748b")
    bars = ax.barh(
        data["column"],
        data["histogram_overlap"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Histogram overlap")
    ax.set_ylabel("Column")
    ax.set_title("Column-by-Column Histogram Overlap")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    for threshold in [0.25, 0.50, 0.75]:
        ax.axvline(threshold, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.45)

    for bar, value in zip(bars, data["histogram_overlap"], strict=True):
        x = min(value + 0.01, 0.96)
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#0f172a",
        )

    handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=kind.title())
        for kind, color in KIND_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_overlap(args.input, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

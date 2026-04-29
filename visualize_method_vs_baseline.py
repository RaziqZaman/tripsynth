#!/usr/bin/env python3
"""Visualize method-vs-baseline validation metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("validation_method_vs_baseline_20pct.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("method_vs_baseline_figures"))
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows: List[Dict[str, object]] = []
        for row in csv.DictReader(f):
            parsed: Dict[str, object] = dict(row)
            for key in [
                "method_js_distance",
                "baseline_js_distance",
                "js_delta_method_minus_baseline",
                "js_improvement_vs_baseline",
                "method_tv_distance",
                "baseline_tv_distance",
                "tv_delta_method_minus_baseline",
                "tv_improvement_vs_baseline",
            ]:
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def save_delta_bar(rows: List[Dict[str, object]], output_path: Path, metric: str, top_n: int) -> None:
    import matplotlib.pyplot as plt

    delta_key = f"{metric}_delta_method_minus_baseline"
    label = metric.upper()
    selected = sorted(rows, key=lambda r: abs(float(r[delta_key])), reverse=True)[:top_n]
    selected = list(reversed(selected))
    labels = [str(r["column"]) for r in selected]
    deltas = np.array([float(r[delta_key]) for r in selected], dtype=np.float64)
    colors = ["#D55E00" if d > 0 else "#009E73" for d in deltas]

    height = max(6.0, 0.28 * len(selected) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    y = np.arange(len(selected))
    ax.barh(y, deltas, color=colors, alpha=0.88)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"{label} delta: method - baseline")
    ax.set_title(f"Largest absolute {label} differences by column")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.99,
        0.01,
        "green: method lower distance | orange: baseline lower distance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_scatter(rows: List[Dict[str, object]], output_path: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    method_key = f"method_{metric}_distance"
    baseline_key = f"baseline_{metric}_distance"
    label = metric.upper()
    types = sorted({str(r["column_type"]) for r in rows})
    palette = {
        "categorical": "#0072B2",
        "numeric": "#D55E00",
        "time": "#009E73",
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    max_val = 0.0
    for typ in types:
        subset = [r for r in rows if r["column_type"] == typ]
        xs = np.array([float(r[baseline_key]) for r in subset], dtype=np.float64)
        ys = np.array([float(r[method_key]) for r in subset], dtype=np.float64)
        max_val = max(max_val, float(xs.max(initial=0)), float(ys.max(initial=0)))
        ax.scatter(xs, ys, label=typ, s=36, alpha=0.78, color=palette.get(typ, "#666666"))

    lim = max(0.05, max_val * 1.05)
    ax.plot([0, lim], [0, lim], color="#222222", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"Baseline {label} distance")
    ax.set_ylabel(f"Method {label} distance")
    ax.set_title(f"Method vs baseline {label} distance")
    ax.grid(alpha=0.2)
    ax.legend(title="Column type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_summary(rows: List[Dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    types = sorted({str(r["column_type"]) for r in rows})
    js_by_type = [
        [float(r["js_delta_method_minus_baseline"]) for r in rows if r["column_type"] == typ]
        for typ in types
    ]
    tv_by_type = [
        [float(r["tv_delta_method_minus_baseline"]) for r in rows if r["column_type"] == typ]
        for typ in types
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, values, title in [
        (axes[0], js_by_type, "JS delta by column type"),
        (axes[1], tv_by_type, "TV delta by column type"),
    ]:
        ax.boxplot(values, tick_labels=types, vert=False, showmeans=True)
        ax.axvline(0, color="#222222", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("method - baseline")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_ylabel("Column type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_delta_bar(rows, args.output_dir / "largest_js_deltas.png", "js", args.top_n)
    save_delta_bar(rows, args.output_dir / "largest_tv_deltas.png", "tv", args.top_n)
    save_scatter(rows, args.output_dir / "js_scatter.png", "js")
    save_scatter(rows, args.output_dir / "tv_scatter.png", "tv")
    save_summary(rows, args.output_dir / "delta_summary_by_type.png")

    js_deltas = [float(r["js_delta_method_minus_baseline"]) for r in rows]
    tv_deltas = [float(r["tv_delta_method_minus_baseline"]) for r in rows]
    js_wins = sum(1 for d in js_deltas if d < 0)
    tv_wins = sum(1 for d in tv_deltas if d < 0)
    print(f"Wrote figures to {args.output_dir}")
    print(f"JS: method lower distance on {js_wins}/{len(rows)} columns; mean delta={np.mean(js_deltas):.6f}")
    print(f"TV: method lower distance on {tv_wins}/{len(rows)} columns; mean delta={np.mean(tv_deltas):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

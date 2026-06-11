from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trip_synth.utils.io import ensure_dir, read_json

from .common import apply_poster_style


POSTER_FILES = [
    "01_architecture_contrastive_vae.png",
    "02_experiment_pipeline.png",
    "03_hparam_heatmap.png",
    "04_marginal_distance_leaderboard.png",
    "05_cross_marginal_error_heatmap.png",
    "06_privacy_copy_rate_by_method.png",
    "07_screenline_concept_map.png",
    "08_aadt_observed_vs_synthetic_log_scatter.png",
    "09_aadt_method_leaderboard.png",
    "10_geh_distribution_by_method.png",
    "11_best_method_summary_panel.png",
]


def _save_flow(path: Path, title: str, labels: list[str], vertical: bool = False) -> None:
    plt.figure(figsize=(12, 6 if not vertical else 9))
    ax = plt.gca()
    ax.axis("off")
    if vertical:
        xs = [0.5] * len(labels)
        ys = np.linspace(0.88, 0.12, len(labels))
    else:
        xs = np.linspace(0.08, 0.92, len(labels))
        ys = [0.55] * len(labels)
    for i, label in enumerate(labels):
        ax.text(
            xs[i],
            ys[i],
            label,
            ha="center",
            va="center",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f6f7f9", edgecolor="#333333", linewidth=1.2),
        )
        if i < len(labels) - 1:
            if vertical:
                ax.annotate("", xy=(xs[i + 1], ys[i + 1] + 0.04), xytext=(xs[i], ys[i] - 0.04), arrowprops=dict(arrowstyle="->", lw=1.6))
            else:
                ax.annotate("", xy=(xs[i + 1] - 0.045, ys[i + 1]), xytext=(xs[i] + 0.045, ys[i]), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.set_title(title, loc="left", pad=20)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    plt.figure(figsize=(9, 5.5))
    x = np.arange(len(labels))
    plt.bar(x, values, color=["#2c7fb8", "#41ab5d", "#fdae61", "#756bb1", "#de2d26"][: len(labels)])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _placeholder(path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.axis("off")
    plt.text(0.5, 0.6, title, ha="center", va="center", fontsize=18, fontweight="bold")
    plt.text(0.5, 0.42, message, ha="center", va="center", fontsize=12, wrap=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _load_table(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / "tables" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def make_poster_figures(run_dir: str | Path, methods: list[str] | None = None) -> list[Path]:
    apply_poster_style()
    run_dir = Path(run_dir)
    poster = ensure_dir(run_dir / "figures" / "poster")
    methods = methods or []
    created: list[Path] = []

    p = poster / "01_architecture_contrastive_vae.png"
    _save_flow(
        p,
        "Contrastive mixed-tabular VAE",
        [
            "Survey trip row",
            "Remove weight from features",
            "Embeddings + normalized numeric features",
            "Wide/deep encoder",
            "Latent mean/logvar",
            "Decoder",
            "Synthetic trip",
        ],
    )
    created.append(p)

    p = poster / "02_experiment_pipeline.png"
    _save_flow(
        p,
        "Experiment pipeline",
        [
            "Survey trips + expansion weights",
            "Five synthesis methods",
            "Synthetic tables without weight",
            "Marginals + privacy",
            "OD-to-screenline proxy",
            "AADT/AAWDT comparison",
            "Method ranking",
        ],
    )
    created.append(p)

    hparam = _load_table(run_dir, "hparam_results.csv")
    p = poster / "03_hparam_heatmap.png"
    if hparam.empty:
        _placeholder(p, "Hyperparameter heatmap", "Run scripts/run_hparam_grid.sh to populate hparam_results.csv.")
    else:
        pivot = hparam.pivot_table(index="latent_dim", columns="lambda_contrastive", values="score", aggfunc="mean")
        plt.figure(figsize=(8, 5))
        plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="Composite score")
        plt.xlabel("lambda_contrastive")
        plt.ylabel("latent_dim")
        plt.title("Mean validation score")
        plt.tight_layout()
        plt.savefig(p, dpi=300)
        plt.close()
    created.append(p)

    summary = _load_table(run_dir, "method_validation_summary.csv")
    p = poster / "04_marginal_distance_leaderboard.png"
    if not summary.empty and "mean_categorical_tv" in summary:
        labels = summary["method"].astype(str).tolist()
        values = summary["mean_categorical_tv"].fillna(0).astype(float).tolist()
        _bar(p, "Marginal distance by method", labels, values, "Mean categorical TV")
    else:
        _placeholder(p, "Marginal leaderboard", "Marginal metrics were not available.")
    created.append(p)

    p = poster / "05_cross_marginal_error_heatmap.png"
    if not summary.empty and "mean_cross_tv" in summary:
        vals = summary[["method", "mean_cross_tv"]].fillna(0)
        plt.figure(figsize=(8, 4.5))
        plt.imshow(vals[["mean_cross_tv"]].to_numpy(), cmap="magma", aspect="auto")
        plt.yticks(range(len(vals)), vals["method"])
        plt.xticks([0], ["Mean cross TV"])
        plt.colorbar(label="Distance")
        plt.title("Cross-marginal error")
        plt.tight_layout()
        plt.savefig(p, dpi=300)
        plt.close()
    else:
        _placeholder(p, "Cross-marginal heatmap", "Cross-marginal metrics were not available.")
    created.append(p)

    privacy = _load_table(run_dir, "privacy_summary.csv")
    p = poster / "06_privacy_copy_rate_by_method.png"
    if not privacy.empty and "exact_row_copy_rate" in privacy:
        _bar(
            p,
            "Exact copy rate",
            privacy["method"].astype(str).tolist(),
            privacy["exact_row_copy_rate"].fillna(0).astype(float).tolist(),
            "Copy rate",
        )
    else:
        _placeholder(p, "Privacy copy rate", "Privacy metrics were not available.")
    created.append(p)

    p = poster / "07_screenline_concept_map.png"
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.set_title("Centroid-line screenline proxy", loc="left")
    ax.plot([0.15, 0.85], [0.25, 0.78], color="#2c7fb8", lw=3, label="OD centroid line")
    ax.axline((0.5, 0.0), (0.5, 1.0), color="#de2d26", lw=2, linestyle="--", label="tract boundary buffer")
    ax.scatter([0.15, 0.85], [0.25, 0.78], s=100, color="#222222")
    ax.scatter([0.48, 0.52], [0.52, 0.56], s=80, color="#41ab5d", label="AADT stations")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close()
    created.append(p)

    aadt = _load_table(run_dir, "aadt_validation_summary.csv")
    comparison_path = run_dir / "metrics" / "aadt_screenline_comparisons.parquet"
    aadt_comparisons = pd.read_parquet(comparison_path) if comparison_path.exists() else pd.DataFrame()
    p = poster / "08_aadt_observed_vs_synthetic_log_scatter.png"
    if not aadt_comparisons.empty and {"observed_count", "synthetic_count"}.issubset(aadt_comparisons.columns):
        plt.figure(figsize=(7, 6))
        for method, group in aadt_comparisons.groupby("method"):
            plt.scatter(np.log1p(group["observed_count"]), np.log1p(group["synthetic_count"]), s=18, alpha=0.55, label=method)
        plt.xlabel("log1p observed AADT/AAWDT")
        plt.ylabel("log1p synthetic virtual crossings")
        plt.title("AADT/AAWDT screenline comparison")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(p, dpi=300)
        plt.close()
    else:
        _placeholder(p, "AADT scatter", "AADT validation skipped or external screenline artifacts unavailable.")
    created.append(p)

    p = poster / "09_aadt_method_leaderboard.png"
    if not aadt.empty and "pearson_log1p" in aadt:
        _bar(p, "AADT log correlation", aadt["method"].astype(str).tolist(), aadt["pearson_log1p"].fillna(0).tolist(), "Pearson log1p")
    else:
        _placeholder(p, "AADT leaderboard", "AADT validation skipped or unavailable.")
    created.append(p)

    p = poster / "10_geh_distribution_by_method.png"
    if not aadt_comparisons.empty and {"observed_count", "synthetic_count"}.issubset(aadt_comparisons.columns):
        obs = aadt_comparisons["observed_count"].to_numpy(float)
        syn = aadt_comparisons["synthetic_count"].to_numpy(float)
        geh_vals = np.sqrt(2.0 * (syn - obs) ** 2 / np.maximum((syn + obs) / 2.0, 1e-9))
        plot_df = aadt_comparisons.assign(geh=geh_vals)
        plt.figure(figsize=(8, 4.8))
        for method, group in plot_df.groupby("method"):
            plt.hist(group["geh"], bins=30, alpha=0.45, label=method)
        plt.xlabel("GEH")
        plt.ylabel("Screenlines")
        plt.title("GEH distribution by method")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(p, dpi=300)
        plt.close()
    else:
        _placeholder(p, "GEH distribution", "Generated after AADT validation.")
    created.append(p)

    p = poster / "11_best_method_summary_panel.png"
    lines = []
    if not summary.empty:
        for metric, label, ascending in [
            ("mean_categorical_tv", "Best marginal", True),
            ("mean_cross_tv", "Best cross-marginal", True),
            ("privacy_score", "Best privacy", False),
        ]:
            if metric in summary:
                ordered = summary.sort_values(metric, ascending=ascending)
                if not ordered.empty:
                    lines.append(f"{label}: {ordered.iloc[0]['method']}")
    if not lines:
        lines = ["Run summary metrics were not available."]
    _placeholder(p, "Best method summary", "\n".join(lines))
    created.append(p)

    captions = poster / "captions.md"
    captions.write_text(
        "\n".join(
            [
                "# Poster Figure Captions",
                "",
                "Figures summarize the synthesis pipeline, validation metrics, privacy diagnostics, and optional AADT/AAWDT screenline validation.",
                "AADT figures are placeholders in quick runs because external geospatial validation is disabled by configuration.",
            ]
        )
        + "\n"
    )
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/runs/quick_test")
    args = parser.parse_args()
    created = make_poster_figures(args.run_dir)
    print(json.dumps([str(p) for p in created], indent=2))

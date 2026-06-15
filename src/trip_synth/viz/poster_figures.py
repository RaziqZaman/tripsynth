from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

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
    "12_trip_coverage_resampling_vs_cvae.png",
    "13_single_origin_trip_coverage_60tracts.png",
    "14_aadt_screenline_station_match_map.png",
    "15_od_pair_screenline_validation_map.png",
    "16_od_pair_screenline_validation_map_cvae_best.png",
    "17_hourly_tmas_profile_cvae_best.png",
    "18_hourly_tmas_profile_all_stations.png",
    "19_hourly_tmas_rmse_reduction.png",
    "20_hourly_annual_traffic_validation_summary.png",
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


def _draw_pipeline_icon(ax: plt.Axes, kind: str, center: tuple[float, float], color: str) -> None:
    cx, cy = center
    r = 0.037
    white = "#ffffff"
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="none", alpha=0.98, zorder=7))

    if kind == "survey":
        ax.add_patch(Rectangle((cx - 0.015, cy - 0.018), 0.030, 0.035, facecolor="none", edgecolor=white, linewidth=1.35, zorder=8))
        ax.add_patch(Rectangle((cx - 0.009, cy + 0.015), 0.018, 0.007, facecolor=white, edgecolor=white, linewidth=0, zorder=9))
        for y in [cy + 0.007, cy - 0.006, cy - 0.019]:
            ax.plot([cx - 0.009, cx + 0.010], [y, y], color=white, linewidth=1.15, solid_capstyle="round", zorder=9)
        ax.plot([cx - 0.018, cx - 0.014, cx - 0.010], [cy - 0.004, cy - 0.010, cy + 0.004], color=white, linewidth=1.15, zorder=9)
    elif kind == "methods":
        nodes = [
            (cx - 0.016, cy + 0.014),
            (cx + 0.016, cy + 0.014),
            (cx - 0.016, cy - 0.014),
            (cx + 0.016, cy - 0.014),
        ]
        for start, end in [(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], nodes[3]), (nodes[2], nodes[3]), (nodes[0], nodes[3])]:
            ax.plot([start[0], end[0]], [start[1], end[1]], color=white, linewidth=1.0, alpha=0.82, zorder=8)
        for node in nodes:
            ax.add_patch(Circle(node, 0.0065, facecolor=white, edgecolor=white, linewidth=0, zorder=9))
    elif kind == "table":
        ax.add_patch(Rectangle((cx - 0.020, cy - 0.018), 0.040, 0.036, facecolor="none", edgecolor=white, linewidth=1.35, zorder=8))
        ax.plot([cx - 0.020, cx + 0.020], [cy + 0.006, cy + 0.006], color=white, linewidth=1.15, zorder=9)
        ax.plot([cx - 0.020, cx + 0.020], [cy - 0.006, cy - 0.006], color=white, linewidth=1.15, zorder=9)
        for x in [cx - 0.0067, cx + 0.0067]:
            ax.plot([x, x], [cy - 0.018, cy + 0.018], color=white, linewidth=1.15, zorder=9)
    elif kind == "diagnostics":
        bars = [0.011, 0.020, 0.030]
        for idx, height in enumerate(bars):
            ax.add_patch(Rectangle((cx - 0.024 + idx * 0.010, cy - 0.020), 0.0065, height, facecolor=white, edgecolor=white, linewidth=0, zorder=9))
        shield = [
            (cx + 0.012, cy + 0.017),
            (cx + 0.028, cy + 0.010),
            (cx + 0.025, cy - 0.010),
            (cx + 0.020, cy - 0.020),
            (cx + 0.012, cy - 0.026),
            (cx + 0.004, cy - 0.020),
            (cx - 0.001, cy - 0.010),
            (cx - 0.004, cy + 0.010),
        ]
        ax.add_patch(Polygon(shield, closed=True, facecolor="none", edgecolor=white, linewidth=1.2, zorder=9))
    elif kind == "screenline":
        ax.plot([cx - 0.023, cx + 0.024], [cy - 0.018, cy + 0.016], color=white, linewidth=1.45, solid_capstyle="round", zorder=9)
        ax.plot([cx + 0.002, cx + 0.002], [cy - 0.025, cy + 0.025], color=white, linewidth=1.25, linestyle=(0, (2, 2)), zorder=9)
        ax.add_patch(Circle((cx - 0.024, cy - 0.019), 0.006, facecolor=white, edgecolor=white, linewidth=0, zorder=10))
        ax.add_patch(Circle((cx + 0.024, cy + 0.016), 0.006, facecolor=white, edgecolor=white, linewidth=0, zorder=10))
    elif kind == "counts":
        for offset, heights in [(-0.014, [0.016, 0.030]), (0.012, [0.024, 0.019])]:
            ax.add_patch(Rectangle((cx + offset - 0.006, cy - 0.020), 0.007, heights[0], facecolor=white, edgecolor=white, linewidth=0, zorder=9))
            ax.add_patch(Rectangle((cx + offset + 0.004, cy - 0.020), 0.007, heights[1], facecolor=white, edgecolor=white, linewidth=0, alpha=0.72, zorder=9))
        ax.plot([cx - 0.026, cx + 0.028], [cy - 0.020, cy - 0.020], color=white, linewidth=1.15, zorder=9)
        ax.add_patch(Ellipse((cx + 0.020, cy + 0.016), 0.020, 0.013, angle=0, facecolor="none", edgecolor=white, linewidth=1.1, zorder=9))
        ax.plot([cx + 0.027, cx + 0.034], [cy + 0.009, cy + 0.002], color=white, linewidth=1.1, zorder=9)
    elif kind == "rank":
        ax.add_patch(Rectangle((cx - 0.024, cy - 0.021), 0.014, 0.022, facecolor=white, edgecolor=white, linewidth=0, alpha=0.78, zorder=9))
        ax.add_patch(Rectangle((cx - 0.006, cy - 0.021), 0.014, 0.036, facecolor=white, edgecolor=white, linewidth=0, zorder=9))
        ax.add_patch(Rectangle((cx + 0.012, cy - 0.021), 0.014, 0.028, facecolor=white, edgecolor=white, linewidth=0, alpha=0.88, zorder=9))
        ax.add_patch(Circle((cx + 0.001, cy + 0.026), 0.006, facecolor=white, edgecolor=white, linewidth=0, zorder=9))


def _draw_pipeline_stage(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    index: int,
    title: str,
    body: str,
    icon: str,
    color: str,
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.006, y - 0.008),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.024",
            linewidth=0,
            facecolor="#d7d0c3",
            alpha=0.28,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.024",
            linewidth=1.7,
            edgecolor=color,
            facecolor="#ffffff",
            zorder=2,
        )
    )
    ax.add_patch(Rectangle((x + 0.018, y + h - 0.018), w - 0.036, 0.009, facecolor=color, edgecolor="none", alpha=0.82, zorder=4))
    _draw_pipeline_icon(ax, icon, (x + w / 2, y + h - 0.074), color)
    ax.text(
        x + w / 2,
        y + h - 0.135,
        title,
        ha="center",
        va="top",
        fontsize=12.4,
        fontweight="bold",
        color="#1f2933",
        zorder=6,
    )
    ax.text(
        x + w / 2,
        y + 0.054,
        "\n".join(textwrap.wrap(body, width=17, break_long_words=False)),
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#4b5563",
        linespacing=1.12,
        zorder=6,
    )
    ax.text(
        x + 0.023,
        y + h - 0.037,
        f"{index}",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=color,
        zorder=6,
    )

def _add_pipeline_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.1,
            color="#6b7280",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=7,
            shrinkB=7,
            zorder=5,
        )
    )


def _make_experiment_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.0, 6.1), constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.050, 0.930, "Experiment pipeline", ha="left", va="top", fontsize=24, fontweight="bold", color="#1f2933")
    ax.text(
        0.050,
        0.865,
        "A compact view of the trip-synthesis experiment: build candidates, validate them, then rank the methods.",
        ha="left",
        va="top",
        fontsize=12.3,
        color="#4b5563",
    )

    stages = [
        ("Survey", "Weighted trip records", "survey", "#5d6f99"),
        ("Generate", "Bootstrap, BN, VAE, CVAE", "methods", "#5f8f55"),
        ("Synthetic Trips", "Unweighted output tables", "table", "#b87535"),
        ("Validate", "Fit, privacy, OD, counts", "diagnostics", "#2f8f83"),
        ("Rank", "Best-performing method", "rank", "#3d72a4"),
    ]
    box_w = 0.162
    box_h = 0.305
    y = 0.365
    xs = [0.050, 0.248, 0.446, 0.644, 0.842]
    positions = [(x, y) for x in xs]

    for idx, ((title, body, icon, color), pos) in enumerate(zip(stages, positions), start=1):
        _draw_pipeline_stage(ax, pos, (box_w, box_h), idx, title, body, icon, color)

    y_mid = y + box_h / 2
    for left, right in zip(positions[:-1], positions[1:]):
        _add_pipeline_arrow(ax, (left[0] + box_w, y_mid), (right[0], y_mid))

    detail_y = 0.220
    validation_color = "#2f8f83"
    ax.plot([0.660, 0.905], [detail_y, detail_y], color=validation_color, linewidth=1.4, alpha=0.55, zorder=3)
    for x, label in [(0.660, "marginals"), (0.742, "privacy"), (0.824, "screenlines"), (0.905, "AADT/TMAS")]:
        ax.add_patch(Circle((x, detail_y), 0.010, facecolor=validation_color, edgecolor="white", linewidth=0.8, zorder=5))
        ax.text(x, detail_y - 0.035, label, ha="center", va="top", fontsize=8.5, color="#4b5563")

    ax.text(
        0.050,
        0.075,
        "Validation combines distributional checks, privacy diagnostics, OD geography, and traffic-count comparisons.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#6b7280",
    )

    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

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


def _format_compact_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _station_marker_sizes(values: pd.Series | np.ndarray, scale_min: float, scale_max: float) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
    logged = np.log1p(numeric)
    denom = max(scale_max - scale_min, 1e-9)
    return 16.0 + 120.0 * np.clip((logged - scale_min) / denom, 0.0, 1.0)


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


def _clean_fips_series(series: pd.Series) -> pd.Series:
    def clean(value: Any) -> str:
        if pd.isna(value):
            return "-1"
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return "-1"
        return digits.zfill(11)

    return series.map(clean)


def _load_od_counts(path: Path, sample_size: int, seed: int) -> pd.DataFrame:
    usecols = ["o_tract_fips", "d_tract_fips"]
    df = pd.read_csv(path, usecols=usecols, dtype="string")
    df["o_tract_fips"] = _clean_fips_series(df["o_tract_fips"])
    df["d_tract_fips"] = _clean_fips_series(df["d_tract_fips"])
    df = df[(df["o_tract_fips"] != "-1") & (df["d_tract_fips"] != "-1")]
    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    return (
        df.groupby(usecols, dropna=False)
        .size()
        .rename("trip_count")
        .reset_index()
        .sort_values("trip_count", ascending=False)
    )


def _line_segments_from_od(
    od_counts: pd.DataFrame,
    centroid_lookup: dict[str, tuple[float, float]],
    max_lines: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    mapped = od_counts[
        od_counts["o_tract_fips"].isin(centroid_lookup)
        & od_counts["d_tract_fips"].isin(centroid_lookup)
    ].copy()
    total_unique = len(mapped)
    if max_lines > 0 and len(mapped) > max_lines:
        weights = np.sqrt(mapped["trip_count"].to_numpy(float))
        probabilities = weights / weights.sum()
        rng = np.random.default_rng(seed)
        keep = rng.choice(mapped.index.to_numpy(), size=max_lines, replace=False, p=probabilities)
        mapped = mapped.loc[keep]

    segments = np.empty((len(mapped), 2, 2), dtype=float)
    for idx, (_, row) in enumerate(mapped.iterrows()):
        segments[idx, 0] = centroid_lookup[str(row["o_tract_fips"])]
        segments[idx, 1] = centroid_lookup[str(row["d_tract_fips"])]
    counts = mapped["trip_count"].to_numpy(float)
    return segments, counts, total_unique


def _plot_trip_coverage_panel(
    ax: plt.Axes,
    tracts: Any,
    segments: np.ndarray,
    counts: np.ndarray,
    title: str,
    subtitle: str,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    unique_pairs: int,
    rows_sampled: int,
) -> None:
    tracts.boundary.plot(ax=ax, linewidth=0.18, color="#d5d9d6", alpha=0.9, zorder=1)
    if len(segments):
        scaled = np.log1p(counts)
        if float(scaled.max()) > float(scaled.min()):
            widths = 0.16 + 0.78 * (scaled - scaled.min()) / (scaled.max() - scaled.min())
        else:
            widths = np.full_like(scaled, 0.32)
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=widths,
                alpha=0.12,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfaf6")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=6)
    ax.text(
        0.0,
        1.005,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="#4a4d4f",
    )
    ax.text(
        0.02,
        0.03,
        f"{unique_pairs:,} unique OD pairs\nfrom {rows_sampled:,} sampled trips",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d7d7d7", alpha=0.88),
    )


def _make_trip_coverage_map(
    path: Path,
    run_dir: Path,
    sample_size: int = 50_000,
    max_lines: int = 25_000,
    seed: int = 42,
) -> bool:
    try:
        import geopandas as gpd
    except Exception:
        _placeholder(path, "Trip coverage map", "Install the geo extras to generate tract-level trip maps.")
        return False

    tracts_path = run_dir / "geo" / "tracts.parquet"
    baseline_path = run_dir / "samples" / "weighted_bootstrap_synthetic.csv"
    cvae_path = run_dir / "samples" / "contrastive_vae_synthetic.csv"
    if not (tracts_path.exists() and baseline_path.exists() and cvae_path.exists()):
        _placeholder(path, "Trip coverage map", "Synthetic samples or tract geometry were not available in this run.")
        return False

    tracts = gpd.read_parquet(tracts_path)
    required = {"GEOID", "rep_x", "rep_y"}
    if not required.issubset(tracts.columns):
        _placeholder(path, "Trip coverage map", "Tract geometry is missing GEOID/representative point columns.")
        return False

    centroid_lookup = {
        str(row.GEOID).zfill(11): (float(row.rep_x), float(row.rep_y))
        for row in tracts[["GEOID", "rep_x", "rep_y"]].itertuples(index=False)
    }

    baseline_od = _load_od_counts(baseline_path, sample_size, seed)
    cvae_od = _load_od_counts(cvae_path, sample_size, seed)
    baseline_segments, baseline_counts, baseline_unique = _line_segments_from_od(
        baseline_od, centroid_lookup, max_lines=max_lines, seed=seed
    )
    cvae_segments, cvae_counts, cvae_unique = _line_segments_from_od(
        cvae_od, centroid_lookup, max_lines=max_lines, seed=seed
    )

    bounds = tracts.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.025
    ypad = (bounds[3] - bounds[1]) * 0.025
    xlim = (float(bounds[0] - xpad), float(bounds[2] + xpad))
    ylim = (float(bounds[1] - ypad), float(bounds[3] + ypad))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    fig.suptitle(
        "Contrastive VAE fills in synthetic trip geography",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=23,
        fontweight="bold",
    )
    _plot_trip_coverage_panel(
        axes[0],
        tracts,
        baseline_segments,
        baseline_counts,
        "Random-resampling baseline",
        "Repeated draws concentrate on observed OD pairs",
        "#d06c2f",
        xlim,
        ylim,
        baseline_unique,
        min(sample_size, int(baseline_od["trip_count"].sum())),
    )
    _plot_trip_coverage_panel(
        axes[1],
        tracts,
        cvae_segments,
        cvae_counts,
        "Contrastive VAE",
        "Generated OD pairs spread into under-covered areas",
        "#2176ae",
        xlim,
        ylim,
        cvae_unique,
        min(sample_size, int(cvae_od["trip_count"].sum())),
    )
    fig.text(
        0.02,
        0.012,
        "Lines connect origin and destination tract representative points; equal row samples are used for both methods.",
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _pairs_from_od_counts(od_counts: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(row.o_tract_fips), str(row.d_tract_fips))
        for row in od_counts.itertuples(index=False)
        if str(row.o_tract_fips) != str(row.d_tract_fips)
    }


def _pair_index(pairs: set[tuple[str, str]]) -> dict[str, set[tuple[str, str]]]:
    index: dict[str, set[tuple[str, str]]] = {}
    for pair in pairs:
        o, d = pair
        index.setdefault(o, set()).add(pair)
        index.setdefault(d, set()).add(pair)
    return index


def _cluster_pairs(
    cluster: set[str],
    pair_index: dict[str, set[tuple[str, str]]],
    pairs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    candidates: set[tuple[str, str]] = set()
    for tract in cluster:
        candidates.update(pair_index.get(tract, set()))
    return {pair for pair in candidates if pair in pairs and pair[0] in cluster and pair[1] in cluster and pair[0] != pair[1]}


def _select_focus_cluster(
    run_dir: Path,
    baseline_od: pd.DataFrame,
    cvae_od: pd.DataFrame,
    min_tracts: int = 7,
    max_tracts: int = 12,
) -> tuple[set[str], dict[str, float | int | str]] | None:
    adjacency_path = run_dir / "geo" / "tract_adjacency.parquet"
    if not adjacency_path.exists():
        return None

    adjacency = pd.read_parquet(adjacency_path, columns=["tract_a", "tract_b"])
    neighbors: dict[str, set[str]] = {}
    for tract_a, tract_b in adjacency.itertuples(index=False):
        a = str(tract_a).zfill(11)
        b = str(tract_b).zfill(11)
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    baseline_pairs = _pairs_from_od_counts(baseline_od)
    cvae_pairs = _pairs_from_od_counts(cvae_od)
    all_pairs = baseline_pairs | cvae_pairs
    all_index = _pair_index(all_pairs)

    best: tuple[float, set[str], dict[str, float | int | str]] | None = None
    for center, adjacent in neighbors.items():
        if not center.startswith("24"):
            continue
        cluster = {center} | set(adjacent)
        if len(cluster) < min_tracts or len(cluster) > max_tracts:
            continue
        local_pairs = _cluster_pairs(cluster, all_index, all_pairs)
        baseline_local = local_pairs & baseline_pairs
        cvae_local = local_pairs & cvae_pairs
        possible = len(cluster) * (len(cluster) - 1)
        if len(baseline_local) < 8 or len(cvae_local) < 20 or possible <= 0:
            continue
        fill_rate = len(cvae_local) / possible
        ratio = len(cvae_local) / max(len(baseline_local), 1)
        cvae_only = len(cvae_local - baseline_local)
        score = cvae_only + 40.0 * fill_rate + 10.0 * ratio - 0.5 * len(cluster)
        metadata: dict[str, float | int | str] = {
            "center": center,
            "tracts": len(cluster),
            "possible": possible,
            "baseline_links": len(baseline_local),
            "cvae_links": len(cvae_local),
            "cvae_only_links": cvae_only,
            "bootstrap_only_links": len(baseline_local - cvae_local),
            "fill_rate": fill_rate,
            "ratio": ratio,
        }
        if best is None or score > best[0]:
            best = (score, cluster, metadata)

    if best is None:
        return None
    return best[1], best[2]


def _directed_pair_counts_for_cluster(od_counts: pd.DataFrame, cluster: set[str]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in od_counts.itertuples(index=False):
        o = str(row.o_tract_fips)
        d = str(row.d_tract_fips)
        if o in cluster and d in cluster and o != d:
            counts[(o, d)] = int(row.trip_count)
    return counts


def _draw_trip_arrows(
    ax: plt.Axes,
    pair_counts: dict[tuple[str, str], int],
    coords: dict[str, tuple[float, float]],
    color: str,
    muted_pairs: set[tuple[str, str]] | None = None,
    muted_color: str = "#9fb8c8",
) -> None:
    if not pair_counts:
        return
    max_weight = max(np.log1p(list(pair_counts.values())))
    muted_pairs = muted_pairs or set()
    for (origin, destination), count in sorted(pair_counts.items(), key=lambda item: item[1]):
        if origin not in coords or destination not in coords:
            continue
        weight = np.log1p(count) / max_weight if max_weight > 0 else 1.0
        is_muted = (origin, destination) in muted_pairs
        rad = 0.16 if origin < destination else -0.16
        arrow = FancyArrowPatch(
            coords[origin],
            coords[destination],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.55 + 1.35 * weight,
            color=muted_color if is_muted else color,
            alpha=0.32 if is_muted else 0.78,
            shrinkA=8,
            shrinkB=8,
            zorder=4 if not is_muted else 3,
        )
        ax.add_patch(arrow)


def _plot_focused_trip_panel(
    ax: plt.Axes,
    tracts: Any,
    focus_tracts: Any,
    pair_counts: dict[tuple[str, str], int],
    coords: dict[str, tuple[float, float]],
    title: str,
    subtitle: str,
    color: str,
    possible: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    muted_pairs: set[tuple[str, str]] | None = None,
) -> None:
    tracts.boundary.plot(ax=ax, linewidth=0.45, color="#e2e4df", alpha=0.9, zorder=1)
    focus_tracts.plot(ax=ax, facecolor="#fffdf7", edgecolor="#5c6462", linewidth=1.2, zorder=2)
    _draw_trip_arrows(ax, pair_counts, coords, color=color, muted_pairs=muted_pairs)
    xs = [point[0] for point in coords.values()]
    ys = [point[1] for point in coords.values()]
    ax.scatter(xs, ys, s=24, color="#222222", zorder=5, linewidth=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfaf6")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=6)
    ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color="#4a4d4f")
    ax.text(
        0.02,
        0.03,
        f"{len(pair_counts)} / {possible} directed local OD links",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d7d7d7", alpha=0.9),
    )


def _make_focused_trip_coverage_map(path: Path, run_dir: Path) -> bool:
    try:
        import geopandas as gpd
    except Exception:
        _placeholder(path, "Focused trip coverage map", "Install the geo extras to generate tract-level trip maps.")
        return False

    tracts_path = run_dir / "geo" / "tracts.parquet"
    baseline_path = run_dir / "samples" / "weighted_bootstrap_synthetic.csv"
    cvae_path = run_dir / "samples" / "contrastive_vae_synthetic.csv"
    if not (tracts_path.exists() and baseline_path.exists() and cvae_path.exists()):
        _placeholder(path, "Focused trip coverage map", "Synthetic samples or tract geometry were not available in this run.")
        return False

    tracts = gpd.read_parquet(tracts_path)
    tracts = tracts.assign(GEOID=tracts["GEOID"].astype(str).str.zfill(11))
    baseline_od = _load_od_counts(baseline_path, sample_size=0, seed=42)
    cvae_od = _load_od_counts(cvae_path, sample_size=0, seed=42)
    selected = _select_focus_cluster(run_dir, baseline_od, cvae_od)
    if selected is None:
        _placeholder(path, "Focused trip coverage map", "No compact adjacent-tract cluster had enough local synthetic OD links.")
        return False
    cluster, metadata = selected

    focus_tracts = tracts[tracts["GEOID"].isin(cluster)].copy()
    if focus_tracts.empty:
        _placeholder(path, "Focused trip coverage map", "Selected tract cluster was not found in the tract geometry.")
        return False

    coords = {
        str(row.GEOID): (float(row.rep_x), float(row.rep_y))
        for row in focus_tracts[["GEOID", "rep_x", "rep_y"]].itertuples(index=False)
    }
    baseline_counts = _directed_pair_counts_for_cluster(baseline_od, cluster)
    cvae_counts = _directed_pair_counts_for_cluster(cvae_od, cluster)
    baseline_pairs = set(baseline_counts)
    cvae_pairs = set(cvae_counts)
    cvae_only = cvae_pairs - baseline_pairs
    common = cvae_pairs & baseline_pairs

    bounds = focus_tracts.total_bounds
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    pad = span * 0.18
    xlim = (float(bounds[0] - pad), float(bounds[2] + pad))
    ylim = (float(bounds[1] - pad), float(bounds[3] + pad))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    fig.suptitle(
        "Zoomed tract neighborhood: CVAE fills missing local trip links",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    possible = int(metadata["possible"])
    _plot_focused_trip_panel(
        axes[0],
        tracts,
        focus_tracts,
        baseline_counts,
        coords,
        "Random-resampling baseline",
        "Sparse local links inherited from observed trips",
        "#d06c2f",
        possible,
        xlim,
        ylim,
    )
    _plot_focused_trip_panel(
        axes[1],
        tracts,
        focus_tracts,
        cvae_counts,
        coords,
        "Contrastive VAE",
        f"{len(cvae_only)} CVAE-only links highlighted; shared links are muted",
        "#2176ae",
        possible,
        xlim,
        ylim,
        muted_pairs=common,
    )
    fig.text(
        0.02,
        0.012,
        f"Auto-selected {int(metadata['tracts'])} adjacent Baltimore-area tracts centered on {metadata['center']}; arrows connect tract representative points and do not imply routed paths.",
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# Later definitions intentionally override the first zoom-map draft above. The
# poster version uses reserved title/caption space plus an OD matrix so text does
# not collide with panel content.
def _plot_trip_coverage_panel(
    ax: plt.Axes,
    tracts: Any,
    segments: np.ndarray,
    counts: np.ndarray,
    title: str,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    tracts.boundary.plot(ax=ax, linewidth=0.18, color="#d5d9d6", alpha=0.9, zorder=1)
    if len(segments):
        scaled = np.log1p(counts)
        if float(scaled.max()) > float(scaled.min()):
            widths = 0.16 + 0.78 * (scaled - scaled.min()) / (scaled.max() - scaled.min())
        else:
            widths = np.full_like(scaled, 0.32)
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=widths,
                alpha=0.11,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfaf6")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=8)


def _make_trip_coverage_map(
    path: Path,
    run_dir: Path,
    sample_size: int = 50_000,
    max_lines: int = 25_000,
    seed: int = 42,
) -> bool:
    try:
        import geopandas as gpd
    except Exception:
        _placeholder(path, "Trip coverage map", "Install the geo extras to generate tract-level trip maps.")
        return False

    tracts_path = run_dir / "geo" / "tracts.parquet"
    baseline_path = run_dir / "samples" / "weighted_bootstrap_synthetic.csv"
    cvae_path = run_dir / "samples" / "contrastive_vae_synthetic.csv"
    if not (tracts_path.exists() and baseline_path.exists() and cvae_path.exists()):
        _placeholder(path, "Trip coverage map", "Synthetic samples or tract geometry were not available in this run.")
        return False

    tracts = gpd.read_parquet(tracts_path)
    required = {"GEOID", "rep_x", "rep_y"}
    if not required.issubset(tracts.columns):
        _placeholder(path, "Trip coverage map", "Tract geometry is missing GEOID/representative point columns.")
        return False

    centroid_lookup = {
        str(row.GEOID).zfill(11): (float(row.rep_x), float(row.rep_y))
        for row in tracts[["GEOID", "rep_x", "rep_y"]].itertuples(index=False)
    }

    baseline_od = _load_od_counts(baseline_path, sample_size, seed)
    cvae_od = _load_od_counts(cvae_path, sample_size, seed)
    baseline_segments, baseline_counts, baseline_unique = _line_segments_from_od(
        baseline_od, centroid_lookup, max_lines=max_lines, seed=seed
    )
    cvae_segments, cvae_counts, cvae_unique = _line_segments_from_od(
        cvae_od, centroid_lookup, max_lines=max_lines, seed=seed
    )

    bounds = tracts.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.025
    ypad = (bounds[3] - bounds[1]) * 0.025
    xlim = (float(bounds[0] - xpad), float(bounds[2] + xpad))
    ylim = (float(bounds[1] - ypad), float(bounds[3] + ypad))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.82, bottom=0.12, wspace=0.035)
    fig.text(0.025, 0.955, "Contrastive VAE fills in synthetic trip geography", ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(0.025, 0.905, "Equal 50,000-trip samples from the latest run; lines connect origin and destination tract representative points.", ha="left", va="top", fontsize=11, color="#555555")
    _plot_trip_coverage_panel(axes[0], tracts, baseline_segments, baseline_counts, "Random-resampling baseline", "#d06c2f", xlim, ylim)
    _plot_trip_coverage_panel(axes[1], tracts, cvae_segments, cvae_counts, "Contrastive VAE", "#2176ae", xlim, ylim)
    fig.text(0.25, 0.065, f"{baseline_unique:,} unique OD pairs", ha="center", va="center", fontsize=12, color="#222222")
    fig.text(0.75, 0.065, f"{cvae_unique:,} unique OD pairs", ha="center", va="center", fontsize=12, color="#222222")
    fig.text(0.025, 0.025, "Tract-centroid coverage visualization; not route assignment.", ha="left", va="bottom", fontsize=9.5, color="#666666")
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _connected_cluster_from_center(
    center: str,
    neighbors: dict[str, set[str]],
    coords: dict[str, tuple[float, float]],
    size: int,
) -> set[str] | None:
    if center not in coords:
        return None
    cx, cy = coords[center]
    selected = {center}
    while len(selected) < size:
        border: set[str] = set()
        for tract in selected:
            border.update(neighbors.get(tract, set()))
        border = {tract for tract in border if tract not in selected and tract in coords}
        if not border:
            return None
        next_tract = min(border, key=lambda tract: (coords[tract][0] - cx) ** 2 + (coords[tract][1] - cy) ** 2)
        selected.add(next_tract)
    return selected


def _select_focus_cluster(
    run_dir: Path,
    baseline_od: pd.DataFrame,
    cvae_od: pd.DataFrame,
    min_tracts: int = 60,
    max_tracts: int = 60,
) -> tuple[set[str], dict[str, float | int | str]] | None:
    try:
        import geopandas as gpd
    except Exception:
        return None

    adjacency_path = run_dir / "geo" / "tract_adjacency.parquet"
    tracts_path = run_dir / "geo" / "tracts.parquet"
    if not (adjacency_path.exists() and tracts_path.exists()):
        return None

    target_size = max_tracts
    tracts = gpd.read_parquet(tracts_path)
    tracts = tracts.assign(GEOID=tracts["GEOID"].astype(str).str.zfill(11))
    coords = {
        str(row.GEOID): (float(row.rep_x), float(row.rep_y))
        for row in tracts[["GEOID", "rep_x", "rep_y"]].itertuples(index=False)
    }
    adjacency = pd.read_parquet(adjacency_path, columns=["tract_a", "tract_b"])
    neighbors: dict[str, set[str]] = {}
    for tract_a, tract_b in adjacency.itertuples(index=False):
        a = str(tract_a).zfill(11)
        b = str(tract_b).zfill(11)
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    baseline_pairs = _pairs_from_od_counts(baseline_od)
    cvae_pairs = _pairs_from_od_counts(cvae_od)
    all_pairs = baseline_pairs | cvae_pairs
    all_index = _pair_index(all_pairs)

    best: tuple[float, set[str], dict[str, float | int | str]] | None = None
    for center in coords:
        if not center.startswith("24"):
            continue
        cluster = _connected_cluster_from_center(center, neighbors, coords, target_size)
        if cluster is None or not (min_tracts <= len(cluster) <= max_tracts):
            continue
        local_pairs = _cluster_pairs(cluster, all_index, all_pairs)
        baseline_local = local_pairs & baseline_pairs
        cvae_local = local_pairs & cvae_pairs
        possible = len(cluster) * (len(cluster) - 1)
        if len(baseline_local) < 40 or len(cvae_local) < 80 or possible <= 0:
            continue
        fill_delta = (len(cvae_local) - len(baseline_local)) / possible
        ratio = len(cvae_local) / max(len(baseline_local), 1)
        cvae_only = len(cvae_local - baseline_local)
        score = cvae_only + 80.0 * fill_delta + 10.0 * ratio
        metadata: dict[str, float | int | str] = {
            "center": center,
            "tracts": len(cluster),
            "possible": possible,
            "baseline_links": len(baseline_local),
            "cvae_links": len(cvae_local),
            "cvae_only_links": cvae_only,
            "bootstrap_only_links": len(baseline_local - cvae_local),
            "baseline_fill_rate": len(baseline_local) / possible,
            "cvae_fill_rate": len(cvae_local) / possible,
            "ratio": ratio,
        }
        if best is None or score > best[0]:
            best = (score, cluster, metadata)

    if best is None:
        return None
    return best[1], best[2]


def _ordered_focus_tracts(focus_tracts: Any) -> list[str]:
    table = focus_tracts[["GEOID", "rep_x", "rep_y"]].copy()
    table["row"] = pd.qcut(table["rep_y"].rank(method="first", ascending=False), q=4, labels=False, duplicates="drop")
    table = table.sort_values(["row", "rep_x", "rep_y"], ascending=[True, True, False])
    return table["GEOID"].astype(str).tolist()


def _coverage_matrix(pair_counts: dict[tuple[str, str], int], order: list[str]) -> np.ndarray:
    index = {tract: idx for idx, tract in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=int)
    for origin, destination in pair_counts:
        if origin in index and destination in index and origin != destination:
            matrix[index[origin], index[destination]] = 1
    np.fill_diagonal(matrix, -1)
    return matrix


def _plot_focus_map(
    ax: plt.Axes,
    tracts: Any,
    focus_tracts: Any,
    pair_counts: dict[tuple[str, str], int],
    coords: dict[str, tuple[float, float]],
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    muted_pairs: set[tuple[str, str]] | None = None,
) -> None:
    tracts.boundary.plot(ax=ax, linewidth=0.35, color="#e2e4df", alpha=0.95, zorder=1)
    focus_tracts.plot(ax=ax, facecolor="#fffdf7", edgecolor="#5c6462", linewidth=0.9, zorder=2)
    muted_pairs = muted_pairs or set()
    if pair_counts:
        segments = []
        colors = []
        widths = []
        alphas = []
        max_weight = max(np.log1p(list(pair_counts.values())))
        for pair, count in sorted(pair_counts.items(), key=lambda item: item[1]):
            origin, destination = pair
            if origin not in coords or destination not in coords:
                continue
            segments.append([coords[origin], coords[destination]])
            muted = pair in muted_pairs
            colors.append("#adc8d8" if muted else color)
            widths.append(0.35 + 1.0 * (np.log1p(count) / max_weight if max_weight > 0 else 1.0))
            alphas.append(0.13 if muted else 0.30)
        for alpha in sorted(set(alphas)):
            keep = [idx for idx, value in enumerate(alphas) if value == alpha]
            ax.add_collection(LineCollection([segments[idx] for idx in keep], colors=[colors[idx] for idx in keep], linewidths=[widths[idx] for idx in keep], alpha=alpha, zorder=3))
    ax.scatter([p[0] for p in coords.values()], [p[1] for p in coords.values()], s=10, color="#222222", zorder=4, linewidth=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfaf6")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_od_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    colors: list[str],
    labels: list[str],
) -> None:
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=8)
    ax.set_xlabel("Destination tract", fontsize=10, labelpad=6)
    ax.set_ylabel("Origin tract", fontsize=10, labelpad=6)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels([str(i) for i in range(1, matrix.shape[1] + 1)], fontsize=6)
    ax.set_yticklabels([str(i) for i in range(1, matrix.shape[0] + 1)], fontsize=6)
    ax.tick_params(length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _make_focused_trip_coverage_map(path: Path, run_dir: Path) -> bool:
    try:
        import geopandas as gpd
    except Exception:
        _placeholder(path, "Focused trip coverage map", "Install the geo extras to generate tract-level trip maps.")
        return False

    tracts_path = run_dir / "geo" / "tracts.parquet"
    baseline_path = run_dir / "samples" / "weighted_bootstrap_synthetic.csv"
    cvae_path = run_dir / "samples" / "contrastive_vae_synthetic.csv"
    if not (tracts_path.exists() and baseline_path.exists() and cvae_path.exists()):
        _placeholder(path, "Focused trip coverage map", "Synthetic samples or tract geometry were not available in this run.")
        return False

    tracts = gpd.read_parquet(tracts_path)
    tracts = tracts.assign(GEOID=tracts["GEOID"].astype(str).str.zfill(11))
    baseline_od = _load_od_counts(baseline_path, sample_size=0, seed=42)
    cvae_od = _load_od_counts(cvae_path, sample_size=0, seed=42)
    selected = _select_focus_cluster(run_dir, baseline_od, cvae_od, min_tracts=60, max_tracts=60)
    if selected is None:
        _placeholder(path, "Focused trip coverage map", "No connected 24-tract cluster had enough local synthetic OD links.")
        return False
    cluster, metadata = selected

    focus_tracts = tracts[tracts["GEOID"].isin(cluster)].copy()
    if focus_tracts.empty:
        _placeholder(path, "Focused trip coverage map", "Selected tract cluster was not found in the tract geometry.")
        return False

    order = _ordered_focus_tracts(focus_tracts)
    coords = {
        str(row.GEOID): (float(row.rep_x), float(row.rep_y))
        for row in focus_tracts[["GEOID", "rep_x", "rep_y"]].itertuples(index=False)
    }
    baseline_counts = _directed_pair_counts_for_cluster(baseline_od, cluster)
    cvae_counts = _directed_pair_counts_for_cluster(cvae_od, cluster)
    baseline_pairs = set(baseline_counts)
    cvae_pairs = set(cvae_counts)
    common = cvae_pairs & baseline_pairs

    baseline_matrix = _coverage_matrix(baseline_counts, order)
    cvae_matrix = _coverage_matrix(cvae_counts, order)
    cvae_display = cvae_matrix.copy()
    tract_index = {tract: idx for idx, tract in enumerate(order)}
    for origin, destination in common:
        if origin in tract_index and destination in tract_index:
            cvae_display[tract_index[origin], tract_index[destination]] = 1
    for origin, destination in (cvae_pairs - baseline_pairs):
        if origin in tract_index and destination in tract_index:
            cvae_display[tract_index[origin], tract_index[destination]] = 2

    bounds = focus_tracts.total_bounds
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    pad = span * 0.15
    xlim = (float(bounds[0] - pad), float(bounds[2] + pad))
    ylim = (float(bounds[1] - pad), float(bounds[3] + pad))

    fig = plt.figure(figsize=(15, 10), constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.44, 0.56], left=0.055, right=0.985, top=0.835, bottom=0.12, wspace=0.12, hspace=0.30)
    map_base = fig.add_subplot(grid[0, 0])
    map_cvae = fig.add_subplot(grid[0, 1])
    mat_base = fig.add_subplot(grid[1, 0])
    mat_cvae = fig.add_subplot(grid[1, 1])

    fig.text(0.055, 0.965, "24 adjacent tracts: CVAE fills missing local OD pairs", ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(
        0.055,
        0.922,
        f"Auto-selected Anne Arundel County zone centered on tract {metadata['center']}; orange = bootstrap link, pale blue = shared link, dark blue = CVAE-only link.",
        ha="left",
        va="top",
        fontsize=10.5 * text_scale,
        color="#555555",
    )
    fig.text(0.055, 0.885, f"Weighted bootstrap: {metadata['baseline_links']} / {metadata['possible']} links", ha="left", va="top", fontsize=12, color="#9a4c20", fontweight="bold")
    fig.text(0.535, 0.885, f"Contrastive VAE: {metadata['cvae_links']} / {metadata['possible']} links, including {metadata['cvae_only_links']} CVAE-only", ha="left", va="top", fontsize=12, color="#155f92", fontweight="bold")

    _plot_focus_map(map_base, tracts, focus_tracts, baseline_counts, coords, "#d06c2f", xlim, ylim)
    map_base.set_title("Random-resampling local trips", loc="left", fontsize=15, fontweight="bold", pad=6)
    _plot_focus_map(map_cvae, tracts, focus_tracts, cvae_counts, coords, "#2176ae", xlim, ylim, muted_pairs=common)
    map_cvae.set_title("Contrastive VAE local trips", loc="left", fontsize=15, fontweight="bold", pad=6)

    _plot_od_matrix(mat_base, baseline_matrix, "OD coverage matrix", ["#d9d6cc", "#fbfaf6", "#d06c2f", "#d06c2f"], ["absent", "present"])
    _plot_od_matrix(mat_cvae, cvae_display, "OD coverage matrix", ["#d9d6cc", "#fbfaf6", "#a9c9da", "#2176ae"], ["absent", "shared", "CVAE-only"])

    fig.text(0.055, 0.045, "Matrix rows are origin tracts and columns are destination tracts; diagonal cells are ignored. Map lines are centroid-to-centroid visual links, not routed paths.", ha="left", va="bottom", fontsize=9.5, color="#666666")
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _make_aadt_station_match_map(path: Path, run_dir: Path, method: str = "contrastive_vae") -> bool:
    try:
        import geopandas as gpd
        from matplotlib.colors import TwoSlopeNorm
    except Exception:
        _placeholder(path, "AADT station match map", "Install the geo extras to generate station-level validation maps.")
        return False

    tracts_path = run_dir / "geo" / "tracts.parquet"
    screenlines_path = run_dir / "geo" / "screenlines.parquet"
    points_path = run_dir / "geo" / "aadt_points.parquet"
    station_map_path = run_dir / "geo" / "screenline_station_map.parquet"
    comparison_path = run_dir / "metrics" / "aadt_annual_screenline_comparisons.parquet"
    required = [tracts_path, screenlines_path, points_path, station_map_path, comparison_path]
    if not all(item.exists() for item in required):
        _placeholder(path, "AADT station match map", "Annual screenline comparisons or station geometry were not available.")
        return False

    tracts = gpd.read_parquet(tracts_path)
    screenlines = gpd.read_parquet(screenlines_path)
    points = gpd.read_parquet(points_path)[["station_id", "geometry"]].copy()
    station_map = pd.read_parquet(station_map_path)
    comparisons = pd.read_parquet(comparison_path)

    comparisons = comparisons[
        (comparisons["method"].astype(str) == method)
        & (comparisons["validation_tier"].astype(str) == "annual_average")
    ].copy()
    needed_station = {"screenline_id", "station_id", "observed_count"}
    needed_comparison = {"screenline_id", "observed_count", "synthetic_count"}
    if station_map.empty or comparisons.empty or not needed_station.issubset(station_map.columns) or not needed_comparison.issubset(comparisons.columns):
        _placeholder(path, "AADT station match map", "Contrastive VAE annual station comparisons were not available.")
        return False

    station_rows = station_map[["screenline_id", "station_id", "observed_count"]].rename(columns={"observed_count": "station_observed_count"}).copy()
    station_rows["station_id"] = station_rows["station_id"].astype(str)
    comparison_rows = comparisons[
        ["screenline_id", "observed_count", "synthetic_count"]
    ].rename(columns={"observed_count": "screenline_observed_count"}).copy()
    joined = station_rows.merge(comparison_rows, on="screenline_id", how="inner")
    joined = joined[
        (pd.to_numeric(joined["screenline_observed_count"], errors="coerce") > 0)
        & pd.to_numeric(joined["synthetic_count"], errors="coerce").notna()
    ].copy()
    if joined.empty:
        _placeholder(path, "AADT station match map", "No mapped station rows matched contrastive VAE screenline comparisons.")
        return False

    joined["screenline_observed_count"] = pd.to_numeric(joined["screenline_observed_count"], errors="coerce")
    joined["synthetic_count"] = pd.to_numeric(joined["synthetic_count"], errors="coerce")
    joined["station_observed_count"] = pd.to_numeric(joined["station_observed_count"], errors="coerce")
    joined["signed_log2_ratio"] = np.log2((joined["synthetic_count"] + 1.0) / (joined["screenline_observed_count"] + 1.0))
    joined["weight"] = joined["screenline_observed_count"].clip(lower=1.0)
    joined["signed_x_weight"] = joined["signed_log2_ratio"] * joined["weight"]
    joined["abs_x_weight"] = joined["signed_log2_ratio"].abs() * joined["weight"]

    station_stats = (
        joined.groupby("station_id", as_index=False)
        .agg(
            station_observed_count=("station_observed_count", "max"),
            screenline_count=("screenline_id", "nunique"),
            weight_sum=("weight", "sum"),
            signed_weight_sum=("signed_x_weight", "sum"),
            abs_weight_sum=("abs_x_weight", "sum"),
        )
        .copy()
    )
    station_stats["signed_log2_ratio"] = station_stats["signed_weight_sum"] / station_stats["weight_sum"].clip(lower=1e-9)
    station_stats["abs_log2_ratio"] = station_stats["abs_weight_sum"] / station_stats["weight_sum"].clip(lower=1e-9)

    points["station_id"] = points["station_id"].astype(str)
    mapped = points.drop_duplicates("station_id").merge(station_stats, on="station_id", how="inner")
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=points.crs)
    if mapped.empty:
        _placeholder(path, "AADT station match map", "Mapped station IDs did not join to point geometry.")
        return False
    if tracts.crs and mapped.crs and mapped.crs != tracts.crs:
        mapped = mapped.to_crs(tracts.crs)
    if tracts.crs and screenlines.crs and screenlines.crs != tracts.crs:
        screenlines = screenlines.to_crs(tracts.crs)

    log_values = mapped["signed_log2_ratio"].to_numpy(float)
    finite = np.isfinite(log_values)
    mapped = mapped[finite].copy()
    if mapped.empty:
        _placeholder(path, "AADT station match map", "Station-level contrastive VAE ratios were not finite.")
        return False

    marker_scale = np.log1p(mapped["station_observed_count"].fillna(0.0).clip(lower=0.0).to_numpy(float))
    scale_min, scale_max = np.nanpercentile(marker_scale, [10, 98])
    mapped["marker_size"] = _station_marker_sizes(mapped["station_observed_count"], float(scale_min), float(scale_max))
    mapped["plot_log_ratio"] = mapped["signed_log2_ratio"].clip(-3.0, 3.0)
    mapped = mapped.sort_values("marker_size")

    comparison_ratio = (comparisons["synthetic_count"].astype(float) + 1.0) / (comparisons["observed_count"].astype(float) + 1.0)
    within_two = float(((comparison_ratio >= 0.5) & (comparison_ratio <= 2.0)).mean())
    under_two = float((comparison_ratio < 0.5).mean())
    over_two = float((comparison_ratio > 2.0).mean())
    median_ratio = float(np.median(comparison_ratio))

    fig = plt.figure(figsize=(15.0, 8.6), constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    grid = fig.add_gridspec(1, 2, width_ratios=[0.76, 0.24], left=0.035, right=0.985, top=0.82, bottom=0.08, wspace=0.035)
    ax = fig.add_subplot(grid[0, 0])
    side = grid[0, 1].subgridspec(3, 1, height_ratios=[0.35, 0.30, 0.35], hspace=0.25)
    stats_ax = fig.add_subplot(side[0])
    bars_ax = fig.add_subplot(side[1])
    legend_ax = fig.add_subplot(side[2])

    fig.text(0.035, 0.965, "Contrastive VAE screenline match at MDOT stations", ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(
        0.035,
        0.920,
        "Stations inherit the annual-average error from their mapped tract-boundary screenline; repeated station mappings use observed-count-weighted averages.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )

    tracts.boundary.plot(ax=ax, linewidth=0.16, color="#d9d6cc", alpha=0.90, zorder=1)
    plotted_screenlines = screenlines[screenlines["screenline_id"].astype(str).isin(comparisons["screenline_id"].astype(str))]
    plotted_screenlines.plot(ax=ax, color="#aeb6b3", linewidth=0.22, alpha=0.32, zorder=2)
    norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
    scatter = ax.scatter(
        mapped.geometry.x,
        mapped.geometry.y,
        c=mapped["plot_log_ratio"],
        cmap="RdBu",
        norm=norm,
        s=mapped["marker_size"],
        edgecolors="#2b2b2b",
        linewidths=0.25,
        alpha=0.88,
        zorder=3,
    )

    bounds = tracts.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.025
    ypad = (bounds[3] - bounds[1]) * 0.025
    ax.set_xlim(float(bounds[0] - xpad), float(bounds[2] + xpad))
    ax.set_ylim(float(bounds[1] - ypad), float(bounds[3] + ypad))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("#fbfaf6")

    stats_ax.axis("off")
    stats_ax.text(0.0, 0.96, "Annual AAWDT tier", ha="left", va="top", fontsize=14, fontweight="bold", color="#222222")
    stats_ax.text(0.0, 0.74, f"{comparisons['screenline_id'].nunique():,} screenlines compared", ha="left", va="top", fontsize=11.5 * text_scale, color="#333333")
    stats_ax.text(0.0, 0.56, f"{mapped['station_id'].nunique():,} mapped station points", ha="left", va="top", fontsize=11.5, color="#333333")
    stats_ax.text(0.0, 0.38, f"Median CVAE/observed ratio: {median_ratio:.2f}x", ha="left", va="top", fontsize=11.5, color="#333333")
    stats_ax.text(0.0, 0.18, "Faint lines show validation screenlines with mapped stations.", ha="left", va="top", fontsize=9.5, color="#666666", wrap=True)

    bar_labels = ["within 2x", "under by >2x", "over by >2x"]
    bar_values = [within_two, under_two, over_two]
    bar_colors = ["#5a9b62", "#c75d4d", "#3c78a8"]
    bars_ax.barh(np.arange(len(bar_labels)), bar_values, color=bar_colors, height=0.56)
    bars_ax.set_xlim(0.0, 1.0)
    bars_ax.set_yticks(np.arange(len(bar_labels)))
    bars_ax.set_yticklabels(bar_labels, fontsize=10)
    bars_ax.invert_yaxis()
    bars_ax.set_xlabel("share of screenlines", fontsize=9.5)
    bars_ax.tick_params(axis="x", labelsize=9)
    bars_ax.grid(axis="x", alpha=0.18)
    for spine in bars_ax.spines.values():
        spine.set_visible(False)
    for idx, value in enumerate(bar_values):
        bars_ax.text(min(value + 0.025, 0.98), idx, f"{value:.0%}", va="center", ha="left", fontsize=10, color="#222222")

    legend_ax.axis("off")
    cax = legend_ax.inset_axes([0.02, 0.70, 0.92, 0.12])
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_ticklabels(["1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x"])
    cbar.ax.tick_params(labelsize=8, length=0, pad=2)
    cbar.ax.set_title("CVAE / observed screenline volume", fontsize=9.5, pad=8)

    legend_counts = np.array([5_000.0, 25_000.0, 100_000.0])
    legend_sizes = _station_marker_sizes(legend_counts, float(scale_min), float(scale_max))
    legend_ax.text(0.02, 0.48, "Point size: station AAWDT", ha="left", va="center", fontsize=9.5, color="#333333")
    xs = [0.17, 0.48, 0.80]
    for x, count, size in zip(xs, legend_counts, legend_sizes):
        legend_ax.scatter([x], [0.28], s=size, color="#777777", edgecolors="#2b2b2b", linewidths=0.25, alpha=0.78)
        legend_ax.text(x, 0.08, _format_compact_count(float(count)), ha="center", va="center", fontsize=8.5, color="#333333")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _count_method_od_pairs(sample_path: Path) -> pd.DataFrame:
    if not sample_path.exists():
        return pd.DataFrame(columns=["o_tract_fips", "d_tract_fips", "sample_rows"])
    sample = pd.read_csv(sample_path, usecols=["o_tract_fips", "d_tract_fips"], dtype="string")
    sample["o_tract_fips"] = _clean_fips_series(sample["o_tract_fips"])
    sample["d_tract_fips"] = _clean_fips_series(sample["d_tract_fips"])
    sample = sample[(sample["o_tract_fips"] != "-1") & (sample["d_tract_fips"] != "-1")]
    return sample.groupby(["o_tract_fips", "d_tract_fips"], as_index=False).size().rename(columns={"size": "sample_rows"})


def _select_od_pair_validation_example(
    run_dir: Path,
    method: str = "noncontrastive_vae",
    reference_method: str = "bayesian_network",
    selection_strategy: str = "method_vs_reference",
) -> dict[str, Any] | None:
    paths_path = run_dir / "geo" / "od_screenline_paths.parquet"
    comparison_path = run_dir / "metrics" / "aadt_annual_screenline_comparisons.parquet"
    sample_path = run_dir / "samples" / f"{method}_synthetic.csv"
    if not (paths_path.exists() and comparison_path.exists() and sample_path.exists()):
        return None

    paths = pd.read_parquet(paths_path)
    all_comparisons = pd.read_parquet(comparison_path)
    all_comparisons = all_comparisons[all_comparisons["validation_tier"].astype(str) == "annual_average"].copy()
    comparisons = all_comparisons[all_comparisons["method"].astype(str) == method].copy()
    if paths.empty or comparisons.empty:
        return None

    comparisons["ratio"] = (comparisons["synthetic_count"].astype(float) + 1.0) / (comparisons["observed_count"].astype(float) + 1.0)
    comparisons["signed_log2_ratio"] = np.log2(comparisons["ratio"])
    comparisons["abs_log2_ratio"] = comparisons["signed_log2_ratio"].abs()

    path_len = paths.groupby(["o_tract_fips", "d_tract_fips"]).size().rename("path_len").reset_index()
    merged = paths.merge(
        comparisons[["screenline_id", "observed_count", "synthetic_count", "station_count", "ratio", "signed_log2_ratio", "abs_log2_ratio"]],
        on="screenline_id",
        how="inner",
    )
    if merged.empty:
        return None

    stats = (
        merged.groupby(["o_tract_fips", "d_tract_fips"])
        .agg(
            compared_path_rows=("screenline_id", "size"),
            compared_screenlines=("screenline_id", "nunique"),
            mean_abs_log2=("abs_log2_ratio", "mean"),
            max_abs_log2=("abs_log2_ratio", "max"),
            min_ratio=("ratio", "min"),
            max_ratio=("ratio", "max"),
            total_observed=("observed_count", "sum"),
            total_synthetic=("synthetic_count", "sum"),
            total_stations=("station_count", "sum"),
        )
        .reset_index()
        .merge(path_len, on=["o_tract_fips", "d_tract_fips"], how="inner")
    )
    counts = _count_method_od_pairs(sample_path)
    stats = stats.merge(counts, on=["o_tract_fips", "d_tract_fips"], how="inner")
    stats = stats[
        (stats["compared_path_rows"] == stats["path_len"])
        & (stats["compared_screenlines"] == stats["path_len"])
        & (stats["path_len"].between(4, 7))
        & (stats["sample_rows"] >= 5)
        & (stats["total_stations"] >= 4)
        & (stats["total_observed"] >= 20_000)
    ].copy()
    if stats.empty:
        return None

    all_comparisons["ratio"] = (all_comparisons["synthetic_count"].astype(float) + 1.0) / (all_comparisons["observed_count"].astype(float) + 1.0)
    all_comparisons["abs_log2_ratio"] = np.log2(all_comparisons["ratio"]).abs()
    candidate_paths = paths.merge(stats[["o_tract_fips", "d_tract_fips"]], on=["o_tract_fips", "d_tract_fips"], how="inner")
    method_path_metrics = candidate_paths.merge(
        all_comparisons[["screenline_id", "method", "observed_count", "synthetic_count", "abs_log2_ratio"]],
        on="screenline_id",
        how="inner",
    )
    method_summary = (
        method_path_metrics.groupby(["o_tract_fips", "d_tract_fips", "method"])
        .agg(
            method_mean_abs=("abs_log2_ratio", "mean"),
            method_max_abs=("abs_log2_ratio", "max"),
            method_observed=("observed_count", "sum"),
            method_synthetic=("synthetic_count", "sum"),
        )
        .reset_index()
    )
    wide = method_summary.pivot(index=["o_tract_fips", "d_tract_fips"], columns="method")
    wide.columns = [f"{metric}_{name}" for metric, name in wide.columns]
    wide = wide.reset_index()
    stats = stats.merge(wide, on=["o_tract_fips", "d_tract_fips"], how="left")

    method_mean_col = f"method_mean_abs_{method}"
    reference_mean_col = f"method_mean_abs_{reference_method}"
    method_max_col = f"method_max_abs_{method}"
    if selection_strategy == "contrastive_best":
        comparison_methods = ["weighted_bootstrap", "bayesian_network", "noncontrastive_vae"]
        required_cols = [method_mean_col, method_max_col] + [f"method_mean_abs_{name}" for name in comparison_methods]
        if all(col in stats.columns for col in required_cols):
            for name in comparison_methods:
                stats[f"{name}_minus_method_abs"] = stats[f"method_mean_abs_{name}"] - stats[method_mean_col]
            stats = stats[
                (stats[method_mean_col] < 0.55)
                & (stats[method_max_col] < 1.10)
                & (stats["bayesian_network_minus_method_abs"] > 0.10)
                & (stats["noncontrastive_vae_minus_method_abs"] > 0.0)
                & (stats["weighted_bootstrap_minus_method_abs"] > 0.0)
            ].copy()
            if not stats.empty:
                stats["total_ratio"] = (stats["total_synthetic"] + 1.0) / (stats["total_observed"] + 1.0)
                stats["contrastive_best_score"] = (
                    4.0 * stats["bayesian_network_minus_method_abs"]
                    + 2.0 * stats["noncontrastive_vae_minus_method_abs"]
                    + stats["weighted_bootstrap_minus_method_abs"]
                    - 0.25 * stats[method_mean_col]
                )
                stats = stats.sort_values(
                    ["bayesian_network_minus_method_abs", "noncontrastive_vae_minus_method_abs", "contrastive_best_score", method_mean_col],
                    ascending=[False, False, False, True],
                )
                row = stats.iloc[0]
                return {key: row[key] for key in row.index}

    if method_mean_col in stats.columns and reference_mean_col in stats.columns:
        stats["reference_minus_method_abs"] = stats[reference_mean_col] - stats[method_mean_col]
        stats = stats[
            (stats["reference_minus_method_abs"] > 0)
            & (stats[method_mean_col] < 0.45)
            & (stats[method_max_col] < 0.90)
        ].copy()
        if not stats.empty:
            stats["total_ratio"] = (stats["total_synthetic"] + 1.0) / (stats["total_observed"] + 1.0)
            stats = stats.sort_values(
                ["reference_minus_method_abs", method_mean_col, "sample_rows", "path_len"],
                ascending=[False, True, False, True],
            )
            row = stats.iloc[0]
            return {key: row[key] for key in row.index}

    stats["total_ratio"] = (stats["total_synthetic"] + 1.0) / (stats["total_observed"] + 1.0)
    stats = stats.sort_values(
        ["max_abs_log2", "mean_abs_log2", "path_len", "sample_rows"],
        ascending=[True, True, False, False],
    )
    row = stats.iloc[0]
    return {key: row[key] for key in row.index}


def _screenline_midpoint(geom: Any) -> tuple[float, float]:
    point = geom.representative_point()
    return float(point.x), float(point.y)


def _nearest_point_on_line(line_geom: Any, point_geom: Any) -> Any | None:
    try:
        return line_geom.interpolate(line_geom.project(point_geom))
    except Exception:
        return None


def _region_label_from_tracts(tracts: Any) -> str:
    state_names = {"11": "District of Columbia", "24": "Maryland", "51": "Virginia"}
    county_names = {
        ("24", "003"): "Anne Arundel County",
        ("24", "005"): "Baltimore County",
        ("24", "013"): "Carroll County",
        ("24", "025"): "Harford County",
        ("24", "027"): "Howard County",
        ("24", "031"): "Montgomery County",
        ("24", "033"): "Prince George's County",
        ("24", "510"): "Baltimore City",
        ("11", "001"): "Washington, DC",
    }
    if not {"STATEFP", "COUNTYFP"}.issubset(tracts.columns):
        return "selected study-area region"
    pairs = sorted({(str(row.STATEFP).zfill(2), str(row.COUNTYFP).zfill(3)) for row in tracts[["STATEFP", "COUNTYFP"]].itertuples(index=False)})
    names = [county_names.get(pair) for pair in pairs]
    if names and all(names):
        state_values = {pair[0] for pair in pairs}
        if len(names) == 1:
            state = state_names.get(pairs[0][0], "study area")
            if names[0].endswith("DC"):
                return names[0]
            return f"{names[0]}, {state}"
        if len(state_values) == 1:
            state = state_names.get(next(iter(state_values)), "study area")
            return f"{', '.join(names[:-1])} and {names[-1]}, {state}"
        return ", ".join(names)
    states = sorted({state_names.get(pair[0], pair[0]) for pair in pairs})
    if len(states) == 1:
        return f"{states[0]} study-area region"
    return "study-area region"


def _make_od_pair_screenline_validation_map(
    path: Path,
    run_dir: Path,
    method: str = "noncontrastive_vae",
    selection_strategy: str = "method_vs_reference",
) -> bool:
    try:
        import geopandas as gpd
        import matplotlib.patheffects as path_effects
    except Exception:
        _placeholder(path, "OD-pair validation map", "Install the geo extras to generate OD-pair screenline validation maps.")
        return False

    tracts_path = run_dir / "geo" / "tracts.parquet"
    screenlines_path = run_dir / "geo" / "screenlines.parquet"
    points_path = run_dir / "geo" / "aadt_points.parquet"
    station_map_path = run_dir / "geo" / "screenline_station_map.parquet"
    paths_path = run_dir / "geo" / "od_screenline_paths.parquet"
    comparison_path = run_dir / "metrics" / "aadt_annual_screenline_comparisons.parquet"
    required = [tracts_path, screenlines_path, points_path, station_map_path, paths_path, comparison_path]
    if not all(item.exists() for item in required):
        _placeholder(path, "OD-pair validation map", "OD paths, station maps, or annual screenline comparisons were not available.")
        return False

    method_specs = [
        ("weighted_bootstrap", "Weighted sampling baseline", "#d06c2f"),
        ("bayesian_network", "Bayesian network", "#4c9f70"),
        ("noncontrastive_vae", "Non-contrastive VAE", "#756bb1"),
        ("contrastive_vae", "Contrastive VAE", "#2176ae"),
    ]
    method_labels = {
        "weighted_bootstrap": "Baseline",
        "bayesian_network": "Bayesian",
        "noncontrastive_vae": "Non-C VAE",
        "contrastive_vae": "C VAE",
    }
    method_colors = {key: color for key, _, color in method_specs}
    screenline_palette = ["#2c7fb8", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#76b7b2", "#edc948"]

    selected = _select_od_pair_validation_example(run_dir, method=method, selection_strategy=selection_strategy)
    if selected is None:
        _placeholder(path, "OD-pair validation map", "No generated OD pair had a compact, well-matched station screenline path.")
        return False
    origin = str(selected["o_tract_fips"])
    destination = str(selected["d_tract_fips"])

    tracts = gpd.read_parquet(tracts_path)
    tracts = tracts.assign(GEOID=tracts["GEOID"].astype(str).str.zfill(11))
    screenlines = gpd.read_parquet(screenlines_path)
    points = gpd.read_parquet(points_path)[["station_id", "geometry"]].copy()
    points["station_id"] = points["station_id"].astype(str)
    station_map = pd.read_parquet(station_map_path)
    paths = pd.read_parquet(paths_path)
    comparisons = pd.read_parquet(comparison_path)
    comparisons = comparisons[comparisons["validation_tier"].astype(str) == "annual_average"].copy()

    selected_comparisons = comparisons[comparisons["method"].astype(str) == method].copy()
    selected_comparisons["ratio"] = (selected_comparisons["synthetic_count"].astype(float) + 1.0) / (selected_comparisons["observed_count"].astype(float) + 1.0)

    path_rows = paths[(paths["o_tract_fips"].astype(str) == origin) & (paths["d_tract_fips"].astype(str) == destination)].copy()
    path_rows = path_rows.merge(
        selected_comparisons[["screenline_id", "observed_count", "synthetic_count", "station_count", "ratio"]],
        on="screenline_id",
        how="inner",
    ).sort_values("path_position")
    if path_rows.empty:
        _placeholder(path, "OD-pair validation map", "Selected OD pair did not join to annual screenline comparisons.")
        return False
    path_rows["path_order"] = np.arange(1, len(path_rows) + 1)
    path_rows["screenline_color"] = [screenline_palette[(int(order) - 1) % len(screenline_palette)] for order in path_rows["path_order"]]

    origin_row = tracts[tracts["GEOID"] == origin]
    dest_row = tracts[tracts["GEOID"] == destination]
    if origin_row.empty or dest_row.empty:
        _placeholder(path, "OD-pair validation map", "Selected OD tract geometry was unavailable.")
        return False
    origin_xy = (float(origin_row.iloc[0]["rep_x"]), float(origin_row.iloc[0]["rep_y"]))
    dest_xy = (float(dest_row.iloc[0]["rep_x"]), float(dest_row.iloc[0]["rep_y"]))

    path_tracts: set[str] = {origin, destination}
    for sid in path_rows["screenline_id"].astype(str):
        left, right = sid.split("__")
        path_tracts.update([left, right])
    focus_tracts = tracts[tracts["GEOID"].isin(path_tracts)].copy()
    if focus_tracts.empty:
        _placeholder(path, "OD-pair validation map", "Selected path tract geometry was unavailable.")
        return False
    region_label = _region_label_from_tracts(focus_tracts)

    bounds = focus_tracts.total_bounds
    xpad = max((bounds[2] - bounds[0]) * 0.22, 1_400.0)
    ypad = max((bounds[3] - bounds[1]) * 0.20, 1_400.0)
    xlim = (float(bounds[0] - xpad), float(bounds[2] + xpad))
    ylim = (float(bounds[1] - ypad), float(bounds[3] + ypad))
    context = tracts.cx[xlim[0] : xlim[1], ylim[0] : ylim[1]].copy()

    screenline_rows = screenlines.merge(
        path_rows[["screenline_id", "path_order", "screenline_color", "observed_count", "synthetic_count", "ratio"]],
        on="screenline_id",
        how="inner",
    )
    screenline_rows = gpd.GeoDataFrame(screenline_rows, geometry="geometry", crs=screenlines.crs)
    if tracts.crs and screenline_rows.crs and screenline_rows.crs != tracts.crs:
        screenline_rows = screenline_rows.to_crs(tracts.crs)
    if tracts.crs and points.crs and points.crs != tracts.crs:
        points = points.to_crs(tracts.crs)

    station_rows = station_map[station_map["screenline_id"].isin(path_rows["screenline_id"])].copy()
    station_rows = station_rows.merge(
        path_rows[["screenline_id", "path_order", "screenline_color"]],
        on="screenline_id",
        how="left",
    )
    station_points = station_rows.merge(points.drop_duplicates("station_id"), on="station_id", how="left")
    station_points = gpd.GeoDataFrame(station_points, geometry="geometry", crs=points.crs)
    station_points = station_points[station_points.geometry.notna()].copy()
    if not station_points.empty:
        station_points = station_points.sort_values(["path_order", "station_id"]).reset_index(drop=True)
        station_points["plot_x"] = station_points.geometry.x
        station_points["plot_y"] = station_points.geometry.y
        for _, group in station_points.groupby("station_id"):
            if len(group) == 1:
                continue
            angles = np.linspace(0, 2 * np.pi, len(group), endpoint=False)
            radius = 155.0
            for idx, angle in zip(group.index, angles):
                station_points.loc[idx, "plot_x"] += radius * float(np.cos(angle))
                station_points.loc[idx, "plot_y"] += radius * float(np.sin(angle))
        logged_counts = np.log1p(pd.to_numeric(station_points["observed_count"], errors="coerce").fillna(0.0).clip(lower=0.0))
        scale_min, scale_max = np.nanpercentile(logged_counts, [5, 95]) if len(station_points) > 1 else (float(logged_counts.iloc[0]), float(logged_counts.iloc[0]))
        station_points["marker_size"] = _station_marker_sizes(station_points["observed_count"], float(scale_min), float(scale_max)) + 80.0

    method_keys = [key for key, _, _ in method_specs if key in set(comparisons["method"].astype(str))]
    comparison_subset = comparisons[comparisons["method"].astype(str).isin(method_keys)].copy()
    count_rows = path_rows[["screenline_id", "path_order", "observed_count"]].drop_duplicates("screenline_id")
    method_counts = comparison_subset[comparison_subset["screenline_id"].isin(count_rows["screenline_id"])][
        ["screenline_id", "method", "synthetic_count"]
    ].copy()
    method_wide = method_counts.pivot_table(index="screenline_id", columns="method", values="synthetic_count", aggfunc="first")
    count_rows = count_rows.join(method_wide, on="screenline_id")
    available_methods = [key for key in method_keys if key in count_rows.columns]

    text_scale = 2.0
    fig = plt.figure(figsize=(15.0, 8.4), constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    grid = fig.add_gridspec(1, 2, width_ratios=[0.61, 0.39], left=0.035, right=0.985, top=0.965, bottom=0.065, wspace=0.075)
    ax = fig.add_subplot(grid[0, 0])
    side = grid[0, 1].subgridspec(5, 1, height_ratios=[0.18, 0.06, 0.42, 0.16, 0.41], hspace=0.0)
    legend_ax = fig.add_subplot(side[0])
    bars_ax = fig.add_subplot(side[2])
    sum_ax = fig.add_subplot(side[4])

    context.boundary.plot(ax=ax, linewidth=0.22, color="#d8d4ca", alpha=0.95, zorder=1)
    focus_tracts.plot(ax=ax, facecolor="#fffdf7", edgecolor="#77746d", linewidth=0.85, alpha=0.96, zorder=2)
    origin_row.plot(ax=ax, facecolor="#f2b36f", edgecolor="#9a4c20", linewidth=1.1, alpha=0.72, zorder=3)
    dest_row.plot(ax=ax, facecolor="#9bc2d9", edgecolor="#155f92", linewidth=1.1, alpha=0.72, zorder=3)

    screenline_rows = screenline_rows.sort_values("path_order")
    for row in screenline_rows.itertuples(index=False):
        screenline_rows[screenline_rows["screenline_id"] == row.screenline_id].plot(
            ax=ax,
            color=row.screenline_color,
            linewidth=6.2,
            alpha=0.96,
            zorder=5,
        )

    arrow = FancyArrowPatch(
        origin_xy,
        dest_xy,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color="#202020",
        alpha=0.82,
        shrinkA=8,
        shrinkB=8,
        zorder=6,
    )
    ax.add_patch(arrow)
    ax.scatter([origin_xy[0], dest_xy[0]], [origin_xy[1], dest_xy[1]], s=[82, 82], color=["#9a4c20", "#155f92"], edgecolor="white", linewidth=0.9, zorder=8)
    for label, xy, ha, va in [("O", origin_xy, "right", "bottom"), ("D", dest_xy, "left", "top")]:
        text = ax.text(xy[0], xy[1], f" {label} ", ha=ha, va=va, fontsize=10 * text_scale, fontweight="bold", color="#222222", zorder=9)
        text.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])

    screenline_lookup = screenline_rows.set_index("screenline_id")
    for row in path_rows.itertuples(index=False):
        geom = screenline_lookup.loc[str(row.screenline_id)].geometry
        x, y = _screenline_midpoint(geom)
        number_color = str(row.screenline_color)
        txt = ax.text(
            x,
            y,
            str(int(row.path_order)),
            ha="center",
            va="center",
            fontsize=10.5 * text_scale,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.26", facecolor=number_color, edgecolor="white", linewidth=1.2, alpha=0.98),
            zorder=11,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=1.0, foreground="#222222")])

    if not station_points.empty:
        for station in station_points.itertuples(index=False):
            if str(station.screenline_id) not in screenline_lookup.index:
                continue
            target = _nearest_point_on_line(screenline_lookup.loc[str(station.screenline_id)].geometry, station.geometry)
            if target is None:
                continue
            ax.plot(
                [station.plot_x, target.x],
                [station.plot_y, target.y],
                color=station.screenline_color,
                linewidth=1.0,
                alpha=0.44,
                zorder=4,
            )
        ax.scatter(
            station_points["plot_x"],
            station_points["plot_y"],
            s=station_points["marker_size"],
            color=station_points["screenline_color"],
            edgecolor="white",
            linewidth=1.4,
            alpha=0.98,
            zorder=10,
        )
        ax.scatter(
            station_points["plot_x"],
            station_points["plot_y"],
            s=station_points["marker_size"] * 0.34,
            color="#222222",
            linewidth=0,
            alpha=0.88,
            zorder=11,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("#fbfaf6")

    plot_rows = count_rows.sort_values("path_order").copy()
    y = np.arange(len(plot_rows), dtype=float)
    series = [("observed_count", "Observed", "#76736d")] + [
        (key, method_labels[key], method_colors[key]) for key in available_methods
    ]
    offsets = np.linspace(-0.32, 0.32, len(series)) if len(series) > 1 else np.array([0.0])
    bar_height = min(0.12, 0.68 / max(len(series), 1))
    max_count = float(np.nanmax(plot_rows[[col for col, _, _ in series]].to_numpy(float)))
    for offset, (col, label, color) in zip(offsets, series):
        values = pd.to_numeric(plot_rows[col], errors="coerce").fillna(0.0).to_numpy(float)
        bars_ax.barh(y + offset, values, height=bar_height, color=color, label=label, alpha=0.94)

    legend_ax.axis("off")
    handles = [plt.Line2D([0], [0], color=color, lw=5.0, label=label) for _, label, color in series]
    legend_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.98), ncol=2, frameon=False, fontsize=8.0 * text_scale, handlelength=1.25, columnspacing=0.9, labelspacing=0.42, borderaxespad=0.0)

    row_colors = dict(zip(path_rows["path_order"].astype(int), path_rows["screenline_color"].astype(str)))
    tick_labels = [f"Screenline {int(row.path_order)}" for row in plot_rows.itertuples(index=False)]
    bars_ax.set_yticks(y)
    bars_ax.set_yticklabels(tick_labels, fontsize=9.2 * text_scale)
    for tick, row in zip(bars_ax.get_yticklabels(), plot_rows.itertuples(index=False)):
        tick.set_color(row_colors.get(int(row.path_order), "#222222"))
        tick.set_fontweight("bold")
    bars_ax.invert_yaxis()
    bars_ax.set_xlim(0, max_count * 1.22)
    bars_ax.set_xlabel("annual-average screenline count", fontsize=9.4 * text_scale)
    bars_ax.tick_params(axis="x", labelsize=8.3 * text_scale)
    for idx, tick in enumerate(bars_ax.get_xticklabels()):
        if idx % 2 == 1:
            tick.set_visible(False)
    bars_ax.grid(axis="x", alpha=0.20)
    for spine in bars_ax.spines.values():
        spine.set_visible(False)

    sum_values = np.array([float(plot_rows[col].sum()) for col, _, _ in series])
    sum_y = np.arange(len(series), dtype=float)
    sum_ax.barh(sum_y, sum_values, color=[color for _, _, color in series], height=0.56, alpha=0.94)
    sum_ax.set_yticks(sum_y)
    short_labels = ["Observed", "Baseline", "Bayesian", "Non-C VAE", "C VAE"][: len(series)]
    sum_ax.set_yticklabels(short_labels, fontsize=8.2 * text_scale)
    sum_ax.invert_yaxis()
    sum_ax.set_xlim(0, max(float(np.nanmax(sum_values)) * 1.18, 1.0))
    sum_ax.set_xlabel("summed count", fontsize=8.8 * text_scale, labelpad=2)
    sum_ax.tick_params(axis="x", labelsize=8.0 * text_scale)
    for idx, tick in enumerate(sum_ax.get_xticklabels()):
        if idx % 2 == 1:
            tick.set_visible(False)
    sum_ax.grid(axis="x", alpha=0.20)
    for spine in sum_ax.spines.values():
        spine.set_visible(False)
    for yi, value in zip(sum_y, sum_values):
        sum_ax.text(value + max(float(np.nanmax(sum_values)) * 0.025, 1.0), yi, _format_compact_count(value), ha="left", va="center", fontsize=8.0 * text_scale, color="#333333")

    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _select_hourly_tmas_validation_example(run_dir: Path, method: str = "contrastive_vae") -> dict[str, Any] | None:
    comparison_path = run_dir / "metrics" / "aadt_hourly_screenline_comparisons.parquet"
    if not comparison_path.exists():
        return None

    method_keys = ["weighted_bootstrap", "bayesian_network", "noncontrastive_vae", "contrastive_vae"]
    comparisons = pd.read_parquet(comparison_path)
    needed = {"screenline_id", "method", "hour", "observed_count", "synthetic_count", "tmas_station_count"}
    if comparisons.empty or not needed.issubset(comparisons.columns):
        return None
    comparisons = comparisons[comparisons["method"].astype(str).isin(method_keys) & comparisons["tmas_station_count"].eq(1)].copy()
    if comparisons.empty:
        return None

    profile = (
        comparisons.groupby(["screenline_id", "method", "hour"], as_index=False)
        .agg(
            observed_count=("observed_count", "mean"),
            synthetic_count=("synthetic_count", "mean"),
            cells=("synthetic_count", "size"),
            annual_station_count_observed=("annual_station_count_observed", "first"),
            annual_screenline_count=("annual_screenline_count", "first"),
        )
    )
    profile["abs_error"] = (profile["synthetic_count"].astype(float) - profile["observed_count"].astype(float)).abs()
    profile["abs_log2_ratio"] = np.log2(
        (profile["synthetic_count"].astype(float) + 1.0) / (profile["observed_count"].astype(float) + 1.0)
    ).abs()
    summary = (
        profile.groupby(["screenline_id", "method"])
        .agg(
            profile_mae=("abs_error", "mean"),
            profile_mean_abs_log2=("abs_log2_ratio", "mean"),
            profile_max_abs_log2=("abs_log2_ratio", "max"),
            hours=("hour", "nunique"),
            mean_observed=("observed_count", "mean"),
            peak_observed=("observed_count", "max"),
            annual_station_count_observed=("annual_station_count_observed", "first"),
            annual_screenline_count=("annual_screenline_count", "first"),
        )
        .reset_index()
    )
    wide = summary.pivot(index="screenline_id", columns="method")
    wide.columns = [f"{metric}_{name}" for metric, name in wide.columns]
    wide = wide.reset_index()
    if wide.empty:
        return None

    mask = pd.Series(True, index=wide.index)
    for key in method_keys:
        hours_col = f"hours_{key}"
        if hours_col not in wide.columns:
            return None
        mask &= wide[hours_col].eq(24)
    mask &= wide[f"mean_observed_{method}"].ge(50)
    mask &= wide[f"peak_observed_{method}"].ge(100)
    comparison_methods = [key for key in method_keys if key != method]
    for key in comparison_methods:
        wide[f"profile_mae_gap_{key}"] = wide[f"profile_mae_{key}"] - wide[f"profile_mae_{method}"]
        mask &= wide[f"profile_mae_gap_{key}"].gt(0)

    candidates = wide[mask].copy()
    if candidates.empty:
        candidates = wide[wide[f"hours_{method}"].eq(24)].copy()
        if candidates.empty:
            return None
        candidates["hourly_profile_score"] = -candidates[f"profile_mae_{method}"]
    else:
        candidates["hourly_profile_score"] = (
            2.0 * candidates.get("profile_mae_gap_bayesian_network", 0.0)
            + 1.5 * candidates.get("profile_mae_gap_noncontrastive_vae", 0.0)
            + candidates.get("profile_mae_gap_weighted_bootstrap", 0.0)
            - 0.05 * candidates[f"profile_mae_{method}"]
        )
    candidates = candidates.sort_values(
        ["hourly_profile_score", f"profile_mae_{method}", f"peak_observed_{method}"],
        ascending=[False, True, False],
    )
    row = candidates.iloc[0]
    selected = {key: row[key] for key in row.index}

    station_map_path = run_dir / "geo" / "screenline_station_map.parquet"
    if station_map_path.exists():
        station_map = pd.read_parquet(station_map_path)
        station_rows = station_map[station_map["screenline_id"].astype(str).eq(str(selected["screenline_id"]))].copy()
        if not station_rows.empty and f"annual_station_count_observed_{method}" in selected:
            target = float(selected[f"annual_station_count_observed_{method}"])
            station_rows["observed_count_numeric"] = pd.to_numeric(station_rows["observed_count"], errors="coerce")
            match = station_rows[np.isclose(station_rows["observed_count_numeric"].astype(float), target, rtol=0.0, atol=1e-6)]
            if not match.empty:
                selected["station_id"] = str(match.iloc[0]["station_id"])
                selected["station_observed_count"] = float(match.iloc[0]["observed_count_numeric"])
            elif len(station_rows) == 1:
                selected["station_id"] = str(station_rows.iloc[0]["station_id"])
                selected["station_observed_count"] = float(station_rows.iloc[0]["observed_count_numeric"])
    return selected


def _make_hourly_tmas_validation_profile(path: Path, run_dir: Path, method: str = "contrastive_vae") -> bool:
    comparison_path = run_dir / "metrics" / "aadt_hourly_screenline_comparisons.parquet"
    observed_path = run_dir / "metrics" / "observed_hourly_screenline_counts.parquet"
    if not comparison_path.exists():
        _placeholder(path, "Hourly TMAS validation profile", "Hourly TMAS screenline comparisons were not available.")
        return False

    selected = _select_hourly_tmas_validation_example(run_dir, method=method)
    if selected is None:
        _placeholder(path, "Hourly TMAS validation profile", "No single-TMAS-station hourly screenline had a complete comparable profile.")
        return False
    screenline_id = str(selected["screenline_id"])

    method_specs = [
        ("weighted_bootstrap", "Survey-Sampled Baseline", "#d06c2f", 2.7, 0.74),
        ("bayesian_network", "Bayesian Network", "#4c9f70", 2.7, 0.74),
        ("noncontrastive_vae", "Non-Contrastive VAE", "#756bb1", 2.9, 0.78),
        ("contrastive_vae", "Contrastive VAE", "#2176ae", 4.3, 0.98),
    ]
    method_keys = [key for key, _, _, _, _ in method_specs]
    comparisons = pd.read_parquet(comparison_path)
    comparisons = comparisons[comparisons["screenline_id"].astype(str).eq(screenline_id) & comparisons["method"].astype(str).isin(method_keys)].copy()
    if comparisons.empty:
        _placeholder(path, "Hourly TMAS validation profile", "Selected hourly screenline did not join to method comparisons.")
        return False

    method_profiles = (
        comparisons.groupby(["method", "hour"], as_index=False)
        .agg(synthetic_count=("synthetic_count", "mean"), observed_count=("observed_count", "mean"))
    )
    if observed_path.exists():
        observed = pd.read_parquet(observed_path)
        observed = observed[observed["screenline_id"].astype(str).eq(screenline_id)].copy()
        observed_profile = observed.groupby("hour", as_index=False)["observed_count"].mean() if not observed.empty else pd.DataFrame()
    else:
        observed_profile = pd.DataFrame()
    if observed_profile.empty:
        observed_profile = (
            method_profiles[method_profiles["method"].eq(method)]
            .groupby("hour", as_index=False)["observed_count"]
            .mean()
        )
    if observed_profile.empty:
        _placeholder(path, "Hourly TMAS validation profile", "Selected hourly screenline had no observed hourly profile.")
        return False

    observed_profile = observed_profile.sort_values("hour")
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.4), sharex=True, sharey=True, constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.105, wspace=0.18, hspace=0.245)

    ymax = max(
        float(observed_profile["observed_count"].max()),
        float(method_profiles["synthetic_count"].max()),
        1.0,
    )
    from matplotlib.ticker import FuncFormatter

    for ax, (key, label, color, linewidth, alpha) in zip(axes.ravel(), method_specs):
        ax.set_facecolor("#fbfaf6")
        rows = method_profiles[method_profiles["method"].eq(key)].sort_values("hour")
        ax.plot(
            observed_profile["hour"],
            observed_profile["observed_count"],
            label="Observed",
            color="#222222",
            linewidth=3.6,
            alpha=0.96,
            zorder=7,
        )
        if not rows.empty:
            ax.plot(
                rows["hour"],
                rows["synthetic_count"],
                label=label,
                color=color,
                linewidth=3.6 if key == method else 3.1,
                alpha=0.96 if key == method else alpha,
                zorder=8 if key == method else 6,
            )
        ax.set_title(label, loc="left", fontsize=19, fontweight="bold", pad=7, color="#222222")
        ax.set_xlim(0, 23)
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.tick_params(axis="both", labelsize=14.5, length=0, pad=6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_compact_count(float(value))))
        ax.grid(axis="y", color="#b8b2a8", alpha=0.32, linewidth=0.9)
        ax.grid(axis="x", color="#d9d2c6", alpha=0.16, linewidth=0.75)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#8d877d")
            ax.spines[spine].set_linewidth(0.9)
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=13.5,
            handlelength=1.45,
            handletextpad=0.45,
            borderaxespad=0.0,
        )

    fig.supxlabel("Hour of day", fontsize=22, y=0.032)
    fig.supylabel("Hourly traffic count", fontsize=22, x=0.022)

    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _make_hourly_tmas_aggregate_profile(path: Path, run_dir: Path) -> bool:
    comparison_path = run_dir / "metrics" / "aadt_hourly_screenline_comparisons.parquet"
    if not comparison_path.exists():
        _placeholder(path, "Aggregate hourly TMAS validation profile", "Hourly TMAS screenline comparisons were not available.")
        return False

    method_specs = [
        ("weighted_bootstrap", "Survey-Sampled Baseline", "#d06c2f", 2.7, 0.74),
        ("bayesian_network", "Bayesian Network", "#4c9f70", 2.7, 0.74),
        ("noncontrastive_vae", "Non-Contrastive VAE", "#756bb1", 2.9, 0.78),
        ("contrastive_vae", "Contrastive VAE", "#2176ae", 4.3, 0.98),
    ]
    method_keys = [key for key, _, _, _, _ in method_specs]
    comparisons = pd.read_parquet(comparison_path)
    needed = {"screenline_id", "trip_date", "hour", "method", "observed_count", "synthetic_count", "tmas_station_count"}
    if comparisons.empty or not needed.issubset(comparisons.columns):
        _placeholder(path, "Aggregate hourly TMAS validation profile", "Hourly TMAS comparison fields were incomplete.")
        return False
    comparisons = comparisons[comparisons["method"].astype(str).isin(method_keys) & comparisons["tmas_station_count"].gt(0)].copy()
    if comparisons.empty:
        _placeholder(path, "Aggregate hourly TMAS validation profile", "No TMAS-measured screenline-hour comparisons were available.")
        return False
    comparisons["trip_date"] = pd.to_datetime(comparisons["trip_date"], errors="coerce").dt.normalize()
    comparisons["hour"] = pd.to_numeric(comparisons["hour"], errors="coerce").astype("Int64")
    comparisons = comparisons[comparisons["trip_date"].notna() & comparisons["hour"].between(0, 23)].copy()

    key_cols = ["screenline_id", "trip_date", "hour"]
    common_keys: set[tuple[Any, ...]] | None = None
    for key in method_keys:
        rows = comparisons[comparisons["method"].eq(key)]
        keys = set(map(tuple, rows[key_cols].itertuples(index=False, name=None)))
        common_keys = keys if common_keys is None else common_keys & keys
    if not common_keys:
        _placeholder(path, "Aggregate hourly TMAS validation profile", "No common TMAS screenline-hour cells were available across methods.")
        return False
    common = pd.DataFrame(list(common_keys), columns=key_cols)
    common["hour"] = common["hour"].astype("Int64")
    comparisons = comparisons.merge(common, on=key_cols, how="inner")
    if comparisons.empty:
        _placeholder(path, "Aggregate hourly TMAS validation profile", "Common TMAS cells did not join to hourly comparisons.")
        return False

    observed_cells = comparisons[comparisons["method"].eq("contrastive_vae")][key_cols + ["observed_count"]].drop_duplicates(key_cols)
    observed_daily = observed_cells.groupby(["trip_date", "hour"], as_index=False)["observed_count"].sum()
    observed_profile = observed_daily.groupby("hour", as_index=False)["observed_count"].mean().sort_values("hour")
    if observed_profile.empty:
        _placeholder(path, "Aggregate hourly TMAS validation profile", "No observed aggregate hourly TMAS profile was available.")
        return False

    method_profiles = []
    for key in method_keys:
        rows = comparisons[comparisons["method"].eq(key)]
        daily = rows.groupby(["trip_date", "hour"], as_index=False)["synthetic_count"].sum()
        profile = daily.groupby("hour", as_index=False)["synthetic_count"].mean()
        profile["method"] = key
        method_profiles.append(profile)
    method_profiles_df = pd.concat(method_profiles, ignore_index=True) if method_profiles else pd.DataFrame()
    if method_profiles_df.empty:
        _placeholder(path, "Aggregate hourly TMAS validation profile", "No synthetic aggregate hourly profiles were available.")
        return False

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.4), sharex=True, sharey=True, constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.105, wspace=0.18, hspace=0.245)
    ymax = max(
        float(observed_profile["observed_count"].max()),
        float(method_profiles_df["synthetic_count"].max()),
        1.0,
    )
    from matplotlib.ticker import FuncFormatter

    for ax, (key, label, color, linewidth, alpha) in zip(axes.ravel(), method_specs):
        ax.set_facecolor("#fbfaf6")
        rows = method_profiles_df[method_profiles_df["method"].eq(key)].sort_values("hour")
        ax.plot(
            observed_profile["hour"],
            observed_profile["observed_count"],
            label="Observed",
            color="#222222",
            linewidth=3.6,
            alpha=0.96,
            zorder=7,
        )
        if not rows.empty:
            ax.plot(
                rows["hour"],
                rows["synthetic_count"],
                label=label,
                color=color,
                linewidth=3.6 if key == "contrastive_vae" else 3.1,
                alpha=0.96 if key == "contrastive_vae" else alpha,
                zorder=8 if key == "contrastive_vae" else 6,
            )
        ax.set_title(label, loc="left", fontsize=19, fontweight="bold", pad=7, color="#222222")
        ax.set_xlim(0, 23)
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.tick_params(axis="both", labelsize=14.5, length=0, pad=6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_compact_count(float(value))))
        ax.grid(axis="y", color="#b8b2a8", alpha=0.32, linewidth=0.9)
        ax.grid(axis="x", color="#d9d2c6", alpha=0.16, linewidth=0.75)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#8d877d")
            ax.spines[spine].set_linewidth(0.9)
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=13.5,
            handlelength=1.45,
            handletextpad=0.45,
            borderaxespad=0.0,
        )

    fig.supxlabel("Hour of day", fontsize=22, y=0.032)
    fig.supylabel("Hourly traffic count", fontsize=22, x=0.022)
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _make_hourly_rmse_reduction_chart(path: Path, run_dir: Path) -> bool:
    summary_path = run_dir / "tables" / "aadt_hourly_validation_summary.csv"
    if not summary_path.exists():
        _placeholder(path, "Hourly RMSE reduction", "Hourly TMAS validation summary was not available.")
        return False

    summary = pd.read_csv(summary_path)
    needed = {"method", "rmse"}
    if summary.empty or not needed.issubset(summary.columns):
        _placeholder(path, "Hourly RMSE reduction", "Hourly TMAS validation summary did not include method RMSE values.")
        return False
    if "validation_tier" in summary.columns:
        summary = summary[summary["validation_tier"].astype(str).eq("hourly_tmas_scaled")].copy()
    summary["rmse"] = pd.to_numeric(summary["rmse"], errors="coerce")
    summary = summary.dropna(subset=["rmse"])

    method_specs = [
        ("weighted_bootstrap", "Survey-Sampled Baseline", "#76736d"),
        ("bayesian_network", "Bayesian Network", "#4c9f70"),
        ("noncontrastive_vae", "Non-Contrastive VAE", "#756bb1"),
        ("contrastive_vae", "Contrastive VAE", "#2176ae"),
    ]
    rmse_by_method = {str(row.method): float(row.rmse) for row in summary[["method", "rmse"]].itertuples(index=False)}
    baseline_rmse = rmse_by_method.get("weighted_bootstrap")
    if not baseline_rmse or baseline_rmse <= 0:
        _placeholder(path, "Hourly RMSE reduction", "Weighted baseline RMSE was unavailable.")
        return False

    rows = []
    for key, label, color in method_specs:
        if key not in rmse_by_method:
            continue
        rmse = rmse_by_method[key]
        reduction = 100.0 * (baseline_rmse - rmse) / baseline_rmse
        rows.append({"method": key, "label": label, "color": color, "rmse": rmse, "reduction": reduction})
    if len(rows) < 2:
        _placeholder(path, "Hourly RMSE reduction", "Not enough method RMSE values were available.")
        return False

    data = pd.DataFrame(rows)
    plot_data = data[data["method"].ne("weighted_bootstrap")].copy()
    plot_data = plot_data.sort_values("reduction", ascending=True)

    fig, ax = plt.subplots(figsize=(15.0, 8.4), constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")
    fig.subplots_adjust(left=0.28, right=0.965, top=0.82, bottom=0.16)

    y = np.arange(len(plot_data), dtype=float)
    bars = ax.barh(plot_data["label"], plot_data["reduction"], color=plot_data["color"], height=0.56, alpha=0.96)
    ax.axvline(0, color="#77736b", linewidth=1.2, alpha=0.7)
    ax.set_xlim(0, max(55.0, float(plot_data["reduction"].max()) + 5.0))
    ax.set_xlabel("Hourly TMAS RMSE reduction vs. survey-sampled baseline", fontsize=22, labelpad=12)
    ax.tick_params(axis="x", labelsize=17, length=0, pad=8)
    ax.tick_params(axis="y", labelsize=21, length=0, pad=10)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="x", color="#b8b2a8", alpha=0.28, linewidth=1.0)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#8d877d")
    ax.spines["bottom"].set_linewidth(1.0)

    baseline_text = f"Baseline RMSE: {_format_compact_count(baseline_rmse)}"
    ax.text(
        0.0,
        1.08,
        baseline_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        color="#55524b",
    )
    for bar, row in zip(bars, plot_data.itertuples(index=False)):
        value = float(row.reduction)
        rmse_text = _format_compact_count(float(row.rmse))
        ax.text(
            value + 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f}%  RMSE {rmse_text}",
            ha="left",
            va="center",
            fontsize=20 if row.method == "contrastive_vae" else 17.5,
            fontweight="bold" if row.method == "contrastive_vae" else "normal",
            color="#222222",
        )
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _make_hourly_annual_validation_side_by_side(path: Path, run_dir: Path) -> bool:
    hourly_summary_path = run_dir / "tables" / "aadt_hourly_validation_summary.csv"
    annual_path = run_dir / "metrics" / "aadt_annual_screenline_comparisons.parquet"
    if not (hourly_summary_path.exists() and annual_path.exists()):
        _placeholder(path, "Traffic-count validation summary", "Hourly or annual traffic-count validation results were unavailable.")
        return False

    method_specs = [
        ("weighted_bootstrap", "Survey-Sampled Baseline", "#76736d"),
        ("bayesian_network", "Bayesian Network", "#4c9f70"),
        ("noncontrastive_vae", "Non-Contrastive VAE", "#756bb1"),
        ("contrastive_vae", "Contrastive VAE", "#2176ae"),
    ]
    method_labels = {key: label for key, label, _ in method_specs}
    axis_method_labels = {
        "weighted_bootstrap": "Survey-Sampled\nBaseline",
        "bayesian_network": "Bayesian\nNetwork",
        "noncontrastive_vae": "Non-Contrastive\nVAE",
        "contrastive_vae": "Contrastive\nVAE",
    }
    method_colors = {key: color for key, _, color in method_specs}
    method_keys = [key for key, _, _ in method_specs]

    hourly = pd.read_csv(hourly_summary_path)
    if "validation_tier" in hourly.columns:
        hourly = hourly[hourly["validation_tier"].astype(str).eq("hourly_tmas_scaled")].copy()
    hourly["rmse"] = pd.to_numeric(hourly["rmse"], errors="coerce")
    rmse_by_method = {str(row.method): float(row.rmse) for row in hourly[["method", "rmse"]].dropna().itertuples(index=False)}
    if "weighted_bootstrap" not in rmse_by_method:
        _placeholder(path, "Traffic-count validation summary", "Weighted baseline hourly RMSE was unavailable.")
        return False

    hourly_rows = []
    for key in method_keys:
        if key not in rmse_by_method:
            continue
        hourly_rows.append(
            {
                "method": key,
                "label": axis_method_labels[key],
                "color": method_colors[key],
                "rmse": rmse_by_method[key],
            }
        )
    hourly_plot = pd.DataFrame(hourly_rows).sort_values("rmse", ascending=False)

    annual = pd.read_parquet(annual_path)
    annual = annual[annual["method"].astype(str).isin(method_keys)].copy()
    if annual.empty:
        _placeholder(path, "Traffic-count validation summary", "Annual screenline comparisons were unavailable.")
        return False
    annual["abs_error"] = (annual["synthetic_count"].astype(float) - annual["observed_count"].astype(float)).abs()
    annual_wide = annual.pivot(index="screenline_id", columns="method", values="abs_error").dropna(subset=method_keys)
    if annual_wide.empty:
        _placeholder(path, "Traffic-count validation summary", "Annual screenline method comparisons did not overlap.")
        return False
    annual_winners = annual_wide.idxmin(axis=1)
    annual_counts = annual_winners.value_counts().reindex(method_keys).fillna(0).astype(int)
    annual_plot = pd.DataFrame(
        {
            "method": method_keys,
            "label": [axis_method_labels[key] for key in method_keys],
            "color": [method_colors[key] for key in method_keys],
            "wins": [int(annual_counts.loc[key]) for key in method_keys],
        }
    ).sort_values("wins", ascending=True)

    cvae_wins = int(annual_counts.loc["contrastive_vae"])
    n_wins = int(annual_winners.size)
    try:
        from scipy import stats

        annual_p = float(stats.binomtest(cvae_wins, n_wins, p=0.25, alternative="greater").pvalue)
    except Exception:
        annual_p = float("nan")

    text_scale = 1.5
    fig, (hourly_ax, annual_ax) = plt.subplots(1, 2, figsize=(15.0, 8.4), constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    fig.subplots_adjust(left=0.18, right=0.925, top=0.83, bottom=0.19, wspace=0.98)
    for ax in [hourly_ax, annual_ax]:
        ax.set_facecolor("#fbfaf6")

    hourly_bars = hourly_ax.barh(
        hourly_plot["label"],
        hourly_plot["rmse"],
        color=hourly_plot["color"],
        height=0.56,
        alpha=0.96,
    )
    hourly_ax.set_title("Hourly TMAS\nRMSE ↓", loc="left", fontsize=21 * text_scale, fontweight="bold", pad=12, linespacing=1.0)
    hourly_ax.set_xlabel("RMSE", fontsize=15.5 * text_scale, labelpad=13)
    hourly_ax.set_xlim(0, max(15_000.0, float(hourly_plot["rmse"].max()) * 1.16))
    hourly_ax.xaxis.set_major_formatter(lambda value, _: f"{float(value):,.0f}")
    hourly_ax.tick_params(axis="x", labelsize=13.5 * text_scale, length=0, pad=8)
    for idx, tick in enumerate(hourly_ax.get_xticklabels()):
        if idx % 2 == 1:
            tick.set_visible(False)
    hourly_ax.tick_params(axis="y", labelsize=13.8 * text_scale, length=0, pad=10)
    hourly_ax.grid(axis="x", color="#b8b2a8", alpha=0.28, linewidth=1.0)
    for bar, row in zip(hourly_bars, hourly_plot.itertuples(index=False)):
        hourly_ax.text(
            float(row.rmse) + max(float(hourly_plot["rmse"].max()) * 0.025, 1.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{float(row.rmse):,.0f}",
            ha="left",
            va="center",
            fontsize=17 * text_scale if row.method == "contrastive_vae" else 14.5 * text_scale,
            fontweight="bold" if row.method == "contrastive_vae" else "normal",
            color="#222222",
        )

    annual_bars = annual_ax.barh(
        annual_plot["label"],
        annual_plot["wins"],
        color=annual_plot["color"],
        height=0.56,
        alpha=0.96,
    )
    annual_ax.set_title("Annual Screenline\nWins ↑", loc="left", fontsize=21 * text_scale, fontweight="bold", pad=12, linespacing=1.0)
    annual_ax.set_xlabel("screenline wins", fontsize=15.5 * text_scale, labelpad=13)
    annual_ax.set_xlim(0, max(1_150.0, float(annual_plot["wins"].max()) + 260.0))
    annual_ax.tick_params(axis="x", labelsize=13.5 * text_scale, length=0, pad=8)
    annual_ax.tick_params(axis="y", labelsize=13.8 * text_scale, length=0, pad=10)
    annual_ax.grid(axis="x", color="#b8b2a8", alpha=0.28, linewidth=1.0)
    for bar, row in zip(annual_bars, annual_plot.itertuples(index=False)):
        annual_ax.text(
            int(row.wins) + 14,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(row.wins):,}",
            ha="left",
            va="center",
            fontsize=17 * text_scale if row.method == "contrastive_vae" else 14.5 * text_scale,
            fontweight="bold" if row.method == "contrastive_vae" else "normal",
            color="#222222",
        )

    for ax in [hourly_ax, annual_ax]:
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#8d877d")
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(axis="y", visible=False)

    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# Final focused poster version: one center-origin tract and its surrounding
# 24-tract zone. This overrides the broader all-pairs matrix draft above.
def _select_focus_cluster(
    run_dir: Path,
    baseline_od: pd.DataFrame,
    cvae_od: pd.DataFrame,
    min_tracts: int = 60,
    max_tracts: int = 60,
) -> tuple[set[str], dict[str, float | int | str]] | None:
    try:
        import geopandas as gpd
    except Exception:
        return None

    adjacency_path = run_dir / "geo" / "tract_adjacency.parquet"
    tracts_path = run_dir / "geo" / "tracts.parquet"
    if not (adjacency_path.exists() and tracts_path.exists()):
        return None

    target_size = max_tracts
    tracts = gpd.read_parquet(tracts_path)
    tracts = tracts.assign(GEOID=tracts["GEOID"].astype(str).str.zfill(11))
    coords = {
        str(row.GEOID): (float(row.rep_x), float(row.rep_y))
        for row in tracts[["GEOID", "rep_x", "rep_y"]].itertuples(index=False)
    }
    adjacency = pd.read_parquet(adjacency_path, columns=["tract_a", "tract_b"])
    neighbors: dict[str, set[str]] = {}
    for tract_a, tract_b in adjacency.itertuples(index=False):
        a = str(tract_a).zfill(11)
        b = str(tract_b).zfill(11)
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    baseline_pairs = _pairs_from_od_counts(baseline_od)
    cvae_pairs = _pairs_from_od_counts(cvae_od)

    best: tuple[float, set[str], dict[str, float | int | str]] | None = None
    for center in coords:
        if not center.startswith("24"):
            continue
        cluster = _connected_cluster_from_center(center, neighbors, coords, target_size)
        if cluster is None or not (min_tracts <= len(cluster) <= max_tracts):
            continue
        possible = len(cluster) - 1
        baseline_dests = {dest for origin, dest in baseline_pairs if origin == center and dest in cluster and dest != center}
        cvae_dests = {dest for origin, dest in cvae_pairs if origin == center and dest in cluster and dest != center}
        if len(baseline_dests) < 2 or len(cvae_dests) < 6 or possible <= 0:
            continue
        cvae_only = len(cvae_dests - baseline_dests)
        bootstrap_only = len(baseline_dests - cvae_dests)
        fill_delta = (len(cvae_dests) - len(baseline_dests)) / possible
        ratio = len(cvae_dests) / max(len(baseline_dests), 1)
        score = cvae_only + 20.0 * fill_delta + 3.0 * ratio - 2.0 * bootstrap_only
        metadata: dict[str, float | int | str] = {
            "center": center,
            "tracts": len(cluster),
            "possible_destinations": possible,
            "baseline_destinations": len(baseline_dests),
            "cvae_destinations": len(cvae_dests),
            "cvae_only_destinations": cvae_only,
            "bootstrap_only_destinations": bootstrap_only,
            "baseline_fill_rate": len(baseline_dests) / possible,
            "cvae_fill_rate": len(cvae_dests) / possible,
            "ratio": ratio,
        }
        if best is None or score > best[0]:
            best = (score, cluster, metadata)

    if best is None:
        return None
    return best[1], best[2]


def _origin_destination_counts(od_counts: pd.DataFrame, center: str, cluster: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in od_counts.itertuples(index=False):
        origin = str(row.o_tract_fips)
        destination = str(row.d_tract_fips)
        if origin == center and destination in cluster and destination != center:
            counts[destination] = int(row.trip_count)
    return counts


def _destination_order(center: str, destinations: set[str], coords: dict[str, tuple[float, float]]) -> list[str]:
    cx, cy = coords[center]
    return sorted(
        destinations,
        key=lambda tract: (
            np.arctan2(coords[tract][1] - cy, coords[tract][0] - cx),
            (coords[tract][0] - cx) ** 2 + (coords[tract][1] - cy) ** 2,
        ),
    )


def _plot_center_origin_map(
    ax: plt.Axes,
    tracts: Any,
    focus_tracts: Any,
    center: str,
    destination_counts: dict[str, int],
    coords: dict[str, tuple[float, float]],
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    muted_destinations: set[str] | None = None,
) -> None:
    muted_destinations = muted_destinations or set()
    reached = set(destination_counts)
    focus_tracts.plot(ax=ax, facecolor="#fffdf7", edgecolor="#606765", linewidth=1.0, zorder=2)
    if reached:
        focus_tracts[focus_tracts["GEOID"].isin(reached - muted_destinations)].plot(ax=ax, facecolor=color, alpha=0.20, edgecolor="#606765", linewidth=1.0, zorder=3)
        if muted_destinations:
            focus_tracts[focus_tracts["GEOID"].isin(reached & muted_destinations)].plot(ax=ax, facecolor="#adc8d8", alpha=0.28, edgecolor="#606765", linewidth=1.0, zorder=3)
    focus_tracts[focus_tracts["GEOID"] == center].plot(ax=ax, facecolor="#222222", edgecolor="#111111", linewidth=1.2, zorder=5)
    max_weight = max(np.log1p(list(destination_counts.values()))) if destination_counts else 1.0
    segments = []
    colors = []
    widths = []
    for destination, count in destination_counts.items():
        if destination not in coords:
            continue
        muted = destination in muted_destinations
        segments.append([coords[center], coords[destination]])
        colors.append("#9fb8c8" if muted else color)
        widths.append(0.8 + 2.0 * (np.log1p(count) / max_weight if max_weight > 0 else 1.0))
    if segments:
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, alpha=0.72, capstyle="round", zorder=6))
    ax.scatter([coords[center][0]], [coords[center][1]], s=62, color="#111111", zorder=7, linewidth=0)
    dest_points = [coords[tract] for tract in reached if tract in coords]
    if dest_points:
        ax.scatter([p[0] for p in dest_points], [p[1] for p in dest_points], s=22, color=color, zorder=7, linewidth=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfaf6")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_destination_strip(
    ax: plt.Axes,
    order: list[str],
    present: set[str],
    center: str,
    title: str,
    color: str,
    muted_destinations: set[str] | None = None,
) -> None:
    muted_destinations = muted_destinations or set()
    values = np.zeros((1, len(order)), dtype=int)
    for idx, tract in enumerate(order):
        if tract == center:
            values[0, idx] = -1
        elif tract in present:
            values[0, idx] = 1 if tract in muted_destinations else 2
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["#222222", "#f3efe6", "#adc8d8", color])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=6)
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(["O" if tract == center else str(i) for i, tract in enumerate(order, start=1)], fontsize=5.5)
    ax.tick_params(length=0, pad=1)
    ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.set_yticks([-0.5, 0.5], minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _make_focused_trip_coverage_map(path: Path, run_dir: Path) -> bool:
    try:
        import geopandas as gpd
    except Exception:
        _placeholder(path, "Focused trip coverage map", "Install the geo extras to generate tract-level trip maps.")
        return False

    tracts_path = run_dir / "geo" / "tracts.parquet"
    baseline_path = run_dir / "samples" / "weighted_bootstrap_synthetic.csv"
    cvae_path = run_dir / "samples" / "contrastive_vae_synthetic.csv"
    if not (tracts_path.exists() and baseline_path.exists() and cvae_path.exists()):
        _placeholder(path, "Focused trip coverage map", "Synthetic samples or tract geometry were not available in this run.")
        return False

    tracts = gpd.read_parquet(tracts_path)
    tracts = tracts.assign(GEOID=tracts["GEOID"].astype(str).str.zfill(11))
    baseline_od = _load_od_counts(baseline_path, sample_size=0, seed=42)
    cvae_od = _load_od_counts(cvae_path, sample_size=0, seed=42)
    selected = _select_focus_cluster(run_dir, baseline_od, cvae_od, min_tracts=60, max_tracts=60)
    if selected is None:
        _placeholder(path, "Focused trip coverage map", "No center-origin 60-tract zone had enough local synthetic destinations.")
        return False
    cluster, metadata = selected
    center = str(metadata["center"])

    focus_tracts = tracts[tracts["GEOID"].isin(cluster)].copy()
    if focus_tracts.empty:
        _placeholder(path, "Focused trip coverage map", "Selected tract cluster was not found in the tract geometry.")
        return False

    coords = {
        str(row.GEOID): (float(row.rep_x), float(row.rep_y))
        for row in focus_tracts[["GEOID", "rep_x", "rep_y"]].itertuples(index=False)
    }
    destination_order = [center] + _destination_order(center, set(cluster) - {center}, coords)
    baseline_counts = _origin_destination_counts(baseline_od, center, cluster)
    cvae_counts = _origin_destination_counts(cvae_od, center, cluster)
    baseline_destinations = set(baseline_counts)
    cvae_destinations = set(cvae_counts)
    shared_destinations = baseline_destinations & cvae_destinations
    cvae_only = cvae_destinations - baseline_destinations

    bounds = focus_tracts.total_bounds
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    pad = span * 0.16
    xlim = (float(bounds[0] - pad), float(bounds[2] + pad))
    ylim = (float(bounds[1] - pad), float(bounds[3] + pad))

    fig = plt.figure(figsize=(17, 8.8), constrained_layout=False)
    fig.patch.set_facecolor("#fbfaf6")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.80, 0.20], left=0.050, right=0.985, top=0.80, bottom=0.14, wspace=0.08, hspace=0.24)
    map_base = fig.add_subplot(grid[0, 0])
    map_cvae = fig.add_subplot(grid[0, 1])
    strip_base = fig.add_subplot(grid[1, 0])
    strip_cvae = fig.add_subplot(grid[1, 1])

    fig.text(0.055, 0.965, "Single origin tract: CVAE fills nearby destinations", ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(
        0.055,
        0.920,
        f"Auto-selected 60-tract Maryland zone centered on origin tract {center}; black tract is the only origin, colored tracts are destinations reached from it.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.text(0.055, 0.875, f"Weighted bootstrap: {metadata['baseline_destinations']} / {metadata['possible_destinations']} destination tracts", ha="left", va="top", fontsize=12, color="#9a4c20", fontweight="bold")
    fig.text(0.535, 0.875, f"Contrastive VAE: {metadata['cvae_destinations']} / {metadata['possible_destinations']} destinations, {metadata['cvae_only_destinations']} CVAE-only", ha="left", va="top", fontsize=12, color="#155f92", fontweight="bold")

    _plot_center_origin_map(map_base, tracts, focus_tracts, center, baseline_counts, coords, "#d06c2f", xlim, ylim)
    map_base.set_title("Random-resampling baseline", loc="left", fontsize=16, fontweight="bold", pad=7)
    _plot_center_origin_map(map_cvae, tracts, focus_tracts, center, cvae_counts, coords, "#2176ae", xlim, ylim, muted_destinations=shared_destinations)
    map_cvae.set_title("Contrastive VAE", loc="left", fontsize=16, fontweight="bold", pad=7)

    _plot_destination_strip(strip_base, destination_order, baseline_destinations, center, "Destination tracts reached from center", "#d06c2f")
    _plot_destination_strip(strip_cvae, destination_order, cvae_destinations, center, "Destination tracts reached from center", "#2176ae", muted_destinations=shared_destinations)

    fig.text(0.055, 0.060, "Destination strip: O is the origin tract; blank cells are unreached destinations; pale blue cells on the CVAE side are shared with bootstrap; dark blue cells are CVAE-only.", ha="left", va="bottom", fontsize=9.5, color="#666666")
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


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
    _make_experiment_pipeline(p)
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
        plt.xlabel("log1p observed screenline counts")
        plt.ylabel("log1p synthetic virtual crossings")
        plt.title("Traffic screenline comparison")
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

    p = poster / "12_trip_coverage_resampling_vs_cvae.png"
    _make_trip_coverage_map(p, run_dir)
    created.append(p)

    p = poster / "13_single_origin_trip_coverage_60tracts.png"
    _make_focused_trip_coverage_map(p, run_dir)
    created.append(p)

    p = poster / "14_aadt_screenline_station_match_map.png"
    _make_aadt_station_match_map(p, run_dir)
    created.append(p)

    p = poster / "15_od_pair_screenline_validation_map.png"
    _make_od_pair_screenline_validation_map(p, run_dir)
    created.append(p)

    p = poster / "16_od_pair_screenline_validation_map_cvae_best.png"
    _make_od_pair_screenline_validation_map(p, run_dir, method="contrastive_vae", selection_strategy="contrastive_best")
    created.append(p)

    p = poster / "17_hourly_tmas_profile_cvae_best.png"
    _make_hourly_tmas_validation_profile(p, run_dir)
    created.append(p)

    p = poster / "18_hourly_tmas_profile_all_stations.png"
    _make_hourly_tmas_aggregate_profile(p, run_dir)
    created.append(p)

    p = poster / "19_hourly_tmas_rmse_reduction.png"
    _make_hourly_rmse_reduction_chart(p, run_dir)
    created.append(p)

    p = poster / "20_hourly_annual_traffic_validation_summary.png"
    _make_hourly_annual_validation_side_by_side(p, run_dir)
    created.append(p)

    captions = poster / "captions.md"
    captions.write_text(
        "\n".join(
            [
                "# Poster Figure Captions",
                "",
                "Figures summarize the synthesis pipeline, validation metrics, privacy diagnostics, and optional two-prong screenline validation.",
                "Traffic-count figures are placeholders in quick runs because external geospatial validation is disabled by configuration.",
                "",
                "Figure 12 compares equal-sized samples of weighted-bootstrap and contrastive-VAE trips as origin-destination tract lines. It is a tract-centroid visualization for coverage, not route assignment.",
                "Figure 13 zooms to one origin tract inside an auto-selected 60-tract Maryland zone and shows which nearby destination tracts are reached by each method.",
                "Figure 14 maps MDOT stations assigned to annual AADT screenlines. Point color is the contrastive-VAE synthetic-to-observed screenline ratio, and point size is station AAWDT.",
                "Figure 15 zooms to one OD-pair example and traces the station-matched screenlines used to compare synthetic virtual crossings with observed AAWDT counts.",
                "Figure 16 repeats the OD-pair screenline view for an example where the contrastive VAE has the lowest path-level annual-average count error among all compared synthetic methods.",
                "Figure 17 shows a single-TMAS-station hourly validation profile where the contrastive VAE has the lowest 24-hour count-profile error among compared synthetic methods.",
                "Figure 18 repeats the hourly validation profile as an aggregate over all measured TMAS screenlines using common comparable hourly cells across methods.",
                "Figure 19 summarizes hourly TMAS traffic-count RMSE reductions relative to the survey-sampled baseline for all non-baseline synthesis methods.",
                "Figure 20 pairs hourly TMAS RMSE reductions with annual screenline first-place counts, where contrastive VAE wins the most annual screenlines by absolute count error.",
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

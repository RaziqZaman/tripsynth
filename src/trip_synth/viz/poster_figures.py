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
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

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
        fontsize=10.5,
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
    _save_flow(
        p,
        "Experiment pipeline",
        [
            "Survey trips + expansion weights",
            "Four synthesis methods",
            "Synthetic tables without weight",
            "Marginals + privacy",
            "OD-to-screenline proxy",
            "Two-prong count comparison",
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

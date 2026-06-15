#!/usr/bin/env python3
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "runs" / "paper_wctr_1m_vae_sweep"
OUT_DIR = ROOT / "latex-figures"

METHODS = [
    "weighted_bootstrap",
    "bayesian_network",
    "noncontrastive_vae",
    "contrastive_vae",
]

METHOD_LABELS = {
    "weighted_bootstrap": "Weighted bootstrap",
    "bayesian_network": "Bayesian network",
    "noncontrastive_vae": "Noncontrastive VAE",
    "contrastive_vae": "Contrastive VAE",
}

SHORT_LABELS = {
    "weighted_bootstrap": "Bootstrap",
    "bayesian_network": "Bayesian\nnetwork",
    "noncontrastive_vae": "Noncontrastive\nVAE",
    "contrastive_vae": "Contrastive\nVAE",
}

COLORS = {
    "survey": "#6B7280",
    "weighted_bootstrap": "#D55E00",
    "bayesian_network": "#009E73",
    "noncontrastive_vae": "#CC79A7",
    "contrastive_vae": "#0072B2",
    "ink": "#1F2933",
    "muted": "#68717A",
    "grid": "#D9DEE3",
    "light": "#F6F8FA",
}

INTERNAL = pd.DataFrame(
    [
        ("weighted_bootstrap", 0.0033, 0.0439, 0.9801, 0.0050, 0.9950, 1.0000, 0.0000),
        ("bayesian_network", 0.0034, 0.0437, 0.9801, 0.0901, 0.9099, 0.0000, 0.3330),
        ("noncontrastive_vae", 0.0621, 0.1261, 0.9117, 0.2556, 0.7444, 0.0000, 0.6856),
        ("contrastive_vae", 0.0652, 0.1103, 0.9163, 0.2265, 0.7735, 0.0000, 0.6852),
    ],
    columns=[
        "method",
        "cat_tv",
        "num_ks",
        "marginal_similarity",
        "cross_tv",
        "cross_similarity",
        "copy_rate",
        "novel_od_share",
    ],
)

OD_COUNTS = pd.DataFrame(
    [
        ("Survey input", "survey", 58_925),
        ("Weighted bootstrap", "weighted_bootstrap", 58_009),
        ("Bayesian network", "bayesian_network", 180_910),
        ("Noncontrastive VAE", "noncontrastive_vae", 398_039),
        ("Contrastive VAE", "contrastive_vae", 395_542),
    ],
    columns=["label", "key", "unique_pairs"],
)

ANNUAL = pd.DataFrame(
    [
        ("weighted_bootstrap", 0.2150, 56_659, 29_855, -19_707, 208.3, 648),
        ("bayesian_network", 0.2168, 55_772, 29_918, -17_060, 207.7, 802),
        ("noncontrastive_vae", 0.1694, 56_145, 30_789, -16_083, 214.1, 817),
        ("contrastive_vae", 0.1733, 57_843, 30_572, -22_120, 215.9, 840),
    ],
    columns=["method", "pearson_log1p", "rmse", "mae", "bias", "geh_mean", "lowest_error_screenlines"],
)

HOURLY = pd.DataFrame(
    [
        ("weighted_bootstrap", 537_936, 0.0536, 14_143, 5_756, -4_266, 126.0, 0.0),
        ("bayesian_network", 624_360, 0.1549, 7_169, 4_985, -4_064, 117.9, 49.3),
        ("noncontrastive_vae", 578_232, 0.1750, 6_960, 4_923, -4_057, 116.6, 50.8),
        ("contrastive_vae", 601_440, 0.1422, 6_944, 4_945, -4_288, 118.2, 50.9),
    ],
    columns=["method", "cells", "pearson_log1p", "rmse", "mae", "bias", "geh_mean", "rmse_reduction"],
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.8,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.6,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.45,
            "grid.alpha": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_output_dir() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for item in OUT_DIR.iterdir():
        if item.is_file():
            item.unlink()


def save_figure(fig: plt.Figure, filename: str, png: bool = False) -> None:
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", pad_inches=0.035, metadata={"Creator": "trip synthesis figure generator"})
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def wrap(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def add_panel_label(ax: plt.Axes, label: str, x: float = 0.0, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", color=COLORS["ink"])


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    face: str = "white",
    edge: str = "#344054",
    fontsize: float = 7.2,
    weight: str = "normal",
    text_color: str = "#1F2933",
    radius: float = 0.025,
    lw: float = 0.8,
    zorder: int = 2,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        linespacing=1.18,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#344054",
    lw: float = 1.0,
    rad: float = 0.0,
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.5,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
            zorder=zorder,
        )
    )


def clean_fips(value: object) -> str:
    if pd.isna(value):
        return "-1"
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(11) if digits else "-1"


def format_k(value: float) -> str:
    return f"{value / 1000:.0f}k"


def strip_axes(ax: plt.Axes) -> None:
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)


def load_geo():
    import geopandas as gpd

    tracts = gpd.read_parquet(RUN_DIR / "geo" / "tracts.parquet")
    tracts = tracts.assign(GEOID=tracts["GEOID"].astype(str).str.zfill(11))
    centroids = {
        row.GEOID: (float(row.rep_x), float(row.rep_y))
        for row in tracts[["GEOID", "rep_x", "rep_y"]].itertuples(index=False)
    }
    return tracts, centroids


def line_segments_from_sample(
    sample_path: Path,
    centroids: dict[str, tuple[float, float]],
    sample_rows: int,
    max_lines: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    df = pd.read_csv(sample_path, usecols=["o_tract_fips", "d_tract_fips"], dtype="string")
    available_rows = len(df)
    if len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=seed)
    df["o_tract_fips"] = df["o_tract_fips"].map(clean_fips)
    df["d_tract_fips"] = df["d_tract_fips"].map(clean_fips)
    df = df[(df["o_tract_fips"] != "-1") & (df["d_tract_fips"] != "-1")]
    counts = (
        df.groupby(["o_tract_fips", "d_tract_fips"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )
    counts = counts[
        counts["o_tract_fips"].isin(centroids)
        & counts["d_tract_fips"].isin(centroids)
        & (counts["o_tract_fips"] != counts["d_tract_fips"])
    ].copy()
    unique_pairs = len(counts)
    if len(counts) > max_lines:
        rng = np.random.default_rng(seed)
        weights = np.sqrt(counts["trip_count"].to_numpy(float))
        probabilities = weights / weights.sum()
        chosen = rng.choice(counts.index.to_numpy(), size=max_lines, replace=False, p=probabilities)
        counts = counts.loc[chosen]
    segments = np.empty((len(counts), 2, 2), dtype=float)
    for idx, row in enumerate(counts.itertuples(index=False)):
        segments[idx, 0] = centroids[str(row.o_tract_fips)]
        segments[idx, 1] = centroids[str(row.d_tract_fips)]
    return segments, counts["trip_count"].to_numpy(float), unique_pairs, min(sample_rows, available_rows)


def figure_workflow_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.965, "Study workflow and validation logic", fontsize=9.2, fontweight="bold", color=COLORS["ink"], ha="left", va="top")
    ax.text(
        0.015,
        0.915,
        "Four synthesis methods are evaluated against internal fidelity, support expansion, and independent traffic-count proxies.",
        fontsize=7.2,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )

    y = 0.55
    h = 0.23
    xs = [0.03, 0.205, 0.405, 0.645, 0.82]
    widths = [0.125, 0.145, 0.175, 0.13, 0.14]
    labels = [
        "Survey trip records\n110,926 transformed trips",
        "Preprocessing and schema\n44 synthesized attributes",
        "Synthetic generators\n1,000,000 rows per method",
        "Validation suite\ninternal and external",
        "Interpretation\nfit, novelty, plausibility",
    ]
    faces = ["#F4F6F8", "#F4F6F8", "#EFF7F5", "#F5F3FA", "#F7F7F2"]
    for x0, w0, label, face in zip(xs, widths, labels, faces):
        box(ax, (x0, y), (w0, h), label, face=face, edge="#4B5563", fontsize=6.9, weight="bold" if x0 in [0.405, 0.645] else "normal")

    for left_x, left_w, right_x in zip(xs[:-1], widths[:-1], xs[1:]):
        arrow(ax, (left_x + left_w + 0.012, y + h / 2), (right_x - 0.014, y + h / 2), color="#4B5563")

    method_y = [0.34, 0.255, 0.17, 0.085]
    for method, my in zip(METHODS, method_y):
        box(
            ax,
            (0.405, my),
            (0.175, 0.058),
            METHOD_LABELS[method],
            face="white",
            edge=COLORS[method],
            fontsize=6.5,
            weight="bold",
            radius=0.016,
            lw=0.9,
        )
    arrow(ax, (0.49, y - 0.015), (0.49, 0.408), color="#4B5563", lw=0.8)
    ax.text(0.49, 0.39, "common sample size", ha="center", va="center", fontsize=6.2, color=COLORS["muted"])

    validation_boxes = [
        ("Internal fit\nmarginals, pairs, copying", 0.635, 0.27, COLORS["bayesian_network"]),
        ("External screenlines\ntract-boundary proxy", 0.635, 0.145, COLORS["contrastive_vae"]),
    ]
    for label, x0, y0, edge in validation_boxes:
        box(ax, (x0, y0), (0.165, 0.08), label, face="white", edge=edge, fontsize=6.5, radius=0.017, lw=0.9)
        arrow(ax, (0.58, y0 + 0.04), (x0 - 0.008, y0 + 0.04), color=edge, lw=0.8)
        arrow(ax, (x0 + 0.165, y0 + 0.04), (0.82, y0 + 0.04), color=edge, lw=0.8)

    summary = [
        ("Bootstrap", "internal upper bound, exact copying"),
        ("Bayesian network", "interpretable dependence baseline"),
        ("Contrastive VAE", "expanded OD support with stronger dependence fit"),
        ("Learned generators", "large hourly RMSE reduction"),
    ]
    for idx, (head, tail) in enumerate(summary):
        yy = 0.325 - idx * 0.072
        ax.scatter([0.835], [yy], s=18, color=["#D55E00", "#009E73", "#0072B2", "#4B5563"][idx], zorder=4)
        ax.text(0.852, yy, f"{head}: {tail}", ha="left", va="center", fontsize=6.3, color=COLORS["ink"])

    ax.text(0.03, 0.05, "Population expansion target: 19.29 million weighted trips", fontsize=6.3, color=COLORS["muted"], ha="left")
    save_figure(fig, "fig_workflow_overview.pdf")


def figure_cvae_architecture() -> None:
    fig = plt.figure(figsize=(7.25, 4.45))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.16)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    add_panel_label(ax1, "(a) Mixed-tabular VAE", 0.0, 0.97)
    add_panel_label(ax2, "(b) Contrastive view construction", 0.0, 0.97)

    box(ax1, (0.03, 0.68), (0.22, 0.13), "Categorical fields\nembeddings", face="#EEF6FB", edge=COLORS["contrastive_vae"], fontsize=6.6)
    box(ax1, (0.03, 0.47), (0.22, 0.13), "Continuous fields\nstandardization", face="#F8F4FB", edge=COLORS["noncontrastive_vae"], fontsize=6.6)
    box(ax1, (0.31, 0.58), (0.15, 0.13), "Concatenated\ninput", face="#F7F8FA", edge="#4B5563", fontsize=6.7)
    box(ax1, (0.52, 0.58), (0.15, 0.13), "Encoder\nhidden layers", face="#F7F8FA", edge="#4B5563", fontsize=6.7)
    box(ax1, (0.72, 0.58), (0.17, 0.13), "Latent\nrepresentation", face="#FFF7ED", edge="#B45309", fontsize=6.7)
    box(ax1, (0.52, 0.30), (0.15, 0.13), "Decoder\nhidden layers", face="#F7F8FA", edge="#4B5563", fontsize=6.7)
    box(ax1, (0.74, 0.22), (0.19, 0.11), "Categorical heads\nclass probabilities", face="#EEF6FB", edge=COLORS["contrastive_vae"], fontsize=6.2)
    box(ax1, (0.74, 0.40), (0.19, 0.11), "Numeric heads\nscaled values", face="#F8F4FB", edge=COLORS["noncontrastive_vae"], fontsize=6.2)

    arrow(ax1, (0.25, 0.745), (0.31, 0.66), color=COLORS["contrastive_vae"])
    arrow(ax1, (0.25, 0.535), (0.31, 0.63), color=COLORS["noncontrastive_vae"])
    arrow(ax1, (0.46, 0.645), (0.52, 0.645))
    arrow(ax1, (0.67, 0.645), (0.72, 0.645))
    arrow(ax1, (0.805, 0.58), (0.60, 0.43), rad=-0.18)
    arrow(ax1, (0.67, 0.365), (0.74, 0.455), color=COLORS["noncontrastive_vae"])
    arrow(ax1, (0.67, 0.365), (0.74, 0.275), color=COLORS["contrastive_vae"])
    ax1.text(0.53, 0.18, "Loss combines reconstruction, KL regularization,\nand validity-preserving column heads.", ha="center", va="center", fontsize=6.4, color=COLORS["muted"])

    row_y = [0.75, 0.55, 0.35]
    row_labels = [
        "Original record",
        "Positive view",
        "Negative view",
    ]
    descriptions = [
        "protected fields and non-key attributes",
        "protected fields fixed; non-key fields lightly perturbed",
        "one or more non-key fields corrupted within schema",
    ]
    for y0, label, desc, edge in zip(row_y, row_labels, descriptions, ["#4B5563", COLORS["bayesian_network"], "#B42318"]):
        box(ax2, (0.03, y0 - 0.055), (0.25, 0.11), label, face="white", edge=edge, fontsize=6.8, weight="bold")
        box(ax2, (0.34, y0 - 0.055), (0.42, 0.11), wrap(desc, 31), face=COLORS["light"], edge=edge, fontsize=6.2, lw=0.75)
        arrow(ax2, (0.28, y0), (0.34, y0), color=edge, lw=0.85)

    box(ax2, (0.07, 0.12), (0.31, 0.11), "Protected key fields\norigin, destination, time", face="#EAF4FB", edge=COLORS["contrastive_vae"], fontsize=6.3)
    box(ax2, (0.44, 0.12), (0.31, 0.11), "Perturbed non-key fields\nhousehold and trip attributes", face="#F7F4FA", edge=COLORS["noncontrastive_vae"], fontsize=6.3)
    box(ax2, (0.80, 0.35), (0.17, 0.33), "InfoNCE\nregularization\n\nalign positives\nseparate negatives", face="#FFF7ED", edge="#B45309", fontsize=6.4, weight="bold")
    for y0, edge in [(0.55, COLORS["bayesian_network"]), (0.35, "#B42318")]:
        arrow(ax2, (0.76, y0), (0.80, 0.515), color=edge, lw=0.9)
    ax2.text(0.50, 0.055, "The contrastive term shapes the latent space without changing the generated schema.", ha="center", va="center", fontsize=6.3, color=COLORS["muted"])

    save_figure(fig, "fig_cvae_architecture.pdf")


def figure_screenline_proxy() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel_label(ax, "Tract-boundary screenline proxy", 0.015, 0.96)
    left_poly = np.array([[0.07, 0.18], [0.42, 0.12], [0.52, 0.31], [0.47, 0.54], [0.52, 0.83], [0.12, 0.88], [0.05, 0.62]])
    right_poly = np.array([[0.52, 0.31], [0.42, 0.12], [0.83, 0.14], [0.94, 0.42], [0.90, 0.76], [0.52, 0.83], [0.47, 0.54]])
    ax.add_patch(Polygon(left_poly, closed=True, facecolor="#F0F5F9", edgecolor="#667085", linewidth=1.0))
    ax.add_patch(Polygon(right_poly, closed=True, facecolor="#F8F4FB", edgecolor="#667085", linewidth=1.0))
    boundary = np.array([[0.52, 0.83], [0.47, 0.54], [0.52, 0.31], [0.42, 0.12]])
    ax.plot(boundary[:, 0], boundary[:, 1], color=COLORS["bayesian_network"], linewidth=3.2, solid_capstyle="round", zorder=4)
    ax.plot(boundary[:, 0], boundary[:, 1], color="white", linewidth=1.0, alpha=0.75, solid_capstyle="round", zorder=5)

    origin = (0.23, 0.57)
    dest = (0.73, 0.45)
    crossing = (0.50, 0.51)
    ax.scatter([origin[0], dest[0]], [origin[1], dest[1]], s=60, color=[COLORS["weighted_bootstrap"], COLORS["contrastive_vae"]], edgecolor="white", linewidth=0.8, zorder=8)
    ax.text(origin[0] - 0.025, origin[1] + 0.055, "Origin\nrepresentative point", ha="center", va="bottom", fontsize=6.4, color=COLORS["ink"])
    ax.text(dest[0] + 0.035, dest[1] - 0.055, "Destination\nrepresentative point", ha="center", va="top", fontsize=6.4, color=COLORS["ink"])
    arrow(ax, origin, dest, color="#1F2933", lw=1.45, zorder=7)
    ax.add_patch(Circle(crossing, 0.023, facecolor="white", edgecolor=COLORS["bayesian_network"], linewidth=1.3, zorder=9))
    ax.text(crossing[0], crossing[1], "1", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLORS["bayesian_network"], zorder=10)

    stations = [(0.55, 0.62), (0.57, 0.39), (0.43, 0.41), (0.58, 0.75)]
    for sx, sy in stations:
        ax.scatter([sx], [sy], s=45, marker="s", facecolor="#111827", edgecolor="white", linewidth=0.6, zorder=8)
        nearest = boundary[np.argmin(np.sum((boundary - np.array([sx, sy])) ** 2, axis=1))]
        ax.plot([sx, nearest[0]], [sy, nearest[1]], color="#9AA4B2", linewidth=0.6, linestyle=(0, (2, 2)), zorder=3)
    ax.text(0.625, 0.71, "Nearby traffic-count\nstations assigned to\nshared boundary", ha="left", va="center", fontsize=6.5, color=COLORS["ink"])
    ax.text(0.34, 0.80, "Tract A", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#344054")
    ax.text(0.74, 0.79, "Tract B", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#344054")
    ax.text(0.39, 0.215, "Candidate\nscreenline", ha="right", va="center", fontsize=6.5, color=COLORS["bayesian_network"], fontweight="bold")

    box(ax, (0.66, 0.18), (0.27, 0.13), "Ordered tract-boundary crossings\nare compared with station counts", face="white", edge="#4B5563", fontsize=6.5)
    arrow(ax, (0.57, 0.35), (0.66, 0.25), color="#4B5563", lw=0.9)
    ax.text(
        0.055,
        0.055,
        "Paper screenline set: 2,232 filtered tracts, 5,927 adjacent pairs, 3,107 station-mapped screenlines.",
        fontsize=6.4,
        color=COLORS["muted"],
        ha="left",
    )
    save_figure(fig, "fig_screenline_proxy.pdf")


def figure_training_curves() -> None:
    paths = {
        "noncontrastive_vae": RUN_DIR / "checkpoints" / "noncontrastive_vae" / "noncontrastive_vae_training_loss.csv",
        "contrastive_vae": RUN_DIR / "checkpoints" / "contrastive_vae" / "contrastive_vae_training_loss.csv",
    }
    histories = {method: pd.read_csv(path) for method, path in paths.items()}
    best_epochs = {"noncontrastive_vae": 195, "contrastive_vae": 196}

    fig, ax = plt.subplots(figsize=(6.95, 3.55))
    for method in ["noncontrastive_vae", "contrastive_vae"]:
        df = histories[method].copy()
        df["smooth"] = df["val_loss"].rolling(11, center=True, min_periods=1).mean()
        ax.plot(df["epoch"], df["val_loss"], color=COLORS[method], alpha=0.17, linewidth=0.7)
        ax.plot(df["epoch"], df["smooth"], color=COLORS[method], linewidth=1.55, label=METHOD_LABELS[method])
        epoch = best_epochs[method]
        row = df.loc[df["epoch"].round().astype(int) == epoch].iloc[0]
        ax.scatter([epoch], [row["val_loss"]], s=38, color=COLORS[method], edgecolor="white", linewidth=0.9, zorder=5)
        ax.axvline(epoch, color=COLORS[method], linewidth=0.7, linestyle=(0, (2, 2)), alpha=0.55)
        ax.annotate(
            f"best epoch {epoch}",
            xy=(epoch, row["val_loss"]),
            xytext=(epoch + 58, row["val_loss"] + (1.8 if method == "contrastive_vae" else -2.4)),
            arrowprops=dict(arrowstyle="-", color=COLORS[method], lw=0.7),
            fontsize=6.5,
            color=COLORS[method],
            ha="left",
            va="center",
        )
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(0, 1260)
    ax.set_ylim(24, 68)
    ax.grid(axis="y")
    strip_axes(ax)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, "Validation-loss trajectories", 0.0, 1.03)
    ax.text(0.01, 0.04, "Faint lines show raw epoch losses; bold lines show 11-epoch rolling means.", transform=ax.transAxes, fontsize=6.3, color=COLORS["muted"], ha="left")
    save_figure(fig, "fig_training_curves.pdf")


def figure_internal_validation_dashboard() -> None:
    metrics = [
        ("marginal_similarity", "Marginal similarity", "higher", (0.70, 1.02), "{:.3f}"),
        ("cross_similarity", "Cross-marginal similarity", "higher", (0.70, 1.02), "{:.3f}"),
        ("copy_rate", "Exact copy rate", "lower", (0.0, 1.05), "{:.2f}"),
        ("cat_tv", "Categorical TV", "lower", (0.0, 0.075), "{:.3f}"),
        ("num_ks", "Numeric KS", "lower", (0.0, 0.145), "{:.3f}"),
        ("cross_tv", "Cross-marginal TV", "lower", (0.0, 0.285), "{:.3f}"),
    ]
    fig = plt.figure(figsize=(7.25, 5.7))
    grid = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.42, left=0.065, right=0.985, top=0.90, bottom=0.14)
    fig.text(0.065, 0.97, "Internal validation: fit, dependence, and copying", ha="left", va="top", fontsize=9.2, fontweight="bold", color=COLORS["ink"])
    fig.text(0.065, 0.932, "Bootstrap is the memorization upper bound; contrastive regularization improves VAE dependence and numeric fidelity.", ha="left", va="top", fontsize=7.1, color=COLORS["muted"])

    x = np.arange(len(METHODS))
    for idx, (col, title, direction, ylim, fmt) in enumerate(metrics):
        ax = fig.add_subplot(grid[idx // 3, idx % 3])
        values = INTERNAL.set_index("method").loc[METHODS, col].to_numpy(float)
        bars = ax.bar(x, values, color=[COLORS[m] for m in METHODS], width=0.68, edgecolor="white", linewidth=0.6)
        ax.set_title(title, loc="left", pad=4)
        ax.set_ylim(*ylim)
        ax.grid(axis="y")
        strip_axes(ax)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.set_xticks(x)
        if idx // 3 == 1:
            ax.set_xticklabels([SHORT_LABELS[m] for m in METHODS], rotation=0)
        else:
            ax.set_xticklabels([])
        ax.text(0.98, 0.91, "higher is better" if direction == "higher" else "lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=5.9, color=COLORS["muted"])
        for bar, value in zip(bars, values):
            if idx in [0, 1]:
                ytxt = max(value - 0.045, ylim[0] + 0.015)
                va = "top"
                color = "white" if value > 0.86 else COLORS["ink"]
            else:
                ytxt = min(value + (ylim[1] - ylim[0]) * 0.035, ylim[1] * 0.96)
                va = "bottom"
                color = COLORS["ink"]
            ax.text(bar.get_x() + bar.get_width() / 2, ytxt, fmt.format(value), ha="center", va=va, fontsize=5.8, color=color)
        add_panel_label(ax, f"({chr(97 + idx)})", -0.02, 1.08)

    # Compact story callouts across the dashboard bottom.
    fig.text(0.075, 0.055, "Copying cost: bootstrap exact-copy rate = 1.00", color=COLORS["weighted_bootstrap"], fontsize=6.8, fontweight="bold")
    fig.text(0.395, 0.055, "Interpretable baseline: Bayesian network preserves one-way marginals", color=COLORS["bayesian_network"], fontsize=6.8, fontweight="bold")
    fig.text(0.725, 0.055, "Contrastive VAE: lower numeric KS and cross-TV than noncontrastive VAE", color=COLORS["contrastive_vae"], fontsize=6.8, fontweight="bold", ha="center")
    save_figure(fig, "fig_internal_validation_dashboard.pdf")


def plot_desire_lines(
    ax: plt.Axes,
    tracts,
    segments: np.ndarray,
    counts: np.ndarray,
    title: str,
    color: str,
    bounds: Iterable[float],
) -> None:
    tracts.boundary.plot(ax=ax, linewidth=0.10, color="#D0D5DD", alpha=0.85, zorder=1)
    if len(segments):
        logged = np.log1p(counts)
        if logged.max() > logged.min():
            widths = 0.10 + 0.52 * (logged - logged.min()) / (logged.max() - logged.min())
        else:
            widths = np.full_like(logged, 0.22)
        collection = LineCollection(
            segments,
            colors=color,
            linewidths=widths,
            alpha=0.10,
            capstyle="round",
            joinstyle="round",
            rasterized=True,
            zorder=2,
        )
        ax.add_collection(collection)
    xmin, ymin, xmax, ymax = bounds
    xpad = (xmax - xmin) * 0.03
    ypad = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=2, fontsize=7.6)


def figure_od_support_panels() -> None:
    tracts, centroids = load_geo()
    base_segments, base_counts, base_unique_sample, row_sample = line_segments_from_sample(
        RUN_DIR / "samples" / "weighted_bootstrap_synthetic.csv",
        centroids,
        sample_rows=50_000,
        max_lines=10_000,
        seed=22,
    )
    cvae_segments, cvae_counts, cvae_unique_sample, _ = line_segments_from_sample(
        RUN_DIR / "samples" / "contrastive_vae_synthetic.csv",
        centroids,
        sample_rows=50_000,
        max_lines=10_000,
        seed=22,
    )

    fig = plt.figure(figsize=(7.25, 4.65))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.05, 1.05], wspace=0.08, left=0.065, right=0.985, top=0.86, bottom=0.16)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_base = fig.add_subplot(grid[0, 1])
    ax_cvae = fig.add_subplot(grid[0, 2])

    fig.text(0.065, 0.96, "OD support expansion and tract-centroid desire lines", fontsize=9.2, fontweight="bold", ha="left", va="top", color=COLORS["ink"])
    fig.text(0.065, 0.918, "The VAE variants generate many more directed tract-pair combinations while the map uses equal-size row samples.", fontsize=7.1, color=COLORS["muted"], ha="left", va="top")

    y = np.arange(len(OD_COUNTS))
    ax_bar.barh(y, OD_COUNTS["unique_pairs"] / 1000, color=[COLORS[key] for key in OD_COUNTS["key"]], height=0.62)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(OD_COUNTS["label"], fontsize=6.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Unique directed OD pairs (thousands)")
    ax_bar.grid(axis="x")
    strip_axes(ax_bar)
    ax_bar.spines["bottom"].set_visible(True)
    for idx, value in enumerate(OD_COUNTS["unique_pairs"]):
        ax_bar.text(value / 1000 + 8, idx, f"{value:,}", va="center", ha="left", fontsize=6.2, color=COLORS["ink"])
    add_panel_label(ax_bar, "(a)", -0.03, 1.03)

    bounds = tracts.total_bounds
    plot_desire_lines(ax_base, tracts, base_segments, base_counts, "Weighted bootstrap", COLORS["weighted_bootstrap"], bounds)
    plot_desire_lines(ax_cvae, tracts, cvae_segments, cvae_counts, "Contrastive VAE", COLORS["contrastive_vae"], bounds)
    add_panel_label(ax_base, "(b)", -0.02, 1.03)
    ax_base.text(0.02, 0.04, f"{row_sample:,} sampled trips\n{base_unique_sample:,} OD pairs", transform=ax_base.transAxes, ha="left", va="bottom", fontsize=6.0, color=COLORS["ink"], bbox=dict(facecolor="white", edgecolor="#D0D5DD", boxstyle="round,pad=0.25", alpha=0.9))
    ax_cvae.text(0.02, 0.04, f"{row_sample:,} sampled trips\n{cvae_unique_sample:,} OD pairs", transform=ax_cvae.transAxes, ha="left", va="bottom", fontsize=6.0, color=COLORS["ink"], bbox=dict(facecolor="white", edgecolor="#D0D5DD", boxstyle="round,pad=0.25", alpha=0.9))
    fig.text(0.46, 0.075, "Lines are straight connections between tract representative points and do not indicate assigned routes.", fontsize=6.4, color=COLORS["muted"], ha="left")
    save_figure(fig, "fig_od_support_panels.pdf", png=True)


def metric_dual_panel(ax: plt.Axes, data: pd.DataFrame, title: str, rmse_label: str) -> None:
    x = np.arange(len(METHODS))
    rmse_k = data.set_index("method").loc[METHODS, "rmse"].to_numpy(float) / 1000
    corr = data.set_index("method").loc[METHODS, "pearson_log1p"].to_numpy(float)
    bars = ax.bar(x, rmse_k, color=[COLORS[m] for m in METHODS], alpha=0.32, edgecolor=[COLORS[m] for m in METHODS], linewidth=0.8)
    ax.set_title(title, loc="left", pad=4)
    ax.set_ylabel(rmse_label)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[m] for m in METHODS], fontsize=5.9)
    ax.grid(axis="y")
    strip_axes(ax)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax2 = ax.twinx()
    ax2.plot(x, corr, color="#111827", marker="o", markersize=3.5, linewidth=1.0, label="Pearson log1p")
    ax2.set_ylabel("Pearson log1p")
    ax2.tick_params(axis="y", labelsize=6.2)
    ax2.spines["top"].set_visible(False)
    for side in ["left", "bottom"]:
        ax2.spines[side].set_visible(False)
    ax2.set_ylim(0, max(0.24, corr.max() + 0.03))
    for bar, value in zip(bars, rmse_k):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(rmse_k) * 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=5.8, color=COLORS["ink"])


def figure_external_validation_panels() -> None:
    import geopandas as gpd

    tracts = gpd.read_parquet(RUN_DIR / "geo" / "tracts.parquet")
    screenlines = gpd.read_parquet(RUN_DIR / "geo" / "screenlines.parquet")
    comparisons = pd.read_parquet(RUN_DIR / "metrics" / "aadt_annual_screenline_comparisons.parquet")
    cvae = comparisons[(comparisons["method"] == "contrastive_vae") & (comparisons["validation_tier"] == "annual_average")].copy()
    cvae["log2_ratio"] = np.log2((cvae["synthetic_count"].astype(float) + 1.0) / (cvae["observed_count"].astype(float) + 1.0))
    cvae["plot_log2_ratio"] = cvae["log2_ratio"].clip(-2.5, 2.5)
    mapped = screenlines.merge(cvae[["screenline_id", "plot_log2_ratio"]], on="screenline_id", how="inner")

    fig = plt.figure(figsize=(7.25, 6.15))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], height_ratios=[1.1, 0.9], wspace=0.28, hspace=0.34, left=0.065, right=0.985, top=0.88, bottom=0.13)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_annual = fig.add_subplot(grid[0, 1])
    ax_hourly = fig.add_subplot(grid[1, 0])
    ax_reduction = fig.add_subplot(grid[1, 1])

    fig.text(0.065, 0.965, "External screenline validation", ha="left", va="top", fontsize=9.2, fontweight="bold", color=COLORS["ink"])
    fig.text(0.065, 0.925, "Annual tract-boundary screenlines and hourly TMAS cells test whether synthetic OD support remains externally plausible.", ha="left", va="top", fontsize=7.1, color=COLORS["muted"])

    tracts.boundary.plot(ax=ax_map, linewidth=0.08, color="#D0D5DD", alpha=0.9, zorder=1)
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)
    before = len(ax_map.collections)
    mapped.plot(ax=ax_map, column="plot_log2_ratio", cmap="RdBu", norm=norm, linewidth=0.48, alpha=0.88, zorder=2)
    for collection in ax_map.collections[before:]:
        collection.set_rasterized(True)
    ax_map.set_axis_off()
    ax_map.set_aspect("equal")
    ax_map.set_title("Annual screenline ratio, Contrastive VAE", loc="left", pad=4)
    add_panel_label(ax_map, "(a)", -0.02, 1.04)
    ax_map.text(0.02, 0.03, "3,107 station-mapped screenlines", transform=ax_map.transAxes, ha="left", va="bottom", fontsize=6.2, color=COLORS["ink"], bbox=dict(facecolor="white", edgecolor="#D0D5DD", boxstyle="round,pad=0.25", alpha=0.9))
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu")
    cbar = fig.colorbar(sm, ax=ax_map, orientation="horizontal", fraction=0.055, pad=0.02)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels(["1/4x", "1/2x", "1x", "2x", "4x"])
    cbar.ax.tick_params(labelsize=5.9, length=0, pad=1)
    cbar.ax.set_title("Synthetic / observed", fontsize=6.2, pad=3)

    metric_dual_panel(ax_annual, ANNUAL, "Annual validation", "RMSE (thousand)")
    add_panel_label(ax_annual, "(b)", -0.02, 1.08)
    ax_annual.text(
        0.02,
        0.88,
        "CVAE has the most lowest-error\nscreenlines: 840",
        transform=ax_annual.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=COLORS["contrastive_vae"],
        fontweight="bold",
    )

    metric_dual_panel(ax_hourly, HOURLY, "Hourly TMAS validation", "RMSE (thousand)")
    add_panel_label(ax_hourly, "(c)", -0.02, 1.08)

    data = HOURLY.set_index("method").loc[METHODS]
    y = np.arange(len(METHODS))
    ax_reduction.barh(y, data["rmse_reduction"], color=[COLORS[m] for m in METHODS], height=0.58)
    ax_reduction.set_yticks(y)
    ax_reduction.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=6.4)
    ax_reduction.invert_yaxis()
    ax_reduction.set_xlabel("Hourly RMSE reduction vs bootstrap (%)")
    ax_reduction.set_xlim(0, 55)
    ax_reduction.grid(axis="x")
    strip_axes(ax_reduction)
    ax_reduction.spines["bottom"].set_visible(True)
    for idx, value in enumerate(data["rmse_reduction"]):
        ax_reduction.text(value + 1.0, idx, f"{value:.1f}%", ha="left", va="center", fontsize=6.2, color=COLORS["ink"])
    ax_reduction.set_title("Hourly gain", loc="left", pad=4)
    add_panel_label(ax_reduction, "(d)", -0.02, 1.08)
    fig.text(0.065, 0.055, "Screenline counts are geometric tract-boundary proxy comparisons, not routed network assignments.", fontsize=6.4, color=COLORS["muted"], ha="left")
    save_figure(fig, "fig_external_validation_panels.pdf", png=True)


def main() -> None:
    setup_style()
    clean_output_dir()
    figure_workflow_overview()
    figure_cvae_architecture()
    figure_screenline_proxy()
    figure_training_curves()
    figure_internal_validation_dashboard()
    figure_od_support_panels()
    figure_external_validation_panels()


if __name__ == "__main__":
    main()

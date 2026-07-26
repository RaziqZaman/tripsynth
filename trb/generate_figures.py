#!/usr/bin/env python3
"""Generate the figures used by the TRB manuscript.

The script reads the frozen ``paper_wctr_1m_vae_sweep`` artifacts.  It does
not retrain or resample any synthesis model.  The external-comparison figure
recomputes hourly metrics on the exact screenline/date/hour intersection
shared by all four methods and uses a paired screenline-block bootstrap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "paper_wctr_1m_vae_sweep"
OUT = ROOT / "trb" / "figures"

METHODS = [
    "weighted_bootstrap",
    "bayesian_network",
    "noncontrastive_vae",
    "contrastive_vae",
]
LABELS = {
    "weighted_bootstrap": "Bootstrap",
    "bayesian_network": "Bayesian\nnetwork",
    "noncontrastive_vae": "Noncontrast.\nVAE",
    "contrastive_vae": "Contrastive\nVAE",
}
SHORT = {
    "weighted_bootstrap": "Bootstrap",
    "bayesian_network": "Bayesian network",
    "noncontrastive_vae": "Noncontrastive VAE",
    "contrastive_vae": "Contrastive VAE",
}
COLORS = {
    "weighted_bootstrap": "#D55E00",
    "bayesian_network": "#009E73",
    "noncontrastive_vae": "#CC79A7",
    "contrastive_vae": "#0072B2",
    "survey": "#6B7280",
    "ink": "#17202A",
    "muted": "#5F6B76",
    "grid": "#D8DEE4",
    "light_blue": "#EAF4FB",
    "light_green": "#EAF7F2",
    "light_orange": "#FFF3E8",
    "light_purple": "#F7EFF6",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "axes.edgecolor": "#30363D",
            "axes.linewidth": 0.65,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str, dpi: int = 350) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Creator": "tripsynth TRB figure generator"},
    )
    fig.savefig(
        OUT / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.065,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=9.2,
        color=COLORS["ink"],
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis)
        ax.set_axisbelow(True)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    face: str = "white",
    edge: str = "#475467",
    fontsize: float = 8.0,
    weight: str = "normal",
    radius: float = 0.02,
    linewidth: float = 0.9,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=2,
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
        color=COLORS["ink"],
        linespacing=1.2,
        zorder=3,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#475467",
    linewidth: float = 1.0,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
            zorder=5,
        )
    )


def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.025, 0.235, 0.515, 0.795]
    ws = [0.165, 0.225, 0.225, 0.18]
    y, h = 0.51, 0.34
    texts = [
        "Weighted travel\nsurvey\n110,926 trips\n44 fields",
        "Four generators\n\nBootstrap\nBayesian network\nVAE ablation pair",
        "Validation lenses\n\nSurvey fidelity\nCopying + OD support\nTraffic-count stress tests",
        "Decision evidence\n\nFidelity\nSupport\nOperational fit",
    ]
    faces = [
        "#F4F6F8",
        COLORS["light_green"],
        COLORS["light_purple"],
        COLORS["light_orange"],
    ]
    for x, w, text, face in zip(xs, ws, texts, faces):
        box(ax, (x, y), (w, h), text, face=face, fontsize=9.4)
    for left, width, right in zip(xs[:-1], ws[:-1], xs[1:]):
        arrow(ax, (left + width + 0.008, y + h / 2), (right - 0.008, y + h / 2))

    box(
        ax,
        (0.13, 0.13),
        (0.74, 0.18),
        "Central test: do better survey dependencies yield better roadside evidence?",
        face=COLORS["light_blue"],
        edge=COLORS["contrastive_vae"],
        fontsize=9.2,
        weight="bold",
    )
    arrow(ax, (0.625, y - 0.015), (0.625, 0.325), color=COLORS["contrastive_vae"])
    ax.text(
        0.025,
        0.93,
        "Trip synthesis is evaluated as a multi-objective transportation problem",
        ha="left",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.975,
        0.055,
        "External counts are diagnostics only; they are not used in training or model selection.",
        ha="right",
        va="center",
        fontsize=8.1,
        color=COLORS["muted"],
    )
    save(fig, "fig_01_workflow")


def figure_architecture() -> None:
    fig = plt.figure(figsize=(7.15, 4.65))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.22)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1])
    for ax in [top, bottom]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    panel(top, "(A)")
    top.set_title("Mixed-tabular variational autoencoder", loc="left", pad=9)
    box(top, (0.02, 0.57), (0.15, 0.20), "26 categorical\nembeddings", face=COLORS["light_blue"], edge=COLORS["contrastive_vae"], fontsize=8.5)
    box(top, (0.02, 0.22), (0.15, 0.20), "18 standardized\nnumeric fields", face=COLORS["light_purple"], edge=COLORS["noncontrastive_vae"], fontsize=8.5)
    box(top, (0.23, 0.40), (0.12, 0.20), "Concatenate\n280 inputs", face="#F5F6F7", fontsize=8.5)
    box(top, (0.41, 0.40), (0.12, 0.20), "4-layer\nencoder", face="#F5F6F7", fontsize=8.5)
    box(top, (0.575, 0.40), (0.145, 0.20), "$\\mu,\\log\\sigma^2$\n512-D latent", face=COLORS["light_orange"], edge="#B45309", fontsize=8.5)
    box(top, (0.77, 0.40), (0.10, 0.20), "4-layer\ndecoder", face="#F5F6F7", fontsize=8.5)
    box(top, (0.91, 0.57), (0.075, 0.18), "26 softmax\nheads", face=COLORS["light_blue"], edge=COLORS["contrastive_vae"], fontsize=8.0)
    box(top, (0.91, 0.20), (0.075, 0.18), "18 numeric\noutputs", face=COLORS["light_purple"], edge=COLORS["noncontrastive_vae"], fontsize=8.0)
    arrow(top, (0.17, 0.67), (0.23, 0.52), color=COLORS["contrastive_vae"])
    arrow(top, (0.17, 0.32), (0.23, 0.48), color=COLORS["noncontrastive_vae"])
    arrow(top, (0.35, 0.50), (0.41, 0.50))
    arrow(top, (0.53, 0.50), (0.575, 0.50))
    arrow(top, (0.72, 0.50), (0.77, 0.50))
    arrow(top, (0.87, 0.50), (0.91, 0.66), color=COLORS["contrastive_vae"])
    arrow(top, (0.87, 0.50), (0.91, 0.29), color=COLORS["noncontrastive_vae"])
    top.text(
        0.5,
        0.08,
        "Survey expansion weights are capped at the 99th percentile and mean-normalized for training; weight is never generated.",
        ha="center",
        va="center",
        fontsize=8.3,
        color=COLORS["muted"],
    )

    panel(bottom, "(B)")
    bottom.set_title("Trip-specific contrastive regularization", loc="left", pad=9)
    box(bottom, (0.03, 0.51), (0.16, 0.23), "Original trip\nrecord", face="#F5F6F7", fontsize=8.7, weight="bold")
    box(bottom, (0.27, 0.67), (0.22, 0.20), "Positive view\nlight non-key changes\nOD + time protected", face=COLORS["light_green"], edge=COLORS["bayesian_network"], fontsize=8.1)
    box(bottom, (0.27, 0.22), (0.22, 0.23), "Negative view\nuniform categories +\nshuffled numerics", face="#FFF1F0", edge="#B42318", fontsize=8.1)
    box(bottom, (0.57, 0.51), (0.15, 0.23), "Shared\nencoder", face="#F5F6F7", fontsize=8.5)
    box(bottom, (0.80, 0.51), (0.16, 0.23), "Contrastive\nbinary loss", face=COLORS["light_orange"], edge="#B45309", fontsize=8.3, weight="bold")
    arrow(bottom, (0.19, 0.63), (0.27, 0.77), color=COLORS["bayesian_network"])
    arrow(bottom, (0.19, 0.61), (0.27, 0.335), color="#B42318")
    arrow(bottom, (0.49, 0.77), (0.57, 0.66), color=COLORS["bayesian_network"])
    arrow(bottom, (0.49, 0.335), (0.57, 0.58), color="#B42318")
    arrow(bottom, (0.72, 0.625), (0.80, 0.625), color="#B45309")
    bottom.text(
        0.5,
        0.08,
        r"Training objective: reconstruction + $0.03\,\mathrm{KL}$ + $0.25\,L_{\mathrm{con}}$; the ablation sets the final coefficient to zero.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    save(fig, "fig_02_architecture")


def figure_internal_tradeoff() -> None:
    validation = pd.read_csv(RUN / "tables" / "method_validation_summary.csv").set_index("method")
    privacy = pd.read_csv(RUN / "tables" / "privacy_summary.csv").set_index("method")
    data = validation.join(privacy[["exact_row_copy_rate", "novel_od_pair_share"]])

    fig = plt.figure(figsize=(7.15, 3.55))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.34)
    ax = fig.add_subplot(grid[0])
    ab = fig.add_subplot(grid[1])

    for method in METHODS:
        row = data.loc[method]
        marker = "X" if row["exact_row_copy_rate"] > 0 else "o"
        size = 105 if marker == "X" else 85
        ax.scatter(
            row["mean_cross_tv"],
            row["novel_od_pair_share"],
            s=size,
            marker=marker,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        offsets = {
            "weighted_bootstrap": (6, 8),
            "bayesian_network": (6, 6),
            "noncontrastive_vae": (-118, 8),
            "contrastive_vae": (-18, -38),
        }
        ax.annotate(
            SHORT[method],
            (row["mean_cross_tv"], row["novel_od_pair_share"]),
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=7.4,
            color=COLORS["ink"],
        )
    nc = data.loc["noncontrastive_vae"]
    cv = data.loc["contrastive_vae"]
    arrow(
        ax,
        (nc["mean_cross_tv"] - 0.001, nc["novel_od_pair_share"] - 0.008),
        (cv["mean_cross_tv"] + 0.003, cv["novel_od_pair_share"] - 0.008),
        color=COLORS["contrastive_vae"],
        linewidth=1.25,
    )
    ax.text(
        0.242,
        0.65,
        "11.4% lower\ncross-TV",
        ha="center",
        va="top",
        fontsize=8.0,
        color=COLORS["contrastive_vae"],
    )
    ax.set_xlabel("Mean cross-marginal total variation (lower is better)")
    ax.set_ylabel("Novel OD-pair share (higher is broader support)")
    ax.set_xlim(-0.01, 0.285)
    ax.set_ylim(-0.04, 0.76)
    clean_axes(ax, "both")
    panel(ax, "(A)")
    ax.set_title("  Dependence–support tradeoff", loc="left", pad=8)
    ax.text(
        0.01,
        0.15,
        "X = exact-copy rate 1.00",
        transform=ax.transAxes,
        fontsize=8.0,
        color=COLORS["weighted_bootstrap"],
        fontweight="bold",
    )

    metrics = [
        ("mean_numeric_ks", "Numeric KS"),
        ("mean_cross_tv", "Cross-marginal TV"),
        ("mean_categorical_tv", "Categorical TV"),
    ]
    changes = []
    for column, label in metrics:
        base = float(validation.loc["noncontrastive_vae", column])
        value = float(validation.loc["contrastive_vae", column])
        changes.append((label, 100.0 * (value - base) / base))
    labels = [item[0] for item in changes]
    values = np.array([item[1] for item in changes])
    colors = [COLORS["contrastive_vae"] if value < 0 else COLORS["weighted_bootstrap"] for value in values]
    y = np.arange(len(labels))
    bars = ab.barh(y, values, color=colors, height=0.56)
    ab.axvline(0, color="#30363D", linewidth=0.75)
    ab.set_yticks(y)
    ab.set_yticklabels(labels)
    ab.invert_yaxis()
    ab.set_xlabel("Change from noncontrastive VAE (%)")
    ab.set_xlim(-16, 8)
    clean_axes(ab, "x")
    panel(ab, "(B)")
    ab.set_title("  Effect of contrastive term", loc="left", pad=8)
    for bar, value in zip(bars, values):
        ab.text(
            value - 0.6 if value < 0 else value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            ha="right" if value < 0 else "left",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=COLORS["ink"],
        )
    save(fig, "fig_03_internal_tradeoff")


def figure_screenline_method() -> None:
    fig = plt.figure(figsize=(7.15, 4.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.12)
    ax = fig.add_subplot(grid[0])
    tiers = fig.add_subplot(grid[1])
    for item in [ax, tiers]:
        item.set_xlim(0, 1)
        item.set_ylim(0, 1)
        item.axis("off")

    panel(ax, "(A)")
    ax.set_title("  Geometric screenline proxy", loc="left", pad=8)
    left_poly = np.array([[0.04, 0.12], [0.48, 0.08], [0.54, 0.34], [0.49, 0.56], [0.55, 0.90], [0.11, 0.92], [0.03, 0.59]])
    right_poly = np.array([[0.48, 0.08], [0.93, 0.12], [0.98, 0.54], [0.90, 0.86], [0.55, 0.90], [0.49, 0.56], [0.54, 0.34]])
    ax.add_patch(Polygon(left_poly, closed=True, facecolor="#F0F5F9", edgecolor="#667085", linewidth=1.0))
    ax.add_patch(Polygon(right_poly, closed=True, facecolor="#F8F4FB", edgecolor="#667085", linewidth=1.0))
    boundary = np.array([[0.55, 0.90], [0.49, 0.56], [0.54, 0.34], [0.48, 0.08]])
    ax.plot(boundary[:, 0], boundary[:, 1], color=COLORS["bayesian_network"], linewidth=3.0, zorder=4)
    origin, destination = (0.22, 0.55), (0.78, 0.42)
    ax.scatter([origin[0], destination[0]], [origin[1], destination[1]], s=62, color=[COLORS["weighted_bootstrap"], COLORS["contrastive_vae"]], edgecolor="white", linewidth=0.8, zorder=8)
    arrow(ax, origin, destination, color=COLORS["ink"], linewidth=1.5)
    ax.text(0.20, 0.64, "Origin tract\nrepresentative point", ha="center", va="bottom", fontsize=7.1)
    ax.text(0.79, 0.33, "Destination tract\nrepresentative point", ha="center", va="top", fontsize=7.1)
    stations = [(0.57, 0.72), (0.58, 0.49), (0.45, 0.40), (0.60, 0.82)]
    for sx, sy in stations:
        ax.scatter([sx], [sy], s=42, marker="s", color=COLORS["ink"], edgecolor="white", linewidth=0.6, zorder=8)
    ax.text(0.66, 0.73, "traffic-count\nstations", ha="left", va="center", fontsize=7.1)
    ax.text(0.48, 0.20, "candidate\nscreenline", ha="right", va="center", fontsize=7.2, color=COLORS["bayesian_network"], fontweight="bold")
    ax.text(
        0.50,
        0.01,
        "Representative-point segments mark\nboundary crossings, not road routes.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    panel(tiers, "(B)")
    tiers.set_title("  External count tiers", loc="left", pad=8)
    box(
        tiers,
        (0.08, 0.62),
        (0.84, 0.23),
        "Annual weekday tier\n\nMDOT AAWDT station totals\n3,107 mapped screenlines",
        face=COLORS["light_blue"],
        edge=COLORS["contrastive_vae"],
        fontsize=8.3,
    )
    box(
        tiers,
        (0.08, 0.26),
        (0.84, 0.23),
        "Hourly tier\n\nFHWA TMAS station-hour counts\nannual-share scaled",
        face=COLORS["light_green"],
        edge=COLORS["bayesian_network"],
        fontsize=8.3,
    )
    arrow(tiers, (0.50, 0.62), (0.50, 0.50), color=COLORS["muted"])
    tiers.text(
        0.5,
        0.10,
        "Crossing hour = departure + travel time\n× boundary-position fraction",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    tiers.text(
        0.5,
        0.02,
        "External counts are diagnostic only.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    save(fig, "fig_04_screenline_method")


def figure_hourly_granularity() -> None:
    observed = pd.read_parquet(RUN / "metrics" / "observed_hourly_screenline_counts.parquet")
    observed["trip_date"] = pd.to_datetime(observed["trip_date"]).dt.normalize()
    observed = observed[observed["trip_date"].dt.dayofweek < 5].copy()
    observed_profile = (
        observed.groupby(["screenline_id", "hour"], as_index=False)
        .agg(observed_count=("observed_count", "mean"))
    )
    screenlines = sorted(observed_profile["screenline_id"].unique())
    common_index = pd.MultiIndex.from_product(
        [screenlines, range(24)], names=["screenline_id", "hour"]
    )
    observed_matrix = (
        observed_profile.set_index(["screenline_id", "hour"])
        .reindex(common_index)["observed_count"]
        .to_numpy()
        .reshape(len(screenlines), 24)
    )

    synthetic_matrices = {}
    for method in METHODS:
        synthetic = pd.read_parquet(
            RUN / "metrics" / f"{method}_hourly_virtual_screenline_counts.parquet"
        )
        synthetic["trip_date"] = pd.to_datetime(synthetic["trip_date"]).dt.normalize()
        synthetic = synthetic[synthetic["trip_date"].dt.dayofweek < 5].copy()
        synthetic["pooled_crossings"] = (
            synthetic["synthetic_count"] / synthetic["temporal_expansion_factor"]
        )
        synthetic_profile = synthetic.groupby(
            ["screenline_id", "hour"]
        )["pooled_crossings"].sum()
        synthetic_matrices[method] = (
            synthetic_profile.reindex(common_index, fill_value=0.0)
            .to_numpy()
            .reshape(len(screenlines), 24)
        )

    observed_share = observed_matrix / observed_matrix.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(20260726)
    draws = rng.integers(0, len(screenlines), size=(10_000, len(screenlines)))
    rmse, rmse_lo, rmse_hi = [], [], []
    profile_tv, tv_lo, tv_hi = [], [], []
    tv_boot = {}
    for method in METHODS:
        synthetic = synthetic_matrices[method]
        squared_error = (synthetic - observed_matrix) ** 2
        rmse_value = np.sqrt(squared_error.mean())
        rmse_draws = np.sqrt(squared_error[draws, :].mean(axis=(1, 2)))
        totals = synthetic.sum(axis=1)
        synthetic_share = np.divide(
            synthetic,
            totals[:, None],
            out=np.zeros_like(synthetic),
            where=totals[:, None] > 0,
        )
        tv = 0.5 * np.abs(synthetic_share - observed_share).sum(axis=1)
        tv[totals == 0] = 1.0
        tv_draws = tv[draws].mean(axis=1)
        tv_boot[method] = tv_draws
        rmse.append(rmse_value)
        rmse_lo.append(np.quantile(rmse_draws, 0.025))
        rmse_hi.append(np.quantile(rmse_draws, 0.975))
        profile_tv.append(tv.mean())
        tv_lo.append(np.quantile(tv_draws, 0.025))
        tv_hi.append(np.quantile(tv_draws, 0.975))

    fig, (ax_rmse, ax_tv) = plt.subplots(1, 2, figsize=(7.15, 3.55))
    x = np.arange(len(METHODS))
    for idx, method in enumerate(METHODS):
        ax_rmse.errorbar(
            idx,
            rmse[idx] / 1000,
            yerr=np.array([[rmse[idx] - rmse_lo[idx]], [rmse_hi[idx] - rmse[idx]]]) / 1000,
            fmt="o",
            markersize=7.0,
            color=COLORS[method],
            ecolor=COLORS[method],
            elinewidth=1.25,
            capsize=3.0,
        )
        ax_rmse.text(idx, rmse_hi[idx] / 1000 + 0.07, f"{rmse[idx]/1000:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels([LABELS[method] for method in METHODS])
    ax_rmse.set_ylabel("Date-pooled hourly RMSE (thousand vehicles)")
    ax_rmse.set_ylim(4.7, 7.5)
    clean_axes(ax_rmse, "y")
    panel(ax_rmse, "(A)")
    ax_rmse.set_title("  Absolute volume remains biased", loc="left", pad=8)
    ax_rmse.text(0.02, 0.04, "1,344 identical screenline-hour cells", transform=ax_rmse.transAxes, fontsize=8.0, color=COLORS["muted"])

    for idx, method in enumerate(METHODS):
        ax_tv.errorbar(
            idx,
            profile_tv[idx],
            yerr=np.array([[profile_tv[idx] - tv_lo[idx]], [tv_hi[idx] - profile_tv[idx]]]),
            fmt="o",
            markersize=7.0,
            color=COLORS[method],
            ecolor=COLORS[method],
            elinewidth=1.25,
            capsize=3.0,
        )
        ax_tv.text(idx, tv_hi[idx] + 0.008, f"{profile_tv[idx]:.3f}", ha="center", va="bottom", fontsize=7.0)
    ax_tv.set_xticks(x)
    ax_tv.set_xticklabels([LABELS[method] for method in METHODS])
    ax_tv.set_ylabel("Mean normalized 24-hour profile TV")
    ax_tv.set_ylim(0.08, 0.39)
    clean_axes(ax_tv, "y")
    panel(ax_tv, "(B)")
    ax_tv.set_title("  Temporal-profile shape", loc="left", pad=8)
    improvement_draws = 1.0 - tv_boot["contrastive_vae"] / tv_boot["noncontrastive_vae"]
    improvement_ci = 100.0 * np.quantile(improvement_draws, [0.025, 0.975])
    ax_tv.text(
        0.02,
        0.04,
        f"Contrastive vs noncontrastive: 21.1% lower TV\n95% CI {improvement_ci[0]:.1f}–{improvement_ci[1]:.1f}%; BN remains best",
        transform=ax_tv.transAxes,
        fontsize=8.0,
        color=COLORS["muted"],
    )

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.91, wspace=0.27)
    fig.text(
        0.5,
        0.005,
        "Dates are pooled before undoing method-specific temporal expansion, reducing the synthetic count quantum from 8.5–9.9 thousand to 19.3 vehicles.\nError bars are paired 95% screenline-block bootstrap intervals; all 24 hours remain within each block.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    save(fig, "fig_05_hourly_granularity")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    figure_workflow()
    figure_architecture()
    figure_internal_tradeoff()
    figure_screenline_method()
    figure_hourly_granularity()


if __name__ == "__main__":
    main()

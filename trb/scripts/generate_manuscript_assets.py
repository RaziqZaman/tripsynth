#!/usr/bin/env python3
"""Generate the evidence-linked assets used by the TRB manuscript.

The script reads the frozen ``paper_wctr_1m_vae_sweep`` artifacts and the
transformed survey table. It does not retrain or resample a synthesis model.
It reconstructs a direct weighted-survey benchmark, recomputes date-pooled
hourly metrics on identical support, calculates paired screenline-block
bootstrap intervals, writes machine-readable figure data and LaTeX macros,
and creates every empirical figure used in the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs" / "runs" / "paper_wctr_1m_vae_sweep"
OUT = ROOT / "trb" / "figures"
TABLES = ROOT / "trb" / "tables"
GENERATED = ROOT / "trb" / "generated"
SURVEY = ROOT / "05_transformed_survey.csv"

GENERATED_METHODS = [
    "weighted_bootstrap",
    "bayesian_network",
    "noncontrastive_vae",
    "contrastive_vae",
]
EXTERNAL_METHODS = ["direct_survey", *GENERATED_METHODS]
LABELS = {
    "direct_survey": "Direct\nsurvey",
    "weighted_bootstrap": "Bootstrap",
    "bayesian_network": "Bayesian\nnetwork",
    "noncontrastive_vae": "Noncontrast.\nVAE",
    "contrastive_vae": "Contrastive\nVAE",
}
SHORT = {
    "direct_survey": "Direct weighted survey",
    "weighted_bootstrap": "Bootstrap",
    "bayesian_network": "Bayesian network",
    "noncontrastive_vae": "Noncontrastive VAE",
    "contrastive_vae": "Contrastive VAE",
}
COLORS = {
    "direct_survey": "#4D4D4D",
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
            "font.family": "Liberation Serif",
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.4,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
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
        fontsize=9.0,
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
    box(top, (0.02, 0.57), (0.15, 0.20), "26 categorical\nembeddings", face=COLORS["light_blue"], edge=COLORS["contrastive_vae"], fontsize=9.0)
    box(top, (0.02, 0.22), (0.15, 0.20), "18 standardized\nnumeric fields", face=COLORS["light_purple"], edge=COLORS["noncontrastive_vae"], fontsize=9.0)
    box(top, (0.23, 0.40), (0.12, 0.20), "Concatenate\n280 inputs", face="#F5F6F7", fontsize=9.0)
    box(top, (0.41, 0.40), (0.12, 0.20), "4-layer\nencoder", face="#F5F6F7", fontsize=9.0)
    box(top, (0.575, 0.40), (0.145, 0.20), "$\\mu,\\log\\sigma^2$\n512-D latent", face=COLORS["light_orange"], edge="#B45309", fontsize=9.0)
    box(top, (0.76, 0.40), (0.10, 0.20), "4-layer\ndecoder", face="#F5F6F7", fontsize=9.0)
    box(top, (0.895, 0.57), (0.09, 0.18), "26 softmax\nheads", face=COLORS["light_blue"], edge=COLORS["contrastive_vae"], fontsize=8.6)
    box(top, (0.895, 0.20), (0.09, 0.18), "18 numeric\noutputs", face=COLORS["light_purple"], edge=COLORS["noncontrastive_vae"], fontsize=8.6)
    arrow(top, (0.17, 0.67), (0.23, 0.52), color=COLORS["contrastive_vae"])
    arrow(top, (0.17, 0.32), (0.23, 0.48), color=COLORS["noncontrastive_vae"])
    arrow(top, (0.35, 0.50), (0.41, 0.50))
    arrow(top, (0.53, 0.50), (0.575, 0.50))
    arrow(top, (0.72, 0.50), (0.76, 0.50))
    arrow(top, (0.86, 0.50), (0.895, 0.66), color=COLORS["contrastive_vae"])
    arrow(top, (0.86, 0.50), (0.895, 0.29), color=COLORS["noncontrastive_vae"])
    top.text(
        0.5,
        0.08,
        "Survey expansion weights are capped at the 99th percentile and mean-normalized for training; weight is never generated.",
        ha="center",
        va="center",
        fontsize=9.0,
        color=COLORS["muted"],
    )

    panel(bottom, "(B)")
    bottom.set_title("Trip-specific contrastive regularization", loc="left", pad=9)
    box(bottom, (0.03, 0.51), (0.16, 0.23), "Original trip\nrecord", face="#F5F6F7", fontsize=9.0, weight="bold")
    box(bottom, (0.27, 0.67), (0.22, 0.20), "Positive view\nlight non-key changes\nOD + time protected", face=COLORS["light_green"], edge=COLORS["bayesian_network"], fontsize=9.0)
    box(bottom, (0.27, 0.22), (0.22, 0.23), "Negative view\nuniform categories +\nshuffled numerics", face="#FFF1F0", edge="#B42318", fontsize=9.0)
    box(bottom, (0.57, 0.51), (0.15, 0.23), "Shared\nencoder", face="#F5F6F7", fontsize=9.0)
    box(bottom, (0.80, 0.51), (0.16, 0.23), "Contrastive\nbinary loss", face=COLORS["light_orange"], edge="#B45309", fontsize=9.0, weight="bold")
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
        fontsize=9.0,
        color=COLORS["muted"],
    )
    save(fig, "fig_02_architecture")


def figure_internal_tradeoff() -> None:
    validation = pd.read_csv(RUN / "tables" / "method_validation_summary.csv").set_index("method")
    privacy = pd.read_csv(RUN / "tables" / "privacy_summary.csv").set_index("method")
    data = validation.join(privacy[["exact_row_copy_rate", "novel_od_pair_share"]])

    fig = plt.figure(figsize=(7.15, 3.55))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.45)
    ax = fig.add_subplot(grid[0])
    ab = fig.add_subplot(grid[1])

    tradeoff_labels = {
        "weighted_bootstrap": "Survey-Sampled Baseline",
        "bayesian_network": "Bayesian network",
        "noncontrastive_vae": "Noncontrastive VAE",
        "contrastive_vae": "Contrastive VAE",
    }
    offsets = {
        "weighted_bootstrap": (7, 10),
        "bayesian_network": (7, 7),
        "noncontrastive_vae": (8, -30),
        "contrastive_vae": (0.18, 0.70),
    }
    horizontal_alignment = {
        "weighted_bootstrap": "left",
        "bayesian_network": "left",
        "noncontrastive_vae": "right",
        "contrastive_vae": "right",
    }
    for method in GENERATED_METHODS:
        row = data.loc[method]
        ax.scatter(
            row["mean_cross_tv"],
            row["novel_od_pair_share"],
            s=85,
            marker="o",
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.annotate(
            tradeoff_labels[method],
            (row["mean_cross_tv"], row["novel_od_pair_share"]),
            xytext=offsets[method],
            textcoords="data" if method == "contrastive_vae" else "offset points",
            fontsize=9.0,
            color=COLORS[method],
            fontweight="bold",
            horizontalalignment=horizontal_alignment[method],
            verticalalignment="center" if method == "contrastive_vae" else "baseline",
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS[method],
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 5,
            },
        )
    ax.set_xlabel("Mean cross-marginal total variation (lower is better)")
    ax.set_ylabel("Novel OD-pair share (higher is broader support)")
    ax.set_xlim(-0.01, 0.285)
    ax.set_ylim(-0.04, 0.76)
    clean_axes(ax, "both")
    panel(ax, "(A)")
    ax.set_title("  Dependence–support tradeoff", loc="left", pad=8)

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
    ab.set_yticklabels(labels, fontsize=8.6)
    ab.invert_yaxis()
    ab.set_xlabel("Change from noncontrastive VAE (%)")
    ab.set_xlim(-16, 8)
    clean_axes(ab, "x")
    panel(ab, "(B)")
    ab.set_title("  Effect of contrastive term", loc="left", pad=8)
    for bar, value in zip(bars, values):
        ab.text(
            value + 0.6 if value < 0 else value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color="white" if value < 0 else COLORS["ink"],
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
    ax.text(0.20, 0.64, "Origin tract\nrepresentative point", ha="center", va="bottom", fontsize=9.0)
    ax.text(0.79, 0.33, "Destination tract\nrepresentative point", ha="center", va="top", fontsize=9.0)
    stations = [(0.57, 0.72), (0.58, 0.49), (0.45, 0.40), (0.60, 0.82)]
    for sx, sy in stations:
        ax.scatter([sx], [sy], s=42, marker="s", color=COLORS["ink"], edgecolor="white", linewidth=0.6, zorder=8)
    ax.text(0.66, 0.73, "traffic-count\nstations", ha="left", va="center", fontsize=9.0)
    ax.text(0.48, 0.20, "candidate\nscreenline", ha="right", va="center", fontsize=9.0, color=COLORS["bayesian_network"], fontweight="bold")
    ax.text(
        0.50,
        0.01,
        "Representative-point segments mark\nboundary crossings, not road routes.",
        ha="center",
        va="bottom",
        fontsize=9.0,
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
        fontsize=9.0,
    )
    box(
        tiers,
        (0.08, 0.26),
        (0.84, 0.23),
        "Hourly tier\n\nFHWA TMAS station-hour counts\nannual-share scaled",
        face=COLORS["light_green"],
        edge=COLORS["bayesian_network"],
        fontsize=9.0,
    )
    arrow(tiers, (0.50, 0.62), (0.50, 0.50), color=COLORS["muted"])
    tiers.text(
        0.5,
        0.10,
        "Crossing hour = departure + travel time\n× boundary-position fraction",
        ha="center",
        va="center",
        fontsize=9.0,
        color=COLORS["muted"],
    )
    tiers.text(
        0.5,
        0.02,
        "External counts are diagnostic only.",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=COLORS["muted"],
    )
    save(fig, "fig_04_screenline_method")


def _clean_fips(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(-1).round().astype("int64")
    return numeric.astype(str)


def direct_survey_crossings() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "o_tract_fips",
        "d_tract_fips",
        "tdate_week",
        "tdate_dow",
        "departure_time_minutes",
        "reported_travel_time",
        "weight",
    ]
    survey = pd.read_csv(SURVEY, usecols=columns, low_memory=False)
    survey["trip_id"] = np.arange(len(survey))
    survey["o_tract_fips"] = _clean_fips(survey["o_tract_fips"])
    survey["d_tract_fips"] = _clean_fips(survey["d_tract_fips"])
    paths = pd.read_parquet(RUN / "geo" / "od_screenline_paths.parquet")
    paths["o_tract_fips"] = paths["o_tract_fips"].astype(str)
    paths["d_tract_fips"] = paths["d_tract_fips"].astype(str)
    expanded = survey.merge(paths, on=["o_tract_fips", "d_tract_fips"], how="inner")
    path_counts = expanded.groupby("trip_id")["path_position"].transform("max") + 1
    path_fraction = (expanded["path_position"] + 1) / (path_counts + 1)
    crossing_minutes = (
        pd.to_numeric(expanded["departure_time_minutes"], errors="coerce")
        + pd.to_numeric(expanded["reported_travel_time"], errors="coerce").clip(lower=0)
        * path_fraction
    )
    expanded["crossing_day_offset"] = np.floor(crossing_minutes / 1440.0).astype(int)
    expanded["hour"] = np.floor(np.mod(crossing_minutes, 1440.0) / 60.0).astype(int).clip(0, 23)
    return survey, expanded



def hourly_station_lookup(screenlines: list[str]) -> pd.Series:
    station_map = pd.read_parquet(RUN / "geo" / "screenline_station_map.parquet")
    hourly_ids = set(
        pd.read_csv(
            ROOT / "data" / "external" / "mdot_daily_counts" / "station_hourly_counts.csv",
            usecols=["station_id"],
        )["station_id"].astype(str)
    )
    mapped = station_map[
        station_map["station_id"].astype(str).isin(hourly_ids)
        & station_map["screenline_id"].isin(screenlines)
    ][["screenline_id", "station_id"]].drop_duplicates()
    counts = mapped.groupby("screenline_id")["station_id"].nunique()
    if len(counts) != len(screenlines) or not counts.eq(1).all():
        raise ValueError("Expected one hourly TMAS station for every evaluated screenline")
    return mapped.set_index("screenline_id")["station_id"].reindex(screenlines).astype(str)


def exact_date_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons = pd.read_parquet(
        RUN / "metrics" / "aadt_hourly_screenline_comparisons.parquet"
    )
    keys = ["screenline_id", "trip_date", "hour"]
    comparisons["trip_date"] = pd.to_datetime(comparisons["trip_date"]).dt.normalize()
    common: pd.DataFrame | None = None
    for method in GENERATED_METHODS:
        method_keys = comparisons.loc[comparisons["method"].eq(method), keys].drop_duplicates()
        common = method_keys if common is None else common.merge(method_keys, on=keys)
    assert common is not None
    common = common.sort_values(keys).reset_index(drop=True)

    observed = (
        comparisons.loc[comparisons["method"].eq("weighted_bootstrap"), keys + ["observed_count"]]
        .merge(common, on=keys, how="inner")
        .drop_duplicates(keys)
    )
    fixed_factor = 423.0
    wide = observed.copy()
    for method in GENERATED_METHODS:
        method_data = comparisons.loc[
            comparisons["method"].eq(method),
            keys + ["synthetic_count", "temporal_expansion_factor"],
        ].merge(common, on=keys, how="inner")
        method_data[method] = (
            method_data["synthetic_count"]
            / method_data["temporal_expansion_factor"]
            * fixed_factor
        )
        wide = wide.merge(method_data[keys + [method]], on=keys, how="left")

    survey, expanded = direct_survey_crossings()
    expanded["trip_date"] = pd.Timestamp("2017-01-01") + pd.to_timedelta(
        pd.to_numeric(expanded["tdate_week"]) * 7
        + np.mod(pd.to_numeric(expanded["tdate_dow"]), 7),
        unit="D",
    ) + pd.to_timedelta(expanded["crossing_day_offset"], unit="D")
    direct = (
        expanded.groupby(keys)["weight"].sum().mul(fixed_factor).rename("direct_survey")
    )
    wide = wide.merge(direct.reset_index(), on=keys, how="left")
    wide["direct_survey"] = wide["direct_survey"].fillna(0.0)
    wide = wide[keys + ["observed_count"] + EXTERNAL_METHODS]

    station_by_screenline = hourly_station_lookup(sorted(wide["screenline_id"].unique()))
    wide["station_id"] = wide["screenline_id"].map(station_by_screenline)
    rows: list[dict[str, float | int | str]] = []
    for method in EXTERNAL_METHODS:
        error = wide[method] - wide["observed_count"]
        rows.append(
            {
                "method": method,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "cells": len(wide),
                "screenlines": wide["screenline_id"].nunique(),
                "stations": wide["station_id"].nunique(),
                "dates": wide["trip_date"].nunique(),
                "fixed_temporal_factor": int(fixed_factor),
            }
        )
    summary = pd.DataFrame(rows).set_index("method")
    summary.to_csv(TABLES / "exact_date_common_support_results.csv")

    cluster_ids = sorted(wide["station_id"].unique())
    cluster_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in EXTERNAL_METHODS:
        work = wide[["station_id", "observed_count", method]].copy()
        work["squared_error"] = (work[method] - work["observed_count"]) ** 2
        grouped = work.groupby("station_id")["squared_error"].agg(["sum", "count"]).reindex(cluster_ids)
        cluster_stats[method] = (grouped["sum"].to_numpy(), grouped["count"].to_numpy())
    rng = np.random.default_rng(20260730)
    draws = rng.integers(0, len(cluster_ids), size=(10_000, len(cluster_ids)))

    contrasts: list[dict[str, float | int | str]] = []
    for baseline in ["direct_survey", "weighted_bootstrap", "noncontrastive_vae"]:
        base_sse, base_n = cluster_stats[baseline]
        cv_sse, cv_n = cluster_stats["contrastive_vae"]
        base_rmse = np.sqrt(base_sse[draws].sum(axis=1) / base_n[draws].sum(axis=1))
        cv_rmse = np.sqrt(cv_sse[draws].sum(axis=1) / cv_n[draws].sum(axis=1))
        if baseline == "noncontrastive_vae":
            values = cv_rmse - base_rmse
            metric = "rmse_difference_vehicles"
            estimate = summary.loc["contrastive_vae", "rmse"] - summary.loc[baseline, "rmse"]
        else:
            values = 100.0 * (1.0 - cv_rmse / base_rmse)
            metric = "rmse_relative_reduction_pct"
            estimate = 100.0 * (
                1.0 - summary.loc["contrastive_vae", "rmse"] / summary.loc[baseline, "rmse"]
            )
        contrasts.append(
            {
                "contrast": f"contrastive_vae_vs_{baseline}",
                "metric": metric,
                "estimate": float(estimate),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "clusters": len(cluster_ids),
                "replicates": len(draws),
            }
        )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(TABLES / "exact_date_clustered_contrasts.csv", index=False)
    wide.to_csv(GENERATED / "exact_date_common_support_predictions.csv", index=False)
    return summary, contrast_frame, wide


def pooled_station_cluster_contrasts(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    screenlines: list[str],
) -> pd.DataFrame:
    station_ids = hourly_station_lookup(screenlines).to_numpy()
    clusters = sorted(set(station_ids))
    cluster_screens = [np.flatnonzero(station_ids == cluster) for cluster in clusters]
    rng = np.random.default_rng(20260730)
    draws = rng.integers(0, len(clusters), size=(10_000, len(clusters)))
    stats: dict[str, dict[str, np.ndarray]] = {}
    for method in EXTERNAL_METHODS:
        squared = (predictions[method] - observed) ** 2
        tv = _profile_tv(observed, predictions[method])
        stats[method] = {
            "sse": np.array([squared[idx].sum() for idx in cluster_screens]),
            "cells": np.array([len(idx) * observed.shape[1] for idx in cluster_screens]),
            "tv_sum": np.array([tv[idx].sum() for idx in cluster_screens]),
            "screenlines": np.array([len(idx) for idx in cluster_screens]),
        }

    rows: list[dict[str, float | int | str]] = []
    for baseline in ["direct_survey", "noncontrastive_vae", "bayesian_network"]:
        base = stats[baseline]
        cv = stats["contrastive_vae"]
        base_rmse = np.sqrt(base["sse"][draws].sum(axis=1) / base["cells"][draws].sum(axis=1))
        cv_rmse = np.sqrt(cv["sse"][draws].sum(axis=1) / cv["cells"][draws].sum(axis=1))
        rmse_delta = cv_rmse - base_rmse
        base_tv = base["tv_sum"][draws].sum(axis=1) / base["screenlines"][draws].sum(axis=1)
        cv_tv = cv["tv_sum"][draws].sum(axis=1) / cv["screenlines"][draws].sum(axis=1)
        tv_reduction = 100.0 * (1.0 - cv_tv / base_tv)
        cv_point_rmse = float(np.sqrt(np.mean((predictions["contrastive_vae"] - observed) ** 2)))
        base_point_rmse = float(np.sqrt(np.mean((predictions[baseline] - observed) ** 2)))
        cv_point_tv = float(_profile_tv(observed, predictions["contrastive_vae"]).mean())
        base_point_tv = float(_profile_tv(observed, predictions[baseline]).mean())
        rows.extend(
            [
                {
                    "contrast": f"contrastive_vae_vs_{baseline}",
                    "metric": "rmse_difference_vehicles",
                    "estimate": cv_point_rmse - base_point_rmse,
                    "ci_low": float(np.quantile(rmse_delta, 0.025)),
                    "ci_high": float(np.quantile(rmse_delta, 0.975)),
                    "clusters": len(clusters),
                    "replicates": len(draws),
                },
                {
                    "contrast": f"contrastive_vae_vs_{baseline}",
                    "metric": "profile_tv_relative_reduction_pct",
                    "estimate": 100.0 * (1.0 - cv_point_tv / base_point_tv),
                    "ci_low": float(np.quantile(tv_reduction, 0.025)),
                    "ci_high": float(np.quantile(tv_reduction, 0.975)),
                    "clusters": len(clusters),
                    "replicates": len(draws),
                },
            ]
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "paired_hourly_effects.csv", index=False)
    return frame


def hourly_evidence() -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    observed = pd.read_parquet(RUN / "metrics" / "observed_hourly_screenline_counts.parquet")
    observed["trip_date"] = pd.to_datetime(observed["trip_date"]).dt.normalize()
    observed = observed[observed["trip_date"].dt.dayofweek < 5].copy()
    observed_profile = observed.groupby(["screenline_id", "hour"], as_index=False).agg(
        observed_count=("observed_count", "mean")
    )
    screenlines = sorted(observed_profile["screenline_id"].unique())
    common_index = pd.MultiIndex.from_product(
        [screenlines, range(24)], names=["screenline_id", "hour"]
    )
    observed_matrix = (
        observed_profile.set_index(["screenline_id", "hour"])
        .reindex(common_index, fill_value=0.0)["observed_count"]
        .to_numpy()
        .reshape(len(screenlines), 24)
    )

    survey, expanded = direct_survey_crossings()
    direct = expanded.groupby(["screenline_id", "hour"])["weight"].sum()
    matrices: dict[str, np.ndarray] = {
        "direct_survey": direct.reindex(common_index, fill_value=0.0)
        .to_numpy()
        .reshape(len(screenlines), 24)
    }
    for method in GENERATED_METHODS:
        synthetic = pd.read_parquet(
            RUN / "metrics" / f"{method}_hourly_virtual_screenline_counts.parquet"
        )
        synthetic["trip_date"] = pd.to_datetime(synthetic["trip_date"]).dt.normalize()
        synthetic = synthetic[synthetic["trip_date"].dt.dayofweek < 5].copy()
        synthetic["pooled_crossings"] = (
            synthetic["synthetic_count"] / synthetic["temporal_expansion_factor"]
        )
        pooled = synthetic.groupby(["screenline_id", "hour"])["pooled_crossings"].sum()
        matrices[method] = (
            pooled.reindex(common_index, fill_value=0.0)
            .to_numpy()
            .reshape(len(screenlines), 24)
        )
    return observed_matrix, matrices, screenlines


def annual_evidence() -> tuple[pd.Series, pd.DataFrame]:
    archived = pd.read_parquet(RUN / "metrics" / "aadt_annual_screenline_comparisons.parquet")
    observed = (
        archived.drop_duplicates("screenline_id")
        .set_index("screenline_id")["observed_count"]
        .sort_index()
    )
    predictions = archived.pivot(
        index="screenline_id", columns="method", values="synthetic_count"
    ).reindex(observed.index)
    _, expanded = direct_survey_crossings()
    direct = expanded.groupby("screenline_id")["weight"].sum()
    predictions["direct_survey"] = direct.reindex(observed.index, fill_value=0.0)
    return observed, predictions[EXTERNAL_METHODS]


def _profile_tv(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed_share = observed / observed.sum(axis=1, keepdims=True)
    totals = predicted.sum(axis=1)
    predicted_share = np.divide(
        predicted,
        totals[:, None],
        out=np.zeros_like(predicted),
        where=totals[:, None] > 0,
    )
    values = 0.5 * np.abs(predicted_share - observed_share).sum(axis=1)
    values[totals == 0] = 1.0
    return values


def compute_evidence() -> dict[str, object]:
    TABLES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    exact_summary, exact_contrasts, exact_long = exact_date_evidence()
    observed_hourly, hourly_matrices, hourly_screenlines = hourly_evidence()
    rng = np.random.default_rng(20260730)
    hourly_draws = rng.integers(
        0, len(hourly_screenlines), size=(10_000, len(hourly_screenlines))
    )
    hourly_rows: list[dict[str, float | int | str]] = []
    hourly_tv: dict[str, np.ndarray] = {}
    hourly_se: dict[str, np.ndarray] = {}
    for method in EXTERNAL_METHODS:
        predicted = hourly_matrices[method]
        error = predicted - observed_hourly
        squared = error**2
        tv = _profile_tv(observed_hourly, predicted)
        hourly_tv[method] = tv
        hourly_se[method] = squared
        rmse_draws = np.sqrt(squared[hourly_draws].mean(axis=(1, 2)))
        tv_draws = tv[hourly_draws].mean(axis=1)
        hourly_rows.append(
            {
                "method": method,
                "rmse": float(np.sqrt(squared.mean())),
                "rmse_ci_low": float(np.quantile(rmse_draws, 0.025)),
                "rmse_ci_high": float(np.quantile(rmse_draws, 0.975)),
                "mae": float(np.abs(error).mean()),
                "bias": float(error.mean()),
                "pearson_log1p": float(
                    np.corrcoef(
                        np.log1p(observed_hourly).ravel(), np.log1p(predicted).ravel()
                    )[0, 1]
                ),
                "profile_tv": float(tv.mean()),
                "profile_tv_ci_low": float(np.quantile(tv_draws, 0.025)),
                "profile_tv_ci_high": float(np.quantile(tv_draws, 0.975)),
                "screenlines": len(hourly_screenlines),
                "hours": 24,
                "cells": int(observed_hourly.size),
            }
        )
    hourly_summary = pd.DataFrame(hourly_rows).set_index("method")
    hourly_summary.to_csv(TABLES / "hourly_validation_results.csv")

    paired_rows: list[dict[str, float | str]] = []
    for baseline, label in [
        ("direct_survey", "contrastive_vae_vs_direct_survey"),
        ("noncontrastive_vae", "contrastive_vae_vs_noncontrastive_vae"),
    ]:
        improvement_draws = 1.0 - (
            hourly_tv["contrastive_vae"][hourly_draws].mean(axis=1)
            / hourly_tv[baseline][hourly_draws].mean(axis=1)
        )
        paired_rows.append(
            {
                "contrast": label,
                "metric": "profile_tv_relative_reduction_pct",
                "estimate": 100.0
                * (
                    1.0
                    - hourly_tv["contrastive_vae"].mean()
                    / hourly_tv[baseline].mean()
                ),
                "ci_low": 100.0 * float(np.quantile(improvement_draws, 0.025)),
                "ci_high": 100.0 * float(np.quantile(improvement_draws, 0.975)),
            }
        )
    rmse_delta_draws = np.sqrt(
        hourly_se["contrastive_vae"][hourly_draws].mean(axis=(1, 2))
    ) - np.sqrt(hourly_se["direct_survey"][hourly_draws].mean(axis=(1, 2)))
    paired_rows.append(
        {
            "contrast": "contrastive_vae_vs_direct_survey",
            "metric": "rmse_difference_vehicles",
            "estimate": hourly_summary.loc["contrastive_vae", "rmse"]
            - hourly_summary.loc["direct_survey", "rmse"],
            "ci_low": float(np.quantile(rmse_delta_draws, 0.025)),
            "ci_high": float(np.quantile(rmse_delta_draws, 0.975)),
        }
    )
    paired = pooled_station_cluster_contrasts(
        observed_hourly, hourly_matrices, hourly_screenlines
    )

    observed_annual, annual_predictions = annual_evidence()
    annual_error = annual_predictions.sub(observed_annual, axis=0)
    annual_abs = annual_error.abs()
    annual_sq = annual_error**2
    annual_draws = rng.integers(
        0, len(observed_annual), size=(10_000, len(observed_annual))
    )
    win_flags = annual_abs.eq(annual_abs.min(axis=1), axis=0)
    unique_win_flags = win_flags & win_flags.sum(axis=1).eq(1).to_numpy()[:, None]
    annual_ties = int(win_flags.sum(axis=1).gt(1).sum())
    annual_rows: list[dict[str, float | int | str]] = []
    direct_rmse_draws = np.sqrt(
        annual_sq["direct_survey"].to_numpy()[annual_draws].mean(axis=1)
    )
    for method in EXTERNAL_METHODS:
        rmse_draws = np.sqrt(annual_sq[method].to_numpy()[annual_draws].mean(axis=1))
        mae_draws = annual_abs[method].to_numpy()[annual_draws].mean(axis=1)
        annual_rows.append(
            {
                "method": method,
                "rmse": float(np.sqrt(annual_sq[method].mean())),
                "rmse_ci_low": float(np.quantile(rmse_draws, 0.025)),
                "rmse_ci_high": float(np.quantile(rmse_draws, 0.975)),
                "mae": float(annual_abs[method].mean()),
                "mae_ci_low": float(np.quantile(mae_draws, 0.025)),
                "mae_ci_high": float(np.quantile(mae_draws, 0.975)),
                "bias": float(annual_error[method].mean()),
                "pearson_log1p": float(
                    np.corrcoef(
                        np.log1p(observed_annual), np.log1p(annual_predictions[method])
                    )[0, 1]
                ),
                "wins": int(unique_win_flags[method].sum()),
                "win_pct": 100.0 * float(unique_win_flags[method].mean()),
                "median_absolute_error": float(annual_abs[method].median()),
                "screenlines": len(observed_annual),
                "rmse_delta_vs_direct": float(
                    np.sqrt(annual_sq[method].mean())
                    - np.sqrt(annual_sq["direct_survey"].mean())
                ),
                "rmse_delta_vs_direct_ci_low": float(
                    np.quantile(rmse_draws - direct_rmse_draws, 0.025)
                ),
                "rmse_delta_vs_direct_ci_high": float(
                    np.quantile(rmse_draws - direct_rmse_draws, 0.975)
                ),
            }
        )
    annual_summary = pd.DataFrame(annual_rows).set_index("method")
    annual_summary.to_csv(TABLES / "annual_validation_results.csv")
    annual_long = annual_abs.sub(annual_abs["direct_survey"], axis=0).reset_index().melt(
        id_vars="screenline_id",
        var_name="method",
        value_name="absolute_error_delta_vs_direct",
    )
    annual_long.to_csv(GENERATED / "annual_screenline_error_differences.csv", index=False)

    exact = pd.read_csv(RUN / "tables" / "aadt_hourly_validation_summary.csv").set_index("method")
    apparent_reduction = 100.0 * (
        1.0
        - exact.loc["contrastive_vae", "rmse"]
        / exact.loc["weighted_bootstrap", "rmse"]
    )

    internal = pd.read_csv(RUN / "tables" / "method_validation_summary.csv").set_index("method")
    privacy = pd.read_csv(RUN / "tables" / "privacy_summary.csv").set_index("method")
    internal.join(
        privacy[["exact_row_copy_rate", "duplicate_rate", "novel_od_pair_share"]]
    ).to_csv(TABLES / "internal_validation_results.csv")

    test_rows = []
    nc_marg = json.loads((RUN / "metrics" / "noncontrastive_vae_marginals.json").read_text())[
        "columns"
    ]
    cv_marg = json.loads((RUN / "metrics" / "contrastive_vae_marginals.json").read_text())[
        "columns"
    ]
    for label, kind, metric in [
        ("categorical", "categorical", "total_variation"),
        ("numeric", "numeric", "ks_statistic"),
    ]:
        keys = [key for key, value in nc_marg.items() if value["type"] == kind]
        base = np.array([nc_marg[key][metric] for key in keys])
        contrastive = np.array([cv_marg[key][metric] for key in keys])
        result = wilcoxon(contrastive, base, alternative="two-sided")
        test_rows.append(
            {
                "comparison": label,
                "n_pairs": len(keys),
                "contrastive_improved": int((contrastive < base).sum()),
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    nc_cross = json.loads(
        (RUN / "metrics" / "noncontrastive_vae_cross_marginals.json").read_text()
    )["crosses"]
    cv_cross = json.loads(
        (RUN / "metrics" / "contrastive_vae_cross_marginals.json").read_text()
    )["crosses"]
    keys = list(nc_cross)
    base = np.array([nc_cross[key]["total_variation"] for key in keys])
    contrastive = np.array([cv_cross[key]["total_variation"] for key in keys])
    result = wilcoxon(contrastive, base, alternative="two-sided")
    test_rows.append(
        {
            "comparison": "cross_marginal",
            "n_pairs": len(keys),
            "contrastive_improved": int((contrastive < base).sum()),
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        }
    )
    pd.DataFrame(test_rows).to_csv(TABLES / "internal_paired_tests.csv", index=False)

    return {
        "exact_summary": exact_summary,
        "exact_contrasts": exact_contrasts,
        "exact_long": exact_long,
        "hourly_summary": hourly_summary,
        "hourly_tv": hourly_tv,
        "hourly_se": hourly_se,
        "hourly_draws": hourly_draws,
        "hourly_screenlines": hourly_screenlines,
        "paired": paired,
        "annual_summary": annual_summary,
        "annual_abs": annual_abs,
        "win_flags": unique_win_flags,
        "annual_ties": annual_ties,
        "apparent_reduction": apparent_reduction,
        "exact": exact,
    }


def figure_hourly_granularity(evidence: dict[str, object]) -> None:
    exact = evidence["exact_summary"]
    pooled = evidence["hourly_summary"]
    methods = [
        "weighted_bootstrap",
        "bayesian_network",
        "noncontrastive_vae",
        "contrastive_vae",
    ]
    tick_labels = [
        "Survey-Sampled\nBaseline",
        "Bayesian\nNetwork",
        "Variational\nAutoencoder",
        "Contrastive\nVariational\nAutoencoder",
    ]

    fig = plt.figure(figsize=(7.15, 3.90))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.38)
    ax_exact = fig.add_subplot(grid[0])
    ax_profile = fig.add_subplot(grid[1])
    x = np.arange(len(methods))
    bars = ax_exact.bar(
        x,
        [exact.loc[method, "rmse"] / 1000.0 for method in methods],
        color=[COLORS[method] for method in methods],
        width=0.68,
    )
    for bar, method in zip(bars, methods):
        ax_exact.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.22,
            f"{exact.loc[method, 'rmse']/1000.0:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
        )
    ax_exact.set_xticks(x)
    ax_exact.set_xticklabels(tick_labels, fontsize=7.8)
    ax_exact.set_ylabel(
        "RMSE on common boundary–date–hour cells\n"
        "(thousand vehicles; lower is better)"
    )
    ax_exact.set_ylim(0, 15.4)
    clean_axes(ax_exact, "y")
    panel(ax_exact, "(A)")
    ax_exact.set_title("  Common-cell RMSE", loc="left", pad=8)

    profile_bars = ax_profile.bar(
        x,
        [pooled.loc[method, "profile_tv"] for method in methods],
        color=[COLORS[method] for method in methods],
        width=0.68,
    )
    for bar, method in zip(profile_bars, methods):
        ax_profile.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{pooled.loc[method, 'profile_tv']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
        )
    ax_profile.set_xticks(x)
    ax_profile.set_xticklabels(tick_labels, fontsize=7.8)
    ax_profile.set_ylabel(
        "Mean 24-hour profile total variation\n"
        "(lower is better)"
    )
    ax_profile.set_ylim(0, 0.345)
    clean_axes(ax_profile, "y")
    panel(ax_profile, "(B)")
    ax_profile.set_title("  Profile Shape", loc="left", pad=8)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.25, top=0.90)
    save(fig, "fig_05_hourly_granularity")

def figure_annual_validation(evidence: dict[str, object]) -> None:
    summary = evidence["annual_summary"]
    annual_abs = evidence["annual_abs"]
    fig, (ax_win, ax_delta) = plt.subplots(1, 2, figsize=(7.15, 3.65))
    x = np.arange(len(EXTERNAL_METHODS))
    bars = ax_win.bar(
        x,
        [summary.loc[method, "wins"] for method in EXTERNAL_METHODS],
        color=[COLORS[method] for method in EXTERNAL_METHODS],
        width=0.68,
    )
    ax_win.set_xticks(x)
    ax_win.set_xticklabels(["Direct", "Boot.", "BN", "NC-\nVAE", "C-\nVAE"], fontsize=9.0)
    ax_win.set_ylabel("Screenlines with minimum absolute error")
    ax_win.set_ylim(0, 980)
    clean_axes(ax_win, "y")
    panel(ax_win, "(A)")
    ax_win.set_title("  Unique local wins", loc="left", pad=8)
    for bar, method in zip(bars, EXTERNAL_METHODS):
        ax_win.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 18,
            f"{int(summary.loc[method, 'wins'])}\n({summary.loc[method, 'win_pct']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.0,
        )

    compare = GENERATED_METHODS
    positions = np.arange(len(compare))
    for idx, method in enumerate(compare):
        delta = annual_abs[method] - annual_abs["direct_survey"]
        q25, median, q75 = np.quantile(delta, [0.25, 0.50, 0.75]) / 1000.0
        ax_delta.errorbar(
            idx,
            median,
            yerr=np.array([[median - q25], [q75 - median]]),
            fmt="o",
            markersize=7.2,
            color=COLORS[method],
            ecolor=COLORS[method],
            elinewidth=2.0,
            capsize=4.0,
        )
        ax_delta.text(
            idx,
            q75 + 0.30,
            f"median {median:+.2f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
        )
    ax_delta.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax_delta.set_xticks(positions)
    ax_delta.set_xticklabels(["Boot.", "BN", "NC-\nVAE", "C-\nVAE"], fontsize=9.0)
    ax_delta.set_ylabel("Absolute-error delta vs. direct\n(thousand vehicles)")
    ax_delta.set_ylim(-4.2, 6.2)
    clean_axes(ax_delta, "y")
    panel(ax_delta, "(B)")
    ax_delta.set_title("  Paired error magnitude (IQR)", loc="left", pad=8)
    ax_delta.text(
        0.02,
        0.04,
        "Negative values favor the method.\nContrastive has most wins but worst RMSE.",
        transform=ax_delta.transAxes,
        fontsize=9.0,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.28, top=0.90, wspace=0.42)
    save(fig, "fig_06_annual_validation")


def write_generated_tables(evidence: dict[str, object]) -> None:
    survey, _ = direct_survey_crossings()
    dates = pd.Timestamp("2017-01-01") + pd.to_timedelta(
        pd.to_numeric(survey["tdate_week"]) * 7
        + np.mod(pd.to_numeric(survey["tdate_dow"]), 7),
        unit="D",
    )
    valid_od = (survey["o_tract_fips"] != "-1") & (survey["d_tract_fips"] != "-1")
    survey_pairs = survey.loc[valid_od, ["o_tract_fips", "d_tract_fips"]].drop_duplicates()
    survey_rows = [
        ("Analytical unit", "Retained vehicle-mode-coded trip row"),
        ("Rows and variables", f"{len(survey):,} rows; 44 synthesized fields plus weight"),
        ("Field types", "26 categorical; 17 ordinal integer; 1 continuous"),
        ("Survey date strata", f"{dates.nunique():,} dates, {dates.min().date()}--{dates.max().date()}"),
        ("Valid directed OD support", f"{len(survey_pairs):,} tract pairs from 110,824 rows"),
        ("Retained weight total", f"{survey['weight'].sum():,.3f} household-final-weight units"),
        ("VAE development split", "88,741 training; 22,185 validation rows"),
        ("Distance, median (IQR)", "4.102 (1.640--10.163) miles"),
        ("Reported time, median (IQR)", "15 (10--30) minutes"),
    ]
    pd.DataFrame(survey_rows, columns=["attribute", "value"]).to_csv(
        TABLES / "survey_data_summary.csv", index=False
    )
    survey_tex = [
        r"\begin{table}[!htbp]",
        r"\caption{Travel-survey analytical table and synthesis task.}",
        r"\label{tab:surveydata}",
        r"\centering",
        r"\begin{tabularx}{\textwidth}{P{0.32\textwidth}Y}",
        r"\toprule",
        r"\textbf{Attribute} & \textbf{Verified value} \\",
        r"\midrule",
    ]
    survey_tex.extend(f"{a} & {v} \\\\" for a, v in survey_rows)
    survey_tex.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    (TABLES / "survey_data_summary.tex").write_text("\n".join(survey_tex) + "\n")

    hourly = pd.read_csv(ROOT / "data" / "external" / "mdot_daily_counts" / "station_hourly_counts.csv")
    screen = json.loads((RUN / "geo" / "screenline_summary.json").read_text())
    traffic_rows = [
        ("Annual target", "2018 MDOT SHA annual average weekday traffic"),
        ("Spatial support", f"{screen['screenlines_with_stations']:,} station-mapped tract boundaries"),
        ("Tracts/candidate boundaries", f"{screen['tracts']:,}/{screen['screenlines']:,}"),
        ("Hourly target", "FHWA TMAS station-hour counts, 2017-10-01--2019-07-31"),
        ("Downloaded hourly coverage", f"{hourly['station_id'].nunique()} stations; {hourly['date'].nunique()} dates; {len(hourly):,} rows"),
        ("Mapped hourly support", "37 stations; 56 boundaries; 1,344 date-pooled cells"),
        ("Hourly count, median (IQR)", f"{hourly['observed_count'].median():,.0f} ({hourly['observed_count'].quantile(.25):,.0f}--{hourly['observed_count'].quantile(.75):,.0f}) vehicles"),
        ("Role in experiment", "Not read by implemented training or checkpoint code; human selection history unavailable"),
    ]
    pd.DataFrame(traffic_rows, columns=["attribute", "value"]).to_csv(
        TABLES / "traffic_count_summary.csv", index=False
    )
    traffic_tex = [
        r"\begin{table}[!htbp]",
        r"\caption{Independent roadway-count data and evaluation support.}",
        r"\label{tab:trafficdata}",
        r"\centering",
        r"\begin{tabularx}{\textwidth}{P{0.32\textwidth}Y}",
        r"\toprule",
        r"\textbf{Attribute} & \textbf{Verified value} \\",
        r"\midrule",
    ]
    traffic_tex.extend(f"{a} & {v} \\\\" for a, v in traffic_rows)
    traffic_tex.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    (TABLES / "traffic_count_summary.tex").write_text("\n".join(traffic_tex) + "\n")

    internal = pd.read_csv(TABLES / "internal_validation_results.csv").set_index("method")
    internal_lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Internal fidelity, copying, and unseen-support diagnostics. Lower is better for TV and KS.}",
        r"\label{tab:internal}",
        r"\centering",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabularx}{\textwidth}{Y r r r r r}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Cat. TV} & \textbf{Num. KS} & \textbf{Cross TV} & \textbf{Exact copy} & \textbf{Unseen-OD rows} \\",
        r"\midrule",
    ]
    for method in GENERATED_METHODS:
        row = internal.loc[method]
        internal_lines.append(
            f"{SHORT[method]} & {row['mean_categorical_tv']:.4f} & {row['mean_numeric_ks']:.4f} & "
            f"{row['mean_cross_tv']:.4f} & {row['exact_row_copy_rate']:.4f} & {row['novel_od_pair_share']:.4f} \\\\"
        )
    internal_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    (TABLES / "internal_results.tex").write_text("\n".join(internal_lines) + "\n")

    annual = evidence["annual_summary"]
    exact_summary = evidence["exact_summary"]
    hourly_summary = evidence["hourly_summary"]
    external_lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Hourly traffic-count diagnostics. Exact-date results use 521,424 common cells and a shared 423-date factor. Pooled results use 1,344 common boundary--hour cells. RMSE, MAE, and bias are vehicles per cell.}",
        r"\label{tab:external}",
        r"\centering",
        r"\setlength{\tabcolsep}{2.1pt}",
        r"\begin{tabularx}{\textwidth}{Y r r r r r r}",
        r"\toprule",
        r"\textbf{Method} & \makecell{\textbf{Exact}\\\textbf{RMSE}} & \makecell{\textbf{Pooled}\\\textbf{RMSE}} & \makecell{\textbf{Pooled}\\\textbf{MAE}} & \makecell{\textbf{Pooled}\\\textbf{bias}} & \makecell{\textbf{Log}\\$\boldsymbol{r}$} & \makecell{\textbf{Profile}\\\textbf{TV}} \\",
        r"\midrule",
    ]
    for method in EXTERNAL_METHODS:
        external_lines.append(
            f"{SHORT[method]} & {exact_summary.loc[method, 'rmse']:,.0f} & "
            f"{hourly_summary.loc[method, 'rmse']:,.0f} & "
            f"{hourly_summary.loc[method, 'mae']:,.0f} & "
            f"{hourly_summary.loc[method, 'bias']:,.0f} & "
            f"{hourly_summary.loc[method, 'pearson_log1p']:.3f} & "
            f"{hourly_summary.loc[method, 'profile_tv']:.3f}" + r" \\"
        )
    external_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    (TABLES / "external_results.tex").write_text("\n".join(external_lines) + "\n")

    annual_lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Annual weekday diagnostics on 3,107 common boundaries. Errors are vehicles per boundary. Unique wins exclude one five-way tie.}",
        r"\label{tab:annual}",
        r"\centering",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabularx}{\textwidth}{Y r r r r r}",
        r"\toprule",
        r"\textbf{Method} & \textbf{RMSE} & \textbf{MAE} & \textbf{Bias} & \makecell{\textbf{Log}\\$\boldsymbol{r}$} & \textbf{Unique wins} \\",
        r"\midrule",
    ]
    for method in EXTERNAL_METHODS:
        annual_lines.append(
            f"{SHORT[method]} & {annual.loc[method, 'rmse']:,.0f} & "
            f"{annual.loc[method, 'mae']:,.0f} & {annual.loc[method, 'bias']:,.0f} & "
            f"{annual.loc[method, 'pearson_log1p']:.3f} & {int(annual.loc[method, 'wins'])}" + r" \\"
        )
    annual_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    (TABLES / "annual_results.tex").write_text("\n".join(annual_lines) + "\n")

    paired = evidence["paired"]
    exact_summary = evidence["exact_summary"]
    exact_contrasts = evidence["exact_contrasts"]
    exact_direct = exact_contrasts[
        exact_contrasts["contrast"].eq("contrastive_vae_vs_direct_survey")
    ].iloc[0]
    direct_profile = paired[
        (paired["contrast"] == "contrastive_vae_vs_direct_survey")
        & (paired["metric"] == "profile_tv_relative_reduction_pct")
    ].iloc[0]
    ablation_profile = paired[
        (paired["contrast"] == "contrastive_vae_vs_noncontrastive_vae")
        & (paired["metric"] == "profile_tv_relative_reduction_pct")
    ].iloc[0]
    rmse_delta = paired[paired["metric"] == "rmse_difference_vehicles"].iloc[0]
    macros = {
        "SurveyRows": f"{len(survey):,}",
        "SurveyFields": "44",
        "WeightedTripTotal": f"{survey['weight'].sum():,.3f}",
        "HourlyCells": "1,344",
        "HourlyScreenlines": "56",
        "AnnualScreenlines": "3,107",
        "ApparentHourlyReduction": f"{evidence['apparent_reduction']:.1f}\\%",
        "ExactHourlyCells": f"{int(exact_summary.loc['contrastive_vae', 'cells']):,}",
        "ExactHourlyDates": f"{int(exact_summary.loc['contrastive_vae', 'dates']):,}",
        "ExactDirectRMSE": f"{exact_summary.loc['direct_survey', 'rmse']:,.0f}",
        "ExactCVAERMSE": f"{exact_summary.loc['contrastive_vae', 'rmse']:,.0f}",
        "ExactNCVAERMSE": f"{exact_summary.loc['noncontrastive_vae', 'rmse']:,.0f}",
        "ExactReductionDirect": f"{exact_direct['estimate']:.1f}\\%",
        "ExactReductionDirectLow": f"{exact_direct['ci_low']:.1f}\\%",
        "ExactReductionDirectHigh": f"{exact_direct['ci_high']:.1f}\\%",
        "CVAEProfileReductionDirect": f"{direct_profile['estimate']:.1f}\\%",
        "CVAEProfileReductionDirectLow": f"{direct_profile['ci_low']:.1f}\\%",
        "CVAEProfileReductionDirectHigh": f"{direct_profile['ci_high']:.1f}\\%",
        "CVAEProfileReductionAblation": f"{ablation_profile['estimate']:.1f}\\%",
        "CVAEProfileReductionAblationLow": f"{ablation_profile['ci_low']:.1f}\\%",
        "CVAEProfileReductionAblationHigh": f"{ablation_profile['ci_high']:.1f}\\%",
        "DirectHourlyRMSE": f"{hourly_summary.loc['direct_survey', 'rmse']:,.0f}",
        "CVAEHourlyRMSE": f"{hourly_summary.loc['contrastive_vae', 'rmse']:,.0f}",
        "BNHourlyRMSE": f"{hourly_summary.loc['bayesian_network', 'rmse']:,.0f}",
        "CVAEDirectRMSEDelta": f"{rmse_delta['estimate']:,.0f}",
        "CVAEDirectRMSEDeltaLow": f"{rmse_delta['ci_low']:,.0f}",
        "CVAEDirectRMSEDeltaHigh": f"{rmse_delta['ci_high']:,.0f}",
        "CVAEAnnualWins": f"{int(annual.loc['contrastive_vae', 'wins'])}",
        "CVAEAnnualWinPct": f"{annual.loc['contrastive_vae', 'win_pct']:.1f}\\%",
        "BNAnnualRMSE": f"{annual.loc['bayesian_network', 'rmse']:,.0f}",
        "CVAEAnnualRMSE": f"{annual.loc['contrastive_vae', 'rmse']:,.0f}",
    }
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    (ROOT / "trb" / "generated_results.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = compute_evidence()
    write_generated_tables(evidence)
    figure_workflow()
    figure_architecture()
    figure_internal_tradeoff()
    figure_screenline_method()
    figure_hourly_granularity(evidence)
    figure_annual_validation(evidence)


if __name__ == "__main__":
    main()

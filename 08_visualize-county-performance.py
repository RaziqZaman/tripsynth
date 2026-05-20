#!/usr/bin/env python3
"""Visualize county-level real-vs-synthetic MATSim validation performance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = Path("08x_county_runs")
OUT_DIR = OUT_ROOT / "performance_visualizations"

LOWER_IS_BETTER = {"mae", "rmse", "mape", "abs_total_error_share"}
HIGHER_IS_BETTER = {"correlation", "geh_lt_5_share"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"].eq("completed")].copy()
    numeric_columns = [
        "observed_counts",
        "network_base_links",
        "real_vehicle_trips",
        "synthetic_vehicle_trips",
        "matched_counts",
        "observed_total",
        "model_total",
        "mean_bias",
        "mae",
        "rmse",
        "mape",
        "correlation",
        "geh_lt_5_share",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["model_observed_ratio"] = df["model_total"] / df["observed_total"]
    df["abs_total_error_share"] = (df["model_total"] - df["observed_total"]).abs() / df[
        "observed_total"
    ]
    return df


def county_order(df: pd.DataFrame) -> list[str]:
    totals = (
        df.groupby("county")["observed_total"]
        .max()
        .sort_values(ascending=False)
    )
    return totals.index.tolist()


def plot_metric_bars(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    order = county_order(df)
    pivot = (
        df.pivot_table(index="county", columns="scenario", values=metric, aggfunc="first")
        .reindex(order)
    )
    ax = pivot[["real", "synthetic"]].plot(
        kind="bar",
        figsize=(12, 5.8),
        color=["#2f6f9f", "#d8892b"],
        width=0.78,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    savefig(out_path)


def metric_winner(row_real: pd.Series, row_synthetic: pd.Series, metric: str) -> str:
    real = row_real[metric]
    synthetic = row_synthetic[metric]
    if pd.isna(real) or pd.isna(synthetic):
        return "missing"
    if metric in LOWER_IS_BETTER:
        if np.isclose(real, synthetic):
            return "tie"
        return "real" if real < synthetic else "synthetic"
    if metric in HIGHER_IS_BETTER:
        if np.isclose(real, synthetic):
            return "tie"
        return "real" if real > synthetic else "synthetic"
    raise ValueError(metric)


def write_winners(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metrics = [
        "mae",
        "rmse",
        "mape",
        "correlation",
        "geh_lt_5_share",
        "abs_total_error_share",
    ]
    rows = []
    for county, group in df.groupby("county", sort=False):
        by_scenario = {row["scenario"]: row for _, row in group.iterrows()}
        if "real" not in by_scenario or "synthetic" not in by_scenario:
            continue
        row = {"county": county}
        for metric in metrics:
            row[f"{metric}_winner"] = metric_winner(
                by_scenario["real"],
                by_scenario["synthetic"],
                metric,
            )
            row[f"{metric}_real"] = by_scenario["real"][metric]
            row[f"{metric}_synthetic"] = by_scenario["synthetic"][metric]
            row[f"{metric}_synthetic_minus_real"] = (
                by_scenario["synthetic"][metric] - by_scenario["real"][metric]
            )
        rows.append(row)
    winners = pd.DataFrame(rows)
    winners.to_csv(out_dir / "county_metric_winners.csv", index=False)
    return winners


def plot_winner_heatmap(winners: pd.DataFrame, out_path: Path) -> None:
    metric_columns = [
        "mae_winner",
        "rmse_winner",
        "mape_winner",
        "correlation_winner",
        "geh_lt_5_share_winner",
        "abs_total_error_share_winner",
    ]
    labels = {
        "mae_winner": "MAE",
        "rmse_winner": "RMSE",
        "mape_winner": "MAPE",
        "correlation_winner": "Correlation",
        "geh_lt_5_share_winner": "GEH < 5",
        "abs_total_error_share_winner": "Total volume",
    }
    code = {"real": 0, "tie": 1, "synthetic": 2, "missing": np.nan}
    matrix = np.array(
        [
            [code.get(value, np.nan) for value in winners[column]]
            for column in metric_columns
        ],
        dtype=float,
    ).T
    cmap = plt.matplotlib.colors.ListedColormap(["#2f6f9f", "#888888", "#d8892b"])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(winners))))
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(metric_columns)))
    ax.set_xticklabels([labels[column] for column in metric_columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(winners)))
    ax.set_yticklabels(winners["county"])
    ax.set_title("Metric Winner By County")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = winners.iloc[y][metric_columns[x]].replace("synthetic", "synth")
            ax.text(x, y, value, ha="center", va="center", color="white", fontsize=8)
    savefig(out_path)


def plot_delta_bars(
    winners: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    column = f"{metric}_synthetic_minus_real"
    data = winners.sort_values(column)
    colors = np.where(data[column] < 0, "#d8892b", "#2f6f9f")
    plt.figure(figsize=(10, 5.5))
    plt.barh(data["county"], data[column], color=colors)
    plt.axvline(0, color="#444444", linewidth=1)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    savefig(out_path)


def read_count_validations(out_root: Path, completed_counties: list[str]) -> pd.DataFrame:
    frames = []
    for county in completed_counties:
        slug = (
            county.lower()
            .replace("'", "")
            .replace(".", "")
            .replace(" ", "_")
            .replace("-", "_")
        )
        path = out_root / slug / "count_validation.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["county"] = county
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    validation = pd.concat(frames, ignore_index=True)
    for column in ["observed_volume", "model_volume", "absolute_error", "geh"]:
        validation[column] = pd.to_numeric(validation[column], errors="coerce")
    return validation


def plot_pooled_scatter(validation: pd.DataFrame, out_path: Path) -> None:
    if validation.empty:
        return
    plt.figure(figsize=(7, 7))
    for scenario, color in [("real", "#2f6f9f"), ("synthetic", "#d8892b")]:
        group = validation[validation["scenario"].eq(scenario)]
        plt.scatter(
            group["observed_volume"],
            group["model_volume"],
            s=9,
            alpha=0.35,
            label=scenario,
            color=color,
        )
    limit = float(np.nanmax(validation[["observed_volume", "model_volume"]].to_numpy()))
    plt.plot([0, limit], [0, limit], color="#444444", linewidth=1)
    plt.xlabel("Observed volume")
    plt.ylabel("Modeled volume")
    plt.title("Pooled County Count Links: Modeled vs Observed")
    plt.legend()
    plt.grid(alpha=0.2)
    savefig(out_path)


def plot_pooled_geh(validation: pd.DataFrame, out_path: Path) -> None:
    if validation.empty:
        return
    cap = min(80.0, float(validation["geh"].quantile(0.99)))
    bins = np.linspace(0, cap, 60)
    plt.figure(figsize=(9, 5))
    for scenario, color in [("real", "#2f6f9f"), ("synthetic", "#d8892b")]:
        group = validation[validation["scenario"].eq(scenario)]
        plt.hist(group["geh"].dropna(), bins=bins, alpha=0.55, density=True, label=scenario, color=color)
    plt.xlabel("GEH")
    plt.ylabel("Density")
    plt.title("Pooled GEH Distribution Across Completed Counties")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    savefig(out_path)


def write_readme(out_dir: Path, winners: pd.DataFrame) -> None:
    winner_cols = [column for column in winners.columns if column.endswith("_winner")]
    counts = []
    for column in winner_cols:
        metric = column.removesuffix("_winner")
        summary = winners[column].value_counts().to_dict()
        counts.append(
            {
                "metric": metric,
                "real_wins": summary.get("real", 0),
                "synthetic_wins": summary.get("synthetic", 0),
                "ties": summary.get("tie", 0),
            }
        )
    with (out_dir / "metric_winner_counts.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["metric", "real_wins", "synthetic_wins", "ties"])
        writer.writeheader()
        writer.writerows(counts)
    (out_dir / "README.md").write_text(
        "# County Performance Visualizations\n\n"
        "Lower is better for MAE, RMSE, MAPE, and total-volume error. "
        "Higher is better for correlation and GEH<5 share. "
        "The winner heatmap marks which input source performed better for each metric and county.\n"
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot()

    df = read_summary(args.out_root / "county_validation_summary.csv")
    if df.empty:
        raise SystemExit("No completed county validation rows found")

    enriched_path = args.out_dir / "county_performance_metrics.csv"
    df.to_csv(enriched_path, index=False)

    metrics = [
        ("mae", "Mean Absolute Error By County", "MAE"),
        ("rmse", "RMSE By County", "RMSE"),
        ("mape", "MAPE By County", "MAPE"),
        ("correlation", "Correlation By County", "Pearson correlation"),
        ("geh_lt_5_share", "Share Of Count Links With GEH < 5", "Share"),
        ("model_observed_ratio", "Modeled / Observed Total Volume", "ratio"),
        ("abs_total_error_share", "Absolute Total Volume Error Share", "abs(model - observed) / observed"),
    ]
    for metric, title, ylabel in metrics:
        plot_metric_bars(
            df,
            metric,
            title,
            ylabel,
            args.out_dir / f"{metric}_by_county.png",
        )

    winners = write_winners(df, args.out_dir)
    plot_winner_heatmap(winners, args.out_dir / "metric_winner_heatmap.png")
    plot_delta_bars(
        winners,
        "mae",
        "Synthetic Minus Real MAE By County",
        "synthetic MAE - real MAE",
        args.out_dir / "mae_delta_synthetic_minus_real.png",
    )
    plot_delta_bars(
        winners,
        "correlation",
        "Synthetic Minus Real Correlation By County",
        "synthetic correlation - real correlation",
        args.out_dir / "correlation_delta_synthetic_minus_real.png",
    )
    plot_delta_bars(
        winners,
        "abs_total_error_share",
        "Synthetic Minus Real Total Volume Error Share",
        "synthetic error share - real error share",
        args.out_dir / "total_volume_error_delta_synthetic_minus_real.png",
    )

    completed_counties = sorted(df["county"].unique().tolist())
    validation = read_count_validations(args.out_root, completed_counties)
    if not validation.empty:
        validation.to_csv(args.out_dir / "pooled_count_validation.csv", index=False)
        plot_pooled_scatter(validation, args.out_dir / "pooled_assigned_vs_observed_scatter.png")
        plot_pooled_geh(validation, args.out_dir / "pooled_geh_distribution.png")

    write_readme(args.out_dir, winners)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

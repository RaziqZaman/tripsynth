#!/usr/bin/env python3
"""Spatial-pattern statistical tests for county MATSim validations."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = Path("08x_county_runs")
OUT_DIR = OUT_ROOT / "spatial_pattern_tests"
RNG_SEED = 20260520


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutation-samples", type=int, default=1000)
    parser.add_argument("--moran-permutations", type=int, default=199)
    parser.add_argument("--pair-samples", type=int, default=40_000)
    parser.add_argument("--top-share", type=float, default=0.10)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
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


def slug_for_county(county: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", county.lower()).strip("_")


def completed_counties(out_root: Path) -> list[str]:
    summary = pd.read_csv(out_root / "county_validation_summary.csv")
    return sorted(summary.loc[summary["status"].eq("completed"), "county"].unique())


def load_county_data(out_root: Path, county: str) -> pd.DataFrame:
    slug = slug_for_county(county)
    validation = pd.read_csv(out_root / slug / "count_validation.csv")
    observed = pd.read_csv(out_root / slug / "observed_counts.csv")
    coords = observed[["link_id", "lon", "lat"]].drop_duplicates("link_id")
    data = validation.merge(coords, on="link_id", how="left")
    for column in ["observed_volume", "model_volume", "error", "lon", "lat"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["observed_volume", "model_volume", "lon", "lat"])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = math.sqrt(float(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method="average").to_numpy(dtype=float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def top_jaccard(observed: np.ndarray, model: np.ndarray, top_share: float) -> float:
    count = max(1, int(round(len(observed) * top_share)))
    observed_top = set(np.argpartition(observed, -count)[-count:].tolist())
    model_top = set(np.argpartition(model, -count)[-count:].tolist())
    union = observed_top | model_top
    return len(observed_top & model_top) / len(union) if union else float("nan")


def top_capture(observed: np.ndarray, model: np.ndarray, top_share: float) -> float:
    count = max(1, int(round(len(observed) * top_share)))
    observed_top = set(np.argpartition(observed, -count)[-count:].tolist())
    model_top = set(np.argpartition(model, -count)[-count:].tolist())
    return len(observed_top & model_top) / len(observed_top) if observed_top else float("nan")


def pairwise_concordance(
    observed: np.ndarray,
    model: np.ndarray,
    rng: np.random.Generator,
    pair_samples: int,
) -> float:
    n = len(observed)
    if n < 2:
        return float("nan")
    left = rng.integers(0, n, size=pair_samples)
    right = rng.integers(0, n, size=pair_samples)
    keep = left != right
    left = left[keep]
    right = right[keep]
    observed_diff = observed[left] - observed[right]
    model_diff = model[left] - model[right]
    keep = (observed_diff != 0) & (model_diff != 0)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.sign(observed_diff[keep]) == np.sign(model_diff[keep])))


def haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    radius_m = 6_371_000.0
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    dlon = lon_rad[:, None] - lon_rad[None, :]
    dlat = lat_rad[:, None] - lat_rad[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, None])
        * np.cos(lat_rad[None, :])
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_m * np.arcsin(np.sqrt(a))


def knn_indices(lon: np.ndarray, lat: np.ndarray, k: int) -> np.ndarray:
    distances = haversine_matrix(lon, lat)
    np.fill_diagonal(distances, np.inf)
    k = min(k, max(1, len(lon) - 1))
    return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]


def moran_i(values: np.ndarray, neighbors: np.ndarray) -> float:
    if len(values) < 3:
        return float("nan")
    centered = values - np.mean(values)
    denom = float(np.sum(centered**2))
    if denom == 0:
        return float("nan")
    neighbor_values = centered[neighbors]
    numerator = float(np.sum(centered[:, None] * neighbor_values / neighbors.shape[1]))
    return numerator / denom


def moran_permutation_p(
    values: np.ndarray,
    neighbors: np.ndarray,
    observed_i: float,
    rng: np.random.Generator,
    permutations: int,
) -> float:
    if pd.isna(observed_i):
        return float("nan")
    hits = 0
    for _ in range(permutations):
        permuted = rng.permutation(values)
        value = moran_i(permuted, neighbors)
        if abs(value) >= abs(observed_i):
            hits += 1
    return (hits + 1) / (permutations + 1)


def metrics(
    observed: np.ndarray,
    model: np.ndarray,
    rng: np.random.Generator,
    top_share: float,
    pair_samples: int,
) -> dict[str, float]:
    return {
        "pearson": pearson(observed, model),
        "spearman": spearman(observed, model),
        "log_pearson": pearson(np.log1p(observed), np.log1p(model)),
        "top_jaccard": top_jaccard(observed, model, top_share),
        "top_capture": top_capture(observed, model, top_share),
        "pairwise_concordance": pairwise_concordance(
            observed,
            model,
            rng,
            pair_samples,
        ),
    }


def paired_metric_diffs(
    observed: np.ndarray,
    real: np.ndarray,
    synthetic: np.ndarray,
    rng: np.random.Generator,
    top_share: float,
    pair_samples: int,
) -> dict[str, float]:
    real_metrics = metrics(observed, real, rng, top_share, pair_samples)
    synthetic_metrics = metrics(observed, synthetic, rng, top_share, pair_samples)
    return {
        metric: synthetic_metrics[metric] - real_metrics[metric]
        for metric in real_metrics
    }


def bootstrap_tests(
    observed: np.ndarray,
    real: np.ndarray,
    synthetic: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    top_share: float,
    pair_samples: int,
) -> dict[str, dict[str, float]]:
    n = len(observed)
    original = paired_metric_diffs(
        observed,
        real,
        synthetic,
        rng,
        top_share,
        pair_samples,
    )
    draws = {metric: [] for metric in original}
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        diff = paired_metric_diffs(
            observed[indices],
            real[indices],
            synthetic[indices],
            rng,
            top_share,
            max(5_000, min(pair_samples, len(indices) * 100)),
        )
        for metric, value in diff.items():
            if not pd.isna(value):
                draws[metric].append(value)
    output: dict[str, dict[str, float]] = {}
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        output[metric] = {
            "diff_synthetic_minus_real": original[metric],
            "bootstrap_ci_low": float(np.quantile(array, 0.025)) if len(array) else float("nan"),
            "bootstrap_ci_high": float(np.quantile(array, 0.975)) if len(array) else float("nan"),
            "bootstrap_p_two_sided": float(
                min(1.0, 2.0 * min(np.mean(array <= 0), np.mean(array >= 0)))
            )
            if len(array)
            else float("nan"),
        }
    return output


def permutation_tests(
    observed: np.ndarray,
    real: np.ndarray,
    synthetic: np.ndarray,
    original_diffs: dict[str, float],
    rng: np.random.Generator,
    samples: int,
    top_share: float,
    pair_samples: int,
) -> dict[str, float]:
    hits = {metric: 0 for metric in original_diffs}
    usable = {metric: 0 for metric in original_diffs}
    n = len(observed)
    for _ in range(samples):
        swap = rng.random(n) < 0.5
        perm_real = real.copy()
        perm_synthetic = synthetic.copy()
        perm_real[swap] = synthetic[swap]
        perm_synthetic[swap] = real[swap]
        diff = paired_metric_diffs(
            observed,
            perm_real,
            perm_synthetic,
            rng,
            top_share,
            pair_samples,
        )
        for metric, value in diff.items():
            if pd.isna(value) or pd.isna(original_diffs[metric]):
                continue
            usable[metric] += 1
            if abs(value) >= abs(original_diffs[metric]):
                hits[metric] += 1
    return {
        metric: (hits[metric] + 1) / (usable[metric] + 1) if usable[metric] else float("nan")
        for metric in original_diffs
    }


def scenario_arrays(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pivot = data.pivot_table(
        index=["link_id", "lon", "lat"],
        columns="scenario",
        values=["observed_volume", "model_volume", "error"],
        aggfunc="first",
    ).dropna()
    observed = pivot[("observed_volume", "real")].to_numpy(dtype=float)
    real = pivot[("model_volume", "real")].to_numpy(dtype=float)
    synthetic = pivot[("model_volume", "synthetic")].to_numpy(dtype=float)
    real_error = pivot[("error", "real")].to_numpy(dtype=float)
    synthetic_error = pivot[("error", "synthetic")].to_numpy(dtype=float)
    lon = pivot.index.get_level_values("lon").to_numpy(dtype=float)
    lat = pivot.index.get_level_values("lat").to_numpy(dtype=float)
    coords = np.column_stack([lon, lat])
    return observed, real, synthetic, real_error, synthetic_error, coords


def write_metric_plots(metric_tests: pd.DataFrame, out_dir: Path) -> None:
    for metric in [
        "pearson",
        "spearman",
        "log_pearson",
        "top_jaccard",
        "top_capture",
        "pairwise_concordance",
    ]:
        data = metric_tests[metric_tests["metric"].eq(metric)].copy()
        data = data.sort_values("diff_synthetic_minus_real")
        colors = np.where(data["diff_synthetic_minus_real"] > 0, "#d8892b", "#2f6f9f")
        plt.figure(figsize=(10, 5.5))
        plt.barh(data["county"], data["diff_synthetic_minus_real"], color=colors)
        plt.axvline(0, color="#444444", linewidth=1)
        plt.xlabel("synthetic minus real")
        plt.title(f"Paired Spatial Pattern Difference: {metric}")
        plt.grid(axis="x", alpha=0.25)
        savefig(out_dir / f"{metric}_diff_by_county.png")


def write_scenario_metric_plots(scenario_metrics: pd.DataFrame, out_dir: Path) -> None:
    for metric in [
        "pearson",
        "spearman",
        "log_pearson",
        "top_jaccard",
        "top_capture",
        "pairwise_concordance",
    ]:
        pivot = scenario_metrics.pivot_table(
            index="county",
            columns="scenario",
            values=metric,
            aggfunc="first",
        )
        pivot = pivot.sort_values("real", ascending=False)
        ax = pivot[["real", "synthetic"]].plot(
            kind="bar",
            figsize=(11, 5.5),
            color=["#2f6f9f", "#d8892b"],
        )
        ax.set_title(f"{metric} by county")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=45, ha="right")
        savefig(out_dir / f"{metric}_by_county.png")


def write_moran_plots(moran: pd.DataFrame, out_dir: Path) -> None:
    pivot = moran.pivot_table(
        index="county",
        columns="scenario",
        values="moran_i_residual",
        aggfunc="first",
    ).sort_values("real")
    ax = pivot[["real", "synthetic"]].plot(
        kind="bar",
        figsize=(11, 5.5),
        color=["#2f6f9f", "#d8892b"],
    )
    ax.set_title("Residual Moran's I by county")
    ax.set_xlabel("")
    ax.set_ylabel("Moran's I of residuals")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    savefig(out_dir / "residual_morans_i_by_county.png")

    diff = moran.pivot_table(
        index="county",
        columns="scenario",
        values="abs_moran_i_residual",
        aggfunc="first",
    )
    diff["synthetic_minus_real_abs_moran"] = diff["synthetic"] - diff["real"]
    diff = diff.sort_values("synthetic_minus_real_abs_moran")
    colors = np.where(diff["synthetic_minus_real_abs_moran"] < 0, "#d8892b", "#2f6f9f")
    plt.figure(figsize=(10, 5.5))
    plt.barh(diff.index, diff["synthetic_minus_real_abs_moran"], color=colors)
    plt.axvline(0, color="#444444", linewidth=1)
    plt.xlabel("synthetic abs(Moran's I) - real abs(Moran's I)")
    plt.title("Residual Spatial Clustering Difference")
    plt.grid(axis="x", alpha=0.25)
    savefig(out_dir / "abs_residual_morans_i_diff.png")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot()
    rng = np.random.default_rng(args.seed)

    scenario_rows = []
    test_rows = []
    moran_rows = []

    for county in completed_counties(args.out_root):
        data = load_county_data(args.out_root, county)
        observed, real, synthetic, real_error, synthetic_error, coords = scenario_arrays(data)
        if len(observed) < 10:
            continue
        real_metrics = metrics(
            observed,
            real,
            rng,
            args.top_share,
            args.pair_samples,
        )
        synthetic_metrics = metrics(
            observed,
            synthetic,
            rng,
            args.top_share,
            args.pair_samples,
        )
        scenario_rows.append({"county": county, "scenario": "real", "n_links": len(observed), **real_metrics})
        scenario_rows.append(
            {"county": county, "scenario": "synthetic", "n_links": len(observed), **synthetic_metrics}
        )

        boot = bootstrap_tests(
            observed,
            real,
            synthetic,
            rng,
            args.bootstrap_samples,
            args.top_share,
            args.pair_samples,
        )
        permutation_p = permutation_tests(
            observed,
            real,
            synthetic,
            {metric: values["diff_synthetic_minus_real"] for metric, values in boot.items()},
            rng,
            args.permutation_samples,
            args.top_share,
            args.pair_samples,
        )
        for metric, values in boot.items():
            diff = values["diff_synthetic_minus_real"]
            test_rows.append(
                {
                    "county": county,
                    "metric": metric,
                    "n_links": len(observed),
                    **values,
                    "paired_permutation_p_two_sided": permutation_p[metric],
                    "winner": "synthetic" if diff > 0 else "real" if diff < 0 else "tie",
                }
            )

        neighbors = knn_indices(coords[:, 0], coords[:, 1], args.knn)
        for scenario, error in [("real", real_error), ("synthetic", synthetic_error)]:
            i_value = moran_i(error, neighbors)
            p_value = moran_permutation_p(
                error,
                neighbors,
                i_value,
                rng,
                args.moran_permutations,
            )
            moran_rows.append(
                {
                    "county": county,
                    "scenario": scenario,
                    "n_links": len(error),
                    "knn": min(args.knn, len(error) - 1),
                    "moran_i_residual": i_value,
                    "abs_moran_i_residual": abs(i_value),
                    "moran_permutation_p_two_sided": p_value,
                }
            )

    scenario_metrics = pd.DataFrame(scenario_rows)
    metric_tests = pd.DataFrame(test_rows)
    moran = pd.DataFrame(moran_rows)
    scenario_metrics.to_csv(args.out_dir / "spatial_pattern_metrics_by_scenario.csv", index=False)
    metric_tests.to_csv(args.out_dir / "paired_spatial_pattern_tests.csv", index=False)
    moran.to_csv(args.out_dir / "residual_morans_i_tests.csv", index=False)

    winner_counts = (
        metric_tests.groupby(["metric", "winner"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    winner_counts.to_csv(args.out_dir / "spatial_pattern_winner_counts.csv", index=False)

    write_scenario_metric_plots(scenario_metrics, args.out_dir)
    write_metric_plots(metric_tests, args.out_dir)
    write_moran_plots(moran, args.out_dir)

    (args.out_dir / "README.md").write_text(
        "# Spatial Pattern Tests\n\n"
        "These tests focus on whether modeled volumes preserve the spatial pattern of observed "
        "traffic counts across count links within each county. `paired_spatial_pattern_tests.csv` "
        "uses paired bootstrap confidence intervals and paired permutation p-values for "
        "synthetic-minus-real metric differences. Positive differences favor synthetic for all "
        "listed pattern metrics. `residual_morans_i_tests.csv` reports Moran's I on residuals; "
        "lower absolute values indicate less spatial clustering in model error.\n"
    )
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

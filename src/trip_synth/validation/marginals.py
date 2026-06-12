from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.utils.io import ensure_dir, write_json
from trip_synth.utils.progress import progress_iter

from .metrics import (
    align_distributions,
    jensen_shannon,
    ks_statistic,
    total_variation,
    wasserstein_1d,
    weighted_numeric_summary,
    weighted_proportions,
)


def _categorical_plot(real_p: pd.Series, synth_p: pd.Series, title: str, out: Path) -> None:
    p, q, keys = align_distributions(real_p, synth_p)
    order = np.argsort(-(p + q))[:12]
    labels = [keys[i] for i in order]
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.18, p[order], width=0.36, label="weighted survey")
    plt.bar(x + 0.18, q[order], width=0.36, label="synthetic")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Proportion")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def _numeric_plot(real: pd.Series, synth: pd.Series, title: str, out: Path) -> None:
    a = pd.to_numeric(real, errors="coerce").dropna()
    b = pd.to_numeric(synth, errors="coerce").dropna()
    plt.figure(figsize=(8, 4.8))
    bins = min(25, max(8, int(np.sqrt(max(len(a), len(b), 1)))))
    plt.hist(a, bins=bins, alpha=0.55, density=True, label="survey")
    plt.hist(b, bins=bins, alpha=0.55, density=True, label="synthetic")
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.title(f"{title} distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def validate_method_marginals(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    schema: FeatureSchema,
    preprocessor: FittedPreprocessor,
    method: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    fig_dir = ensure_dir(run_dir / "figures" / "marginals" / method)
    metrics: dict[str, Any] = {"method": method, "columns": {}}
    weights = real_df[schema.weight_column] if schema.weight_column in real_df.columns else None

    columns = list(preprocessor.feature_columns)
    for col in progress_iter(columns, desc=f"{method} marginals", total=len(columns), unit="column"):
        if col not in synthetic_df.columns or col not in real_df.columns:
            continue
        if col in preprocessor.categorical_columns:
            real_p = weighted_proportions(real_df[col], weights)
            synth_p = weighted_proportions(synthetic_df[col], None)
            p, q, keys = align_distributions(real_p, synth_p)
            errors = sorted(
                [{"category": keys[i], "abs_error": float(abs(p[i] - q[i]))} for i in range(len(keys))],
                key=lambda row: row["abs_error"],
                reverse=True,
            )[:10]
            metrics["columns"][col] = {
                "type": "categorical",
                "total_variation": total_variation(p, q),
                "jensen_shannon": jensen_shannon(p, q),
                "top_category_abs_errors": errors,
            }
            _categorical_plot(real_p, synth_p, f"{method}: {col}", fig_dir / f"{col}_marginal.png")
        else:
            real_summary = weighted_numeric_summary(real_df[col], weights)
            synth_summary = weighted_numeric_summary(synthetic_df[col], None)
            metrics["columns"][col] = {
                "type": "numeric",
                "survey": real_summary,
                "synthetic": synth_summary,
                "mean_abs_error": abs(real_summary["mean"] - synth_summary["mean"]),
                "std_abs_error": abs(real_summary["std"] - synth_summary["std"]),
                "wasserstein": wasserstein_1d(real_df[col], synthetic_df[col]),
                "ks_statistic": ks_statistic(real_df[col], synthetic_df[col]),
            }
            _numeric_plot(real_df[col], synthetic_df[col], f"{method}: {col}", fig_dir / f"{col}_marginal.png")

    cat_tvs = [
        row["total_variation"]
        for row in metrics["columns"].values()
        if row.get("type") == "categorical" and np.isfinite(row.get("total_variation", np.nan))
    ]
    num_ks = [
        row["ks_statistic"]
        for row in metrics["columns"].values()
        if row.get("type") == "numeric" and np.isfinite(row.get("ks_statistic", np.nan))
    ]
    metrics["summary"] = {
        "mean_categorical_tv": float(np.mean(cat_tvs)) if cat_tvs else float("nan"),
        "mean_numeric_ks": float(np.mean(num_ks)) if num_ks else float("nan"),
        "marginal_similarity_score": float(1.0 - np.nanmean(cat_tvs + num_ks)) if (cat_tvs or num_ks) else 0.0,
    }
    write_json(metrics, run_dir / "metrics" / f"{method}_marginals.json")
    return metrics

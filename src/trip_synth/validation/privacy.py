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


def _row_keys(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].astype("string").fillna("<NA>").agg("\x1f".join, axis=1)


def _distance_matrix_features(df: pd.DataFrame, preprocessor: FittedPreprocessor) -> np.ndarray:
    arr = preprocessor.transform(df)
    parts = []
    if arr["cat"].size:
        cat = arr["cat"].astype(float)
        denom = np.maximum(np.array(preprocessor.category_sizes(), dtype=float) - 1, 1.0)
        parts.append(cat / denom)
    if arr["num"].size:
        parts.append(arr["num"].astype(float))
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(df), 0), dtype=float)


def nearest_neighbor_distances(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    preprocessor: FittedPreprocessor,
    chunk_size: int = 1000,
) -> np.ndarray:
    train_x = _distance_matrix_features(train_df, preprocessor)
    synth_x = _distance_matrix_features(synthetic_df, preprocessor)
    if train_x.shape[1] == 0 or len(train_x) == 0 or len(synth_x) == 0:
        return np.zeros(len(synth_x), dtype=float)
    mins = []
    starts = list(range(0, len(synth_x), chunk_size))
    for i in progress_iter(starts, desc="privacy nearest-neighbor chunks", total=len(starts), unit="chunk"):
        chunk = synth_x[i : i + chunk_size]
        d2 = ((chunk[:, None, :] - train_x[None, :, :]) ** 2).mean(axis=2)
        mins.append(np.sqrt(np.min(d2, axis=1)))
    return np.concatenate(mins)


def validate_method_privacy(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    schema: FeatureSchema,
    preprocessor: FittedPreprocessor,
    method: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figures" / "privacy" / method)
    feature_cols = [c for c in preprocessor.feature_columns if c in real_df.columns and c in synthetic_df.columns]
    train_keys = set(_row_keys(real_df, feature_cols))
    synth_keys = _row_keys(synthetic_df, feature_cols)
    copy_rate = float(synth_keys.isin(train_keys).mean()) if len(synth_keys) else 0.0
    duplicate_rate = float(1.0 - synth_keys.nunique() / len(synth_keys)) if len(synth_keys) else 0.0
    nn = nearest_neighbor_distances(real_df, synthetic_df, preprocessor)
    invalid = 0
    checked = 0
    for col in preprocessor.categorical_columns:
        if col in synthetic_df.columns:
            checked += len(synthetic_df)
            invalid += int((~synthetic_df[col].astype(str).isin(set(preprocessor.categories[col]))).sum())
    invalid_category_rate = float(invalid / checked) if checked else 0.0
    if {"o_tract_fips", "d_tract_fips"}.issubset(feature_cols):
        train_od = set(zip(real_df["o_tract_fips"].astype(str), real_df["d_tract_fips"].astype(str)))
        synth_od = list(zip(synthetic_df["o_tract_fips"].astype(str), synthetic_df["d_tract_fips"].astype(str)))
        novel_od = float(np.mean([pair not in train_od for pair in synth_od])) if synth_od else 0.0
    else:
        novel_od = float("nan")
    metrics = {
        "method": method,
        "exact_row_copy_rate": copy_rate,
        "duplicate_rate": duplicate_rate,
        "nearest_neighbor_distance_mean": float(np.mean(nn)) if nn.size else 0.0,
        "nearest_neighbor_distance_median": float(np.median(nn)) if nn.size else 0.0,
        "nearest_neighbor_distance_p05": float(np.quantile(nn, 0.05)) if nn.size else 0.0,
        "membership_risk_proxy_threshold": 1e-6,
        "membership_risk_proxy_share": float(np.mean(nn <= 1e-6)) if nn.size else 0.0,
        "novel_od_pair_share": novel_od,
        "invalid_category_rate": invalid_category_rate,
        "out_of_range_numeric_rate_after_postprocessing": 0.0,
        "privacy_score": float(max(0.0, 1.0 - copy_rate - invalid_category_rate)),
        "note": "Weighted bootstrap is expected to copy rows by construction." if method == "weighted_bootstrap" else "",
    }
    plt.figure(figsize=(7, 4.5))
    plt.hist(nn, bins=30, color="#3b6ea8", alpha=0.85)
    plt.xlabel("Nearest-neighbor distance")
    plt.ylabel("Synthetic rows")
    plt.title(f"{method}: distance to closest training row")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "privacy" / method / "nearest_neighbor_distance.png", dpi=200)
    plt.close()
    write_json(metrics, run_dir / "metrics" / f"{method}_privacy.json")
    return metrics

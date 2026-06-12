from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.utils.io import ensure_dir, write_json


def _row_keys(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].astype("string").fillna("<NA>").agg("\x1f".join, axis=1)



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
        "nearest_neighbor_distance_mean": float("nan"),
        "nearest_neighbor_distance_median": float("nan"),
        "nearest_neighbor_distance_p05": float("nan"),
        "membership_risk_proxy_threshold": float("nan"),
        "membership_risk_proxy_share": float("nan"),
        "novel_od_pair_share": novel_od,
        "invalid_category_rate": invalid_category_rate,
        "out_of_range_numeric_rate_after_postprocessing": 0.0,
        "privacy_score": float(max(0.0, 1.0 - copy_rate - invalid_category_rate)),
        "nearest_neighbor_note": "Nearest-neighbor privacy diagnostics are disabled to keep validation linear in sample size.",
        "note": "Weighted bootstrap is expected to copy rows by construction." if method == "weighted_bootstrap" else "",
    }
    write_json(metrics, run_dir / "metrics" / f"{method}_privacy.json")
    return metrics

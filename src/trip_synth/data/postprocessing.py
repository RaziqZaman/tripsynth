from __future__ import annotations

import pandas as pd

from .preprocessing import FittedPreprocessor
from .schema import FeatureSchema


def finalize_synthetic(
    df: pd.DataFrame,
    schema: FeatureSchema,
    preprocessor: FittedPreprocessor | None,
    method: str,
    run_id: str,
) -> pd.DataFrame:
    out = df.copy()
    for col in schema.exclude_from_synthesis:
        if col in out.columns:
            out = out.drop(columns=[col])
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").round().astype("Int64")
    if preprocessor is not None:
        for col in preprocessor.categorical_columns:
            if col in out.columns:
                valid = set(preprocessor.categories[col])
                fallback = preprocessor.categories[col][0]
                out[col] = out[col].astype(str).where(out[col].astype(str).isin(valid), fallback)
        ordered = [c for c in preprocessor.feature_columns if c in out.columns]
        extra = [c for c in out.columns if c not in ordered and not c.startswith("synthetic_")]
        out = out[ordered + extra]
    out["synthetic_method"] = method
    out["synthetic_run_id"] = run_id
    return out


def assert_synthetic_contract(df: pd.DataFrame, schema: FeatureSchema) -> None:
    missing = [c for c in schema.exclude_from_synthesis if c in df.columns]
    if missing:
        raise AssertionError(f"Synthetic output includes excluded columns: {missing}")
    if "year" in df.columns and not pd.api.types.is_integer_dtype(df["year"].dropna()):
        raise AssertionError("Synthetic output year must be integer typed")

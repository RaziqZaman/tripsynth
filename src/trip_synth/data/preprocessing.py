from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .schema import FeatureSchema, clean_category_value, clean_fips_value


def transform_weights(weights: pd.Series | np.ndarray, config: dict[str, Any] | None) -> np.ndarray:
    cfg = config or {}
    arr = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).to_numpy(float)
    arr = np.clip(arr, 0.0, None)
    mode = str(cfg.get("transform", "none"))
    if mode == "sqrt":
        arr = np.sqrt(arr)
    elif mode == "log1p":
        arr = np.log1p(arr)
    elif mode == "capped":
        q = float(cfg.get("cap_quantile", 0.99))
        cap = np.quantile(arr[arr > 0], q) if np.any(arr > 0) else 1.0
        arr = np.minimum(arr, cap)
    elif mode in {"capped_normalized", "capped-normalized"}:
        q = float(cfg.get("cap_quantile", 0.99))
        cap = np.quantile(arr[arr > 0], q) if np.any(arr > 0) else 1.0
        arr = np.minimum(arr, cap)
    elif mode == "normalized-to-mean-1":
        pass
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown weight transform: {mode}")

    if cfg.get("normalize_mean_to_one", mode in {"capped_normalized", "normalized-to-mean-1"}):
        mean = arr.mean()
        if mean > 0:
            arr = arr / mean
    if not np.any(arr > 0):
        arr = np.ones_like(arr, dtype=float)
    return arr.astype("float32")


@dataclass
class FittedPreprocessor:
    schema_dict: dict[str, Any]
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    ordinal_integer_columns: list[str]
    continuous_columns: list[str]
    categories: dict[str, list[str]]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    numeric_min: dict[str, float]
    numeric_max: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame, schema: FeatureSchema) -> "FittedPreprocessor":
        df, _ = schema.clean_dataframe(df)
        feature_columns = schema.synthesis_columns(df)
        cat_cols = [c for c in schema.categorical_present(df) if c in feature_columns]
        int_cols = [c for c in schema.integer_present(df) if c in feature_columns]
        cont_cols = [c for c in schema.continuous_present(df) if c in feature_columns]
        num_cols = int_cols + cont_cols

        categories: dict[str, list[str]] = {}
        for col in cat_cols:
            vals = df[col].map(clean_category_value)
            if col in schema.fips_columns:
                vals = df[col].map(clean_fips_value)
            cats = sorted(pd.Series(vals).dropna().astype(str).unique().tolist())
            categories[col] = cats or ["-1"]

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        mins: dict[str, float] = {}
        maxs: dict[str, float] = {}
        for col in num_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            mean = float(vals.mean()) if vals.notna().any() else 0.0
            std = float(vals.std(ddof=0)) if vals.notna().sum() > 1 else 1.0
            if not np.isfinite(std) or std <= 1e-9:
                std = 1.0
            means[col] = mean
            stds[col] = std
            mins[col] = float(vals.min()) if vals.notna().any() else mean
            maxs[col] = float(vals.max()) if vals.notna().any() else mean

        return cls(
            schema_dict=schema.to_dict(),
            feature_columns=feature_columns,
            categorical_columns=cat_cols,
            numeric_columns=num_cols,
            ordinal_integer_columns=int_cols,
            continuous_columns=cont_cols,
            categories=categories,
            numeric_mean=means,
            numeric_std=stds,
            numeric_min=mins,
            numeric_max=maxs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_dict": self.schema_dict,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "ordinal_integer_columns": self.ordinal_integer_columns,
            "continuous_columns": self.continuous_columns,
            "categories": self.categories,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "numeric_min": self.numeric_min,
            "numeric_max": self.numeric_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FittedPreprocessor":
        return cls(**data)

    def category_sizes(self) -> list[int]:
        return [len(self.categories[col]) for col in self.categorical_columns]

    def transform(self, df: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
        schema = FeatureSchema.from_dict(self.schema_dict)
        df, _ = schema.clean_dataframe(df)

        cat_arrays: list[np.ndarray] = []
        for col in self.categorical_columns:
            cats = self.categories[col]
            lookup = {val: i for i, val in enumerate(cats)}
            series = df[col].map(clean_category_value).astype(str)
            if col in schema.fips_columns:
                series = df[col].map(clean_fips_value).astype(str)
            codes = series.map(lookup).fillna(0).astype("int64").to_numpy()
            cat_arrays.append(codes)

        num_arrays: list[np.ndarray] = []
        for col in self.numeric_columns:
            vals = pd.to_numeric(df[col], errors="coerce").astype(float)
            vals = vals.fillna(self.numeric_mean[col])
            arr = ((vals.to_numpy() - self.numeric_mean[col]) / self.numeric_std[col]).astype("float32")
            num_arrays.append(arr)

        n = len(df)
        cat = np.stack(cat_arrays, axis=1).astype("int64") if cat_arrays else np.zeros((n, 0), dtype="int64")
        num = np.stack(num_arrays, axis=1).astype("float32") if num_arrays else np.zeros((n, 0), dtype="float32")
        if weights is None:
            weights = np.ones(n, dtype="float32")
        return {"cat": cat, "num": num, "weights": weights.astype("float32")}

    def inverse_transform(
        self,
        cat_codes: np.ndarray,
        normalized_numeric: np.ndarray,
        config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        config = config or {}
        data: dict[str, Any] = {}
        n = cat_codes.shape[0] if cat_codes.size else normalized_numeric.shape[0]

        for j, col in enumerate(self.categorical_columns):
            cats = self.categories[col]
            codes = np.asarray(cat_codes[:, j] if cat_codes.size else np.zeros(n), dtype=int)
            codes = np.clip(codes, 0, len(cats) - 1)
            data[col] = [cats[i] for i in codes]

        clip_numeric = bool(config.get("clip_numeric", True))
        for j, col in enumerate(self.numeric_columns):
            vals = normalized_numeric[:, j] * self.numeric_std[col] + self.numeric_mean[col]
            if clip_numeric:
                vals = np.clip(vals, self.numeric_min[col], self.numeric_max[col])
            if col in self.ordinal_integer_columns:
                data[col] = np.rint(vals).astype("int64")
            else:
                data[col] = vals.astype(float)

        frame = pd.DataFrame(data)
        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = pd.NA
        return frame[self.feature_columns]

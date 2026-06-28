"""Shared lightweight tabular synthesis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


MODELED_CATEGORICAL_COLUMNS = ["__od_pair", "mode", "purpose", "day_of_week"]
MODELED_NUMERIC_COLUMNS = [
    "departure_time",
    "travel_time_min",
    "vehicle_available",
    "vehicle_occupancy",
]
DEFAULT_EXCLUDED_COLUMNS = {"synthetic_trip_id"}
MISSING_CATEGORY = "__missing__"


def normalize_fips_text(value: Any, width: int = 11) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace(".0", "")
    if not text or text.lower() == "nan":
        return None
    try:
        text = str(int(float(text)))
    except ValueError:
        text = "".join(ch for ch in text if ch.isdigit())
    if len(text) < width:
        return None
    return text.zfill(width)[:width]


def od_pair_series(trips: pd.DataFrame) -> pd.Series:
    origins = trips["origin_tract"].map(normalize_fips_text)
    destinations = trips["destination_tract"].map(normalize_fips_text)
    return origins.fillna("missing") + "|" + destinations.fillna("missing")


@dataclass
class CategoricalSpec:
    column: str
    categories: list[str]


@dataclass
class NumericSpec:
    column: str
    mean: float
    std: float
    minimum: float
    maximum: float
    integer_like: bool


class TabularTripEncoder:
    """Encode trip rows to a compact numeric matrix and decode them back.

    By default this encoder models every usable column in the input frame. It uses
    one ordinal feature per categorical column rather than one-hot expansion, which
    keeps full raw-survey synthesis tractable for the lightweight VAE/diffusion
    prototypes.
    """

    def __init__(
        self,
        *,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        model_all_columns: bool = True,
        weight_field: str = "weight",
        max_categorical_cardinality: int = 50,
        canonical_aliases: dict[str, str] | None = None,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.model_all_columns = bool(model_all_columns)
        self.weight_field = weight_field
        self.max_categorical_cardinality = int(max_categorical_cardinality)
        self.canonical_aliases = canonical_aliases or {}

    def fit(self, trips: pd.DataFrame) -> "TabularTripEncoder":
        self.columns_ = list(trips.columns)
        self.trips_ = trips.reset_index(drop=True).copy()
        model = self._model_frame(self.trips_)
        self.fit_weights_ = self._fit_weights(model)
        self.categorical_specs_: list[CategoricalSpec] = []
        self.numeric_specs_: list[NumericSpec] = []

        categorical_columns, numeric_columns = self._selected_columns(model)
        for column in categorical_columns:
            if not self._usable_column(model, column):
                continue
            values = self._categorical_values(model[column])
            categories = sorted(values.unique().tolist())
            if categories:
                self.categorical_specs_.append(CategoricalSpec(column, categories))

        for column in numeric_columns:
            if not self._usable_column(model, column):
                continue
            numeric = pd.to_numeric(model[column], errors="coerce")
            if numeric.notna().sum() == 0:
                continue
            mean, std = self._weighted_mean_std(numeric)
            finite = numeric.dropna()
            self.numeric_specs_.append(
                NumericSpec(
                    column=column,
                    mean=mean,
                    std=std,
                    minimum=float(finite.min()),
                    maximum=float(finite.max()),
                    integer_like=bool((finite.dropna() % 1 == 0).all()),
                )
            )
        self.n_features_ = len(self.categorical_specs_) + len(self.numeric_specs_)
        if self.n_features_ == 0:
            raise ValueError("No usable columns were found for tabular synthesis.")
        self.modeled_columns_ = [spec.column for spec in self.categorical_specs_] + [
            spec.column for spec in self.numeric_specs_
        ]
        return self

    def _model_frame(self, trips: pd.DataFrame) -> pd.DataFrame:
        model = trips.copy()
        if "origin_tract" in model and "destination_tract" in model:
            model["__od_pair"] = od_pair_series(model)
        return model

    def _fit_weights(self, model: pd.DataFrame) -> np.ndarray:
        weights = pd.Series(1.0, index=model.index, dtype=float)
        if self.weight_field in model:
            configured = pd.to_numeric(model[self.weight_field], errors="coerce").fillna(0).clip(lower=0)
            if configured.sum() > 0:
                weights = configured.astype(float)
        mean = float(weights.mean())
        if mean <= 0 or not np.isfinite(mean):
            return np.ones(len(model), dtype="float32")
        return (weights.to_numpy(dtype=float) / mean).astype("float32")

    def _weighted_mean_std(self, numeric: pd.Series) -> tuple[float, float]:
        valid = numeric.notna().to_numpy()
        values = numeric.loc[valid].to_numpy(dtype=float)
        weights = self.fit_weights_[valid]
        if len(values) == 0 or weights.sum() <= 0:
            mean = float(numeric.mean())
            std = float(numeric.std(ddof=0)) or 1.0
            return mean, std
        mean = float(np.average(values, weights=weights))
        variance = float(np.average((values - mean) ** 2, weights=weights))
        std = variance**0.5 or 1.0
        return mean, std

    def _selected_columns(self, model: pd.DataFrame) -> tuple[list[str], list[str]]:
        if self.categorical_columns is not None or self.numeric_columns is not None:
            categorical = self.categorical_columns or []
            numeric = self.numeric_columns or []
            return list(categorical), list(numeric)
        if not self.model_all_columns:
            return MODELED_CATEGORICAL_COLUMNS, MODELED_NUMERIC_COLUMNS
        return self._infer_all_columns(model)

    def _infer_all_columns(self, model: pd.DataFrame) -> tuple[list[str], list[str]]:
        categorical: list[str] = []
        numeric_columns: list[str] = []
        for column in model.columns:
            if not self._usable_column(model, column):
                continue
            if self._force_categorical(column, model[column]):
                categorical.append(column)
                continue
            numeric = pd.to_numeric(model[column], errors="coerce")
            non_missing = int(model[column].notna().sum())
            numeric_ratio = float(numeric.notna().sum() / max(non_missing, 1))
            unique_count = int(model[column].nunique(dropna=True))
            if numeric_ratio >= 0.95 and unique_count > self.max_categorical_cardinality:
                numeric_columns.append(column)
            else:
                categorical.append(column)
        return categorical, numeric_columns

    def _usable_column(self, model: pd.DataFrame, column: str) -> bool:
        return column in model and column not in DEFAULT_EXCLUDED_COLUMNS and not model[column].isna().all()

    def _force_categorical(self, column: str, values: pd.Series) -> bool:
        lower = column.lower()
        if column == "__od_pair":
            return True
        if any(token in lower for token in ["fips", "tract", "_id", "id_"]):
            return True
        if pd.api.types.is_datetime64_any_dtype(values):
            return True
        if pd.api.types.is_bool_dtype(values) or isinstance(values.dtype, pd.CategoricalDtype):
            return True
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce")
            non_missing = int(values.notna().sum())
            return numeric.notna().sum() / max(non_missing, 1) < 0.95
        return False

    def _categorical_values(self, values: pd.Series) -> pd.Series:
        return values.astype("string").fillna(MISSING_CATEGORY)

    def transform(self, trips: pd.DataFrame) -> np.ndarray:
        model = self._model_frame(trips.reset_index(drop=True))
        arrays = []
        for spec in self.categorical_specs_:
            if spec.column in model:
                values = self._categorical_values(model[spec.column])
            else:
                values = pd.Series(MISSING_CATEGORY, index=model.index, dtype="string")
            lookup = {value: i for i, value in enumerate(spec.categories)}
            codes = values.map(lookup).fillna(0).to_numpy(dtype=float)
            if len(spec.categories) > 1:
                encoded = 2.0 * (codes / (len(spec.categories) - 1)) - 1.0
            else:
                encoded = np.zeros(len(model), dtype=float)
            arrays.append(encoded)
        for spec in self.numeric_specs_:
            if spec.column in model:
                values = pd.to_numeric(model[spec.column], errors="coerce").fillna(spec.mean)
            else:
                values = pd.Series(spec.mean, index=model.index)
            arrays.append(((values.to_numpy(dtype=float) - spec.mean) / spec.std).clip(-5, 5))
        return np.column_stack(arrays).astype("float32")

    def decode(
        self,
        matrix: np.ndarray,
        *,
        seed: int | None = None,
        donor_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        matrix = np.asarray(matrix, dtype=float)
        n = matrix.shape[0]
        if donor_indices is None:
            donor_indices = rng.choice(self.trips_.index.to_numpy(), size=n, replace=True)
        out = self.trips_.iloc[donor_indices].reset_index(drop=True).copy()
        cursor = 0
        for spec in self.categorical_specs_:
            values = matrix[:, cursor]
            cursor += 1
            if len(spec.categories) > 1:
                codes = np.rint(((values + 1.0) / 2.0) * (len(spec.categories) - 1)).astype(int)
                codes = np.clip(codes, 0, len(spec.categories) - 1)
            else:
                codes = np.zeros(n, dtype=int)
            decoded = pd.Series([spec.categories[code] for code in codes], index=out.index)
            decoded = decoded.replace(MISSING_CATEGORY, pd.NA)
            if spec.column == "__od_pair":
                split = decoded.astype("string").str.split(r"\|", n=1, expand=True)
                if split.shape[1] >= 2:
                    out["origin_tract"] = split[0].replace("missing", pd.NA)
                    out["destination_tract"] = split[1].replace("missing", pd.NA)
            else:
                out[spec.column] = decoded
        for spec in self.numeric_specs_:
            values = matrix[:, cursor] * spec.std + spec.mean
            cursor += 1
            values = np.clip(values, spec.minimum, spec.maximum)
            if spec.integer_like:
                values = np.rint(values)
            out[spec.column] = values
        self._sync_canonical_aliases(out)
        return out

    def _sync_canonical_aliases(self, out: pd.DataFrame) -> None:
        for canonical, source in self.canonical_aliases.items():
            if not source or source == canonical:
                continue
            if canonical in out and source in out:
                out[source] = out[canonical]


def sample_donor_indices(
    trips: pd.DataFrame,
    n: int,
    *,
    seed: int | None,
    weight_field: str = "weight",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = None
    if weight_field in trips:
        weights = pd.to_numeric(trips[weight_field], errors="coerce").fillna(0).clip(lower=0)
        if weights.sum() > 0:
            probabilities = (weights / weights.sum()).to_numpy(dtype=float)
    return rng.choice(trips.reset_index(drop=True).index.to_numpy(), size=n, replace=True, p=probabilities)

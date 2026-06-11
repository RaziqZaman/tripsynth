from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trip_synth.utils.io import load_yaml


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(v) for v in value]


def clean_fips_value(value: Any) -> str:
    if pd.isna(value):
        return "-1"
    text = str(value).strip()
    if text in {"", "nan", "None", "<NA>"}:
        return "-1"
    try:
        number = float(text)
        if np.isfinite(number) and abs(number - round(number)) < 1e-6:
            text = str(int(round(number)))
    except ValueError:
        pass
    if text.endswith(".0"):
        text = text[:-2]
    if text in {"-1", "0"}:
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    return digits.zfill(11)


def clean_category_value(value: Any) -> str:
    if pd.isna(value):
        return "-1"
    text = str(value).strip()
    if text in {"", "nan", "None", "<NA>"}:
        return "-1"
    return text


@dataclass
class FeatureSchema:
    id_columns: list[str] = field(default_factory=list)
    weight_column: str = "weight"
    exclude_from_synthesis: list[str] = field(default_factory=lambda: ["weight"])
    categorical_columns: list[str] = field(default_factory=list)
    ordinal_integer_columns: list[str] = field(default_factory=list)
    continuous_columns: list[str] = field(default_factory=list)
    fips_columns: list[str] = field(default_factory=list)
    integer_cast_columns: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeatureSchema":
        data = load_yaml(path)
        return cls(
            id_columns=_as_list(data.get("id_columns")),
            weight_column=str(data.get("weight_column", "weight")),
            exclude_from_synthesis=_as_list(data.get("exclude_from_synthesis")),
            categorical_columns=_as_list(data.get("categorical_columns")),
            ordinal_integer_columns=_as_list(data.get("ordinal_integer_columns")),
            continuous_columns=_as_list(data.get("continuous_columns")),
            fips_columns=_as_list(data.get("fips_columns")),
            integer_cast_columns=data.get("integer_cast_columns", {}) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_columns": self.id_columns,
            "weight_column": self.weight_column,
            "exclude_from_synthesis": self.exclude_from_synthesis,
            "categorical_columns": self.categorical_columns,
            "ordinal_integer_columns": self.ordinal_integer_columns,
            "continuous_columns": self.continuous_columns,
            "fips_columns": self.fips_columns,
            "integer_cast_columns": self.integer_cast_columns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSchema":
        return cls(
            id_columns=_as_list(data.get("id_columns")),
            weight_column=str(data.get("weight_column", "weight")),
            exclude_from_synthesis=_as_list(data.get("exclude_from_synthesis")),
            categorical_columns=_as_list(data.get("categorical_columns")),
            ordinal_integer_columns=_as_list(data.get("ordinal_integer_columns")),
            continuous_columns=_as_list(data.get("continuous_columns")),
            fips_columns=_as_list(data.get("fips_columns")),
            integer_cast_columns=data.get("integer_cast_columns", {}) or {},
        )

    def synthesis_columns(self, df: pd.DataFrame) -> list[str]:
        excluded = set(self.exclude_from_synthesis) | set(self.id_columns)
        return [c for c in df.columns if c not in excluded]

    def categorical_present(self, df: pd.DataFrame) -> list[str]:
        cols = set(df.columns)
        return [c for c in self.categorical_columns if c in cols]

    def integer_present(self, df: pd.DataFrame) -> list[str]:
        cols = set(df.columns)
        return [c for c in self.ordinal_integer_columns if c in cols]

    def continuous_present(self, df: pd.DataFrame) -> list[str]:
        cols = set(df.columns)
        return [c for c in self.continuous_columns if c in cols]

    def clean_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        out = df.copy()
        warnings: list[str] = []

        for col in self.fips_columns:
            if col in out.columns:
                out[col] = out[col].map(clean_fips_value).astype("string")

        for col in self.categorical_present(out):
            if col not in self.fips_columns:
                out[col] = out[col].map(clean_category_value).astype("string")

        for col, cast_config in self.integer_cast_columns.items():
            if col not in out.columns:
                continue
            numeric = pd.to_numeric(out[col], errors="coerce")
            rounded = numeric.round()
            bad = numeric.notna() & ~np.isclose(numeric, rounded, atol=1e-6)
            if bad.any():
                mode = cast_config.get("mode", "safe_round")
                warnings.append(
                    f"{col}: rounded {int(bad.sum())} non-integer values using mode={mode}"
                )
            if cast_config.get("nullable", False):
                out[col] = rounded.astype("Int64")
            else:
                out[col] = rounded.fillna(0).astype("int64")

        for col in self.integer_present(out):
            if col in self.integer_cast_columns:
                continue
            numeric = pd.to_numeric(out[col], errors="coerce")
            out[col] = numeric.round().astype("Int64")

        for col in self.continuous_present(out):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

        if self.weight_column in out.columns:
            out[self.weight_column] = pd.to_numeric(out[self.weight_column], errors="coerce")
        return out, warnings

    def required_column_report(self, df: pd.DataFrame) -> dict[str, list[str]]:
        present = set(df.columns)
        expected = (
            set(self.categorical_columns)
            | set(self.ordinal_integer_columns)
            | set(self.continuous_columns)
            | {self.weight_column}
        )
        return {
            "missing": sorted(expected - present),
            "extra": sorted(present - expected - set(self.id_columns)),
        }

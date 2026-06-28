"""Survey ingestion and configurable schema mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tripsynth.config import resolve_path


CANONICAL_SURVEY_FIELDS = [
    "trip_id",
    "household_id",
    "person_id",
    "origin_tract",
    "destination_tract",
    "trip_date",
    "survey_date",
    "survey_year",
    "survey_month",
    "day_of_week",
    "departure_time",
    "arrival_time",
    "travel_time_min",
    "mode",
    "purpose",
    "weight",
    "vehicle_available",
]


@dataclass(frozen=True)
class SurveyData:
    raw: pd.DataFrame
    canonical: pd.DataFrame
    schema_report: list[dict[str, Any]]
    path: Path


def load_table(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suffix.endswith(".parquet") or suffix.endswith(".geoparquet"):
        return pd.read_parquet(path)
    if suffix.endswith(".json") or suffix.endswith(".jsonl"):
        return pd.read_json(path, lines=suffix.endswith(".jsonl"))
    if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported survey file format for {path}")


def _resolve_source_field(source_field: str | None, columns: list[str]) -> str | None:
    if not source_field:
        return None
    if source_field in columns:
        return source_field
    lower_lookup = {column.lower(): column for column in columns}
    return lower_lookup.get(str(source_field).lower())


def apply_schema_mapping(
    raw: pd.DataFrame, schema: dict[str, str | None]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    columns = list(raw.columns)
    canonical = pd.DataFrame(index=raw.index)
    report: list[dict[str, Any]] = []

    for field in CANONICAL_SURVEY_FIELDS:
        configured = schema.get(field)
        source = _resolve_source_field(configured, columns)
        found = source is not None
        if found:
            canonical[field] = raw[source]
        else:
            canonical[field] = pd.NA
        report.append(
            {
                "canonical_field": field,
                "configured_source_field": configured,
                "resolved_source_field": source,
                "configured": configured is not None,
                "field_found": found,
            }
        )

    return canonical, report


def load_survey(config: dict[str, Any]) -> SurveyData:
    survey_config = config.get("survey", {})
    path = resolve_path(config, survey_config.get("path"))
    if path is None:
        raise FileNotFoundError("survey.path is not configured")
    if not path.exists():
        raise FileNotFoundError(
            f"Survey file not found: {path}. Update survey.path in the config."
        )

    raw = load_table(path)
    canonical, report = apply_schema_mapping(raw, survey_config.get("schema", {}))
    return SurveyData(raw=raw, canonical=canonical, schema_report=report, path=path)

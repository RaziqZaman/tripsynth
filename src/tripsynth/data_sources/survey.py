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
    "survey_week",
    "day_of_week",
    "departure_time",
    "arrival_time",
    "travel_time_min",
    "mode",
    "purpose",
    "weight",
    "vehicle_available",
    "vehicle_occupancy",
]


@dataclass(frozen=True)
class SurveyData:
    raw: pd.DataFrame
    canonical: pd.DataFrame
    schema_report: list[dict[str, Any]]
    path: Path


RAW_COLUMN_PREFIX = "raw__"


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


def _derive_trip_date_from_week_dow(
    canonical: pd.DataFrame, derivation: dict[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    reference = pd.Timestamp(derivation.get("reference_date", "2017-01-01"))
    week_field = derivation.get("week_field", "survey_week")
    dow_field = derivation.get("day_of_week_field", "day_of_week")
    week_base = int(derivation.get("week_index_base", 0))

    weeks = pd.to_numeric(canonical.get(week_field), errors="coerce")
    dows = pd.to_numeric(canonical.get(dow_field), errors="coerce")
    valid = weeks.notna() & dows.between(1, 7)
    dates = pd.Series(pd.NaT, index=canonical.index, dtype="datetime64[ns]")

    target_weekday = (dows.loc[valid].astype(int) - 1) % 7
    offset_from_reference_weekday = (target_weekday - reference.dayofweek) % 7
    days_after_reference = (
        (weeks.loc[valid].astype(int) - week_base) * 7 + offset_from_reference_weekday
    )
    dates.loc[valid] = reference + pd.to_timedelta(days_after_reference, unit="D")
    report = {
        "derived": True,
        "reference_date": reference.date().isoformat(),
        "week_field": week_field,
        "day_of_week_field": dow_field,
        "week_index_base": week_base,
        "valid_derived_dates": int(dates.notna().sum()),
    }
    return dates, report


def apply_temporal_derivations(
    canonical: pd.DataFrame,
    report: list[dict[str, Any]],
    survey_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    derivation = survey_config.get("temporal_derivation", {})
    if not derivation.get("derive_trip_date_from_week_dow"):
        return canonical, report
    if canonical["trip_date"].notna().any():
        return canonical, report

    dates, derivation_report = _derive_trip_date_from_week_dow(canonical, derivation)
    if not dates.notna().any():
        return canonical, report

    canonical = canonical.copy()
    canonical["trip_date"] = dates
    for row in report:
        if row["canonical_field"] == "trip_date":
            row.update(
                {
                    "configured_source_field": "derived:survey_week+day_of_week",
                    "resolved_source_field": None,
                    "configured": True,
                    "field_found": True,
                    "derivation": derivation_report,
                }
            )
            break
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
    canonical, report = apply_temporal_derivations(canonical, report, survey_config)
    return SurveyData(raw=raw, canonical=canonical, schema_report=report, path=path)


def _same_values(left: pd.Series, right: pd.Series) -> bool:
    return left.astype("string").fillna("__missing__").equals(
        right.astype("string").fillna("__missing__")
    )


def build_synthesis_frame(survey: SurveyData, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a route-ready synthesis table containing canonical and raw survey columns."""
    del config  # Reserved for future column-selection options.
    canonical = survey.canonical.reset_index(drop=True).copy()
    raw = survey.raw.reset_index(drop=True)
    frame = canonical.copy()
    collisions: dict[str, str] = {}

    for column in raw.columns:
        raw_name = str(column)
        target = raw_name
        if target in frame.columns:
            if _same_values(frame[target], raw[column]):
                continue
            target = f"{RAW_COLUMN_PREFIX}{raw_name}"
            suffix = 2
            while target in frame.columns:
                target = f"{RAW_COLUMN_PREFIX}{raw_name}_{suffix}"
                suffix += 1
            collisions[raw_name] = target
        frame[target] = raw[column]

    frame.attrs["raw_survey_columns"] = [str(column) for column in raw.columns]
    frame.attrs["canonical_survey_columns"] = list(canonical.columns)
    frame.attrs["raw_column_collisions"] = collisions
    return frame


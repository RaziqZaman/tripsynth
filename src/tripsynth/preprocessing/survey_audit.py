"""Survey temporal and schema audit command implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tripsynth.config import ensure_standard_directories, write_json
from tripsynth.data_sources.survey import SurveyData, load_survey
from tripsynth.preprocessing.temporal_coverage import (
    day_of_week_counts,
    departure_hour_counts,
    infer_temporal_coverage,
)


@dataclass(frozen=True)
class SurveyAuditResult:
    metadata: dict[str, Any]
    missingness: pd.DataFrame
    temporal_coverage: pd.DataFrame
    output_paths: dict[str, str]


def _field_found(schema_report: list[dict[str, Any]], field: str) -> bool:
    return any(row["canonical_field"] == field and row["field_found"] for row in schema_report)


def _unique_count(frame: pd.DataFrame, field: str) -> int | None:
    if field not in frame or frame[field].isna().all():
        return None
    return int(frame[field].nunique(dropna=True))


def build_missingness(survey: SurveyData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(survey.canonical)
    for row in survey.schema_report:
        field = row["canonical_field"]
        missing_count = int(survey.canonical[field].isna().sum())
        rows.append(
            {
                **row,
                "rows": total,
                "missing_count": missing_count,
                "missing_share": missing_count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _append_time_coverage(
    temporal: pd.DataFrame, canonical: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [temporal]

    dow = day_of_week_counts(canonical["day_of_week"])
    if not dow.empty:
        dow_rows = dow.assign(
            coverage_type="day_of_week",
            year=pd.NA,
            month=pd.NA,
            hour=pd.NA,
            status="observed",
            note="from configured day_of_week field",
        )[
            ["coverage_type", "year", "month", "day_of_week", "hour", "count", "status", "note"]
        ]
        frames.append(dow_rows)

    hours = departure_hour_counts(canonical["departure_time"])
    if not hours.empty:
        hour_rows = hours.assign(
            coverage_type="hour",
            year=pd.NA,
            month=pd.NA,
            day_of_week=pd.NA,
            status="observed",
            note="from configured departure_time field",
        )[
            ["coverage_type", "year", "month", "day_of_week", "hour", "count", "status", "note"]
        ]
        frames.append(hour_rows)

    return pd.concat(frames, ignore_index=True), dow, hours


def _write_figures(
    temporal: pd.DataFrame, hour_counts: pd.DataFrame, figure_dir: Path
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if temporal["coverage_type"].eq("year_month").any():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        month_rows = temporal.loc[temporal["coverage_type"] == "year_month"].copy()
        month_rows["period"] = month_rows["year"].astype(str) + "-" + month_rows[
            "month"
        ].astype(int).astype(str).str.zfill(2)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(month_rows["period"], month_rows["count"])
        ax.set_title("Survey trips by month")
        ax.set_xlabel("Survey month")
        ax.set_ylabel("Trips")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = figure_dir / "survey_trips_by_month.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["trips_by_month_png"] = str(path)

    if not hour_counts.empty:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(hour_counts["hour"], hour_counts["count"], width=0.8)
        ax.set_title("Survey trips by departure hour")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Trips")
        ax.set_xticks(range(0, 24, 2))
        fig.tight_layout()
        path = figure_dir / "survey_trips_by_hour.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["trips_by_hour_png"] = str(path)

    return paths


def run_survey_audit(config: dict[str, Any]) -> SurveyAuditResult:
    ensure_standard_directories()
    survey = load_survey(config)
    missingness = build_missingness(survey)
    temporal = infer_temporal_coverage(survey.canonical)
    coverage_table, dow_counts, hour_counts = _append_time_coverage(
        temporal.coverage_table, survey.canonical
    )

    table_dir = Path("reports/tables")
    figure_dir = Path("reports/figures")
    metadata_dir = Path("data/metadata")

    temporal_path = table_dir / "survey_temporal_coverage.csv"
    missingness_path = table_dir / "survey_field_missingness.csv"
    coverage_table.to_csv(temporal_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    figure_paths = _write_figures(coverage_table, hour_counts, figure_dir)

    metadata: dict[str, Any] = {
        "survey_path": str(survey.path),
        "rows_loaded": int(len(survey.raw)),
        "trip_count": _unique_count(survey.canonical, "trip_id") or int(len(survey.raw)),
        "household_count": _unique_count(survey.canonical, "household_id"),
        "person_count": _unique_count(survey.canonical, "person_id"),
        "date_time_fields_found": {
            "trip_date": _field_found(survey.schema_report, "trip_date"),
            "survey_date": _field_found(survey.schema_report, "survey_date"),
            "survey_year": _field_found(survey.schema_report, "survey_year"),
            "survey_month": _field_found(survey.schema_report, "survey_month"),
            "day_of_week": _field_found(survey.schema_report, "day_of_week"),
            "departure_time": _field_found(survey.schema_report, "departure_time"),
            "arrival_time": _field_found(survey.schema_report, "arrival_time"),
            "travel_time_min": _field_found(survey.schema_report, "travel_time_min"),
        },
        "exact_date_field": temporal.exact_date_field,
        "calendar_coverage_status": (
            "exact_dates_available"
            if temporal.has_exact_dates
            else "month_level_available"
            if temporal.has_year_month
            else "unknown"
        ),
        "covered_years": temporal.covered_years,
        "covered_months_by_year": temporal.covered_months_by_year,
        "first_survey_date": temporal.first_date,
        "last_survey_date": temporal.last_date,
        "full_year_coverage_by_year": temporal.full_year_coverage_by_year,
        "year_status": temporal.year_status,
        "usable_for_calendar_temporal_validation": bool(
            temporal.usable_for_calendar_temporal_validation
        ),
        "usable_for_spatial_aadt_proxy_validation": bool(
            _field_found(survey.schema_report, "origin_tract")
            and _field_found(survey.schema_report, "destination_tract")
        ),
        "default_validation_label": "spatial roadway-volume proxy validation against AADT",
        "warning": (
            "Do not assume complete annual coverage unless full_year_coverage_by_year "
            "is true for the relevant year and the paper documents that assumption."
        ),
        "day_of_week_counts": dow_counts.to_dict(orient="records"),
        "departure_hour_counts": hour_counts.to_dict(orient="records"),
        "schema_report": survey.schema_report,
    }

    audit_path = metadata_dir / "survey_audit.json"
    write_json(audit_path, metadata)

    output_paths = {
        "survey_audit_json": str(audit_path),
        "survey_temporal_coverage_csv": str(temporal_path),
        "survey_field_missingness_csv": str(missingness_path),
        **figure_paths,
    }
    return SurveyAuditResult(
        metadata=metadata,
        missingness=missingness,
        temporal_coverage=coverage_table,
        output_paths=output_paths,
    )


def format_audit_summary(result: SurveyAuditResult) -> str:
    metadata = result.metadata
    fields = metadata["date_time_fields_found"]
    years = metadata["covered_years"] or "unknown"
    months = metadata["covered_months_by_year"] or "unknown"
    year_status = metadata["year_status"] or "unknown"
    return "\n".join(
        [
            "Survey audit complete",
            f"  rows/trips loaded: {metadata['rows_loaded']} rows / {metadata['trip_count']} trips",
            f"  households/persons: {metadata['household_count']} / {metadata['person_count']}",
            f"  date/time fields found: {fields}",
            f"  years covered: {years}",
            f"  months covered by year: {months}",
            f"  year coverage status: {year_status}",
            (
                "  temporal validation supported: "
                f"{metadata['usable_for_calendar_temporal_validation']}"
            ),
            (
                "  spatial AADT proxy validation supported: "
                f"{metadata['usable_for_spatial_aadt_proxy_validation']}"
            ),
            "  default validation label: spatial roadway-volume proxy validation against AADT",
            f"  outputs: {metadata.get('survey_path')} -> {result.output_paths}",
        ]
    )

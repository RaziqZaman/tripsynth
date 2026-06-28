"""Temporal coverage helpers that avoid full-year assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TemporalCoverage:
    exact_date_field: str | None
    has_exact_dates: bool
    has_year_month: bool
    covered_years: list[int]
    covered_months_by_year: dict[str, list[int]]
    first_date: str | None
    last_date: str | None
    full_year_coverage_by_year: dict[str, bool] | str
    year_status: dict[str, str]
    usable_for_calendar_temporal_validation: bool
    coverage_table: pd.DataFrame
    date_series: pd.Series


def parse_date_series(series: pd.Series) -> pd.Series:
    values = series.dropna().astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if values.empty:
        return parsed

    eight_digit = values.str.fullmatch(r"\d{8}").mean() >= 0.8
    if eight_digit:
        parsed_values = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    else:
        parsed_values = pd.to_datetime(values, errors="coerce")
    parsed.loc[parsed_values.index] = parsed_values
    return parsed


def _valid_int_series(series: pd.Series, low: int, high: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.between(low, high))
    return numeric.astype("Int64")


def infer_temporal_coverage(canonical: pd.DataFrame) -> TemporalCoverage:
    exact_date_field = None
    dates = pd.Series(pd.NaT, index=canonical.index, dtype="datetime64[ns]")
    for field in ["trip_date", "survey_date"]:
        parsed = parse_date_series(canonical[field])
        if parsed.notna().any():
            exact_date_field = field
            dates = parsed
            break

    rows: list[dict[str, Any]] = []
    months_by_year: dict[str, list[int]] = {}
    covered_years: list[int] = []
    first_date: str | None = None
    last_date: str | None = None
    has_exact_dates = exact_date_field is not None
    has_year_month = False

    if has_exact_dates:
        valid_dates = dates.dropna()
        years = valid_dates.dt.year.astype(int)
        months = valid_dates.dt.month.astype(int)
        first_date = valid_dates.min().date().isoformat()
        last_date = valid_dates.max().date().isoformat()
        grouped = (
            pd.DataFrame({"year": years, "month": months})
            .groupby(["year", "month"])
            .size()
            .reset_index(name="count")
        )
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "coverage_type": "year_month",
                    "year": int(row.year),
                    "month": int(row.month),
                    "day_of_week": pd.NA,
                    "hour": pd.NA,
                    "count": int(row.count),
                    "status": "observed",
                    "note": f"from {exact_date_field}",
                }
            )
        covered_years = sorted(int(year) for year in grouped["year"].unique())
        for year in covered_years:
            months_present = sorted(
                int(month) for month in grouped.loc[grouped["year"] == year, "month"]
            )
            months_by_year[str(year)] = months_present
    else:
        years = _valid_int_series(canonical["survey_year"], 1900, 2100)
        months = _valid_int_series(canonical["survey_month"], 1, 12)
        if years.notna().any() and months.notna().any():
            ym = pd.DataFrame({"year": years, "month": months}).dropna()
            has_year_month = not ym.empty
            grouped = ym.groupby(["year", "month"]).size().reset_index(name="count")
            for row in grouped.itertuples(index=False):
                year = int(row.year)
                month = int(row.month)
                rows.append(
                    {
                        "coverage_type": "year_month",
                        "year": year,
                        "month": month,
                        "day_of_week": pd.NA,
                        "hour": pd.NA,
                        "count": int(row.count),
                        "status": "observed_month_level",
                        "note": "from configured survey_year/survey_month fields",
                    }
                )
            covered_years = sorted(int(year) for year in grouped["year"].unique())
            for year in covered_years:
                months_by_year[str(year)] = sorted(
                    int(month) for month in grouped.loc[grouped["year"] == year, "month"]
                )

    if not rows:
        rows.append(
            {
                "coverage_type": "calendar",
                "year": pd.NA,
                "month": pd.NA,
                "day_of_week": pd.NA,
                "hour": pd.NA,
                "count": int(len(canonical)),
                "status": "unknown",
                "note": "No exact trip date or configured year/month fields were available.",
            }
        )

    if months_by_year:
        full_year: dict[str, bool] = {}
        year_status: dict[str, str] = {}
        for year, months_present in months_by_year.items():
            is_full = set(months_present) == set(range(1, 13))
            full_year[year] = is_full
            year_status[year] = (
                "complete_by_month_presence" if is_full else "partial_by_month_presence"
            )
    else:
        full_year = "unknown"
        year_status = {}

    usable_for_temporal = has_exact_dates or has_year_month

    return TemporalCoverage(
        exact_date_field=exact_date_field,
        has_exact_dates=has_exact_dates,
        has_year_month=has_year_month,
        covered_years=covered_years,
        covered_months_by_year=months_by_year,
        first_date=first_date,
        last_date=last_date,
        full_year_coverage_by_year=full_year,
        year_status=year_status,
        usable_for_calendar_temporal_validation=usable_for_temporal,
        coverage_table=pd.DataFrame(rows),
        date_series=dates,
    )


def departure_hour_counts(series: pd.Series) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce")
    minutes = numeric.where(numeric.between(0, 1439))

    if not minutes.notna().any():
        text = series.dropna().astype("string").str.strip()
        parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
        minutes = pd.Series(pd.NA, index=series.index, dtype="Float64")
        minutes.loc[parsed.index] = parsed.dt.hour * 60 + parsed.dt.minute

    hours = (minutes // 60).astype("Int64")
    counts = hours.dropna().astype(int).value_counts().sort_index()
    return pd.DataFrame({"hour": counts.index, "count": counts.values})


def day_of_week_counts(series: pd.Series) -> pd.DataFrame:
    values = series.dropna()
    if values.empty:
        return pd.DataFrame(columns=["day_of_week", "count"])
    counts = values.value_counts().sort_index()
    return pd.DataFrame({"day_of_week": counts.index.astype(str), "count": counts.values})

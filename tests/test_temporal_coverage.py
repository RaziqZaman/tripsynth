import pandas as pd

from tripsynth.data_sources.survey import CANONICAL_SURVEY_FIELDS
from tripsynth.preprocessing.temporal_coverage import infer_temporal_coverage


def canonical(**values):
    n = len(next(iter(values.values()))) if values else 0
    data = {field: [pd.NA] * n for field in CANONICAL_SURVEY_FIELDS}
    data.update(values)
    return pd.DataFrame(data)


def test_full_year_by_month_presence():
    dates = pd.date_range("2018-01-01", "2018-12-01", freq="MS")
    coverage = infer_temporal_coverage(canonical(trip_date=dates))
    assert coverage.covered_years == [2018]
    assert coverage.full_year_coverage_by_year == {"2018": True}
    assert coverage.usable_for_calendar_temporal_validation is True


def test_partial_year_dates_are_not_full_year():
    dates = pd.to_datetime(["2018-01-02", "2018-02-03", "2018-03-04"])
    coverage = infer_temporal_coverage(canonical(trip_date=dates))
    assert coverage.full_year_coverage_by_year == {"2018": False}
    assert coverage.year_status["2018"] == "partial_by_month_presence"


def test_no_date_field_is_unknown():
    coverage = infer_temporal_coverage(canonical(day_of_week=[1, 2, 3]))
    assert coverage.full_year_coverage_by_year == "unknown"
    assert coverage.usable_for_calendar_temporal_validation is False

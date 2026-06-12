from __future__ import annotations

import numpy as np

from trip_synth.validation.aadt_validation import compute_aadt_metrics
from trip_synth.validation.route_assignment import adjacent_pairs_from_ordered_tracts, screenline_id


def test_screenline_ids_are_undirected() -> None:
    assert screenline_id("b", "a") == screenline_id("a", "b")
    assert adjacent_pairs_from_ordered_tracts(["1", "2", "2", "3"]) == [("1", "2"), ("2", "3")]


def test_aadt_metrics_include_geh() -> None:
    metrics = compute_aadt_metrics(np.array([1000, 2000, 3000]), np.array([980, 2200, 2800]))
    assert metrics["screenlines"] == 3
    assert "share_geh_lt_5" in metrics
    assert 0 <= metrics["share_geh_lt_5"] <= 1


def test_virtual_counts_average_weekday() -> None:
    import pandas as pd

    from trip_synth.validation.aadt_validation import _synthetic_average_screenline_counts

    synthetic = pd.DataFrame(
        {
            "o_tract_fips": ["24001000100", "24001000100", "24001000100", "24001000100"],
            "d_tract_fips": ["24001000200", "24001000200", "24001000200", "24001000200"],
            "tdate_dow": [1, 1, 2, 6],
        }
    )
    paths = pd.DataFrame(
        {
            "o_tract_fips": ["24001000100"],
            "d_tract_fips": ["24001000200"],
            "path_position": [0],
            "screenline_id": ["24001000100__24001000200"],
        }
    )
    counts = _synthetic_average_screenline_counts(synthetic, paths, "average_weekday")
    assert counts.loc[0, "screenline_id"] == "24001000100__24001000200"
    assert counts.loc[0, "synthetic_count"] == 1.5


def test_trip_date_reconstruction_uses_iso_dow() -> None:
    import pandas as pd

    from trip_synth.validation.aadt_validation import trip_dates_from_week_dow

    frame = pd.DataFrame({"tdate_week": [75, 75, 75], "tdate_dow": [1, 2, 7]})
    dates = trip_dates_from_week_dow(
        frame,
        {"aadt_validation": {"trip_week_origin_date": "2017-01-01", "tdate_dow_encoding": "iso_monday_1"}},
    )
    assert dates.dt.strftime("%Y-%m-%d").tolist() == ["2018-06-11", "2018-06-12", "2018-06-10"]


def test_virtual_counts_exact_day() -> None:
    import pandas as pd

    from trip_synth.validation.aadt_validation import _synthetic_exact_day_screenline_counts

    synthetic = pd.DataFrame(
        {
            "o_tract_fips": ["24001000100", "24001000100", "24001000100"],
            "d_tract_fips": ["24001000200", "24001000200", "24001000200"],
            "tdate_week": [75, 75, 75],
            "tdate_dow": [1, 1, 2],
        }
    )
    paths = pd.DataFrame(
        {
            "o_tract_fips": ["24001000100"],
            "d_tract_fips": ["24001000200"],
            "path_position": [0],
            "screenline_id": ["24001000100__24001000200"],
        }
    )
    counts = _synthetic_exact_day_screenline_counts(
        synthetic,
        paths,
        {"aadt_validation": {"trip_week_origin_date": "2017-01-01", "tdate_dow_encoding": "iso_monday_1"}},
    ).sort_values("trip_date")
    assert counts["screenline_id"].tolist() == ["24001000100__24001000200", "24001000100__24001000200"]
    assert counts["trip_date"].dt.strftime("%Y-%m-%d").tolist() == ["2018-06-11", "2018-06-12"]
    assert counts["synthetic_count"].tolist() == [2, 1]


def test_daily_station_counts_loader_uses_configured_columns(tmp_path) -> None:
    import pandas as pd

    from trip_synth.validation.aadt_validation import load_daily_station_counts

    path = tmp_path / "daily_counts.csv"
    pd.DataFrame(
        {
            "LOCATION_ID": ["A1", "A1", "A2"],
            "COUNT_DATE": ["2018-06-11", "2018-06-11", "2018-06-12"],
            "VOL": [10, 5, 7],
        }
    ).to_csv(path, index=False)
    counts = load_daily_station_counts(
        {
            "aadt_validation": {
                "daily_counts": {
                    "file": str(path),
                    "station_id_column": "LOCATION_ID",
                    "date_column": "COUNT_DATE",
                    "count_column": "VOL",
                }
            }
        }
    ).sort_values(["station_id", "trip_date"])
    assert counts["station_id"].tolist() == ["A1", "A2"]
    assert counts["trip_date"].dt.strftime("%Y-%m-%d").tolist() == ["2018-06-11", "2018-06-12"]
    assert counts["observed_count"].tolist() == [15, 7]


def test_observed_count_field_priority_prefers_2018() -> None:
    from trip_synth.validation.aadt_screenlines import choose_observed_count_field, comparison_basis_for_field

    config = {
        "aadt_validation": {
            "observed_count_field_priority": ["AAWDT_2018", "AADT_2018", "AAWDT", "AADT"],
            "comparison_basis": "average_weekday",
        }
    }
    field = choose_observed_count_field(["OBJECTID", "AADT", "AAWDT", "AAWDT_2018"], config)
    assert field == "AAWDT_2018"
    assert comparison_basis_for_field(config, field) == "average_weekday"


def test_station_boundary_membership_requires_boundary_closer_than_centroid() -> None:
    from shapely.geometry import LineString, Point

    from trip_synth.validation.aadt_screenlines import _station_boundary_membership

    boundary = LineString([(0, 0), (0, 10)])
    tract_a_centroid = Point(-100, 5)
    tract_b_centroid = Point(100, 5)

    near_boundary, boundary_dist, centroid_dist = _station_boundary_membership(
        Point(5, 5), boundary, tract_a_centroid, tract_b_centroid
    )
    near_centroid, _, _ = _station_boundary_membership(Point(-90, 5), boundary, tract_a_centroid, tract_b_centroid)

    assert near_boundary is True
    assert boundary_dist < centroid_dist
    assert near_centroid is False



def test_hourly_station_counts_loader_uses_configured_columns(tmp_path) -> None:
    import pandas as pd

    from trip_synth.validation.aadt_validation import load_hourly_station_counts

    path = tmp_path / "hourly_counts.csv"
    pd.DataFrame(
        {
            "LOC": ["P1", "P1", "P1"],
            "DAY": ["2018-06-11", "2018-06-11", "2018-06-11"],
            "HR": [8, 8, 9],
            "VOL": [10, 5, 7],
        }
    ).to_csv(path, index=False)
    counts = load_hourly_station_counts(
        {
            "aadt_validation": {
                "hourly_counts": {
                    "file": str(path),
                    "station_id_column": "LOC",
                    "date_column": "DAY",
                    "hour_column": "HR",
                    "count_column": "VOL",
                }
            }
        }
    ).sort_values(["station_id", "trip_date", "hour"])
    assert counts["station_id"].tolist() == ["P1", "P1"]
    assert counts["hour"].tolist() == [8, 9]
    assert counts["observed_count"].tolist() == [15, 7]


def test_synthetic_hourly_counts_use_path_fraction() -> None:
    import pandas as pd

    from trip_synth.validation.aadt_validation import _synthetic_hourly_screenline_counts

    synthetic = pd.DataFrame(
        {
            "o_tract_fips": ["24001000100"],
            "d_tract_fips": ["24001000300"],
            "tdate_week": [75],
            "tdate_dow": [1],
            "departure_time_minutes": [60],
            "reported_travel_time": [120],
        }
    )
    paths = pd.DataFrame(
        {
            "o_tract_fips": ["24001000100", "24001000100"],
            "d_tract_fips": ["24001000300", "24001000300"],
            "path_position": [0, 1],
            "screenline_id": ["A__B", "B__C"],
        }
    )
    counts = _synthetic_hourly_screenline_counts(
        synthetic,
        paths,
        {"aadt_validation": {"trip_week_origin_date": "2017-01-01", "tdate_dow_encoding": "iso_monday_1"}},
    ).sort_values("screenline_id")
    assert counts["screenline_id"].tolist() == ["A__B", "B__C"]
    assert counts["hour"].tolist() == [1, 2]
    assert counts["synthetic_count"].tolist() == [1, 1]


def test_observed_hourly_counts_scale_by_annual_station_share(tmp_path) -> None:
    import pandas as pd

    from trip_synth.validation.aadt_validation import build_observed_hourly_screenline_counts

    run_dir = tmp_path / "run"
    (run_dir / "geo").mkdir(parents=True)
    (run_dir / "metrics").mkdir(parents=True)
    pd.DataFrame(
        {
            "screenline_id": ["L1", "L1"],
            "station_id": ["P1", "P2"],
            "observed_count": [25, 75],
        }
    ).to_parquet(run_dir / "geo" / "screenline_station_map.parquet")
    hourly_path = tmp_path / "hourly.csv"
    pd.DataFrame(
        {
            "station_id": ["P1", "P2", "P1"],
            "date": ["2018-06-11", "2018-06-11", "2018-06-11"],
            "hour": [8, 8, 9],
            "observed_count": [10, 30, 10],
        }
    ).to_csv(hourly_path, index=False)
    screenlines = pd.DataFrame({"screenline_id": ["L1"], "observed_count": [100]})
    observed = build_observed_hourly_screenline_counts(
        {
            "aadt_validation": {
                "hourly_counts": {
                    "file": str(hourly_path),
                    "station_id_column": "station_id",
                    "date_column": "date",
                    "hour_column": "hour",
                    "count_column": "observed_count",
                }
            }
        },
        run_dir,
        screenlines,
    ).sort_values("hour")
    assert observed["hour"].tolist() == [8, 9]
    assert observed["annual_share_observed"].round(2).tolist() == [1.0, 0.25]
    assert observed["observed_count"].round(6).tolist() == [40.0, 40.0]

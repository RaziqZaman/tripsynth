from pathlib import Path

import yaml

from tripsynth.config import load_config
from tripsynth.experiments.baseline import run_weighted_resampling_desire_line_baseline


def test_weighted_resampling_desire_line_baseline_with_local_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "o,d,week,dow,dep,mode,weight\n"
        "11001000100,24031000100,40,2,480,4,2.0\n"
        "24031000100,11001000100,40,3,500,4,1.0\n"
        "11001000100,11001000200,40,4,520,1,1.0\n",
        encoding="utf-8",
    )
    counts = tmp_path / "counts.csv"
    counts.write_text(
        "segment_id,county_fips,observed_aadt,wkt\n"
        "dc_1,11001,1000,\"LINESTRING (-77.05 38.9, -77.04 38.91)\"\n"
        "md_1,24031,800,\"LINESTRING (-77.20 39.0, -77.19 39.01)\"\n",
        encoding="utf-8",
    )
    config = {
        "project": {"seed": 7, "crs_projected": "EPSG:26918"},
        "survey": {
            "path": "survey.csv",
            "schema": {
                "origin_tract": "o",
                "destination_tract": "d",
                "survey_week": "week",
                "day_of_week": "dow",
                "departure_time": "dep",
                "mode": "mode",
                "weight": "weight",
            },
            "temporal_derivation": {
                "derive_trip_date_from_week_dow": True,
                "reference_date": "2017-01-01",
                "week_field": "survey_week",
                "day_of_week_field": "day_of_week",
                "week_index_base": 0,
            },
        },
        "study_area": {"counties": {"dc": ["11001"], "md": ["24031"]}},
        "observed_counts": {
            "default_source": "local_user_counts",
            "sources": {
                "local_user_counts": {
                    "enabled": True,
                    "type": "local_file",
                    "path": "counts.csv",
                    "state": "dmv",
                    "temporal_type": "annual_average",
                    "field_map": {
                        "segment_id": "segment_id",
                        "county_fips": "county_fips",
                        "observed_aadt": "observed_aadt",
                        "geometry": "wkt",
                    },
                }
            },
        },
        "validation": {
            "vehicle_trip_filter": {
                "mode_field": "mode",
                "auto_modes": [4],
                "include_when_mode_missing": False,
            },
            "baseline": {
                "scale_factor": 1,
                "routing_method": "tract_desire_line_proxy",
                "desire_line_buffer_m": 10000,
                "route_volume_col": None,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_weighted_resampling_desire_line_baseline(
        load_config(config_path), run_dir=tmp_path / "run", seed=7
    )
    assert result.summary["vehicle_filter"]["vehicle_trip_count"] == 2
    assert result.summary["routing"]["routes"] >= 1
    assert Path(result.summary["outputs"]["metrics_by_run"]).exists()
    assert Path(result.summary["outputs"]["metrics_by_segment"]).exists()

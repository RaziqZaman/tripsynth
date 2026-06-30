from pathlib import Path

import pandas as pd
import yaml

from tripsynth.config import load_config
from tripsynth.experiments.method_comparison import run_synthesis_method_comparison


def test_method_comparison_writes_compact_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "o,d,week,dow,dep,mode,purpose,weight\n"
        "24031000100,24033000100,40,2,480,4,1,2.0\n"
        "24031000200,24033000200,40,3,500,3,2,1.0\n"
        "24033000100,24031000100,40,4,520,4,1,1.0\n"
        "24033000200,24031000200,40,5,540,3,3,1.0\n",
        encoding="utf-8",
    )
    counts = tmp_path / "counts.csv"
    counts.write_text(
        "segment_id,county_fips,observed_aadt,wkt\n"
        "md_1,24031,1000,\"LINESTRING (-77.20 39.0, -77.19 39.01)\"\n"
        "md_2,24033,800,\"LINESTRING (-76.90 38.9, -76.89 38.91)\"\n",
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
                "purpose": "purpose",
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
        "study_area": {"counties": {"md": ["24031", "24033"]}},
        "observed_counts": {
            "default_source": "local_user_counts",
            "sources": {
                "local_user_counts": {
                    "enabled": True,
                    "type": "local_file",
                    "path": "counts.csv",
                    "state": "md",
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
            "vehicle_trip_filter": {"mode_field": "mode", "auto_modes": [3, 4]},
            "baseline": {"scale_factor": 1, "routing_method": "tract_desire_line_proxy"},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_synthesis_method_comparison(
        load_config(config_path),
        methods=["weighted_resampling", "bayesian_network"],
        seeds=[7],
        run_dir=tmp_path / "comparison",
    )

    assert len(result.comparison) == 2
    assert set(result.comparison["synthesis_method"]) == {"weighted_resampling", "bayesian_network"}
    assert set(result.comparison["routing_method"]) == {"tract_desire_line_proxy"}
    assert Path(result.summary["comparison_by_run"]).exists()
    assert Path(result.summary["comparison_compact"]).exists()
    compact = pd.read_csv(result.summary["comparison_compact"])
    assert "routing_method" in compact
    assert "routing_edge_weight_strategy" in compact

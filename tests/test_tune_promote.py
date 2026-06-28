from pathlib import Path

import yaml

from tripsynth.config import load_config
from tripsynth.experiments.tune_promote import run_tune_promote_synthesis


def test_tune_promote_selects_and_promotes_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "\n".join(
            [
                "o,d,week,dow,dep,mode,purpose,weight,raw_extra",
                "24031000100,24033000100,40,2,480,4,1,2.0,a",
                "24031000200,24033000200,40,3,500,3,2,1.0,b",
                "24033000100,24031000100,40,4,520,4,1,1.0,c",
                "24033000200,24031000200,40,5,540,3,3,1.0,d",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    counts = tmp_path / "counts.csv"
    counts.write_text(
        "\n".join(
            [
                "segment_id,county_fips,observed_aadt,wkt",
                "md_1,24031,1000,\"LINESTRING (-77.20 39.0, -77.19 39.01)\"",
                "md_2,24033,800,\"LINESTRING (-76.90 38.9, -76.89 38.91)\"",
            ]
        )
        + "\n",
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
        "experiment": {
            "synthesis_hyperparameter_grids": {
                "weighted_resampling": {"weight_field": ["weight"]},
                "bayesian_network": {"smoothing": [0.1], "weight_field": ["weight"]},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_tune_promote_synthesis(
        load_config(config_path),
        methods=["weighted_resampling", "bayesian_network"],
        sweep_seeds=[7],
        promotion_seeds=[7],
        target_trip_count=4,
        target_population_share=None,
        run_dir=tmp_path / "tune_promote",
    )

    assert len(result.selected_configs) == 2
    assert len(result.promotion.leaderboard) == 2
    assert Path(result.summary["summary_path"]).exists()

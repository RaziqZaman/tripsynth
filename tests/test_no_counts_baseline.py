import yaml

from tripsynth.config import load_config
from tripsynth.experiments.baseline_modes import run_weighted_resampling_desire_line_baseline


def test_default_baseline_runs_without_observed_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "o,d,week,dow,dep,mode,purpose,weight\n"
        "11001000100,24031000100,40,2,480,4,1,2.0\n"
        "24031000100,11001000100,40,3,500,4,2,1.0\n"
        "11001000100,11001000200,41,4,520,1,3,1.0\n"
        "11001000200,11001000100,41,5,600,3,4,1.0\n",
        encoding="utf-8",
    )
    config = {
        "project": {"seed": 7},
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
        "validation": {
            "default_mode": "survey_holdout",
            "survey_holdout": {
                "holdout_share": 0.25,
                "features": ["origin_county_fips", "destination_county_fips", "mode"],
                "weight_field": "weight",
            },
            "baseline": {"scale_factor": 1},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_weighted_resampling_desire_line_baseline(
        load_config(config_path), run_dir=tmp_path / "run", seed=7
    )

    assert result.summary["validation_mode"] == "survey_holdout"
    assert result.summary["observed_counts"]["required"] is False
    assert (tmp_path / "run" / "metrics" / "metrics_by_run.csv").exists()
    assert (tmp_path / "run" / "metrics" / "metrics_by_feature.csv").exists()

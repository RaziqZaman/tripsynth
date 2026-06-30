from pathlib import Path

import pandas as pd

from tripsynth.data_sources.survey import SurveyData, build_synthesis_frame
from tripsynth.synthesis.factory import synthesize_trips


def _toy_trips():
    return pd.DataFrame(
        {
            "origin_tract": ["24031000100", "24031000200", "24033000100", "24033000200"],
            "destination_tract": ["24033000100", "24033000200", "24031000100", "24031000200"],
            "mode": [3, 4, 3, 4],
            "purpose": [1, 2, 1, 3],
            "day_of_week": [1, 2, 3, 4],
            "departure_time": [480, 510, 600, 720],
            "travel_time_min": [25, 30, 35, 45],
            "vehicle_available": [1, 1, 2, 2],
            "vehicle_occupancy": [1, 2, 1, 3],
            "weight": [1.0, 2.0, 1.0, 3.0],
            "o_tract_fips": ["24031000100", "24031000200", "24033000100", "24033000200"],
            "d_tract_fips": ["24033000100", "24033000200", "24031000100", "24031000200"],
            "home_tract_fips": ["24031000100", "24031000200", "24033000100", "24033000200"],
            "hh_income_detailed": [3, 5, 7, 9],
            "license": [1, 1, 0, 1],
        }
    )


def test_all_synthesis_methods_sample_canonical_tract_trips():
    config = {
        "synthesis": {
            "vae": {"epochs": 1, "hidden_dim": 8, "latent_dim": 3, "batch_size": 4},
            "contrastive_vae": {
                "epochs": 1,
                "hidden_dim": 8,
                "latent_dim": 3,
                "batch_size": 4,
                "contrastive_batch_size": 4,
                "contrastive_weight": 0.2,
                "sample_batch_size": 8,
            },
        }
    }
    for method in ["weighted_resampling", "bayesian_network", "vae", "contrastive_vae", "diffusion"]:
        result = synthesize_trips(method, _toy_trips(), n=8, seed=11, config=config)
        assert result.method == method
        assert len(result.synthetic_trips) == 8
        assert result.synthetic_trips["origin_tract"].notna().all()
        assert result.synthetic_trips["destination_tract"].notna().all()
        assert "synthetic_trip_id" in result.synthetic_trips
        assert "home_tract_fips" in result.synthetic_trips
        assert result.metadata["n"] == 8
        if method in {"vae", "contrastive_vae", "diffusion"}:
            assert result.metadata["model_all_columns"] is True
            assert "home_tract_fips" in result.metadata["modeled_columns"]
        if method == "contrastive_vae":
            assert result.metadata["contrastive_weight"] == 0.2
            assert result.metadata["sample_from_training_latent"] is True


def test_build_synthesis_frame_includes_raw_and_canonical_columns():
    raw = pd.DataFrame(
        {
            "o": ["24031000100"],
            "d": ["24033000100"],
            "mode_raw": [3],
            "extra_behavior": ["commute"],
            "weight": [2.0],
        }
    )
    canonical = pd.DataFrame(
        {
            "origin_tract": ["24031000100"],
            "destination_tract": ["24033000100"],
            "mode": [3],
            "weight": [2.0],
        }
    )
    survey = SurveyData(raw=raw, canonical=canonical, schema_report=[], path=Path("survey.csv"))

    frame = build_synthesis_frame(survey, {})

    assert "origin_tract" in frame
    assert "o" in frame
    assert "extra_behavior" in frame
    assert frame.attrs["raw_survey_columns"] == ["o", "d", "mode_raw", "extra_behavior", "weight"]


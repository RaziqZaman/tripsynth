from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from trip_synth.baselines import weighted_bootstrap
from trip_synth.data.load import read_survey_csv
from trip_synth.data.postprocessing import assert_synthetic_contract
from trip_synth.data.schema import FeatureSchema


def test_weighted_bootstrap_contract() -> None:
    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    df, _ = read_survey_csv("05sample_transformed_survey.csv", schema)
    with tempfile.TemporaryDirectory() as td:
        config = {"seed": 123, "run_name": "unit"}
        art = weighted_bootstrap.fit(df, schema, config, Path(td))
        synth = weighted_bootstrap.sample(art, 25, schema, config, Path(td))
    assert len(synth) == 25
    assert "weight" not in synth.columns
    assert "synthetic_method" in synth.columns
    assert pd.api.types.is_integer_dtype(synth["year"])
    assert_synthetic_contract(synth, schema)

from __future__ import annotations

import tempfile
from pathlib import Path

from trip_synth.baselines import weighted_bootstrap
from trip_synth.data.load import read_survey_csv
from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.validation.privacy import validate_method_privacy


def test_privacy_detects_bootstrap_copying() -> None:
    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    df, _ = read_survey_csv("05sample_transformed_survey.csv", schema)
    pre = FittedPreprocessor.fit(df, schema)
    with tempfile.TemporaryDirectory() as td:
        config = {"seed": 123, "run_name": "unit"}
        art = weighted_bootstrap.fit(df, schema, config, Path(td))
        synth = weighted_bootstrap.sample(art, 50, schema, config, Path(td))
        metrics = validate_method_privacy(df, synth, schema, pre, "weighted_bootstrap", Path(td))
    assert metrics["exact_row_copy_rate"] == 1.0
    assert metrics["invalid_category_rate"] == 0.0

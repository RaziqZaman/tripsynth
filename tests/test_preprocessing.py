from __future__ import annotations

from trip_synth.data.load import read_survey_csv
from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema


def test_preprocessor_round_trip_valid_categories() -> None:
    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    df, _ = read_survey_csv("05sample_transformed_survey.csv", schema)
    pre = FittedPreprocessor.fit(df, schema)
    arrays = pre.transform(df.head(10))
    out = pre.inverse_transform(arrays["cat"], arrays["num"])
    assert list(out.columns) == pre.feature_columns
    assert "weight" not in out.columns
    for col in pre.categorical_columns:
        assert set(out[col].astype(str)).issubset(set(pre.categories[col]))
    assert out["year"].dtype.kind in {"i", "u"}

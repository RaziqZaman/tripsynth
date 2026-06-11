from __future__ import annotations

import pandas as pd

from trip_synth.data.load import read_survey_csv
from trip_synth.data.schema import FeatureSchema, clean_fips_value


def test_schema_casts_year_and_fips() -> None:
    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    df, warnings = read_survey_csv("05sample_transformed_survey.csv", schema)
    assert "weight" in df.columns
    assert pd.api.types.is_integer_dtype(df["year"])
    assert clean_fips_value("24005411303.0") == "24005411303"
    assert df["o_tract_fips"].astype(str).str.len().ge(2).all()
    assert warnings == []


def test_synthesis_columns_exclude_weight() -> None:
    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    df, _ = read_survey_csv("05sample_transformed_survey.csv", schema)
    cols = schema.synthesis_columns(df)
    assert "weight" not in cols
    assert "year" in cols

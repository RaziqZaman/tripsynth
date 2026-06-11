from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import FeatureSchema


def read_survey_csv(path: str | Path, schema: FeatureSchema) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    dtype = {col: "string" for col in schema.categorical_columns}
    use_dtype = {col: typ for col, typ in dtype.items()}
    df = pd.read_csv(path, dtype=use_dtype, low_memory=False)
    return schema.clean_dataframe(df)

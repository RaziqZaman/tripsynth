"""Feature engineering helpers that avoid unsafe calendar assumptions."""

from __future__ import annotations

import pandas as pd


def allowed_temporal_features(frame: pd.DataFrame, *, allow_year: bool = False, allow_month: bool = False) -> list[str]:
    features = []
    if "departure_time" in frame:
        features.append("departure_time")
    if "day_of_week" in frame:
        features.append("day_of_week")
    if allow_year and "survey_year" in frame:
        features.append("survey_year")
    if allow_month and "survey_month" in frame:
        features.append("survey_month")
    return features

"""Survey cleaning placeholders."""

from __future__ import annotations

import pandas as pd


def passthrough_cleaning(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy()

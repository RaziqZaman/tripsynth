"""Common synthesis interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class SynthesisResult:
    method: str
    synthetic_trips: pd.DataFrame
    metadata: dict[str, Any]


class TripSynthesizer(Protocol):
    method: str

    def fit(self, trips: pd.DataFrame) -> "TripSynthesizer":
        ...

    def sample(self, n: int, *, seed: int | None = None) -> SynthesisResult:
        ...

"""Weighted random resampling baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tripsynth.synthesis.base import SynthesisResult


@dataclass
class WeightedResampler:
    weight_field: str = "weight"
    method: str = "weighted_resampling"

    def fit(self, trips: pd.DataFrame) -> "WeightedResampler":
        self.trips_ = trips.copy()
        self.input_columns_ = list(trips.columns)
        return self

    def sample(self, n: int, *, seed: int | None = None) -> SynthesisResult:
        if not hasattr(self, "trips_"):
            raise RuntimeError("WeightedResampler.fit must be called before sample.")
        trips = self.trips_
        if self.weight_field in trips:
            weights = pd.to_numeric(trips[self.weight_field], errors="coerce").fillna(0)
            weights = weights.where(weights > 0, 0)
            probabilities = weights / weights.sum() if weights.sum() > 0 else None
        else:
            probabilities = None
        rng = np.random.default_rng(seed)
        indices = rng.choice(trips.index.to_numpy(), size=n, replace=True, p=probabilities)
        synthetic = trips.loc[indices].reset_index(drop=True)
        synthetic["synthetic_trip_id"] = [f"wr_{seed or 0}_{i}" for i in range(n)]
        return SynthesisResult(
            method=self.method,
            synthetic_trips=synthetic,
            metadata={
                "n": n,
                "seed": seed,
                "used_weights": probabilities is not None,
                "weight_field": self.weight_field,
                "input_columns": len(self.input_columns_),
                "row_level_full_table_resampling": True,
            },
        )


def sample_by_scale(
    trips: pd.DataFrame, *, scale_factor: float, weight_field: str = "weight", seed: int = 42
) -> SynthesisResult:
    n = max(1, int(math.ceil(len(trips) * scale_factor)))
    return WeightedResampler(weight_field=weight_field).fit(trips).sample(n, seed=seed)

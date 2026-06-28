"""Bayesian-network synthesis placeholder with a conservative fallback."""

from __future__ import annotations

import pandas as pd

from tripsynth.synthesis.base import SynthesisResult
from tripsynth.synthesis.weighted_resampling import WeightedResampler


class BayesianNetworkSynthesizer:
    method = "bayesian_network"

    def fit(self, trips: pd.DataFrame) -> "BayesianNetworkSynthesizer":
        self.trips_ = trips.copy()
        return self

    def sample(self, n: int, *, seed: int | None = None) -> SynthesisResult:
        result = WeightedResampler().fit(self.trips_).sample(n, seed=seed)
        return SynthesisResult(
            method=self.method,
            synthetic_trips=result.synthetic_trips,
            metadata={
                **result.metadata,
                "fallback": "weighted_resampling",
                "note": "Install/configure pgmpy for learned Bayesian-network synthesis.",
            },
        )

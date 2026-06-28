"""Tabular VAE synthesis placeholder."""

from __future__ import annotations


class VAESynthesizer:
    method = "vae"

    def fit(self, trips):
        self.trips_ = trips.copy()
        return self

    def sample(self, n: int, *, seed: int | None = None):
        raise NotImplementedError(
            "VAE synthesis is scaffolded but not part of the first deliverable. "
            "Use weighted_resampling for the current smoke pipeline."
        )

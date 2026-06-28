"""Lightweight empirical Bayesian-network style synthesizer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from tripsynth.progress import progress
from tripsynth.synthesis.base import SynthesisResult
from tripsynth.synthesis.tabular_utils import normalize_fips_text


class BayesianNetworkSynthesizer:
    """Sample a tract OD chain plus donor attributes from empirical conditionals.

    This is intentionally dependency-light: it does not require pgmpy. The learned
    structure is origin tract -> destination tract -> donor attributes, with small
    Dirichlet smoothing on origin/destination probabilities.
    """

    method = "bayesian_network"

    def __init__(self, *, smoothing: float = 0.5, weight_field: str = "weight") -> None:
        self.smoothing = float(smoothing)
        self.weight_field = weight_field

    def fit(self, trips: pd.DataFrame) -> "BayesianNetworkSynthesizer":
        self.trips_ = trips.reset_index(drop=True).copy()
        self.input_columns_ = list(trips.columns)
        working = self.trips_.copy()
        working["_origin"] = working["origin_tract"].map(normalize_fips_text)
        working["_destination"] = working["destination_tract"].map(normalize_fips_text)
        working = working.dropna(subset=["_origin", "_destination"])
        if working.empty:
            raise ValueError("BayesianNetworkSynthesizer requires origin and destination tracts.")

        if self.weight_field in working:
            weights = pd.to_numeric(working[self.weight_field], errors="coerce").fillna(0).clip(lower=0)
            working["_fit_weight"] = weights.where(weights > 0, 1.0)
        else:
            working["_fit_weight"] = 1.0

        self.origins_ = sorted(working["_origin"].unique().tolist())
        origin_counts = working.groupby("_origin")["_fit_weight"].sum().reindex(self.origins_, fill_value=0.0)
        origin_probs = origin_counts + self.smoothing
        self.origin_probabilities_ = (origin_probs / origin_probs.sum()).to_numpy(dtype=float)

        self.destinations_by_origin_: dict[str, list[str]] = {}
        self.destination_probabilities_: dict[str, np.ndarray] = {}
        self.indices_by_pair_: dict[tuple[str, str], np.ndarray] = {}
        self.probabilities_by_pair_: dict[tuple[str, str], np.ndarray | None] = {}

        global_destinations = sorted(working["_destination"].unique().tolist())
        global_counts = working.groupby("_destination")["_fit_weight"].sum().reindex(global_destinations, fill_value=0.0)
        global_probs = (global_counts + self.smoothing) / (global_counts.sum() + self.smoothing * len(global_counts))

        for origin in self.origins_:
            sub = working.loc[working["_origin"] == origin]
            destinations = sorted(sub["_destination"].unique().tolist())
            counts = sub.groupby("_destination")["_fit_weight"].sum().reindex(destinations, fill_value=0.0)
            smoothed = counts + self.smoothing
            # Blend very lightly toward the global destination distribution.
            global_aligned = global_probs.reindex(destinations, fill_value=0.0).to_numpy(dtype=float)
            probs = 0.98 * (smoothed / smoothed.sum()).to_numpy(dtype=float) + 0.02 * global_aligned
            probs = probs / probs.sum()
            self.destinations_by_origin_[origin] = destinations
            self.destination_probabilities_[origin] = probs

            for destination in destinations:
                pair_sub = sub.loc[sub["_destination"] == destination]
                indices = pair_sub.index.to_numpy(dtype=int)
                self.indices_by_pair_[(origin, destination)] = indices
                pair_weights = pair_sub["_fit_weight"].to_numpy(dtype=float)
                self.probabilities_by_pair_[(origin, destination)] = (
                    pair_weights / pair_weights.sum() if pair_weights.sum() > 0 else None
                )

        self.fallback_indices_ = working.index.to_numpy(dtype=int)
        fallback_weights = working["_fit_weight"].to_numpy(dtype=float)
        self.fallback_probabilities_ = fallback_weights / fallback_weights.sum() if fallback_weights.sum() > 0 else None
        return self

    def sample(self, n: int, *, seed: int | None = None) -> SynthesisResult:
        if not hasattr(self, "trips_"):
            raise RuntimeError("BayesianNetworkSynthesizer.fit must be called before sample.")
        rng = np.random.default_rng(seed)
        origins = rng.choice(self.origins_, size=n, replace=True, p=self.origin_probabilities_)
        donor_indices = []
        sampled_destinations = []
        for origin in progress(origins, total=int(n), desc="bayesian sample", unit="trip"):
            destinations = self.destinations_by_origin_[origin]
            destination = rng.choice(destinations, p=self.destination_probabilities_[origin])
            sampled_destinations.append(destination)
            pair = (origin, destination)
            candidates = self.indices_by_pair_.get(pair, self.fallback_indices_)
            probabilities = self.probabilities_by_pair_.get(pair, self.fallback_probabilities_)
            donor_indices.append(int(rng.choice(candidates, p=probabilities)))

        synthetic = self.trips_.iloc[donor_indices].reset_index(drop=True).copy()
        synthetic["origin_tract"] = origins
        synthetic["destination_tract"] = sampled_destinations
        synthetic["synthetic_trip_id"] = [f"bn_{seed or 0}_{i}" for i in range(n)]
        return SynthesisResult(
            method=self.method,
            synthetic_trips=synthetic,
            metadata={
                "n": int(n),
                "seed": seed,
                "implementation": "weighted_od_conditional_full_row_donor_sampler",
                "smoothing": self.smoothing,
                "weight_field": self.weight_field,
                "input_columns": len(self.input_columns_),
                "row_level_full_table_donor_sampling": True,
                "unique_origins": len(self.origins_),
                "unique_od_pairs": len(self.indices_by_pair_),
            },
        )

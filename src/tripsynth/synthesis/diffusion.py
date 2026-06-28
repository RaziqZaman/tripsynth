"""Lightweight tabular diffusion-style synthesizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from tripsynth.progress import progress
from tripsynth.synthesis.base import SynthesisResult
from tripsynth.synthesis.tabular_utils import TabularTripEncoder, sample_donor_indices


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class DiffusionSynthesizer:
    """Generate trips by adding and partially denoising encoded empirical samples.

    This is a runnable tabular diffusion proxy: it samples observed encoded trips,
    injects Gaussian noise through a short schedule, and decodes back to a full
    raw-plus-canonical trip table. OD is also encoded as a tract-pair category to
    keep route assignment inside empirical support.
    """

    method = "diffusion"

    def __init__(
        self,
        *,
        noise_scale: float = 0.25,
        steps: int = 6,
        weight_field: str = "weight",
        model_all_columns: bool = True,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        max_categorical_cardinality: int = 50,
        canonical_aliases: dict[str, str] | None = None,
        device: str = "auto",
        sample_batch_size: int = 524288,
    ) -> None:
        self.noise_scale = float(noise_scale)
        self.steps = int(steps)
        self.weight_field = weight_field
        self.model_all_columns = bool(model_all_columns)
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.max_categorical_cardinality = int(max_categorical_cardinality)
        self.canonical_aliases = canonical_aliases or {}
        self.device = device
        self.sample_batch_size = int(sample_batch_size)

    def fit(self, trips: pd.DataFrame) -> "DiffusionSynthesizer":
        self.device_ = _resolve_device(self.device)
        if self.device_.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.encoder_ = TabularTripEncoder(
            categorical_columns=self.categorical_columns,
            numeric_columns=self.numeric_columns,
            model_all_columns=self.model_all_columns,
            weight_field=self.weight_field,
            max_categorical_cardinality=self.max_categorical_cardinality,
            canonical_aliases=self.canonical_aliases,
        ).fit(trips)
        self.encoded_ = self.encoder_.transform(trips)
        self.encoded_tensor_ = torch.tensor(self.encoded_, dtype=torch.float32, device=self.device_)
        return self

    def sample(self, n: int, *, seed: int | None = None) -> SynthesisResult:
        if not hasattr(self, "encoded_"):
            raise RuntimeError("DiffusionSynthesizer.fit must be called before sample.")
        if seed is not None:
            torch.manual_seed(int(seed))
        donor_indices = sample_donor_indices(
            self.encoder_.trips_, int(n), seed=seed, weight_field=self.weight_field
        )
        frames = []
        sample_batch_size = max(1, int(self.sample_batch_size))
        sample_starts = range(0, int(n), sample_batch_size)
        sample_chunks = (int(n) + sample_batch_size - 1) // sample_batch_size
        with torch.no_grad():
            for start in progress(sample_starts, total=sample_chunks, desc="diffusion sample", unit="chunk"):
                end = min(int(n), start + sample_batch_size)
                index = torch.tensor(donor_indices[start:end], dtype=torch.long, device=self.device_)
                x = self.encoded_tensor_.index_select(0, index).clone()
                for step in range(max(1, self.steps)):
                    fraction = 1.0 - step / max(1, self.steps)
                    noise = torch.randn_like(x) * (self.noise_scale * fraction)
                    x = 0.92 * x + 0.08 * torch.tanh(x + noise)
                decoded = x.detach().cpu().numpy()
                frames.append(
                    self.encoder_.decode(decoded, seed=seed, donor_indices=donor_indices[start:end])
                )
        synthetic = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        synthetic["synthetic_trip_id"] = [f"diff_{seed or 0}_{i}" for i in range(int(n))]
        return SynthesisResult(
            method=self.method,
            synthetic_trips=synthetic,
            metadata={
                "n": int(n),
                "seed": seed,
                "implementation": "weighted_full_table_diffusion_proxy",
                "noise_scale": self.noise_scale,
                "steps": self.steps,
                "weight_field": self.weight_field,
                "model_all_columns": self.model_all_columns,
                "device": str(self.device_),
                "sample_batch_size": self.sample_batch_size,
                "input_columns": len(self.encoder_.columns_),
                "modeled_features": int(self.encoder_.n_features_),
                "modeled_columns": self.encoder_.modeled_columns_,
            },
        )

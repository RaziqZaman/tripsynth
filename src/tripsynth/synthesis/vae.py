"""Small runnable tabular VAE-style synthesizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from tripsynth.progress import progress
from tripsynth.synthesis.base import SynthesisResult
from tripsynth.synthesis.tabular_utils import TabularTripEncoder, sample_donor_indices


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(max(1, int(num_layers))):
        layers.extend([nn.Linear(current, hidden_dim), nn.ReLU()])
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class _TinyVAE(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.encoder = _mlp(n_features, hidden_dim, hidden_dim, num_layers)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = _mlp(latent_dim, hidden_dim, n_features, num_layers)

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-8, 8)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar


class VAESynthesizer:
    method = "vae"

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        hidden_dim: int = 48,
        num_layers: int = 1,
        epochs: int = 4,
        batch_size: int = 1024,
        sample_batch_size: int = 262144,
        learning_rate: float = 1e-3,
        beta: float = 0.01,
        weight_field: str = "weight",
        model_all_columns: bool = True,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        max_categorical_cardinality: int = 50,
        canonical_aliases: dict[str, str] | None = None,
        device: str = "auto",
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.sample_batch_size = int(sample_batch_size)
        self.learning_rate = float(learning_rate)
        self.beta = float(beta)
        self.weight_field = weight_field
        self.model_all_columns = bool(model_all_columns)
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.max_categorical_cardinality = int(max_categorical_cardinality)
        self.canonical_aliases = canonical_aliases or {}
        self.device = device

    def fit(self, trips: pd.DataFrame) -> "VAESynthesizer":
        torch.manual_seed(0)
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
        x = torch.tensor(self.encoder_.transform(trips), dtype=torch.float32)
        weights = torch.tensor(self.encoder_.fit_weights_, dtype=torch.float32)
        self.model_ = _TinyVAE(
            x.shape[1], self.latent_dim, self.hidden_dim, self.num_layers
        ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(0)
        dataset = torch.utils.data.TensorDataset(x, weights)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        self.model_.train()
        losses = []
        epochs = max(1, self.epochs)
        for epoch_index in progress(range(epochs), desc="vae train", unit="epoch"):
            epoch_loss = 0.0
            weight_total = 0.0
            batches = progress(
                loader,
                total=len(loader),
                desc=f"vae epoch {epoch_index + 1}/{epochs}",
                unit="batch",
                leave=False,
            )
            for batch, batch_weights in batches:
                batch = batch.to(self.device_, non_blocking=True)
                batch_weights = batch_weights.to(self.device_, non_blocking=True)
                reconstructed, mu, logvar = self.model_(batch)
                reconstruction = nn.functional.mse_loss(reconstructed, batch, reduction="none").mean(dim=1)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                row_loss = reconstruction + self.beta * kld
                loss = (row_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-6)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float((row_loss.detach() * batch_weights).sum().cpu())
                weight_total += float(batch_weights.sum().cpu())
            losses.append(epoch_loss / max(weight_total, 1e-6))
        self.training_losses_ = losses
        return self

    def sample(self, n: int, *, seed: int | None = None) -> SynthesisResult:
        if not hasattr(self, "model_"):
            raise RuntimeError("VAESynthesizer.fit must be called before sample.")
        torch.manual_seed(int(seed or 0))
        self.model_.eval()
        donor_indices = sample_donor_indices(
            self.encoder_.trips_, int(n), seed=seed, weight_field=self.weight_field
        )
        frames = []
        sample_batch_size = max(1, int(self.sample_batch_size))
        sample_starts = range(0, int(n), sample_batch_size)
        sample_chunks = (int(n) + sample_batch_size - 1) // sample_batch_size
        with torch.no_grad():
            for start in progress(sample_starts, total=sample_chunks, desc="vae sample", unit="chunk"):
                end = min(int(n), start + sample_batch_size)
                batch_n = end - start
                z = torch.randn(batch_n, self.latent_dim, device=self.device_)
                decoded = self.model_.decoder(z).detach().cpu().numpy()
                frames.append(
                    self.encoder_.decode(decoded, seed=seed, donor_indices=donor_indices[start:end])
                )
        synthetic = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        synthetic["synthetic_trip_id"] = [f"vae_{seed or 0}_{i}" for i in range(int(n))]
        return SynthesisResult(
            method=self.method,
            synthetic_trips=synthetic,
            metadata={
                "n": int(n),
                "seed": seed,
                "implementation": "weighted_full_table_vae",
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "sample_batch_size": self.sample_batch_size,
                "beta": self.beta,
                "learning_rate": self.learning_rate,
                "weight_field": self.weight_field,
                "weighted_training_loss": True,
                "model_all_columns": self.model_all_columns,
                "device": str(self.device_),
                "input_columns": len(self.encoder_.columns_),
                "modeled_features": int(self.encoder_.n_features_),
                "modeled_columns": self.encoder_.modeled_columns_,
                "final_training_loss": self.training_losses_[-1] if self.training_losses_ else None,
            },
        )

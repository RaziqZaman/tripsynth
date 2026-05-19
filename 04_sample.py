#!/usr/bin/env python3
"""Train a weighted tabular VAE and synthesize trip records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for bare environments.
    def tqdm(iterable, **_: object):
        return iterable


INPUT_CSV = Path("03x_filled-survey.csv")
OUTPUT_CSV = Path("04x_synthetic-trips.csv")
MODEL_DIR = Path("04_model")
WEIGHT_COLUMN = "wthhfin"

NUMERIC_COLUMNS = [
    "departure_time_min",
    "travelers_hh",
    "vehicle_occupancy",
    "distance",
    "reported_travel_time",
    "year",
    "hhsize",
    "numstudents",
    "numdrivers",
    "numworkers",
    "numdisabilities",
    "numvehicle",
    "numvehicle_transponder",
    "numbicycle",
    "hh_income_detailed",
    "tdate_days",
    "hhtrips",
    "age",
    "jobs_count",
    "j1_telecommute_days",
    "td_telecommute_time",
    "td_shop_time",
    "person_tripcount",
    "walk_bike_loop_trips",
]


@dataclass
class Preprocessor:
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    model_categorical_columns: list[str]
    numeric_missing_columns: list[str]
    missing_indicator_columns: dict[str, str]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    numeric_min: np.ndarray
    numeric_max: np.ndarray
    categories: dict[str, list[str]]


class TripDataset(Dataset):
    def __init__(self, numeric: np.ndarray, categorical: np.ndarray) -> None:
        self.numeric = torch.as_tensor(numeric, dtype=torch.float32)
        self.categorical = torch.as_tensor(categorical, dtype=torch.long)

    def __len__(self) -> int:
        return self.numeric.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.numeric[index], self.categorical[index]


class TabularVAE(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cardinalities: list[int],
        latent_dim: int,
        hidden_dims: list[int],
        embedding_cap: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_numeric = num_numeric
        self.cardinalities = cardinalities
        self.embedding_dims = [
            min(embedding_cap, max(2, math.ceil(math.sqrt(cardinality))))
            for cardinality in cardinalities
        ]

        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dim)
            for cardinality, dim in zip(cardinalities, self.embedding_dims)
        )

        encoder_layers: list[nn.Module] = []
        encoder_input_dim = num_numeric + sum(self.embedding_dims)
        last_dim = encoder_input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            last_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(last_dim, latent_dim)
        self.logvar = nn.Linear(last_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        last_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            last_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        self.numeric_head = nn.Linear(last_dim, num_numeric)
        self.categorical_heads = nn.ModuleList(
            nn.Linear(last_dim, cardinality) for cardinality in cardinalities
        )

    def encode(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        features = torch.cat([numeric, *embedded], dim=1) if embedded else numeric
        hidden = self.encoder(features)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = self.decoder(z)
        numeric = self.numeric_head(hidden)
        categorical = [head(hidden) for head in self.categorical_heads]
        return numeric, categorical

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(numeric, categorical)
        z = self.reparameterize(mu, logvar)
        numeric_out, categorical_out = self.decode(z)
        return numeric_out, categorical_out, mu, logvar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--samples", type=int, default=110_880)
    parser.add_argument("--epochs", type=int, default=5040)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=1800)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[2048, 1920])
    parser.add_argument("--embedding-cap", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_encode(input_csv: Path) -> tuple[TripDataset, np.ndarray, Preprocessor]:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    missing_numeric = sorted(set(NUMERIC_COLUMNS) - set(df.columns))
    if missing_numeric:
        raise SystemExit(f"Missing numeric columns: {missing_numeric}")
    if WEIGHT_COLUMN not in df.columns:
        raise SystemExit(f"Missing survey weight column: {WEIGHT_COLUMN}")

    feature_columns = [column for column in df.columns if column != WEIGHT_COLUMN]
    numeric_columns = [column for column in NUMERIC_COLUMNS if column in feature_columns]
    categorical_columns = [
        column for column in feature_columns if column not in set(numeric_columns)
    ]

    numeric_frame = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_missing = numeric_frame.isna()
    numeric_missing_columns = [
        column for column in numeric_columns if numeric_missing[column].any()
    ]
    fill_values = numeric_frame.median(numeric_only=True).fillna(0)
    numeric_frame = numeric_frame.fillna(fill_values)
    numeric_raw = numeric_frame.to_numpy(dtype=np.float32)
    numeric_mean = numeric_raw.mean(axis=0)
    numeric_std = numeric_raw.std(axis=0)
    numeric_std[numeric_std == 0] = 1.0
    numeric_min = numeric_raw.min(axis=0)
    numeric_max = numeric_raw.max(axis=0)
    numeric = (numeric_raw - numeric_mean) / numeric_std

    categories: dict[str, list[str]] = {}
    model_categorical_columns: list[str] = []
    categorical_arrays = []
    for column in categorical_columns:
        values = df[column].astype(str)
        column_categories = sorted(values.unique().tolist())
        categories[column] = column_categories
        mapping = {value: index for index, value in enumerate(column_categories)}
        categorical_arrays.append(values.map(mapping).to_numpy(dtype=np.int64))
        model_categorical_columns.append(column)

    missing_indicator_columns = {
        column: f"__missing__{column}" for column in numeric_missing_columns
    }
    for column, indicator_column in missing_indicator_columns.items():
        values = np.where(numeric_missing[column].to_numpy(), "1", "0")
        categories[indicator_column] = ["0", "1"]
        categorical_arrays.append(values.astype(np.int64))
        model_categorical_columns.append(indicator_column)

    categorical = (
        np.stack(categorical_arrays, axis=1)
        if categorical_arrays
        else np.empty((len(df), 0), dtype=np.int64)
    )

    weights = pd.to_numeric(df[WEIGHT_COLUMN], errors="coerce").fillna(0).to_numpy()
    weights = np.clip(weights, a_min=0, a_max=None).astype(np.float64)
    if not np.any(weights > 0):
        raise SystemExit(f"{WEIGHT_COLUMN} has no positive weights")

    preprocessor = Preprocessor(
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_categorical_columns=model_categorical_columns,
        numeric_missing_columns=numeric_missing_columns,
        missing_indicator_columns=missing_indicator_columns,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        categories=categories,
    )
    return TripDataset(numeric, categorical), weights, preprocessor


def vae_loss(
    numeric_pred: torch.Tensor,
    categorical_pred: list[torch.Tensor],
    numeric_true: torch.Tensor,
    categorical_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    numeric_loss = F.mse_loss(numeric_pred, numeric_true)
    if categorical_pred:
        categorical_losses = [
            F.cross_entropy(logits, categorical_true[:, index])
            for index, logits in enumerate(categorical_pred)
        ]
        categorical_loss = torch.stack(categorical_losses).mean()
    else:
        categorical_loss = numeric_loss.new_tensor(0.0)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    loss = numeric_loss + categorical_loss + beta * kl_loss
    return loss, numeric_loss, categorical_loss, kl_loss


def train(
    model: TabularVAE,
    dataset: TripDataset,
    weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "num": 0.0, "cat": 0.0, "kl": 0.0}
        batches = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
        for numeric, categorical in progress:
            numeric = numeric.to(device, non_blocking=True)
            categorical = categorical.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            numeric_pred, categorical_pred, mu, logvar = model(numeric, categorical)
            loss, numeric_loss, categorical_loss, kl_loss = vae_loss(
                numeric_pred,
                categorical_pred,
                numeric,
                categorical,
                mu,
                logvar,
                args.beta,
            )
            loss.backward()
            optimizer.step()

            batches += 1
            totals["loss"] += loss.item()
            totals["num"] += numeric_loss.item()
            totals["cat"] += categorical_loss.item()
            totals["kl"] += kl_loss.item()
            progress.set_postfix(
                loss=f"{totals['loss'] / batches:.4f}",
                num=f"{totals['num'] / batches:.4f}",
                cat=f"{totals['cat'] / batches:.4f}",
                kl=f"{totals['kl'] / batches:.4f}",
            )

        if batches:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": totals["loss"] / batches,
                    "numeric_loss": totals["num"] / batches,
                    "categorical_loss": totals["cat"] / batches,
                    "kl_loss": totals["kl"] / batches,
                }
            )

    return history


def metadata_from_preprocessor(preprocessor: Preprocessor) -> dict[str, object]:
    return {
        "feature_columns": preprocessor.feature_columns,
        "numeric_columns": preprocessor.numeric_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "model_categorical_columns": preprocessor.model_categorical_columns,
        "numeric_missing_columns": preprocessor.numeric_missing_columns,
        "missing_indicator_columns": preprocessor.missing_indicator_columns,
        "numeric_mean": preprocessor.numeric_mean.tolist(),
        "numeric_std": preprocessor.numeric_std.tolist(),
        "numeric_min": preprocessor.numeric_min.tolist(),
        "numeric_max": preprocessor.numeric_max.tolist(),
        "categories": preprocessor.categories,
        "weight_column": WEIGHT_COLUMN,
    }


def save_training_history(history: list[dict[str, float]], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    history_csv = model_dir / "training_loss.csv"
    fieldnames = ["epoch", "loss", "numeric_loss", "categorical_loss", "kl_loss"]
    with history_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    if not history:
        return

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ["loss", "numeric_loss", "categorical_loss", "kl_loss"]:
        ax.plot(epochs, [row[key] for row in history], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("VAE training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir / "training_loss_curve.png", dpi=150)
    plt.close(fig)


def save_artifacts(
    model: TabularVAE,
    preprocessor: Preprocessor,
    args: argparse.Namespace,
) -> None:
    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metadata": metadata_from_preprocessor(preprocessor),
        },
        args.model_dir / "vae.pt",
    )
    with (args.model_dir / "preprocess.json").open("w") as metadata_file:
        json.dump(metadata_from_preprocessor(preprocessor), metadata_file, indent=2)


def sample_numeric(
    numeric_pred: torch.Tensor,
    preprocessor: Preprocessor,
) -> np.ndarray:
    numeric = numeric_pred.detach().cpu().numpy()
    numeric = numeric * preprocessor.numeric_std + preprocessor.numeric_mean
    numeric = np.clip(numeric, preprocessor.numeric_min, preprocessor.numeric_max)
    return np.rint(numeric).astype(np.int64)


def write_samples(
    model: TabularVAE,
    preprocessor: Preprocessor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    numeric_lookup = {
        column: index for index, column in enumerate(preprocessor.numeric_columns)
    }
    missing_lookup = {
        column: preprocessor.missing_indicator_columns[column]
        for column in preprocessor.numeric_missing_columns
    }

    model.eval()
    with args.output_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=preprocessor.feature_columns)
        writer.writeheader()

        progress = tqdm(
            range(0, args.samples, args.sample_batch_size),
            desc="sampling",
            unit="batch",
        )
        with torch.no_grad():
            for start in progress:
                batch_size = min(args.sample_batch_size, args.samples - start)
                z = torch.randn(batch_size, args.latent_dim, device=device)
                numeric_pred, categorical_pred = model.decode(z)

                numeric_values = sample_numeric(numeric_pred, preprocessor)
                categorical_values = {}
                for column, logits in zip(
                    preprocessor.model_categorical_columns,
                    categorical_pred,
                ):
                    probabilities = torch.softmax(logits, dim=1)
                    sampled = torch.multinomial(probabilities, num_samples=1).squeeze(1)
                    values = preprocessor.categories[column]
                    categorical_values[column] = [
                        values[index] for index in sampled.cpu().tolist()
                    ]

                for row_index in range(batch_size):
                    row = {}
                    for column in preprocessor.feature_columns:
                        if column in numeric_lookup:
                            indicator_column = missing_lookup.get(column)
                            if (
                                indicator_column is not None
                                and categorical_values[indicator_column][row_index] == "1"
                            ):
                                row[column] = "N/A"
                            else:
                                row[column] = str(
                                    numeric_values[row_index, numeric_lookup[column]]
                                )
                        else:
                            row[column] = categorical_values[column][row_index]
                    writer.writerow(row)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"using cuda: {torch.cuda.get_device_name(device)}")
    else:
        print(f"using device: {device}")

    dataset, weights, preprocessor = load_and_encode(args.input_csv)
    cardinalities = [
        len(preprocessor.categories[column])
        for column in preprocessor.model_categorical_columns
    ]
    model = TabularVAE(
        num_numeric=len(preprocessor.numeric_columns),
        cardinalities=cardinalities,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        embedding_cap=args.embedding_cap,
        dropout=args.dropout,
    ).to(device)

    print(f"input rows: {len(dataset)}")
    print(f"numeric columns: {len(preprocessor.numeric_columns)}")
    print(f"categorical columns: {len(preprocessor.categorical_columns)}")
    if preprocessor.numeric_missing_columns:
        print(
            "numeric columns with N/A modeled: "
            + ", ".join(preprocessor.numeric_missing_columns)
        )
    print(f"output samples: {args.samples}")

    history = train(model, dataset, weights, args, device)
    save_training_history(history, args.model_dir)
    save_artifacts(model, preprocessor, args)
    write_samples(model, preprocessor, args, device)
    print(f"wrote {args.output_csv}")
    print(f"saved model artifacts in {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

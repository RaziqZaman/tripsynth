#!/usr/bin/env python3
"""Sample the trained VAE at population trip scale."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


INPUT_CSV = Path("03x_filled-survey.csv")
OUTPUT_CSV = Path("07x_population_scale/synthetic_population_trips.csv")
MODEL_DIR = Path("04_model")
WEIGHT_COLUMN = "wthhfin"
SAMPLE_MODULE_PATH = Path("04_sample.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--weight-column", default=WEIGHT_COLUMN)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_sample_module():
    spec = importlib.util.spec_from_file_location("trip_vae_sampler", SAMPLE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SAMPLE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def population_trip_count(input_csv: Path, weight_column: str) -> int:
    total = 0.0
    for chunk in pd.read_csv(input_csv, usecols=[weight_column], chunksize=250_000):
        total += pd.to_numeric(chunk[weight_column], errors="coerce").fillna(0).clip(lower=0).sum()
    return int(round(total))


def preprocessor_from_metadata(sample_module, metadata: dict[str, object]):
    return sample_module.Preprocessor(
        feature_columns=list(metadata["feature_columns"]),
        numeric_columns=list(metadata["numeric_columns"]),
        categorical_columns=list(metadata["categorical_columns"]),
        model_categorical_columns=list(metadata["model_categorical_columns"]),
        numeric_missing_columns=list(metadata["numeric_missing_columns"]),
        missing_indicator_columns=dict(metadata["missing_indicator_columns"]),
        numeric_mean=np.asarray(metadata["numeric_mean"], dtype=np.float32),
        numeric_std=np.asarray(metadata["numeric_std"], dtype=np.float32),
        numeric_min=np.asarray(metadata["numeric_min"], dtype=np.float32),
        numeric_max=np.asarray(metadata["numeric_max"], dtype=np.float32),
        categories={
            str(column): [str(value) for value in values]
            for column, values in dict(metadata["categories"]).items()
        },
    )


def main() -> int:
    args = parse_args()
    sample_module = load_sample_module()
    sample_module.seed_everything(args.seed)

    samples = args.samples
    if samples is None:
        samples = population_trip_count(args.input_csv, args.weight_column)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"using cuda: {torch.cuda.get_device_name(device)}")
    else:
        print(f"using device: {device}")

    checkpoint_path = args.model_dir / "vae.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    saved_args = checkpoint.get("args", {})
    preprocessor = preprocessor_from_metadata(sample_module, metadata)
    cardinalities = [
        len(preprocessor.categories[column])
        for column in preprocessor.model_categorical_columns
    ]

    latent_dim = int(saved_args.get("latent_dim", 1800))
    model = sample_module.TabularVAE(
        num_numeric=len(preprocessor.numeric_columns),
        cardinalities=cardinalities,
        latent_dim=latent_dim,
        hidden_dims=list(saved_args.get("hidden_dims", [2048, 1920, 1536, 1024])),
        embedding_cap=int(saved_args.get("embedding_cap", 32)),
        dropout=float(saved_args.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    writer_args = SimpleNamespace(
        output_csv=args.output_csv,
        samples=samples,
        sample_batch_size=args.sample_batch_size,
        latent_dim=latent_dim,
    )
    print(f"population-scale samples: {samples:,}")
    sample_module.write_samples(model, preprocessor, writer_args, device)
    print(f"wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

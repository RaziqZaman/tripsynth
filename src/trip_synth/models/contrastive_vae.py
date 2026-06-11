from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trip_synth.data.schema import FeatureSchema

from .sample import sample_vae_method
from .train import fit_vae_method


def fit(
    train_df: pd.DataFrame,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    return fit_vae_method(train_df, schema, config, output_dir, method="contrastive_vae")


def sample(
    model_or_artifacts: dict[str, Any],
    n_rows: int,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
) -> pd.DataFrame:
    return sample_vae_method(
        model_or_artifacts,
        n_rows,
        schema,
        config,
        output_dir,
        run_id=str(config.get("run_name", "run")),
    )

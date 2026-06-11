from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trip_synth.data.postprocessing import finalize_synthetic
from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.utils.io import ensure_dir, write_json


def fit(
    train_df: pd.DataFrame,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    cleaned, warnings = schema.clean_dataframe(train_df)
    preprocessor = FittedPreprocessor.fit(cleaned, schema)
    weights = pd.to_numeric(cleaned[schema.weight_column], errors="coerce").fillna(0.0).clip(lower=0)
    if weights.sum() <= 0:
        probs = np.ones(len(cleaned)) / max(1, len(cleaned))
    else:
        probs = weights.to_numpy(float) / float(weights.sum())
    artifacts = {
        "method": "weighted_bootstrap",
        "train_df": cleaned,
        "probabilities": probs,
        "preprocessor": preprocessor,
        "warnings": warnings,
    }
    write_json(
        {"method": "weighted_bootstrap", "n_train": len(cleaned), "warnings": warnings},
        output_dir / "weighted_bootstrap_artifact.json",
    )
    return artifacts


def sample(
    model_or_artifacts: dict[str, Any],
    n_rows: int,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.get("seed", 0)))
    train_df = model_or_artifacts["train_df"]
    probs = model_or_artifacts["probabilities"]
    idx = rng.choice(np.arange(len(train_df)), size=int(n_rows), replace=True, p=probs)
    features = train_df.iloc[idx].reset_index(drop=True)
    return finalize_synthetic(
        features,
        schema,
        model_or_artifacts["preprocessor"],
        "weighted_bootstrap",
        str(config.get("run_name", "run")),
    )

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trip_synth.data.postprocessing import finalize_synthetic
from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.utils.io import ensure_dir, write_json


def _probabilities(values: pd.Series, weights: np.ndarray) -> tuple[list[Any], np.ndarray]:
    tmp = pd.DataFrame({"value": values.astype("string"), "weight": weights})
    grouped = tmp.groupby("value", dropna=False)["weight"].sum()
    if grouped.sum() <= 0:
        probs = np.ones(len(grouped)) / max(1, len(grouped))
    else:
        probs = grouped.to_numpy(float) / grouped.sum()
    return grouped.index.astype(str).tolist(), probs


def fit(
    train_df: pd.DataFrame,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    cleaned, warnings = schema.clean_dataframe(train_df)
    preprocessor = FittedPreprocessor.fit(cleaned, schema)
    features = cleaned[preprocessor.feature_columns].copy()
    weights = pd.to_numeric(cleaned[schema.weight_column], errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
    if not np.any(weights > 0):
        weights = np.ones(len(cleaned), dtype=float)

    cfg = config.get("gibbs", {})
    context_columns = [c for c in cfg.get("conditioning_columns", []) if c in features.columns]
    if not context_columns:
        context_columns = [c for c in ["o_activity", "d_activity", "tdate_dow"] if c in features.columns]

    marginals: dict[str, tuple[list[Any], np.ndarray]] = {}
    conditionals: dict[str, dict[tuple[str, ...], tuple[list[Any], np.ndarray]]] = {}
    for target in preprocessor.feature_columns:
        marginals[target] = _probabilities(features[target], weights)
        parents = [c for c in context_columns if c != target][:2]
        table: dict[tuple[str, ...], tuple[list[Any], np.ndarray]] = {}
        if parents:
            tmp = features[parents + [target]].astype("string").copy()
            tmp["_weight"] = weights
            for ctx, group in tmp.groupby(parents, dropna=False):
                if not isinstance(ctx, tuple):
                    ctx = (ctx,)
                vals, probs = _probabilities(group[target], group["_weight"].to_numpy(float))
                table[tuple(str(v) for v in ctx)] = (vals, probs)
        conditionals[target] = table

    write_json(
        {
            "method": "gibbs",
            "n_train": len(cleaned),
            "sweeps": int(cfg.get("sweeps", 3)),
            "context_columns": context_columns,
            "warnings": warnings,
        },
        output_dir / "gibbs_artifact.json",
    )
    return {
        "method": "gibbs",
        "train_df": cleaned,
        "features": features,
        "weights": weights,
        "preprocessor": preprocessor,
        "marginals": marginals,
        "conditionals": conditionals,
        "context_columns": context_columns,
    }


def sample(
    model_or_artifacts: dict[str, Any],
    n_rows: int,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.get("seed", 0)) + 17)
    artifacts = model_or_artifacts
    train_df = artifacts["train_df"]
    weights = artifacts["weights"]
    probs = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
    start_idx = rng.choice(np.arange(len(train_df)), size=int(n_rows), replace=True, p=probs)
    synth = train_df.iloc[start_idx][artifacts["preprocessor"].feature_columns].reset_index(drop=True).copy()
    sweeps = int(config.get("gibbs", {}).get("sweeps", 3))
    context_columns = artifacts["context_columns"]

    for _ in range(sweeps):
        for target in artifacts["preprocessor"].feature_columns:
            parents = [c for c in context_columns if c != target][:2]
            marginal_values, marginal_probs = artifacts["marginals"][target]
            table = artifacts["conditionals"].get(target, {})
            new_values = []
            for _, row in synth.iterrows():
                key = tuple(str(row[c]) for c in parents)
                values, probs_for_values = table.get(key, (marginal_values, marginal_probs))
                new_values.append(rng.choice(values, p=probs_for_values))
            synth[target] = new_values

    final = finalize_synthetic(
        synth,
        schema,
        artifacts["preprocessor"],
        "gibbs",
        str(config.get("run_name", "run")),
    )
    for col in artifacts["preprocessor"].ordinal_integer_columns:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors="coerce").round().astype("Int64")
    for col in artifacts["preprocessor"].continuous_columns:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors="coerce")
    return final

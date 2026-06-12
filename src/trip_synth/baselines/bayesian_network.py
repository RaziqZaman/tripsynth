from __future__ import annotations

from collections import defaultdict
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trip_synth.data.postprocessing import finalize_synthetic
from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.utils.io import ensure_dir, write_json
from trip_synth.utils.progress import progress_iter


def _weighted_counts(values: pd.Series, weights: np.ndarray, smoothing: float = 0.0) -> tuple[list[str], np.ndarray]:
    tmp = pd.DataFrame({"value": values.astype("string"), "weight": weights})
    grouped = tmp.groupby("value", dropna=False)["weight"].sum()
    vals = grouped.index.astype(str).tolist()
    probs = grouped.to_numpy(float) + smoothing
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(vals)) / max(1, len(vals))
    return vals, probs


def _mutual_information(x: pd.Series, y: pd.Series, weights: np.ndarray) -> float:
    frame = pd.DataFrame({"x": x.astype("string"), "y": y.astype("string"), "w": weights})
    joint = frame.groupby(["x", "y"])["w"].sum()
    total = float(joint.sum())
    if total <= 0:
        return 0.0
    px = joint.groupby(level=0).sum() / total
    py = joint.groupby(level=1).sum() / total
    mi = 0.0
    for (xv, yv), w in joint.items():
        pxy = w / total
        if pxy > 0:
            mi += pxy * np.log(pxy / (px[xv] * py[yv]))
    return float(mi)


def _maximum_spanning_tree(mi: np.ndarray, columns: list[str]) -> dict[str, str | None]:
    n = len(columns)
    parent: dict[str, str | None] = {columns[0]: None}
    selected = {0}
    while len(selected) < n:
        best = None
        for i in selected:
            for j in range(n):
                if j in selected:
                    continue
                score = mi[i, j]
                if best is None or score > best[0]:
                    best = (score, i, j)
        if best is None:
            break
        _, i, j = best
        parent[columns[j]] = columns[i]
        selected.add(j)
    for col in columns:
        parent.setdefault(col, None)
    return parent


def _discretize(
    features: pd.DataFrame,
    preprocessor: FittedPreprocessor,
    max_bins: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[Any]]]]:
    discrete = pd.DataFrame(index=features.index)
    decoders: dict[str, dict[str, list[Any]]] = {}
    for col in preprocessor.feature_columns:
        if col in preprocessor.numeric_columns:
            vals = pd.to_numeric(features[col], errors="coerce")
            nunique = vals.nunique(dropna=True)
            if nunique <= max_bins:
                bins = vals.astype("string").fillna("-1")
            else:
                try:
                    bins = pd.qcut(vals.rank(method="first"), q=max_bins, labels=False, duplicates="drop").astype("Int64")
                    bins = bins.astype("string").fillna("-1")
                except ValueError:
                    bins = vals.round().astype("string").fillna("-1")
            discrete[col] = bins
            dec: dict[str, list[Any]] = defaultdict(list)  # type: ignore[assignment]
            for b, v in zip(bins.astype(str), features[col]):
                dec[b].append(v)
            decoders[col] = dict(dec)
        else:
            discrete[col] = features[col].astype("string").fillna("-1")
    return discrete, decoders


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
    cfg = config.get("bayesian_network", {})
    max_bins = int(cfg.get("max_bins", 8))
    smoothing = float(cfg.get("smoothing", 1.0))
    discrete, decoders = _discretize(features, preprocessor, max_bins=max_bins)
    columns = preprocessor.feature_columns
    mi = np.zeros((len(columns), len(columns)), dtype=float)
    deadline = config.get("_method_deadline_monotonic")
    pairs = [(i, j) for i in range(len(columns)) for j in range(i + 1, len(columns))]
    stopped_by_runtime_cap = False
    for i, j in progress_iter(pairs, desc="bayesian_network mutual information", total=len(pairs), unit="pair"):
        if deadline is not None and time.monotonic() >= float(deadline):
            stopped_by_runtime_cap = True
            warnings.append("Bayesian network fit stopped early because max_runtime_seconds was reached.")
            break
        a = columns[i]
        b = columns[j]
        val = _mutual_information(discrete[a], discrete[b], weights)
        mi[i, j] = mi[j, i] = val
    parents = _maximum_spanning_tree(mi, columns)

    root = next(col for col, parent in parents.items() if parent is None)
    root_values, root_probs = _weighted_counts(discrete[root], weights, smoothing=smoothing)
    cpts: dict[str, Any] = {root: {"root": (root_values, root_probs)}}
    for col in columns:
        parent = parents[col]
        if parent is None:
            continue
        tmp = pd.DataFrame({"parent": discrete[parent], "value": discrete[col], "weight": weights})
        table: dict[str, tuple[list[str], np.ndarray]] = {}
        for parent_value, group in tmp.groupby("parent", dropna=False):
            vals, probs = _weighted_counts(group["value"], group["weight"].to_numpy(float), smoothing=smoothing)
            table[str(parent_value)] = (vals, probs)
        cpts[col] = table

    write_json(
        {
            "method": "bayesian_network",
            "n_train": len(cleaned),
            "structure": parents,
            "max_bins": max_bins,
            "warnings": warnings,
            "stopped_by_runtime_cap": stopped_by_runtime_cap,
        },
        output_dir / "bayesian_network_artifact.json",
    )
    return {
        "method": "bayesian_network",
        "preprocessor": preprocessor,
        "features": features,
        "weights": weights,
        "discrete": discrete,
        "decoders": decoders,
        "parents": parents,
        "cpts": cpts,
        "root": root,
        "stopped_by_runtime_cap": stopped_by_runtime_cap,
    }


def _sample_discrete_rows(art: dict[str, Any], n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    preprocessor: FittedPreprocessor = art["preprocessor"]
    sampled_disc = pd.DataFrame(index=np.arange(int(n_rows)))
    root = art["root"]
    root_values, root_probs = art["cpts"][root]["root"]
    sampled_disc[root] = rng.choice(root_values, size=int(n_rows), p=root_probs)
    pending = set(preprocessor.feature_columns) - {root}
    while pending:
        progressed = False
        pending_columns = list(pending)
        for col in progress_iter(pending_columns, desc="bayesian_network sampling columns", total=len(pending_columns), unit="column", leave=False):
            parent = art["parents"][col]
            if parent is None or parent in sampled_disc.columns:
                if parent is None:
                    vals, probs = art["cpts"][col]["root"]
                    sampled_disc[col] = rng.choice(vals, size=int(n_rows), p=probs)
                else:
                    table = art["cpts"].get(col, {})
                    fallback = next(iter(table.values()))
                    parent_values = sampled_disc[parent].astype(str).to_numpy()
                    sampled = np.empty(int(n_rows), dtype=object)
                    unique_values, inverse = np.unique(parent_values, return_inverse=True)
                    for idx_value, parent_value in enumerate(unique_values):
                        idx = np.flatnonzero(inverse == idx_value)
                        vals, probs = table.get(str(parent_value), fallback)
                        sampled[idx] = rng.choice(vals, size=len(idx), p=probs)
                    sampled_disc[col] = sampled
                pending.remove(col)
                progressed = True
        if not progressed:
            for col in pending:
                observed = art["discrete"][col].to_numpy()
                sampled_disc[col] = rng.choice(observed, size=int(n_rows), replace=True)
            break
    return sampled_disc


def _decode_discrete_rows(art: dict[str, Any], sampled_disc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    preprocessor: FittedPreprocessor = art["preprocessor"]
    n_rows = len(sampled_disc)
    decoded: dict[str, Any] = {}
    feature_columns = list(preprocessor.feature_columns)
    for col in progress_iter(feature_columns, desc="bayesian_network decoding", total=len(feature_columns), unit="column"):
        if col in preprocessor.numeric_columns:
            dec = art["decoders"].get(col, {})
            observed = pd.to_numeric(art["features"][col], errors="coerce").dropna().to_numpy()
            fallback = float(np.nanmean(observed)) if observed.size else 0.0
            bins = sampled_disc[col].astype(str).to_numpy()
            vals_out = np.empty(n_rows, dtype=object)
            unique_bins, inverse = np.unique(bins, return_inverse=True)
            for idx_value, bin_value in enumerate(unique_bins):
                idx = np.flatnonzero(inverse == idx_value)
                choices = dec.get(str(bin_value), [])
                if choices:
                    vals_out[idx] = rng.choice(np.asarray(choices, dtype=object), size=len(idx))
                else:
                    vals_out[idx] = fallback
            decoded[col] = vals_out
        else:
            decoded[col] = sampled_disc[col].astype(str).to_numpy()
    return pd.DataFrame(decoded)[preprocessor.feature_columns]


def sample(
    model_or_artifacts: dict[str, Any],
    n_rows: int,
    schema: FeatureSchema,
    config: dict[str, Any],
    output_dir: str | Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.get("seed", 0)) + 31)
    art = model_or_artifacts
    preprocessor: FittedPreprocessor = art["preprocessor"]
    cfg = config.get("bayesian_network", {})
    chunk_size = max(1, int(cfg.get("sample_chunk_size", int(n_rows))))
    deadline = config.get("_method_deadline_monotonic")
    chunks: list[pd.DataFrame] = []
    generated = 0
    while generated < int(n_rows):
        if chunks and deadline is not None and time.monotonic() >= float(deadline):
            break
        take = min(chunk_size, int(n_rows) - generated)
        sampled_disc = _sample_discrete_rows(art, take, rng)
        synth = _decode_discrete_rows(art, sampled_disc, rng)
        final = finalize_synthetic(
            synth,
            schema,
            preprocessor,
            "bayesian_network",
            str(config.get("run_name", "run")),
        )
        for col in preprocessor.ordinal_integer_columns:
            if col in final.columns:
                final[col] = pd.to_numeric(final[col], errors="coerce").round().astype("Int64")
        for col in preprocessor.continuous_columns:
            if col in final.columns:
                final[col] = pd.to_numeric(final[col], errors="coerce")
        chunks.append(final)
        generated += take
    return pd.concat(chunks, ignore_index=True)

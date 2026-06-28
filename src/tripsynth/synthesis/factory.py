"""Synthesis method factory."""

from __future__ import annotations

from typing import Any

from tripsynth.synthesis.bayesian_network import BayesianNetworkSynthesizer
from tripsynth.synthesis.diffusion import DiffusionSynthesizer
from tripsynth.synthesis.vae import VAESynthesizer
from tripsynth.synthesis.weighted_resampling import WeightedResampler


def _canonical_aliases(config: dict[str, Any]) -> dict[str, str]:
    schema = config.get("survey", {}).get("schema", {})
    return {str(canonical): str(source) for canonical, source in schema.items() if source}


def _column_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def build_synthesizer(
    method: str,
    config: dict[str, Any] | None = None,
    *,
    overrides: dict[str, Any] | None = None,
):
    config = config or {}
    synth_config = {**config.get("synthesis", {}).get(method, {}), **(overrides or {})}
    if method == "weighted_resampling":
        return WeightedResampler(weight_field=synth_config.get("weight_field", "weight"))
    if method == "bayesian_network":
        return BayesianNetworkSynthesizer(
            smoothing=float(synth_config.get("smoothing", 0.5)),
            weight_field=synth_config.get("weight_field", "weight"),
        )
    if method == "vae":
        return VAESynthesizer(
            latent_dim=int(synth_config.get("latent_dim", 8)),
            hidden_dim=int(synth_config.get("hidden_dim", 48)),
            num_layers=int(synth_config.get("num_layers", 1)),
            epochs=int(synth_config.get("epochs", 4)),
            batch_size=int(synth_config.get("batch_size", 1024)),
            sample_batch_size=int(synth_config.get("sample_batch_size", 262144)),
            learning_rate=float(synth_config.get("learning_rate", 1e-3)),
            beta=float(synth_config.get("beta", 0.01)),
            weight_field=synth_config.get("weight_field", "weight"),
            model_all_columns=bool(synth_config.get("model_all_columns", True)),
            categorical_columns=_column_list(synth_config.get("categorical_columns")),
            numeric_columns=_column_list(synth_config.get("numeric_columns")),
            max_categorical_cardinality=int(synth_config.get("max_categorical_cardinality", 50)),
            canonical_aliases=_canonical_aliases(config),
            device=synth_config.get("device", "auto"),
        )
    if method == "diffusion":
        return DiffusionSynthesizer(
            noise_scale=float(synth_config.get("noise_scale", 0.25)),
            steps=int(synth_config.get("steps", 6)),
            weight_field=synth_config.get("weight_field", "weight"),
            model_all_columns=bool(synth_config.get("model_all_columns", True)),
            categorical_columns=_column_list(synth_config.get("categorical_columns")),
            numeric_columns=_column_list(synth_config.get("numeric_columns")),
            max_categorical_cardinality=int(synth_config.get("max_categorical_cardinality", 50)),
            canonical_aliases=_canonical_aliases(config),
            device=synth_config.get("device", "auto"),
            sample_batch_size=int(synth_config.get("sample_batch_size", 524288)),
        )
    raise ValueError(f"Unsupported synthesis method {method!r}.")


def synthesize_trips(
    method: str,
    trips,
    *,
    n: int,
    seed: int,
    config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
):
    synthesizer = build_synthesizer(method, config, overrides=overrides)
    return synthesizer.fit(trips).sample(n, seed=seed)

"""Experiment matrix helpers."""

from __future__ import annotations

from itertools import product
from typing import Any


def build_experiment_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiment = config.get("experiment", {})
    methods = experiment.get("synthesis_methods", [])
    routes = experiment.get("routing_methods", [])
    datasets = experiment.get("observed_validation_datasets", [])
    rows = []
    for method, route, dataset in product(methods, routes, datasets):
        seeds = (
            experiment.get("seeds", {}).get("deterministic", [1])
            if method == "weighted_resampling"
            else experiment.get("seeds", {}).get("stochastic", [1])
        )
        for seed in seeds:
            rows.append(
                {
                    "synthesis_method": method,
                    "routing_method": route,
                    "observed_validation_dataset": dataset,
                    "seed": seed,
                    "validation_mode": config.get("validation", {}).get(
                        "default_mode", "spatial_aadt_proxy"
                    ),
                }
            )
    return rows

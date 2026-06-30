"""Hyperparameter sweeps for synthesis methods under routed AADT validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tripsynth.config import ensure_standard_directories, write_json, read_json
from tripsynth.data_sources.observed_counts.loaders import load_observed_counts_for_validation
from tripsynth.data_sources.survey import build_synthesis_frame, load_survey
from tripsynth.experiments.baseline import _filter_vehicle_trips
from tripsynth.experiments.method_comparison import (
    _metric_row,
    _route_synthetic_trips,
    _routing_metadata_row,
    _write_method_run,
)
from tripsynth.preprocessing.survey_audit import run_survey_audit
from tripsynth.progress import progress
from tripsynth.reporting.baseline_diagnostics import run_baseline_diagnostics
from tripsynth.synthesis.factory import synthesize_trips
from tripsynth.validation.spatial_proxy import validate_spatial_aadt_proxy


DEFAULT_SWEEP_GRIDS: dict[str, dict[str, list[Any]]] = {
    "weighted_resampling": {"weight_field": ["weight"]},
    "bayesian_network": {"smoothing": [0.01, 0.05, 0.1, 0.5, 2.0], "weight_field": ["weight"]},
    "vae": {
        "latent_dim": [64, 128, 256],
        "hidden_dim": [1024, 2048],
        "num_layers": [3, 4],
        "epochs": [30, 60],
        "batch_size": [65536],
        "sample_batch_size": [262144],
        "learning_rate": [0.0005, 0.001],
        "beta": [0.001, 0.005],
        "weight_field": ["weight"],
        "model_all_columns": [True],
        "max_categorical_cardinality": [50],
        "device": ["auto"],
    },
    "contrastive_vae": {
        "latent_dim": [64, 128],
        "hidden_dim": [1024],
        "num_layers": [3, 4],
        "epochs": [30, 60],
        "batch_size": [65536],
        "sample_batch_size": [262144],
        "learning_rate": [0.0005],
        "beta": [0.005],
        "contrastive_weight": [0.05, 0.15],
        "contrastive_temperature": [0.2],
        "contrastive_batch_size": [512],
        "sample_from_training_latent": [True],
        "sample_latent_noise_scale": [0.25, 0.5],
        "weight_field": ["weight"],
        "model_all_columns": [True],
        "max_categorical_cardinality": [50],
        "device": ["auto"],
    },
    "diffusion": {
        "noise_scale": [0.02, 0.05, 0.1, 0.2, 0.35],
        "steps": [12, 24, 48],
        "sample_batch_size": [524288],
        "weight_field": ["weight"],
        "model_all_columns": [True],
        "max_categorical_cardinality": [50],
        "device": ["auto"],
    },
}


@dataclass(frozen=True)
class SynthesisSweepResult:
    run_dir: Path
    leaderboard: pd.DataFrame
    summary: dict[str, Any]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _grid_from_config(config: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    configured = config.get("experiment", {}).get("synthesis_hyperparameter_grids")
    if not configured:
        return DEFAULT_SWEEP_GRIDS
    grids: dict[str, dict[str, list[Any]]] = {}
    for method, params in configured.items():
        grids[method] = {}
        for key, values in params.items():
            grids[method][key] = values if isinstance(values, list) else [values]
    return grids


def _method_specs(methods: list[str], grids: dict[str, dict[str, list[Any]]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for method in methods:
        grid = grids.get(method, {})
        if not grid:
            specs.append({"method": method, "params": {}})
            continue
        keys = list(grid)
        for values in product(*(grid[key] for key in keys)):
            specs.append({"method": method, "params": dict(zip(keys, values))})
    return specs


def _params_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _slug(method: str, params: dict[str, Any], seed: int, scale: float) -> str:
    return f"{method}_scale_{scale:g}_seed_{seed}_{_params_hash(params)}".replace(".", "p")


def _resolve_scale_factors(
    vehicle_trips: pd.DataFrame,
    config: dict[str, Any],
    *,
    scale_factors: list[float] | None,
    target_trip_count: int | None,
    target_population_share: float | None,
) -> tuple[list[float], dict[str, Any]]:
    if target_trip_count is not None and target_population_share is not None:
        raise ValueError("Use either target_trip_count or target_population_share, not both.")
    if scale_factors is not None and (target_trip_count is not None or target_population_share is not None):
        raise ValueError("Use scale_factors or a target-count option, not both.")

    metadata: dict[str, Any] = {}
    if target_population_share is not None:
        share = float(target_population_share)
        if share <= 0 or share > 1:
            raise ValueError("target_population_share must be in the interval (0, 1].")
        weight_field = str(config.get("survey", {}).get("population_weight_field") or "weight")
        if weight_field not in vehicle_trips:
            raise ValueError(
                f"Cannot use target_population_share because {weight_field!r} is not in vehicle trips."
            )
        weight_total = float(pd.to_numeric(vehicle_trips[weight_field], errors="coerce").fillna(0.0).sum())
        if weight_total <= 0:
            raise ValueError(f"Cannot use target_population_share because {weight_field!r} sums to zero.")
        target_trip_count = max(1, int(round(weight_total * share)))
        metadata.update(
            {
                "target_population_share": share,
                "population_weight_field": weight_field,
                "weighted_vehicle_trip_total": weight_total,
            }
        )

    if target_trip_count is not None:
        target = int(target_trip_count)
        if target <= 0:
            raise ValueError("target_trip_count must be positive.")
        scale = target / max(1, len(vehicle_trips))
        metadata.update({"target_trip_count": target, "resolved_scale_factor": scale})
        return [scale], metadata

    if scale_factors is not None:
        return [float(scale) for scale in scale_factors], metadata

    return [float(config.get("validation", {}).get("baseline", {}).get("scale_factor", 1))], metadata


def _load_existing_row(run_path: Path, method: str, params: dict[str, Any], seed: int, scale: float) -> dict[str, Any] | None:
    summary_path = run_path / "tables" / "baseline_summary.json"
    metrics_path = run_path / "metrics" / "metrics_by_run.csv"
    if not (summary_path.exists() and metrics_path.exists()):
        return None
    summary = read_json(summary_path)
    metrics = pd.read_csv(metrics_path)
    routing = summary.get("routing", {})
    synthesis = summary.get("synthesis", {})
    return {
        "synthesis_method": method,
        "seed": int(seed),
        "scale_factor": float(scale),
        "params_json": json.dumps(params, sort_keys=True, default=str),
        "run_dir": str(run_path),
        "synthetic_trips": int(synthesis.get("n", 0) or 0),
        "supported_vehicle_trips": int(
            routing.get("vehicle_trips_with_supported_tracts")
            or routing.get("vehicle_trips_with_supported_counties")
            or 0
        ),
        "od_pairs": int(routing.get("od_pairs", 0) or 0),
        "routes": int(routing.get("routes", 0) or 0),
        "failed_routes": int(routing.get("failed_routes", 0) or 0),
        "routed_observed_segments": int(routing.get("routed_observed_segments", 0) or 0),
        "synthesis_implementation": synthesis.get("implementation", method),
        **_routing_metadata_row(routing),
        **_metric_row(metrics),
    }


def _run_one(
    *,
    run_path: Path,
    method: str,
    params: dict[str, Any],
    seed: int,
    scale: float,
    n: int,
    config: dict[str, Any],
    audit,
    observed,
    observed_meta: dict[str, Any],
    vehicle_trips: pd.DataFrame,
    vehicle_filter_meta: dict[str, Any],
    diagnostics: bool,
) -> dict[str, Any]:
    synthesis = synthesize_trips(
        method, vehicle_trips, n=n, seed=int(seed), config=config, overrides=params
    )
    routed = _route_synthetic_trips(synthesis.synthetic_trips, observed, config)
    metrics_by_segment, metrics_by_run = validate_spatial_aadt_proxy(
        routed.routed_volumes, observed, config
    )
    summary = _write_method_run(
        run_path=run_path,
        method=method,
        seed=int(seed),
        validation_label=config.get("validation", {}).get("baseline", {}).get(
            "validation_label", "spatial roadway-volume proxy validation against AADT"
        ),
        audit=audit,
        observed=observed,
        observed_meta=observed_meta,
        vehicle_filter_meta=vehicle_filter_meta,
        synthesis=synthesis,
        routed=routed,
        metrics_by_segment=metrics_by_segment,
        metrics_by_run=metrics_by_run,
    )
    summary["synthesis_hyperparameters"] = params
    summary["scale_factor"] = float(scale)
    write_json(run_path / "tables" / "baseline_summary.json", summary)
    if diagnostics:
        run_baseline_diagnostics(run_path)
    routing = summary.get("routing", {})
    return {
        "synthesis_method": method,
        "seed": int(seed),
        "scale_factor": float(scale),
        "params_json": json.dumps(params, sort_keys=True, default=str),
        "run_dir": str(run_path),
        "synthetic_trips": int(len(synthesis.synthetic_trips)),
        "supported_vehicle_trips": int(
            routing.get("vehicle_trips_with_supported_tracts")
            or routing.get("vehicle_trips_with_supported_counties")
            or 0
        ),
        "od_pairs": int(routing.get("od_pairs", 0) or 0),
        "routes": int(routing.get("routes", 0) or 0),
        "failed_routes": int(routing.get("failed_routes", 0) or 0),
        "routed_observed_segments": int(routing.get("routed_observed_segments", 0) or 0),
        "synthesis_implementation": synthesis.metadata.get("implementation", method),
        **_routing_metadata_row(routing),
        **_metric_row(metrics_by_run),
    }


def run_synthesis_hyperparameter_sweep(
    config: dict[str, Any],
    *,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    scale_factors: list[float] | None = None,
    run_dir: str | Path | None = None,
    force_counts: bool = False,
    observed_source: str | None = None,
    diagnostics: bool = False,
    max_runs: int | None = None,
    max_runs_per_method: int | None = None,
    target_trip_count: int | None = None,
    target_population_share: float | None = None,
    resume: bool = True,
) -> SynthesisSweepResult:
    ensure_standard_directories()
    audit = run_survey_audit(config)
    survey = load_survey(config)
    observed, observed_meta = load_observed_counts_for_validation(
        config, source_name=observed_source, force_fetch=force_counts
    )
    synthesis_frame = build_synthesis_frame(survey, config)
    vehicle_trips, vehicle_filter_meta = _filter_vehicle_trips(synthesis_frame, config)

    grids = _grid_from_config(config)
    methods = methods or list(grids)
    seeds = seeds or [int(config.get("project", {}).get("seed", 42))]
    scale_factors, target_metadata = _resolve_scale_factors(
        vehicle_trips,
        config,
        scale_factors=scale_factors,
        target_trip_count=target_trip_count,
        target_population_share=target_population_share,
    )
    specs = _method_specs(methods, grids)

    root = Path(run_dir) if run_dir else Path("outputs/runs") / f"synthesis_sweep_{_timestamp()}"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    attempted = 0
    attempted_by_method: dict[str, int] = {method: 0 for method in methods}
    total_candidates = len(scale_factors) * len(specs) * len(seeds)
    if max_runs_per_method is not None:
        total_candidates = min(total_candidates, len(methods) * int(max_runs_per_method))
    if max_runs is not None:
        total_candidates = min(total_candidates, int(max_runs))
    with progress(total=total_candidates, desc="synthesis sweep", unit="run") as sweep_bar:
        stop_requested = False
        for scale in scale_factors:
            n = max(1, int(round(len(vehicle_trips) * float(scale))))
            for spec in specs:
                for seed in seeds:
                    if max_runs is not None and attempted >= max_runs:
                        stop_requested = True
                        break
                    method = spec["method"]
                    if max_runs_per_method is not None and attempted_by_method.get(method, 0) >= max_runs_per_method:
                        continue
                    params = spec["params"]
                    sweep_bar.set_postfix(method=method, seed=int(seed), scale=f"{float(scale):g}")
                    run_path = root / _slug(method, params, int(seed), float(scale))
                    existing = _load_existing_row(run_path, method, params, int(seed), float(scale)) if resume else None
                    if existing is not None:
                        rows.append(existing)
                        sweep_bar.update(1)
                        continue
                    attempted += 1
                    attempted_by_method[method] = attempted_by_method.get(method, 0) + 1
                    rows.append(
                        _run_one(
                            run_path=run_path,
                            method=method,
                            params=params,
                            seed=int(seed),
                            scale=float(scale),
                            n=n,
                            config=config,
                            audit=audit,
                            observed=observed,
                            observed_meta=observed_meta,
                            vehicle_trips=vehicle_trips,
                            vehicle_filter_meta=vehicle_filter_meta,
                            diagnostics=diagnostics,
                        )
                    )
                    sweep_bar.update(1)
                if stop_requested:
                    break
            if stop_requested:
                break

    leaderboard = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in [
            "uncalibrated_absolute_proxy.spearman_correlation",
            "uncalibrated_absolute_proxy.pearson_correlation",
            "uncalibrated_absolute_proxy.top_10pct_overlap",
        ]
        if column in leaderboard.columns
    ]
    if sort_columns:
        leaderboard = leaderboard.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    leaderboard_path = root / "sweep_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    compact_columns = [
        column
        for column in [
            "synthesis_method",
            "seed",
            "scale_factor",
            "params_json",
            "synthesis_implementation",
            "routing_method",
            "routing_od_geography",
            "routing_edge_weight_strategy",
            "routing_aadt_preference_alpha",
            "synthetic_trips",
            "supported_vehicle_trips",
            "od_pairs",
            "routes",
            "failed_routes",
            "routed_observed_segments",
            "uncalibrated_absolute_proxy.pearson_correlation",
            "uncalibrated_absolute_proxy.spearman_correlation",
            "uncalibrated_absolute_proxy.jensen_shannon_divergence",
            "uncalibrated_absolute_proxy.top_10pct_overlap",
            "run_dir",
        ]
        if column in leaderboard.columns
    ]
    compact_path = root / "sweep_compact.csv"
    leaderboard[compact_columns].to_csv(compact_path, index=False)
    summary = {
        "run_dir": str(root),
        "methods": methods,
        "seeds": [int(seed) for seed in seeds],
        "scale_factors": [float(scale) for scale in scale_factors],
        "candidate_specs": specs,
        "completed_runs": int(len(leaderboard)),
        "max_runs": max_runs,
        "max_runs_per_method": max_runs_per_method,
        **target_metadata,
        "sweep_leaderboard": str(leaderboard_path),
        "sweep_compact": str(compact_path),
    }
    summary_path = root / "sweep_summary.json"
    write_json(summary_path, summary)
    summary["sweep_summary"] = str(summary_path)
    return SynthesisSweepResult(run_dir=root, leaderboard=leaderboard, summary=summary)


def format_synthesis_sweep_summary(result: SynthesisSweepResult) -> str:
    lines = ["Synthesis hyperparameter sweep complete", f"  run dir: {result.run_dir}"]
    lines.append(f"  completed runs: {len(result.leaderboard)}")
    if not result.leaderboard.empty:
        metric = "uncalibrated_absolute_proxy.spearman_correlation"
        if metric in result.leaderboard:
            best = result.leaderboard.iloc[0]
            lines.append(
                f"  best Spearman: {best['synthesis_method']} seed {best['seed']} scale {best['scale_factor']} = {best[metric]:.3f}"
            )
            lines.append(f"  best params: {best['params_json']}")
    lines.append(f"  leaderboard: {result.summary['sweep_leaderboard']}")
    lines.append(f"  compact: {result.summary['sweep_compact']}")
    return "\n".join(lines)

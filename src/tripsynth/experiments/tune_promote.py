"""Tune synthesis methods and promote winning configs to population-scale validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tripsynth.config import write_json
from tripsynth.progress import progress
from tripsynth.experiments.synthesis_sweep import (
    SynthesisSweepResult,
    run_synthesis_hyperparameter_sweep,
)


DEFAULT_TARGET_METRIC = "uncalibrated_absolute_proxy.spearman_correlation"
SORT_METRICS = [
    "uncalibrated_absolute_proxy.spearman_correlation",
    "uncalibrated_absolute_proxy.pearson_correlation",
    "uncalibrated_absolute_proxy.top_10pct_overlap",
]


@dataclass(frozen=True)
class TunePromoteResult:
    run_dir: Path
    sweep: SynthesisSweepResult
    promotion: SynthesisSweepResult
    selected_configs: list[dict[str, Any]]
    summary: dict[str, Any]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _params_dict(params_json: str) -> dict[str, Any]:
    if not params_json or pd.isna(params_json):
        return {}
    return json.loads(str(params_json))


def _sort_metric_columns(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in SORT_METRICS if metric in frame.columns]


def _select_best_config_per_method(leaderboard: pd.DataFrame) -> list[dict[str, Any]]:
    if leaderboard.empty:
        return []
    sort_metrics = _sort_metric_columns(leaderboard)
    if not sort_metrics:
        raise ValueError("Sweep leaderboard has no supported selection metrics.")

    metric_aggs = {metric: "mean" for metric in sort_metrics}
    grouped = (
        leaderboard.groupby(["synthesis_method", "params_json"], dropna=False)
        .agg(metric_aggs | {"run_dir": "count"})
        .rename(columns={"run_dir": "sweep_runs"})
        .reset_index()
    )
    selected: list[dict[str, Any]] = []
    for method, method_rows in grouped.groupby("synthesis_method", sort=False):
        ranked = method_rows.sort_values(sort_metrics, ascending=[False] * len(sort_metrics))
        best = ranked.iloc[0]
        selected.append(
            {
                "synthesis_method": str(method),
                "params": _params_dict(str(best["params_json"])),
                "params_json": str(best["params_json"]),
                "sweep_runs": int(best["sweep_runs"]),
                "selection_metrics": {
                    metric: float(best[metric]) for metric in sort_metrics if pd.notna(best[metric])
                },
            }
        )
    return selected


def _config_with_selected_grids(config: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = deepcopy(config)
    experiment = promoted.setdefault("experiment", {})
    grids: dict[str, dict[str, list[Any]]] = {}
    for row in selected:
        params = row["params"]
        grids[row["synthesis_method"]] = {key: [value] for key, value in params.items()}
    experiment["synthesis_hyperparameter_grids"] = grids
    return promoted


def run_tune_promote_synthesis(
    config: dict[str, Any],
    *,
    methods: list[str] | None = None,
    sweep_seeds: list[int] | None = None,
    promotion_seeds: list[int] | None = None,
    sweep_scale_factors: list[float] | None = None,
    run_dir: str | Path | None = None,
    observed_source: str | None = None,
    force_counts: bool = False,
    diagnostics: bool = False,
    max_runs: int | None = None,
    max_runs_per_method: int | None = None,
    target_population_share: float | None = 0.10,
    target_trip_count: int | None = None,
    target_correlation: float = 0.75,
    target_metric: str = DEFAULT_TARGET_METRIC,
    resume: bool = True,
) -> TunePromoteResult:
    root = Path(run_dir) if run_dir else Path("outputs/runs") / f"tune_promote_{_timestamp()}"
    root.mkdir(parents=True, exist_ok=True)
    if sweep_scale_factors is None:
        sweep_scale_factors = [1.0]
    if sweep_seeds is None:
        sweep_seeds = [int(config.get("project", {}).get("seed", 42))]
    if promotion_seeds is None:
        promotion_seeds = [int(config.get("project", {}).get("seed", 42))]

    with progress(total=2, desc="tune/promote", unit="phase") as phase_bar:
        phase_bar.set_postfix(phase="sweep")
        sweep = run_synthesis_hyperparameter_sweep(
            config,
            methods=methods,
            seeds=sweep_seeds,
            scale_factors=sweep_scale_factors,
            run_dir=root / "sweep",
            force_counts=force_counts,
            observed_source=observed_source,
            diagnostics=diagnostics,
            max_runs=max_runs,
            max_runs_per_method=max_runs_per_method,
            resume=resume,
        )
        phase_bar.update(1)
        selected = _select_best_config_per_method(sweep.leaderboard)
        if not selected:
            raise ValueError("No sweep configurations completed; cannot promote winners.")

        promoted_config = _config_with_selected_grids(config, selected)
        phase_bar.set_postfix(phase="promote")
        promotion = run_synthesis_hyperparameter_sweep(
            promoted_config,
            methods=[row["synthesis_method"] for row in selected],
            seeds=promotion_seeds,
            run_dir=root / "promoted_10pct",
            force_counts=force_counts,
            observed_source=observed_source,
            diagnostics=diagnostics,
            target_population_share=target_population_share,
            target_trip_count=target_trip_count,
            resume=resume,
        )
        phase_bar.update(1)
    best_promotion: dict[str, Any] | None = None
    target_met = False
    if not promotion.leaderboard.empty and target_metric in promotion.leaderboard:
        ranked = promotion.leaderboard.sort_values(target_metric, ascending=False)
        best = ranked.iloc[0]
        best_promotion = {
            "synthesis_method": str(best["synthesis_method"]),
            "seed": int(best["seed"]),
            "scale_factor": float(best["scale_factor"]),
            "params_json": str(best["params_json"]),
            target_metric: float(best[target_metric]),
            "run_dir": str(best["run_dir"]),
        }
        target_met = bool((promotion.leaderboard[target_metric] >= float(target_correlation)).any())

    summary = {
        "run_dir": str(root),
        "sweep_run_dir": str(sweep.run_dir),
        "promotion_run_dir": str(promotion.run_dir),
        "methods": methods,
        "sweep_seeds": [int(seed) for seed in sweep_seeds],
        "promotion_seeds": [int(seed) for seed in promotion_seeds],
        "sweep_scale_factors": [float(scale) for scale in sweep_scale_factors],
        "target_population_share": target_population_share,
        "target_trip_count": target_trip_count,
        "target_metric": target_metric,
        "target_correlation": float(target_correlation),
        "target_met": target_met,
        "selected_configs": selected,
        "best_promotion": best_promotion,
        "sweep_compact": sweep.summary["sweep_compact"],
        "promotion_compact": promotion.summary["sweep_compact"],
    }
    summary_path = root / "tune_promote_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return TunePromoteResult(
        run_dir=root,
        sweep=sweep,
        promotion=promotion,
        selected_configs=selected,
        summary=summary,
    )


def format_tune_promote_summary(result: TunePromoteResult) -> str:
    lines = ["Tune/promote synthesis run complete", f"  run dir: {result.run_dir}"]
    lines.append(f"  sweep runs: {len(result.sweep.leaderboard)}")
    lines.append(f"  promoted runs: {len(result.promotion.leaderboard)}")
    lines.append(f"  selected methods: {', '.join(row['synthesis_method'] for row in result.selected_configs)}")
    best = result.summary.get("best_promotion")
    if best:
        metric = result.summary["target_metric"]
        lines.append(
            f"  best promoted {metric}: {best['synthesis_method']} seed {best['seed']} = {best[metric]:.3f}"
        )
    lines.append(f"  target met: {result.summary['target_met']}")
    lines.append(f"  sweep compact: {result.summary['sweep_compact']}")
    lines.append(f"  promotion compact: {result.summary['promotion_compact']}")
    lines.append(f"  summary: {result.summary['summary_path']}")
    return "\n".join(lines)

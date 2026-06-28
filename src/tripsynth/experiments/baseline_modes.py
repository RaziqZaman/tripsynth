"""Baseline experiment modes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from tripsynth.config import ensure_standard_directories, write_json
from tripsynth.data_sources.survey import build_synthesis_frame, load_survey
from tripsynth.experiments.baseline import (
    BaselineRunResult,
    run_weighted_resampling_desire_line_baseline as _run_spatial_aadt_baseline,
)
from tripsynth.preprocessing.survey_audit import run_survey_audit
from tripsynth.synthesis.weighted_resampling import sample_by_scale
from tripsynth.validation.survey_holdout import (
    split_survey_holdout,
    validate_synthetic_against_survey_holdout,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _survey_coverage_from_audit(audit) -> dict[str, Any]:
    return {
        "covered_years": audit.metadata.get("covered_years"),
        "covered_months_by_year": audit.metadata.get("covered_months_by_year"),
        "full_year_coverage_by_year": audit.metadata.get("full_year_coverage_by_year"),
        "first_survey_date": audit.metadata.get("first_survey_date"),
        "last_survey_date": audit.metadata.get("last_survey_date"),
    }


def _run_survey_holdout_baseline(
    config: dict[str, Any],
    *,
    scale_factor: float | None,
    seed: int | None,
    run_dir: str | Path | None,
) -> BaselineRunResult:
    ensure_standard_directories()
    audit = run_survey_audit(config)
    survey = load_survey(config)
    baseline_config = config.get("validation", {}).get("baseline", {})
    scale = float(scale_factor if scale_factor is not None else baseline_config.get("scale_factor", 1))
    run_seed = int(seed if seed is not None else config.get("project", {}).get("seed", 42))

    synthesis_frame = build_synthesis_frame(survey, config)
    split = split_survey_holdout(synthesis_frame, config, seed=run_seed)
    synthesis = sample_by_scale(split.train, scale_factor=scale, seed=run_seed)
    metrics_by_feature, metrics_by_run = validate_synthetic_against_survey_holdout(
        synthesis.synthetic_trips, split.holdout, config
    )

    run_path = Path(run_dir) if run_dir else Path("outputs/runs") / f"survey_holdout_{_timestamp()}"
    for child in ["synthetic_trips", "metrics", "tables"]:
        (run_path / child).mkdir(parents=True, exist_ok=True)

    synthetic_path = run_path / "synthetic_trips" / "weighted_resampling.csv"
    feature_metrics_path = run_path / "metrics" / "metrics_by_feature.csv"
    metrics_path = run_path / "metrics" / "metrics_by_run.csv"
    summary_path = run_path / "tables" / "baseline_summary.json"

    synthesis.synthetic_trips.to_csv(synthetic_path, index=False)
    metrics_by_feature.to_csv(feature_metrics_path, index=False)
    metrics_by_run.to_csv(metrics_path, index=False)

    summary = {
        "run_dir": str(run_path),
        "validation_mode": "survey_holdout",
        "validation_label": "survey-internal holdout distribution validation",
        "survey_coverage": _survey_coverage_from_audit(audit),
        "holdout_split": split.metadata,
        "observed_counts": {
            "required": False,
            "load_mode": "not_used_for_survey_holdout",
        },
        "synthesis": synthesis.metadata,
        "outputs": {
            "synthetic_trips": str(synthetic_path),
            "metrics_by_feature": str(feature_metrics_path),
            "metrics_by_run": str(metrics_path),
            "summary": str(summary_path),
        },
        "interpretation_caveat": (
            "This validation compares synthetic trips to a held-out survey sample. "
            "It does not validate routed roadway volumes against observed traffic counts."
        ),
    }
    write_json(summary_path, summary)
    return BaselineRunResult(run_dir=run_path, summary=summary)


def run_weighted_resampling_desire_line_baseline(
    config: dict[str, Any],
    *,
    scale_factor: float | None = None,
    seed: int | None = None,
    run_dir: str | Path | None = None,
    force_counts: bool = False,
    observed_source: str | None = None,
    validation_mode: str | None = None,
) -> BaselineRunResult:
    mode = validation_mode or config.get("validation", {}).get("default_mode", "survey_holdout")
    if mode == "survey_holdout":
        return _run_survey_holdout_baseline(
            config,
            scale_factor=scale_factor,
            seed=seed,
            run_dir=run_dir,
        )
    if mode == "spatial_aadt_proxy":
        return _run_spatial_aadt_baseline(
            config,
            scale_factor=scale_factor,
            seed=seed,
            run_dir=run_dir,
            force_counts=force_counts,
            observed_source=observed_source,
        )
    raise ValueError(
        f"Unsupported baseline validation mode {mode!r}. Use survey_holdout or spatial_aadt_proxy."
    )

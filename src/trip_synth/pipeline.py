from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trip_synth.baselines import bayesian_network, weighted_bootstrap
from trip_synth.data.download import download_external_data
from trip_synth.data.load import read_survey_csv
from trip_synth.data.postprocessing import assert_synthetic_contract
from trip_synth.data.preprocessing import FittedPreprocessor
from trip_synth.data.schema import FeatureSchema
from trip_synth.models.sample import sample_vae_method
from trip_synth.models.train import fit_vae_method
from trip_synth.utils.checkpoints import mark_stage_done, stage_done
from trip_synth.utils.io import copy_if_exists, ensure_dir, load_yaml, write_json
from trip_synth.utils.logging import setup_logger
from trip_synth.utils.progress import progress_iter
from trip_synth.utils.seed import set_seed
from trip_synth.validation.aadt_screenlines import build_screenlines
from trip_synth.validation.aadt_validation import run_aadt_validation
from trip_synth.validation.cross_marginals import validate_method_cross_marginals
from trip_synth.validation.marginals import validate_method_marginals
from trip_synth.validation.privacy import validate_method_privacy
from trip_synth.viz.poster_figures import make_poster_figures


METHOD_MODULES = {
    "weighted_bootstrap": weighted_bootstrap,
    "bayesian_network": bayesian_network,
}


def _output_dirs(run_dir: Path) -> None:
    for name in ["logs", "checkpoints", "samples", "metrics", "figures", "tables", "geo", "data_cache"]:
        ensure_dir(run_dir / name)


def _sample_n(config: dict[str, Any], df: pd.DataFrame, schema: FeatureSchema, logger) -> int:
    requested = config.get("sample_n", len(df))
    if requested == "population_from_weight_sum":
        total = float(pd.to_numeric(df[schema.weight_column], errors="coerce").fillna(0).sum())
        n = int(round(total))
        if n > 5_000_000:
            logger.warning(
                "population_from_weight_sum resolved to %s rows. Full run will be large; consider chunked validation.",
                n,
            )
        return max(1, n)
    return int(requested)


def _method_sample_n(method: str, target_rows: int, config: dict[str, Any], logger) -> int:
    caps = config.get("method_sample_caps", {}) or {}
    cap = caps.get(method)
    if cap is None:
        return int(target_rows)
    cap_rows = max(1, int(cap))
    if target_rows > cap_rows:
        logger.warning(
            "Capping %s sample at %s rows from target %s rows; count validations will use expansion factor %.6f.",
            method,
            cap_rows,
            target_rows,
            target_rows / cap_rows,
        )
        return cap_rows
    return int(target_rows)


def _write_sample_scaling(run_dir: Path, method: str, target_rows: int, generated_rows: int, sample_path: Path) -> None:
    generated = max(1, int(generated_rows))
    target = max(1, int(target_rows))
    write_json(
        {
            "method": method,
            "target_rows": target,
            "generated_rows": generated,
            "sample_expansion_factor": float(target / generated),
            "sample_path": str(sample_path),
            "note": "Synthetic count validations multiply this method's raw screenline counts by sample_expansion_factor.",
        },
        run_dir / "metrics" / f"{method}_sample_scaling.json",
    )


def _fit_and_sample(
    method: str,
    real_df: pd.DataFrame,
    schema: FeatureSchema,
    config: dict[str, Any],
    run_dir: Path,
    n_rows: int,
    resume: bool,
    logger,
) -> pd.DataFrame:
    sample_path = run_dir / "samples" / f"{method}_synthetic.csv"
    method_rows = _method_sample_n(method, n_rows, config, logger)
    if resume and sample_path.exists():
        synthetic = pd.read_csv(sample_path, low_memory=False)
        if len(synthetic) == method_rows:
            logger.info("Reusing existing sample for %s: %s", method, sample_path)
            _write_sample_scaling(run_dir, method, n_rows, len(synthetic), sample_path)
            return synthetic
        logger.warning(
            "Regenerating %s because existing sample has %s rows but config now requests %s rows.",
            method,
            len(synthetic),
            method_rows,
        )

    logger.info("Fitting method: %s", method)
    method_dir = ensure_dir(run_dir / "checkpoints" / method)
    if method in {"contrastive_vae", "noncontrastive_vae"}:
        artifact = fit_vae_method(real_df, schema, config, method_dir, method=method)
        synthetic = sample_vae_method(
            artifact,
            method_rows,
            schema,
            config,
            run_dir / "samples",
            run_id=str(config.get("run_name", "run")),
        )
    else:
        module = METHOD_MODULES[method]
        artifact = module.fit(real_df, schema, config, method_dir)
        synthetic = module.sample(artifact, method_rows, schema, config, run_dir / "samples")

    assert_synthetic_contract(synthetic, schema)
    synthetic.to_csv(sample_path, index=False)
    _write_sample_scaling(run_dir, method, n_rows, len(synthetic), sample_path)
    logger.info("Saved %s synthetic rows for %s to %s", len(synthetic), method, sample_path)
    return synthetic


def _best(summary: pd.DataFrame, metric: str, ascending: bool = True) -> str:
    if summary.empty or metric not in summary:
        return "not available"
    sub = summary.dropna(subset=[metric])
    if sub.empty:
        return "not available"
    return str(sub.sort_values(metric, ascending=ascending).iloc[0]["method"])


def _write_run_summary(
    run_dir: Path,
    config: dict[str, Any],
    sample_rows: dict[str, int],
    summary: pd.DataFrame,
    privacy_summary: pd.DataFrame,
    warnings: list[str],
    runtime_seconds: float,
    aadt_status: dict[str, Any] | None,
) -> None:
    contrastive_delta = "not available"
    if not summary.empty and {"method", "mean_categorical_tv"}.issubset(summary.columns):
        rows = summary.set_index("method")
        if "contrastive_vae" in rows.index and "noncontrastive_vae" in rows.index:
            delta = rows.loc["contrastive_vae", "mean_categorical_tv"] - rows.loc["noncontrastive_vae", "mean_categorical_tv"]
            contrastive_delta = f"marginal TV delta = {delta:.4f} (negative favors contrastive)"

    lines = [
        f"# Run Summary: {config.get('run_name', 'run')}",
        "",
        f"Runtime seconds: {runtime_seconds:.1f}",
        "",
        "## Methods",
    ]
    for method in config.get("methods", []):
        lines.append(f"- {method}: {sample_rows.get(method, 0)} synthetic rows")
    lines.extend(
        [
            "",
            "## Best Methods",
            f"- Best marginal method: {_best(summary, 'mean_categorical_tv', ascending=True)}",
            f"- Best cross-marginal method: {_best(summary, 'mean_cross_tv', ascending=True)}",
            f"- Best privacy/non-copying method: {_best(privacy_summary, 'privacy_score', ascending=False)}",
            f"- Best AADT method: {'not run' if not aadt_status or aadt_status.get('status') == 'skipped' else 'see tables/aadt_validation_summary.csv'}",
            f"- Contrastive VAE vs non-contrastive VAE: {contrastive_delta}",
            "",
            "## Warnings",
        ]
    )
    lines.extend([f"- {w}" for w in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "## Key Figure Paths",
            "- figures/poster/01_architecture_contrastive_vae.png",
            "- figures/poster/04_marginal_distance_leaderboard.png",
            "- figures/poster/05_cross_marginal_error_heatmap.png",
            "- figures/poster/06_privacy_copy_rate_by_method.png",
            "- figures/poster/11_best_method_summary_panel.png",
            "",
            "## AADT Status",
            f"- {aadt_status or {'status': 'not requested'}}",
        ]
    )
    (run_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n")


def run_pipeline(args: argparse.Namespace) -> Path:
    root = Path.cwd()
    config = load_yaml(args.config)
    if args.skip_aadt:
        config.setdefault("validation", {})["run_aadt"] = False
    run_name = str(config.get("run_name", "run"))
    run_dir = root / "outputs" / "runs" / run_name
    _output_dirs(run_dir)
    logger = setup_logger("trip_synth", run_dir / "logs" / "pipeline.log")
    start = time.time()
    warnings: list[str] = []
    set_seed(int(config.get("seed", 0)) if config.get("seed") is not None else None)

    logger.info("Starting run %s", run_name)
    copy_if_exists(args.config, run_dir / "logs" / "config_snapshot.yaml")
    copy_if_exists("configs/schema.yaml", run_dir / "logs" / "schema_snapshot.yaml")
    write_json(config, run_dir / "logs" / "config_snapshot.json")

    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    input_csv = root / str(config["input_csv"])
    real_df, load_warnings = read_survey_csv(input_csv, schema)
    warnings.extend(load_warnings)
    report = schema.required_column_report(real_df)
    if report["missing"]:
        warnings.append(f"Missing expected schema columns: {report['missing']}")
    write_json(report, run_dir / "logs" / "schema_check.json")
    mark_stage_done(run_dir, "schema_check")

    n_rows = _sample_n(config, real_df, schema, logger)
    reference_preprocessor = FittedPreprocessor.fit(real_df, schema)
    methods = [str(m) for m in config.get("methods", [])]
    synthetic_by_method: dict[str, pd.DataFrame] = {}
    sample_rows: dict[str, int] = {}
    for method in progress_iter(methods, desc="Pipeline synthesis methods", total=len(methods), unit="method"):
        synthetic = _fit_and_sample(
            method,
            real_df,
            schema,
            config,
            run_dir,
            n_rows,
            resume=bool(args.resume),
            logger=logger,
        )
        synthetic_by_method[method] = synthetic
        sample_rows[method] = len(synthetic)
    mark_stage_done(run_dir, "synthesis")

    validation_cfg = config.get("validation", {})
    summary_rows: list[dict[str, Any]] = []
    privacy_rows: list[dict[str, Any]] = []
    validation_items = list(synthetic_by_method.items())
    for method, synthetic in progress_iter(validation_items, desc="Internal validation methods", total=len(validation_items), unit="method"):
        row: dict[str, Any] = {"method": method, "synthetic_rows": len(synthetic)}
        if validation_cfg.get("run_marginals", True):
            marg = validate_method_marginals(real_df, synthetic, schema, reference_preprocessor, method, run_dir)
            row.update(marg.get("summary", {}))
        if validation_cfg.get("run_cross_marginals", True):
            cross = validate_method_cross_marginals(real_df, synthetic, schema, reference_preprocessor, method, run_dir)
            row.update(cross.get("summary", {}))
        if validation_cfg.get("run_privacy", True):
            privacy = validate_method_privacy(real_df, synthetic, schema, reference_preprocessor, method, run_dir)
            row["privacy_score"] = privacy.get("privacy_score")
            privacy_rows.append(privacy)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    privacy_summary = pd.DataFrame(privacy_rows)
    summary.to_csv(run_dir / "tables" / "method_validation_summary.csv", index=False)
    privacy_summary.to_csv(run_dir / "tables" / "privacy_summary.csv", index=False)
    mark_stage_done(run_dir, "validation")

    aadt_status: dict[str, Any] | None = None
    if validation_cfg.get("run_aadt", False):
        logger.info("Starting AADT/two-prong traffic validation")
        geo_files = config.get("geo", {})
        missing_geo = [
            p for p in geo_files.get("tract_files", []) if not Path(p).exists()
        ]
        points_file = geo_files.get("aadt_points_file")
        if points_file and not Path(points_file).exists():
            missing_geo.append(points_file)
        if missing_geo and config.get("data", {}).get("auto_download_external", False):
            logger.info("Attempting external data download for AADT validation")
            manifest = download_external_data("configs/data_sources.yaml")
            write_json(manifest, run_dir / "data_cache" / "DATA_MANIFEST.json")
        logger.info("Building AADT screenlines")
        screenline_status = build_screenlines(config, run_dir)
        logger.info("Running AADT validation tiers")
        aadt_status = run_aadt_validation(config, run_dir)
        aadt_status["screenlines"] = screenline_status
    else:
        aadt_status = {"status": "skipped", "reason": "run_aadt is false in config"}
        write_json(aadt_status, run_dir / "metrics" / "aadt_validation_skipped.json")
    mark_stage_done(run_dir, "aadt")

    make_poster_figures(run_dir, methods)
    mark_stage_done(run_dir, "poster_figures")
    runtime = time.time() - start
    _write_run_summary(
        run_dir,
        config,
        sample_rows,
        summary,
        privacy_summary,
        warnings,
        runtime,
        aadt_status,
    )
    logger.info("Run complete: %s", run_dir)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", default=None)
    parser.add_argument("--skip-aadt", action="store_true")
    args = parser.parse_args(argv)
    run_dir = run_pipeline(args)
    print(f"Completed run at {run_dir}")


if __name__ == "__main__":
    main()

"""Command-line interface for the Trip Synthesis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tripsynth.config import load_config
from tripsynth.data_sources.observed_counts.fhwa_hpms import (
    fetch_fhwa_hpms_2018,
    format_fetch_summary,
)
from tripsynth.data_sources.survey import load_survey
from tripsynth.experiments.matrix import build_experiment_matrix
from tripsynth.logging_utils import configure_logging
from tripsynth.preprocessing.survey_audit import format_audit_summary, run_survey_audit
from tripsynth.synthesis.bayesian_network import BayesianNetworkSynthesizer
from tripsynth.synthesis.weighted_resampling import sample_by_scale
from tripsynth.validation.temporal_alignment import require_temporal_overlap


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")


def cmd_audit_survey(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_survey_audit(config)
    print(format_audit_summary(result))
    return 0


def cmd_fetch_counts(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = fetch_fhwa_hpms_2018(config, force=args.force)
    print(format_fetch_summary(result))
    return 0


def cmd_prep_survey(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_survey_audit(config)
    print("Survey prep currently runs the required audit step.")
    print(format_audit_summary(result))
    return 0


def cmd_prep_network(args: argparse.Namespace) -> int:
    print("Network preparation is scaffolded. Configure tract geometries and OSM/network inputs first.")
    return 0


def cmd_run_synthesis(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    survey = load_survey(config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    scale = float(args.scale_factor)

    if args.method == "weighted_resampling":
        result = sample_by_scale(survey.canonical, scale_factor=scale, seed=seed)
    elif args.method == "bayesian_network":
        n = max(1, int(len(survey.canonical) * scale))
        result = BayesianNetworkSynthesizer().fit(survey.canonical).sample(n, seed=seed)
    else:
        raise NotImplementedError(
            f"{args.method} synthesis is scaffolded but not runnable in the first deliverable."
        )

    output_dir = Path("outputs/runs/manual/synthetic_trips")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.method}_scale_{scale:g}_seed_{seed}.csv"
    result.synthetic_trips.to_csv(output_path, index=False)
    print(f"Synthesis complete: {args.method}")
    print(f"  synthetic trips: {len(result.synthetic_trips)}")
    print(f"  output: {output_path}")
    print(f"  metadata: {result.metadata}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    print(
        "Routing is scaffolded for tract-border, OSMnx, MATSim, and SUMO/AequilibraE adapters. "
        "Run audit-survey and fetch-counts before enabling route assignment."
    )
    print(f"  synth-method: {args.synth_method}")
    print(f"  routing-method: {args.routing_method}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    import json

    survey_audit_path = Path("data/metadata/survey_audit.json")
    manifest_path = Path("data/metadata/observed_counts_manifest.json")
    if args.mode == "temporal_overlap_counts":
        if not survey_audit_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                "Run audit-survey and fetch-counts before temporal_overlap_counts validation."
            )
        survey_audit = json.loads(survey_audit_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require_temporal_overlap(survey_audit, manifest, mode=args.mode)
    print(
        f"Validation mode '{args.mode}' is recognized. Full routed-volume validation follows "
        "the first deliverable; HPMS baseline is labeled spatial roadway-volume proxy validation against AADT."
    )
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    matrix = build_experiment_matrix(config)
    output_path = Path("outputs/experiment_matrix.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix).to_csv(output_path, index=False)
    print(f"Experiment matrix written: {output_path} ({len(matrix)} rows)")
    print("Full orchestration is scaffolded after the audit/fetch first deliverable.")
    return 0


def cmd_make_report_assets(args: argparse.Namespace) -> int:
    print(
        "Report assets are produced by audit-survey and fetch-counts in this deliverable. "
        "Run those commands to refresh tables, figures, and maps."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tripsynth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-survey", help="Audit survey schema and temporal coverage.")
    _add_config(audit)
    audit.set_defaults(func=cmd_audit_survey)

    fetch = subparsers.add_parser("fetch-counts", help="Download and normalize observed counts.")
    _add_config(fetch)
    fetch.add_argument("--force", action="store_true", help="Ignore cached downloads/outputs.")
    fetch.set_defaults(func=cmd_fetch_counts)

    prep_survey = subparsers.add_parser("prep-survey", help="Prepare survey data.")
    _add_config(prep_survey)
    prep_survey.set_defaults(func=cmd_prep_survey)

    prep_network = subparsers.add_parser("prep-network", help="Prepare routing network data.")
    _add_config(prep_network)
    prep_network.set_defaults(func=cmd_prep_network)

    synth = subparsers.add_parser("run-synthesis", help="Run a synthesis method.")
    _add_config(synth)
    synth.add_argument("--method", required=True, choices=["weighted_resampling", "bayesian_network", "vae", "diffusion"])
    synth.add_argument("--scale-factor", type=float, default=1.0)
    synth.add_argument("--seed", type=int, default=None)
    synth.set_defaults(func=cmd_run_synthesis)

    route = subparsers.add_parser("route", help="Route synthesized trips.")
    _add_config(route)
    route.add_argument("--synth-method", required=True)
    route.add_argument("--routing-method", required=True)
    route.set_defaults(func=cmd_route)

    validate = subparsers.add_parser("validate", help="Validate routed volumes.")
    _add_config(validate)
    validate.add_argument(
        "--mode",
        default="spatial_aadt_proxy",
        choices=["spatial_aadt_proxy", "temporal_overlap_counts", "relative_share_validation"],
    )
    validate.set_defaults(func=cmd_validate)

    experiment = subparsers.add_parser("run-experiment", help="Run the experiment matrix.")
    _add_config(experiment)
    experiment.set_defaults(func=cmd_run_experiment)

    report = subparsers.add_parser("make-report-assets", help="Refresh report tables/figures/maps.")
    _add_config(report)
    report.set_defaults(func=cmd_make_report_assets)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

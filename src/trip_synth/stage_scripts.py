from __future__ import annotations

import argparse

from trip_synth.pipeline import main as pipeline_main
from trip_synth.validation.aadt_validation import main as aadt_main
from trip_synth.validation.marginals import validate_method_marginals
from trip_synth.validation.privacy import validate_method_privacy


def train_method_main() -> None:
    parser = argparse.ArgumentParser(description="Train a method through the generic pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default=None)
    args, extra = parser.parse_known_args()
    pipeline_args = ["--config", args.config, *extra]
    pipeline_main(pipeline_args)


def sample_method_main() -> None:
    parser = argparse.ArgumentParser(description="Sample a method through the generic pipeline.")
    parser.add_argument("--config", required=True)
    args, extra = parser.parse_known_args()
    pipeline_main(["--config", args.config, "--resume", *extra])


def validate_marginals_main() -> None:
    parser = argparse.ArgumentParser(description="Run validation through the generic pipeline.")
    parser.add_argument("--config", required=True)
    args, extra = parser.parse_known_args()
    pipeline_main(["--config", args.config, "--resume", *extra])


def validate_privacy_main() -> None:
    parser = argparse.ArgumentParser(description="Run privacy validation through the generic pipeline.")
    parser.add_argument("--config", required=True)
    args, extra = parser.parse_known_args()
    pipeline_main(["--config", args.config, "--resume", *extra])

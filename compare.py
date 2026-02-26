#!/usr/bin/env python3
"""Plot comparative histograms for shared numeric columns across two CSV files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_SYNTH = Path("sample_synthetic_trips.csv")
DEFAULT_SURVEY_CANDIDATES = [
    Path("cobined-flat-survey.csv"),
    Path("combined-flat-survey.csv"),
]
DEFAULT_OUTPUT_DIR = Path("comparison_histograms")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "unnamed_column"


def resolve_default_survey_path() -> Path:
    for candidate in DEFAULT_SURVEY_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_SURVEY_CANDIDATES[0]


def find_shared_numeric_columns(synth_df: pd.DataFrame, survey_df: pd.DataFrame) -> list[str]:
    shared = sorted(set(synth_df.columns) & set(survey_df.columns))
    numeric_shared: list[str] = []
    for col in shared:
        synth_numeric = pd.to_numeric(synth_df[col], errors="coerce")
        survey_numeric = pd.to_numeric(survey_df[col], errors="coerce")
        if synth_numeric.notna().any() and survey_numeric.notna().any():
            numeric_shared.append(col)
    return numeric_shared


def plot_histograms(
    synth_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    columns: list[str],
    output_dir: Path,
    bins: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in columns:
        synth = pd.to_numeric(synth_df[col], errors="coerce").dropna()
        survey = pd.to_numeric(survey_df[col], errors="coerce").dropna()
        if synth.empty or survey.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(survey, bins=bins, alpha=0.55, label="combined-flat-survey.csv", density=True)
        ax.hist(synth, bins=bins, alpha=0.55, label="synthetic_trips.csv", density=True)
        ax.set_title(f"Comparative Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_filename(col)}.png", dpi=140)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=DEFAULT_SYNTH,
        help=f"Synthetic CSV path (default: {DEFAULT_SYNTH})",
    )
    parser.add_argument(
        "--survey",
        type=Path,
        default=resolve_default_survey_path(),
        help=(
            "Survey CSV path (default: first existing of "
            "cobined-flat-survey.csv, combined-flat-survey.csv)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store histogram PNGs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins (default: 40)",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=None,
        help="Optional cap on number of columns plotted.",
    )
    return parser.parse_args()


def main() -> int:
    global pd, plt
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        raise ModuleNotFoundError(
            f"Missing Python dependency: {missing}. "
            "Install required packages with: pip install pandas matplotlib"
        ) from exc

    args = parse_args()
    if not args.synthetic.exists():
        raise FileNotFoundError(f"Missing synthetic CSV: {args.synthetic}")
    if not args.survey.exists():
        raise FileNotFoundError(f"Missing survey CSV: {args.survey}")
    if args.bins < 1:
        raise ValueError("--bins must be >= 1")

    synth_df = pd.read_csv(args.synthetic, low_memory=False)
    survey_df = pd.read_csv(args.survey, low_memory=False)

    numeric_columns = find_shared_numeric_columns(synth_df, survey_df)
    if args.max_columns is not None:
        numeric_columns = numeric_columns[: args.max_columns]

    if not numeric_columns:
        raise ValueError("No shared numeric columns found between the two files.")

    plot_histograms(
        synth_df=synth_df,
        survey_df=survey_df,
        columns=numeric_columns,
        output_dir=args.output_dir,
        bins=args.bins,
    )
    print(
        f"Saved {len(numeric_columns)} histogram(s) to {args.output_dir} "
        f"using synthetic={args.synthetic} survey={args.survey}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the full tuple -> synthesize -> untuple -> validate -> counts pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PIPELINE = [
    ["python3", "tuple.py"],
    ["python3", "synthesize.py"],
    ["python3", "untuple.py"],
    ["python3", "validate.py"],
    ["python3", "get_counts.py"],
    ["python3", "get_tupled_counts.py"],
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    for cmd in PIPELINE:
        print(f"$ {' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=repo_root)
        if completed.returncode != 0:
            print(
                f"Pipeline stopped because {' '.join(cmd)} failed with exit code {completed.returncode}.",
                file=sys.stderr,
                flush=True,
            )
            return completed.returncode

    print("Pipeline completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

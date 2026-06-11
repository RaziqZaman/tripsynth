#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT"

"$PYTHON" -m trip_synth.pipeline "$@"

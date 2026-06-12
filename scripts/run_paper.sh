#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "${BASH_SOURCE[0]}")/run_pipeline.sh" --config configs/paper.yaml "$@"

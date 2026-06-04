#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

echo "[$(date --iso-8601=seconds)] Running 09_run-vae-speed-pipeline.sh"
./09_run-vae-speed-pipeline.sh
status=$?

echo
echo "[$(date --iso-8601=seconds)] Pipeline exited with status $status"
echo "Log: 06x_stage1_va_speed/rerun.log"
echo
echo "Leaving this tmux pane open for inspection. Type exit to close it."
exec bash

#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-tripsynth_vae_speed}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or is not on PATH" >&2
  exit 127
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" "./09_tmux-vae-speed-entrypoint.sh"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"

#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-tripsynth_vae_speed}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux send-keys -t "$SESSION" "cd $WORKDIR && ./09_resume-vae-speed-after-demand.sh" C-m
  echo "sent resume command to tmux session: $SESSION"
else
  tmux new-session -d -s "$SESSION" -c "$WORKDIR" ./09_resume-vae-speed-after-demand.sh
  echo "started tmux session: $SESSION"
fi

echo "attach with: tmux attach -t $SESSION"

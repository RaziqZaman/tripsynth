#!/usr/bin/env bash
set -euo pipefail

SCENARIO_DIR="${SCENARIO_DIR:-06x_stage1_md_internal}"
NETWORK_DIR="${NETWORK_DIR:-06x_stage1_osm}"
OBSERVED_SPEEDS="${OBSERVED_SPEEDS:-$NETWORK_DIR/observed_speeds.csv}"

choose_events() {
  local primary="$1"
  local fallback="$2"
  if [[ -s "$primary" ]]; then
    printf '%s\n' "$primary"
  elif [[ -s "$fallback" ]]; then
    printf '%s\n' "$fallback"
  else
    printf '%s\n' "$primary"
  fi
}

REAL_EVENTS="${REAL_EVENTS:-$(choose_events "$SCENARIO_DIR/matsim_real/ITERS/it.0/0.events.xml.gz" "$SCENARIO_DIR/matsim_real/output_events.xml.gz")}"
SYNTHETIC_EVENTS="${SYNTHETIC_EVENTS:-$(choose_events "$SCENARIO_DIR/matsim_synthetic/ITERS/it.0/0.events.xml.gz" "$SCENARIO_DIR/matsim_synthetic/output_events.xml.gz")}"

if [[ ! -s "$OBSERVED_SPEEDS" ]]; then
  echo "missing observed speed matches: $OBSERVED_SPEEDS" >&2
  echo "run 06_prepare-osm-stage1.sh with VALIDATION_TARGET=speed first" >&2
  exit 2
fi

.venv/bin/python 06_extract-matsim-link-speeds.py \
  --events "$REAL_EVENTS" \
  --network-links "$NETWORK_DIR/network_links.csv" \
  --out "$SCENARIO_DIR/real_matsim_link_speeds.csv"

.venv/bin/python 06_extract-matsim-link-speeds.py \
  --events "$SYNTHETIC_EVENTS" \
  --network-links "$NETWORK_DIR/network_links.csv" \
  --out "$SCENARIO_DIR/synthetic_matsim_link_speeds.csv"

.venv/bin/python 06_compare-stage1-speeds.py \
  --observed-speeds "$OBSERVED_SPEEDS" \
  --real-link-speeds "$SCENARIO_DIR/real_matsim_link_speeds.csv" \
  --synthetic-link-speeds "$SCENARIO_DIR/synthetic_matsim_link_speeds.csv" \
  --out "$SCENARIO_DIR/speed_validation.csv" \
  --summary-out "$SCENARIO_DIR/speed_validation_summary.csv"

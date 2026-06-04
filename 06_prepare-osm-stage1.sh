#!/usr/bin/env bash
set -euo pipefail

NETWORK_DIR="${NETWORK_DIR:-06x_stage1_osm}"
SCENARIO_DIR="${SCENARIO_DIR:-06x_stage1_md_internal}"
OSM_URL="${OSM_URL:-https://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf}"
OSM_PBF="${OSM_PBF:-$NETWORK_DIR/$(basename "$OSM_URL")}"
OSM_BBOX="${OSM_BBOX:-}"
VALIDATION_TARGET="${VALIDATION_TARGET:-speed}"
MAX_MATCH_DISTANCE_M="${MAX_MATCH_DISTANCE_M:-500}"
OBSERVED_SPEEDS_RAW="${OBSERVED_SPEEDS_RAW:-$NETWORK_DIR/observed_speeds_raw.csv}"
OBSERVED_SPEEDS="${OBSERVED_SPEEDS:-$NETWORK_DIR/observed_speeds.csv}"

.venv/bin/python 06_download-osm-network.py \
  --url "$OSM_URL" \
  --out "$OSM_PBF"

network_args=(
  --pbf "$OSM_PBF"
  --out "$NETWORK_DIR/network.xml.gz"
  --links-out "$NETWORK_DIR/network_links.csv"
  --link-map-out "$NETWORK_DIR/matsim_link_count_map.csv"
  --stats-out "$NETWORK_DIR/network_stats.json"
)
if [[ -n "$OSM_BBOX" ]]; then
  network_args+=("--bbox=$OSM_BBOX")
fi
.venv/bin/python 06_make-osm-matsim-network.py "${network_args[@]}"

if [[ "$VALIDATION_TARGET" == "speed" || "$VALIDATION_TARGET" == "both" ]]; then
  .venv/bin/python 06_fetch-observed-speeds.py \
    --out "$OBSERVED_SPEEDS_RAW"

  .venv/bin/python 06_match-speeds-to-links.py \
    --speeds "$OBSERVED_SPEEDS_RAW" \
    --links "$NETWORK_DIR/network_links.csv" \
    --out "$OBSERVED_SPEEDS" \
    --max-distance-m "$MAX_MATCH_DISTANCE_M"
fi

if [[ "$VALIDATION_TARGET" == "count" || "$VALIDATION_TARGET" == "both" ]]; then
  .venv/bin/python 06_match-counts-to-links.py \
    --links "$NETWORK_DIR/network_links.csv" \
    --out "$NETWORK_DIR/observed_counts.csv" \
    --max-distance-m "$MAX_MATCH_DISTANCE_M"
fi

if [[ "$VALIDATION_TARGET" != "speed" && "$VALIDATION_TARGET" != "count" && "$VALIDATION_TARGET" != "both" ]]; then
  echo "VALIDATION_TARGET must be speed, count, or both; got $VALIDATION_TARGET" >&2
  exit 2
fi

.venv/bin/python 06_write-matsim-configs.py \
  --stage1-dir "$NETWORK_DIR" \
  --scenario-dir "$SCENARIO_DIR"

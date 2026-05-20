#!/usr/bin/env bash
set -euo pipefail

NETWORK_DIR="${NETWORK_DIR:-06x_stage1_osm}"
SCENARIO_DIR="${SCENARIO_DIR:-06x_stage1_md_internal}"
MAX_MATCH_DISTANCE_M="${MAX_MATCH_DISTANCE_M:-100}"

.venv/bin/python 06_download-osm-network.py   --out "$NETWORK_DIR/maryland-latest.osm.pbf"

.venv/bin/python 06_make-osm-matsim-network.py   --pbf "$NETWORK_DIR/maryland-latest.osm.pbf"   --out "$NETWORK_DIR/network.xml.gz"   --links-out "$NETWORK_DIR/network_links.csv"   --link-map-out "$NETWORK_DIR/matsim_link_count_map.csv"   --stats-out "$NETWORK_DIR/network_stats.json"

.venv/bin/python 06_match-counts-to-links.py   --links "$NETWORK_DIR/network_links.csv"   --out "$NETWORK_DIR/observed_counts.csv"   --max-distance-m "$MAX_MATCH_DISTANCE_M"

.venv/bin/python 06_write-matsim-configs.py   --stage1-dir "$NETWORK_DIR"   --scenario-dir "$SCENARIO_DIR"

#!/usr/bin/env bash
set -euo pipefail

SCENARIO_DIR="${SCENARIO_DIR:-06x_stage1_md_internal}"
NETWORK_DIR="${NETWORK_DIR:-06x_stage1_osm}"
REAL_EVENTS="${REAL_EVENTS:-$SCENARIO_DIR/matsim_real/ITERS/it.0/0.events.xml.gz}"
SYNTHETIC_EVENTS="${SYNTHETIC_EVENTS:-$SCENARIO_DIR/matsim_synthetic/ITERS/it.0/0.events.xml.gz}"

.venv/bin/python 06_extract-matsim-link-volumes.py   --events "$REAL_EVENTS"   --out "$SCENARIO_DIR/real_matsim_link_volumes.csv"

.venv/bin/python 06_extract-matsim-link-volumes.py   --events "$SYNTHETIC_EVENTS"   --out "$SCENARIO_DIR/synthetic_matsim_link_volumes.csv"

.venv/bin/python 06_aggregate-matsim-volumes.py   --link-volumes "$SCENARIO_DIR/real_matsim_link_volumes.csv"   --link-map "$NETWORK_DIR/matsim_link_count_map.csv"   --out "$SCENARIO_DIR/real_assigned_link_counts.csv"

.venv/bin/python 06_aggregate-matsim-volumes.py   --link-volumes "$SCENARIO_DIR/synthetic_matsim_link_volumes.csv"   --link-map "$NETWORK_DIR/matsim_link_count_map.csv"   --out "$SCENARIO_DIR/synthetic_assigned_link_counts.csv"

.venv/bin/python 06_compare-stage1-counts.py   --scenario-totals "$SCENARIO_DIR/scenario_totals.csv"   --observed-counts "$NETWORK_DIR/observed_counts.csv"   --real-assigned "$SCENARIO_DIR/real_assigned_link_counts.csv"   --synthetic-assigned "$SCENARIO_DIR/synthetic_assigned_link_counts.csv"   --out "$SCENARIO_DIR/count_validation.csv"   --summary-out "$SCENARIO_DIR/count_validation_summary.csv"

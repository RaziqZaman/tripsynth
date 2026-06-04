#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND="${MPLBACKEND:-Agg}"

BBOX="${BBOX:--77.95,38.20,-76.75,39.35}"
NETWORK_DIR="${NETWORK_DIR:-06x_stage1_va_speed_osm}"
SCENARIO_DIR="${SCENARIO_DIR:-06x_stage1_va_speed}"
POP_DIR="${POP_DIR:-07x_population_scale}"
CENTROIDS="${CENTROIDS:-06x_stage1/taz_centroids.csv}"
MATSIM_HEAP="${MATSIM_HEAP:-96g}"
THREADS="${THREADS:-8}"

mkdir -p "$SCENARIO_DIR" "$POP_DIR"
exec > >(tee -a "$SCENARIO_DIR/rerun.log") 2>&1

echo "[$(date --iso-8601=seconds)] Starting deeper-VAE + VDOT speed pipeline"
echo "BBOX=$BBOX"
echo "NETWORK_DIR=$NETWORK_DIR"
echo "SCENARIO_DIR=$SCENARIO_DIR"

.venv/bin/python 00_consolidate-survey.py
.venv/bin/python 01_distill-survey.py
.venv/bin/python 02_filter-survey.py
.venv/bin/python 03_fill-blanks.py

.venv/bin/python 04_sample.py \
  --hidden-dims 2048 1920 1536 1024 \
  --batch-size 512 \
  --sample-batch-size 2048 \
  --grad-clip 5 \
  --plateau-patience 80 \
  --plateau-factor 0.5 \
  --early-stopping-patience 400 \
  --min-epochs 400 \
  --min-delta 1e-4

.venv/bin/python 05_validate.py
.venv/bin/python 05_visualize-total-variations.py

.venv/bin/python 07_synthesize-population-scale.py \
  --output-csv "$POP_DIR/synthetic_population_trips.csv" \
  --sample-batch-size 2048

.venv/bin/python 07_compare-population-scale.py \
  --synthetic-csv "$POP_DIR/synthetic_population_trips.csv" \
  --out-dir "$POP_DIR/comparison"

.venv/bin/python 06_fetch-taz-centroids.py

.venv/bin/python 06_prepare-stage1-demand.py \
  --real-csv 03x_filled-survey.csv \
  --synthetic-csv 04x_synthetic-trips.csv \
  --output-dir "$SCENARIO_DIR" \
  --synthetic-scale 12 \
  --taz-centroids "$CENTROIDS" \
  --state-filter Virginia \
  --state-filter-mode both \
  --bbox="$BBOX" \
  --bbox-filter-mode both

NETWORK_DIR="$NETWORK_DIR" \
SCENARIO_DIR="$SCENARIO_DIR" \
OSM_URL="https://download.geofabrik.de/north-america/us/virginia-latest.osm.pbf" \
OSM_BBOX="$BBOX" \
VALIDATION_TARGET="speed" \
MAX_MATCH_DISTANCE_M="500" \
./06_prepare-osm-stage1.sh

.venv/bin/python 06_make-matsim-population.py \
  --vehicle-trips "$SCENARIO_DIR/real_vehicle_trips.csv" \
  --taz-centroids "$CENTROIDS" \
  --out "$SCENARIO_DIR/real_population.xml.gz"

.venv/bin/python 06_make-matsim-population.py \
  --vehicle-trips "$SCENARIO_DIR/synthetic_vehicle_trips.csv" \
  --taz-centroids "$CENTROIDS" \
  --out "$SCENARIO_DIR/synthetic_population.xml.gz"

.venv/bin/python 06_write-matsim-configs.py \
  --stage1-dir "$NETWORK_DIR" \
  --scenario-dir "$SCENARIO_DIR" \
  --threads "$THREADS"

SCENARIO_DIR="$SCENARIO_DIR" MATSIM_HEAP="$MATSIM_HEAP" ./06_run_matsim_real.sh
SCENARIO_DIR="$SCENARIO_DIR" MATSIM_HEAP="$MATSIM_HEAP" ./06_run_matsim_synthetic.sh

SCENARIO_DIR="$SCENARIO_DIR" NETWORK_DIR="$NETWORK_DIR" ./06_postprocess_matsim.sh

.venv/bin/python 06_visualize-stage1-results.py \
  --scenario-dir "$SCENARIO_DIR" \
  --network-dir "$NETWORK_DIR" \
  --out-dir "$SCENARIO_DIR/stage1_comparison"

echo "[$(date --iso-8601=seconds)] DONE"
echo "Speed summary: $SCENARIO_DIR/speed_validation_summary.csv"
echo "Log: $SCENARIO_DIR/rerun.log"

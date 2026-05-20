#!/usr/bin/env bash
set -euo pipefail

SCENARIO_DIR="${SCENARIO_DIR:-07x_population_scale_matsim}"
POP_DIR="${POP_DIR:-07x_population_scale}"
NETWORK_DIR="${NETWORK_DIR:-06x_stage1_osm}"
CENTROIDS="${CENTROIDS:-06x_stage1/taz_centroids.csv}"
STATE_FILTER="${STATE_FILTER:-Maryland}"
STATE_FILTER_MODE="${STATE_FILTER_MODE:-both}"
MATSIM_HEAP="${MATSIM_HEAP:-112g}"
THREADS="${THREADS:-8}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8192}"
ALLOW_FULL_POP_MATSIM="${ALLOW_FULL_POP_MATSIM:-0}"

mkdir -p "$POP_DIR" "$SCENARIO_DIR"
LOG="$POP_DIR/population_scale_matsim_validation.log"
exec > >(tee -a "$LOG") 2>&1

timestamp() {
  date --iso-8601=seconds
}

step() {
  echo
  echo "[$(timestamp)] $*"
}

run_matsim() {
  local scenario="$1"
  local config="$SCENARIO_DIR/config_${scenario}.xml"
  local log_file="$SCENARIO_DIR/matsim_${scenario}.log"
  step "Building MATSim runner for ${scenario}"
  /usr/bin/time -f "build elapsed: %E" mvn -f 06_matsim/pom.xml -DskipTests package
  step "Starting MATSim ${scenario}: ${config}"
  export MAVEN_OPTS="-Xmx${MATSIM_HEAP} -XX:+UseG1GC"
  echo "MAVEN_OPTS=$MAVEN_OPTS"
  /usr/bin/time -f "matsim ${scenario} elapsed: %E" mvn -f 06_matsim/pom.xml exec:java \
    -Dexec.args="$config" \
    2>&1 | tee "$log_file"
  step "Finished MATSim ${scenario}"
}

step "Population-scale synthetic survey sampling"
.venv/bin/python 07_synthesize-population-scale.py \
  --output-csv "$POP_DIR/synthetic_population_trips.csv" \
  --sample-batch-size "$SAMPLE_BATCH_SIZE"

step "Population-scale marginal comparison: synthetic survey vs real survey duplicated by wthhfin"
.venv/bin/python 07_compare-population-scale.py \
  --synthetic-csv "$POP_DIR/synthetic_population_trips.csv" \
  --out-dir "$POP_DIR/comparison"

step "Preparing population-scale vehicle-trip demand for MATSim"
.venv/bin/python 06_prepare-stage1-demand.py \
  --real-csv 03x_filled-survey.csv \
  --synthetic-csv "$POP_DIR/synthetic_population_trips.csv" \
  --output-dir "$SCENARIO_DIR" \
  --synthetic-scale 1 \
  --real-scale 1 \
  --taz-centroids "$CENTROIDS" \
  --state-filter "$STATE_FILTER" \
  --state-filter-mode "$STATE_FILTER_MODE"

step "Checking population-scale MATSim safety"
if [[ "$ALLOW_FULL_POP_MATSIM" != "1" ]]; then
  echo "Refusing to launch full population-scale MATSim by default."
  echo "This run expands to roughly 3-5 million agents per scenario on an 8.7M-link network."
  echo "The previous 218k-agent runs already reached about 62-67 GB Java heap."
  echo "Set ALLOW_FULL_POP_MATSIM=1 only if you intentionally want to risk an overnight OOM."
  exit 2
fi

step "Writing MATSim populations"
.venv/bin/python 06_make-matsim-population.py \
  --vehicle-trips "$SCENARIO_DIR/real_vehicle_trips.csv" \
  --taz-centroids "$CENTROIDS" \
  --out "$SCENARIO_DIR/real_population.xml.gz"
.venv/bin/python 06_make-matsim-population.py \
  --vehicle-trips "$SCENARIO_DIR/synthetic_vehicle_trips.csv" \
  --taz-centroids "$CENTROIDS" \
  --out "$SCENARIO_DIR/synthetic_population.xml.gz"

step "Writing MATSim configs"
.venv/bin/python 06_write-matsim-configs.py \
  --stage1-dir "$NETWORK_DIR" \
  --scenario-dir "$SCENARIO_DIR" \
  --threads "$THREADS"

run_matsim real
run_matsim synthetic

step "Extracting MATSim link volumes"
.venv/bin/python 06_extract-matsim-link-volumes.py \
  --events "$SCENARIO_DIR/matsim_real/output_events.xml.gz" \
  --out "$SCENARIO_DIR/real_matsim_link_volumes.csv"
.venv/bin/python 06_extract-matsim-link-volumes.py \
  --events "$SCENARIO_DIR/matsim_synthetic/output_events.xml.gz" \
  --out "$SCENARIO_DIR/synthetic_matsim_link_volumes.csv"

step "Aggregating link volumes to observed count links"
.venv/bin/python 06_aggregate-matsim-volumes.py \
  --link-volumes "$SCENARIO_DIR/real_matsim_link_volumes.csv" \
  --link-map "$NETWORK_DIR/matsim_link_count_map.csv" \
  --out "$SCENARIO_DIR/real_assigned_link_counts.csv"
.venv/bin/python 06_aggregate-matsim-volumes.py \
  --link-volumes "$SCENARIO_DIR/synthetic_matsim_link_volumes.csv" \
  --link-map "$NETWORK_DIR/matsim_link_count_map.csv" \
  --out "$SCENARIO_DIR/synthetic_assigned_link_counts.csv"

step "Comparing assigned volumes against observed counts"
.venv/bin/python 06_compare-stage1-counts.py \
  --scenario-totals "$SCENARIO_DIR/scenario_totals.csv" \
  --observed-counts "$NETWORK_DIR/observed_counts.csv" \
  --real-assigned "$SCENARIO_DIR/real_assigned_link_counts.csv" \
  --synthetic-assigned "$SCENARIO_DIR/synthetic_assigned_link_counts.csv" \
  --out "$SCENARIO_DIR/count_validation.csv" \
  --summary-out "$SCENARIO_DIR/count_validation_summary.csv"

step "Building comparison charts and report bundle"
.venv/bin/python 06_visualize-stage1-results.py \
  --scenario-dir "$SCENARIO_DIR" \
  --network-dir "$NETWORK_DIR" \
  --out-dir "$SCENARIO_DIR/stage1_comparison"

step "DONE"
echo "Key outputs:"
echo "  $SCENARIO_DIR/count_validation_summary.csv"
echo "  $SCENARIO_DIR/count_validation.csv"
echo "  $SCENARIO_DIR/stage1_comparison/"
echo "  $POP_DIR/comparison/"

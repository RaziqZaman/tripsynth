#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-08x_county_runs}"
POP_DIR="${POP_DIR:-07x_population_scale}"
SYNTHETIC_POP="${SYNTHETIC_POP:-$POP_DIR/synthetic_population_trips.csv}"
TAZ_CENTROIDS="${TAZ_CENTROIDS:-06x_stage1/taz_centroids.csv}"
NETWORK_LINKS="${NETWORK_LINKS:-06x_stage1_osm/network_links.csv}"
OBSERVED_COUNTS="${OBSERVED_COUNTS:-06x_stage1_osm/observed_counts.csv}"
COUNTIES="${COUNTIES:-}"
BBOX_BUFFER_DEG="${BBOX_BUFFER_DEG:-0.15}"
DEMAND_MODE="${DEMAND_MODE:-both}"
MATSIM_HEAP="${MATSIM_HEAP:-96g}"
THREADS="${THREADS:-8}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8192}"
MAX_COUNTIES="${MAX_COUNTIES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$OUT_ROOT" "$POP_DIR"
LOG="$OUT_ROOT/county_matsim_validation.log"
exec > >(tee -a "$LOG") 2>&1

timestamp() { date --iso-8601=seconds; }
step() { echo; echo "[$(timestamp)] $*"; }

run_matsim() {
  local county_dir="$1"
  local scenario="$2"
  local config="$county_dir/config_${scenario}.xml"
  local log_file="$county_dir/matsim_${scenario}.log"
  step "Starting MATSim ${scenario}: ${county_dir}"
  export MAVEN_OPTS="-Xmx${MATSIM_HEAP} -XX:+UseG1GC"
  echo "MAVEN_OPTS=$MAVEN_OPTS"
  /usr/bin/time -f "matsim ${scenario} elapsed: %E" mvn -f 06_matsim/pom.xml exec:java \
    -Dexec.args="$config" \
    2>&1 | tee "$log_file"
  step "Finished MATSim ${scenario}: ${county_dir}"
}

if [[ ! -s "$SYNTHETIC_POP" ]]; then
  step "Population-scale synthetic survey is missing; generating $SYNTHETIC_POP"
  .venv/bin/python 07_synthesize-population-scale.py \
    --output-csv "$SYNTHETIC_POP" \
    --sample-batch-size "$SAMPLE_BATCH_SIZE"
else
  step "Using existing population-scale synthetic survey: $SYNTHETIC_POP"
fi

step "Preparing county scenarios"
.venv/bin/python 08_prepare-county-runs.py \
  --synthetic-csv "$SYNTHETIC_POP" \
  --taz-centroids "$TAZ_CENTROIDS" \
  --network-links "$NETWORK_LINKS" \
  --observed-counts "$OBSERVED_COUNTS" \
  --out-root "$OUT_ROOT" \
  --counties "$COUNTIES" \
  --bbox-buffer-deg "$BBOX_BUFFER_DEG" \
  --demand-mode "$DEMAND_MODE"

step "Building MATSim runner once"
/usr/bin/time -f "build elapsed: %E" mvn -f 06_matsim/pom.xml -DskipTests package

count=0
while IFS= read -r slug; do
  [[ -z "$slug" ]] && continue
  county_dir="$OUT_ROOT/$slug"
  if [[ "$MAX_COUNTIES" != "0" && "$count" -ge "$MAX_COUNTIES" ]]; then
    step "MAX_COUNTIES=$MAX_COUNTIES reached; stopping loop"
    break
  fi
  count=$((count + 1))

  if [[ "$SKIP_EXISTING" == "1" && -s "$county_dir/count_validation_summary.csv" ]]; then
    step "Skipping completed county $slug"
    continue
  fi

  step "Writing MATSim populations for $slug"
  .venv/bin/python 06_make-matsim-population.py \
    --vehicle-trips "$county_dir/real_vehicle_trips.csv" \
    --taz-centroids "$TAZ_CENTROIDS" \
    --out "$county_dir/real_population.xml.gz"
  .venv/bin/python 06_make-matsim-population.py \
    --vehicle-trips "$county_dir/synthetic_vehicle_trips.csv" \
    --taz-centroids "$TAZ_CENTROIDS" \
    --out "$county_dir/synthetic_population.xml.gz"

  step "Writing MATSim configs for $slug"
  .venv/bin/python 06_write-matsim-configs.py \
    --stage1-dir "$county_dir" \
    --scenario-dir "$county_dir" \
    --threads "$THREADS"

  run_matsim "$county_dir" real
  run_matsim "$county_dir" synthetic

  step "Extracting and aggregating link volumes for $slug"
  .venv/bin/python 06_extract-matsim-link-volumes.py \
    --events "$county_dir/matsim_real/output_events.xml.gz" \
    --out "$county_dir/real_matsim_link_volumes.csv"
  .venv/bin/python 06_extract-matsim-link-volumes.py \
    --events "$county_dir/matsim_synthetic/output_events.xml.gz" \
    --out "$county_dir/synthetic_matsim_link_volumes.csv"
  .venv/bin/python 06_aggregate-matsim-volumes.py \
    --link-volumes "$county_dir/real_matsim_link_volumes.csv" \
    --link-map "$county_dir/matsim_link_count_map.csv" \
    --out "$county_dir/real_assigned_link_counts.csv"
  .venv/bin/python 06_aggregate-matsim-volumes.py \
    --link-volumes "$county_dir/synthetic_matsim_link_volumes.csv" \
    --link-map "$county_dir/matsim_link_count_map.csv" \
    --out "$county_dir/synthetic_assigned_link_counts.csv"

  step "Comparing counts for $slug"
  .venv/bin/python 06_compare-stage1-counts.py \
    --scenario-totals "$county_dir/scenario_totals.csv" \
    --observed-counts "$county_dir/observed_counts.csv" \
    --real-assigned "$county_dir/real_assigned_link_counts.csv" \
    --synthetic-assigned "$county_dir/synthetic_assigned_link_counts.csv" \
    --out "$county_dir/count_validation.csv" \
    --summary-out "$county_dir/count_validation_summary.csv"

  step "Rendering report for $slug"
  .venv/bin/python 06_visualize-stage1-results.py \
    --scenario-dir "$county_dir" \
    --network-dir "$county_dir" \
    --out-dir "$county_dir/stage1_comparison"
done < "$OUT_ROOT/county_run_list.txt"

step "Collecting county summaries"
.venv/bin/python 08_collect-county-results.py --out-root "$OUT_ROOT"

step "DONE"
echo "County rollup: $OUT_ROOT/county_validation_summary.csv"
echo "Log: $LOG"

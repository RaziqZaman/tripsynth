# Stage 1 Travel Simulation Comparison

This stage compares network loading from the real filled survey against the
synthetic trips. It is intentionally trip-based: each eligible row becomes a
vehicle-trip contribution, not a full household/person daily activity chain.

## 1. Prepare Vehicle-Trip Demand

```bash
.venv/bin/python 06_prepare-stage1-demand.py --synthetic-scale 12
```

Outputs:

- `06x_stage1/real_vehicle_trips.csv`
- `06x_stage1/synthetic_vehicle_trips.csv`
- `06x_stage1/scenario_totals.csv`

By default, every row with positive `vehicle_occupancy` is included. Each real
row contributes:

```text
wthhfin / vehicle_occupancy * real_scale
```

Each synthetic row contributes:

```text
12 / vehicle_occupancy
```

`real_scale` is chosen so the real and synthetic scenarios have the same total
scaled vehicle demand.

If we decide only certain `travel_mode` codes should load onto the road network,
use:

```bash
.venv/bin/python 06_prepare-stage1-demand.py --synthetic-scale 12 --road-modes 4
```

## 2. Fetch TAZ Centroids

Fetch TPB TAZ centroids from the public MWCOG ArcGIS layer:

```bash
.venv/bin/python 06_fetch-taz-centroids.py
```

Output:

- `06x_stage1/taz_centroids.csv`

By default this uses `outSR=4326`, so `x/y` are `lon/lat`. The MATSim network
must use the same coordinate system.

## 3. Convert To MATSim Population

The prepared demand is still TAZ-to-TAZ. Once `taz_centroids.csv` exists, run:

```bash
.venv/bin/python 06_make-matsim-population.py \
  --vehicle-trips 06x_stage1/real_vehicle_trips.csv \
  --taz-centroids 06x_stage1/taz_centroids.csv \
  --out 06x_stage1/real_population.xml.gz

.venv/bin/python 06_make-matsim-population.py \
  --vehicle-trips 06x_stage1/synthetic_vehicle_trips.csv \
  --taz-centroids 06x_stage1/taz_centroids.csv \
  --out 06x_stage1/synthetic_population.xml.gz
```

The exporter expands fractional `vehicle_weight` values stochastically. This
keeps the MATSim inputs as ordinary unweighted car agents.

## 4. Observed Counts

`06_fetch-observed-counts.py` downloads raw count attributes from an ArcGIS-style
query endpoint. Its default endpoint is the MDOT SHA annual average daily
traffic point layer.

```bash
.venv/bin/python 06_fetch-observed-counts.py \
  --out 06x_stage1/mdot_aadt_points_raw.csv \
  --out-sr 4326

.venv/bin/python 06_prepare-observed-counts.py
```

Outputs:

- `06x_stage1/mdot_aadt_points_raw.csv`
- `06x_stage1/observed_counts_points.csv`

`observed_counts_points.csv` contains `count_id,observed_volume,lon,lat` plus
route/county metadata.

## 5. Match Counts To Network Links

After preparing a road network links CSV, map count points to nearest links:

```bash
.venv/bin/python 06_match-counts-to-links.py \
  --links 06x_stage1/network_links.csv \
  --max-distance-m 100
```

The expected network links CSV columns are:

```text
link_id,from_lon,from_lat,to_lon,to_lat
```

Output:

- `06x_stage1/observed_counts.csv`

with:

```text
link_id,observed_volume,count_id,match_distance_m,...
```

## 6. Compare Assigned Volumes

After running MATSim or another assignment engine, export assigned link volumes
with:

```text
link_id,volume
```

Then compare:

```bash
.venv/bin/python 06_compare-stage1-counts.py \
  --observed-counts 06x_stage1/observed_counts.csv \
  --real-assigned 06x_stage1/real_assigned_link_counts.csv \
  --synthetic-assigned 06x_stage1/synthetic_assigned_link_counts.csv
```

Outputs:

- `06x_stage1/count_validation.csv`
- `06x_stage1/count_validation_summary.csv`

The comparison script automatically scales observed counts by `real_scale` from
`scenario_totals.csv`, so observed counts and simulated volumes are in the same
reduced-demand universe.

## Current Runnable Network Path

For the first Maryland traffic-count validation pass, use the MDOT AADT line
layer as the count-bearing network:

```bash
.venv/bin/python 06_fetch-aadt-line-network.py --page-size 1000
.venv/bin/python 06_make-matsim-network.py
```

Outputs:

- `06x_stage1/network_links.csv`
- `06x_stage1/observed_counts.csv`
- `06x_stage1/network.xml.gz`
- `06x_stage1/matsim_link_count_map.csv`

This network is Maryland-only, so use the Maryland-internal demand set for the
first run:

```bash
.venv/bin/python 06_prepare-stage1-demand.py \
  --synthetic-scale 12 \
  --taz-centroids 06x_stage1/taz_centroids.csv \
  --state-filter Maryland \
  --state-filter-mode both \
  --output-dir 06x_stage1_md_internal

.venv/bin/python 06_make-matsim-population.py \
  --vehicle-trips 06x_stage1_md_internal/real_vehicle_trips.csv \
  --taz-centroids 06x_stage1/taz_centroids.csv \
  --out 06x_stage1_md_internal/real_population.xml.gz

.venv/bin/python 06_make-matsim-population.py \
  --vehicle-trips 06x_stage1_md_internal/synthetic_vehicle_trips.csv \
  --taz-centroids 06x_stage1/taz_centroids.csv \
  --out 06x_stage1_md_internal/synthetic_population.xml.gz
```

The MD-internal scenario keeps `synthetic_scale = 12` and rescales the real
survey demand to the same total simulated vehicle demand.

After MATSim or another assignment engine writes directed link volumes, aggregate
them back to count-link IDs before comparison:

```bash
.venv/bin/python 06_aggregate-matsim-volumes.py \
  --link-volumes 06x_stage1_md_internal/real_matsim_link_volumes.csv \
  --out 06x_stage1_md_internal/real_assigned_link_counts.csv

.venv/bin/python 06_aggregate-matsim-volumes.py \
  --link-volumes 06x_stage1_md_internal/synthetic_matsim_link_volumes.csv \
  --out 06x_stage1_md_internal/synthetic_assigned_link_counts.csv
```

Then compare against MDOT AADT counts:

```bash
.venv/bin/python 06_compare-stage1-counts.py \
  --scenario-totals 06x_stage1_md_internal/scenario_totals.csv \
  --observed-counts 06x_stage1/observed_counts.csv \
  --real-assigned 06x_stage1_md_internal/real_assigned_link_counts.csv \
  --synthetic-assigned 06x_stage1_md_internal/synthetic_assigned_link_counts.csv \
  --out 06x_stage1_md_internal/count_validation.csv \
  --summary-out 06x_stage1_md_internal/count_validation_summary.csv
```

## MATSim Runner

This repo now has a tiny Maven runner under `06_matsim/`. It uses the standard
MATSim Controler entry point and the `org.matsim:matsim:2025.0` dependency.
MATSim itself documents command-line runs through `org.matsim.run.Controler`, and
the official install docs recommend Maven-based project setup.

This machine currently needs Java and Maven installed first:

```bash
./06_matsim_prereqs.sh
```

Generate/update configs whenever paths or iteration settings change:

```bash
.venv/bin/python 06_write-matsim-configs.py
```

Run the two Maryland-internal Stage 1 scenarios. These scripts print timestamps,
MATSim logs, and elapsed runtime. MATSim itself does not provide a reliable
shell-level ETA for queue simulation progress, so these are the commands to put
in tmux if you want them to keep running in the background:

```bash
tmux new -s matsim-real './06_run_matsim_real.sh'
tmux new -s matsim-synth './06_run_matsim_synthetic.sh'
```

After both runs finish, extract link-entry volumes, aggregate directed MATSim
links back to MDOT AADT count links, and compare against scaled observed counts.
The event extraction step shows byte-based progress bars with ETA for the MATSim
events files:

```bash
./06_postprocess_matsim.sh
```

Expected comparison outputs:

- `06x_stage1_md_internal/count_validation.csv`
- `06x_stage1_md_internal/count_validation_summary.csv`

The default configs run only iteration `0`, which is a first network-loading
comparison rather than a calibrated multi-iteration MATSim study.

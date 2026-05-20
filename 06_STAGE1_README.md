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

The MDOT AADT line layer is useful as an observed-count source, but it is not a
fully routable road network. The runnable MATSim path now uses a Maryland OSM
road network from Geofabrik, cleans it to the largest connected drivable
component, and maps MDOT AADT count points onto the OSM links.

Prepare the OSM network, matched observed counts, and MATSim configs with:

```bash
./06_prepare-osm-stage1.sh
```

The wrapper runs these steps with progress bars:

```bash
.venv/bin/python 06_download-osm-network.py
.venv/bin/python 06_make-osm-matsim-network.py
.venv/bin/python 06_match-counts-to-links.py   --links 06x_stage1_osm/network_links.csv   --out 06x_stage1_osm/observed_counts.csv   --max-distance-m 100
.venv/bin/python 06_write-matsim-configs.py   --stage1-dir 06x_stage1_osm   --scenario-dir 06x_stage1_md_internal
```

Current OSM outputs:

- `06x_stage1_osm/maryland-latest.osm.pbf`
- `06x_stage1_osm/network.xml.gz`
- `06x_stage1_osm/network_links.csv`
- `06x_stage1_osm/matsim_link_count_map.csv`
- `06x_stage1_osm/observed_counts.csv`
- `06x_stage1_osm/network_stats.json`

The current build matched 8,762 of 8,773 MDOT AADT count points within 100 m.
The generated configs in `06x_stage1_md_internal/` point at
`06x_stage1_osm/network.xml.gz`.

The Maryland-internal demand set is still the right first run, because the count
validation target is Maryland AADT:

```bash
.venv/bin/python 06_prepare-stage1-demand.py   --synthetic-scale 12   --taz-centroids 06x_stage1/taz_centroids.csv   --state-filter Maryland   --state-filter-mode both   --output-dir 06x_stage1_md_internal

.venv/bin/python 06_make-matsim-population.py   --vehicle-trips 06x_stage1_md_internal/real_vehicle_trips.csv   --taz-centroids 06x_stage1/taz_centroids.csv   --out 06x_stage1_md_internal/real_population.xml.gz

.venv/bin/python 06_make-matsim-population.py   --vehicle-trips 06x_stage1_md_internal/synthetic_vehicle_trips.csv   --taz-centroids 06x_stage1/taz_centroids.csv   --out 06x_stage1_md_internal/synthetic_population.xml.gz
```

After MATSim writes directed link volumes, `06_postprocess_matsim.sh` now uses
`NETWORK_DIR=06x_stage1_osm` by default, so it aggregates simulated volumes back
to the OSM links that carry matched MDOT counts.

## MATSim Runner

This repo now has a tiny Maven runner under `06_matsim/`. It uses the standard
MATSim Controler entry point and the `org.matsim:matsim:15.0` dependency.
MATSim itself documents command-line runs through `org.matsim.run.Controler`, and
the official install docs recommend Maven-based project setup.

This machine currently uses Java 17, Maven, and MATSim 15.0. If Java or Maven is missing, install them first:

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

## Stage 1 Comparison Bundle

Build failure-aware synthetic-vs-real summaries and charts with:

```bash
.venv/bin/python 06_visualize-stage1-results.py
```

Outputs are written to:

- `06x_stage1_md_internal/stage1_comparison/*.csv`
- `06x_stage1_md_internal/stage1_comparison/charts/*.png`

If MATSim has not completed successfully, the comparison bundle still writes run
status diagnostics and weighted-demand overlap metrics, while count-validation
outputs are marked unavailable. The MATSim run scripts now default to
`MATSIM_HEAP=96g` via `MAVEN_OPTS`; override with, for example,
`MATSIM_HEAP=64g ./06_run_matsim_real.sh`.

The default configs run only iteration `0`, which is a first network-loading
comparison rather than a calibrated multi-iteration MATSim study.

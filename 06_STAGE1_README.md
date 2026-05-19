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

## 2. Convert To MATSim Population

The prepared demand is still TAZ-to-TAZ. To create MATSim plans, provide a TAZ
centroid CSV with at least:

```text
taz,x,y
```

Then run, for example:

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

## 3. Observed Counts

`06_fetch-observed-counts.py` can download raw count attributes from an
ArcGIS-style query endpoint. Its default endpoint is the MDOT SHA annual average
daily traffic layer:

```bash
.venv/bin/python 06_fetch-observed-counts.py \
  --out 06x_stage1/observed_counts_raw.csv
```

The raw count points still need to be map-matched to the simulation network.
After map matching, create:

```text
06x_stage1/observed_counts.csv
```

with:

```text
link_id,observed_volume
```

## 4. Compare Assigned Volumes

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

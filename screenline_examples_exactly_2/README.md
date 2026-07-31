# Screenline examples

This folder contains **120 reproducibly sampled OD census-tract maps** from the frozen
`paper_wctr_1m_vae_sweep` run. The random seed is `20260801` and the eligible observed-survey
universe contains 2,738 unique directed OD pairs.

Every sampled pair satisfies all three requested constraints:

1. exactly two unique >1 m shared tract-boundary screenlines;
2. at least one mapped MDOT AADT station on each screenline; and
3. two or more mapped MDOT AADT stations on at least one of the two screenlines.

Pairs from the prior manifest, including their reverse directions, were excluded so this is a
visually distinct additional batch.

The figure and map background is exactly `#FFFFFF`.

## Reading each map

- Numbered colored lines are shared census-tract boundaries, ordered from origin to destination.
- A colored MDOT circle is assigned to the same-color boundary by the run's geometric station rule.
- A triangle identifies an FHWA/TMAS station. When it is mapped, its outline matches the boundary color.
- Gray station symbols are in the displayed map extent but are not used by that OD path.
- Multiple colored symbols joined to one location preserve legitimate multi-boundary station assignments.
- The dashed O–D line connects tract representative points. It is a geometric screenline proxy, not a
  roadway route or network assignment.

The maps reconstruct every >1 m shared tract boundary along that proxy from
`tract_adjacency.parquet`, including boundaries omitted by the validation candidate screenline cache.

## Files

- `screenline_*.png`: the 120 requested maps.
- `manifest.csv`: one row per sampled OD pair, with map and station counts and projected extent.
- `screenlines.csv`: every displayed boundary, its exact color, order, and mapped-station summary.
- `station_assignments.csv`: every colored station–boundary link used in the maps.
- `contact_sheets/`: four visual indexes for quick review.
- `generation_summary.json`: source paths, seed, CRS, and QA totals.

Generated with `scripts/generate_screenline_examples.py`. All coordinates and distance fields are in
EPSG:26985 (NAD83 / Maryland, meters). The generation summary reports
240 displayed boundaries and 710
colored station–boundary links across the map set.

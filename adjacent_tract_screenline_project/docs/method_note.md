# Method note: adjacent tract-pair daily screenline validation

This test converts dense AADT count locations into small spatial validation units.

1. Start with MDOT SHA AADT count locations.
2. Keep mainline locations in Montgomery County and Prince George's County that are plausible for DC-Maryland, PG-DC, Montgomery-DC, or PG-Montgomery validation.
3. Download Census TIGER/Line 2019 tract polygons for Maryland and DC.
4. Assign each count location to its containing tract.
5. Find the nearest adjacent tract sharing a boundary with that tract.
6. Keep only stations close to the shared tract boundary.
7. Treat each accepted station as a station-level tract-pair screenline.

This is more localized than a county-cordon screenline and is much more appropriate for the dense AADT station layer. It still validates daily/annualized volume, not hourly profiles.

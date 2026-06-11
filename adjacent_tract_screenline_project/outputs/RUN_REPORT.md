# Adjacent Census Tract AADT Station Screenline Report

Station candidates after filters: 1,023

Accepted tract-pair screenlines: 771

Rejected/ambiguous candidates: 252


## Screenline summary

| screenline_family         | assignment_quality   |   n_station_screenlines |
|:--------------------------|:---------------------|------------------------:|
| montgomery_dc             | high                 |                       2 |
| montgomery_dc             | low_review           |                       6 |
| montgomery_dc             | medium               |                       1 |
| other_adjacent_tract_pair | high                 |                     485 |
| other_adjacent_tract_pair | low_review           |                     144 |
| other_adjacent_tract_pair | medium               |                     121 |
| pg_dc                     | low_review           |                       4 |
| pg_montgomery             | high                 |                       4 |
| pg_montgomery             | low_review           |                       2 |
| pg_montgomery             | medium               |                       2 |


## Method

Each MDOT AADT count location is assigned to its containing Census tract. A station is accepted as a tract-pair screenline only if it lies within the configured distance of the containing tract boundary and has a nearby adjacent tract sharing that boundary. This makes the output a station-level tract-boundary screenline inventory, not an OD matrix.
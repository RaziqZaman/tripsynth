#!/usr/bin/env python3
"""Compare survey/model trip records against tract-pair screenline AADT.

This is a conservative direct-adjacent-OD comparison: it only counts trips whose
origin tract and destination tract are exactly the two tracts attached to a
station screenline. That is stricter than a true boundary-crossing assignment
and will undercount through trips that cross the boundary but start/end farther
away.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np


def geoid_series(county, tract):
    county = pd.to_numeric(county, errors="coerce")
    tract = pd.to_numeric(tract, errors="coerce")
    out = pd.Series(pd.NA, index=county.index, dtype="object")
    ok = county.notna() & tract.notna()
    out.loc[ok] = county.loc[ok].astype(int).astype(str).str.zfill(5) + tract.loc[ok].astype(int).astype(str).str.zfill(6)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--survey-csv', required=True, help='Survey/synthetic long trip file with o/d county+tract fields.')
    ap.add_argument('--screenlines-csv', default='outputs/tract_pair_aadt_station_screenlines.csv')
    ap.add_argument('--config', default='config.json')
    ap.add_argument('--out', default='outputs/tract_pair_direct_adjacent_od_comparison.csv')
    args=ap.parse_args()
    cfg=json.load(open(args.config))
    trips=pd.read_csv(args.survey_csv)
    sl=pd.read_csv(args.screenlines_csv, dtype=str)
    # Build tract GEOIDs. Input already has o_state_county_fips + o_tract_fips style.
    trips['o_geoid']=geoid_series(trips['o_state_county_fips'], trips['o_tract_fips'])
    trips['d_geoid']=geoid_series(trips['d_state_county_fips'], trips['d_tract_fips'])
    # Optional auto mode filter.
    auto_modes=cfg.get('auto_mode_codes', [])
    if auto_modes:
        trips=trips[pd.to_numeric(trips['travel_mode'], errors='coerce').isin(auto_modes)].copy()
    # Weights.
    wt=pd.to_numeric(trips.get('wttrdfin', 1.0), errors='coerce').fillna(0)
    occ=pd.to_numeric(trips.get('vehicle_occupancy', 1.0), errors='coerce').replace(0,np.nan).fillna(1)
    if cfg.get('vehicle_weight_mode') == 'auto_driver_only':
        trips['vehicle_weight']=wt
    else:
        trips['vehicle_weight']=wt/occ
    dep=pd.to_numeric(trips.get('departure_time_hhmm', np.nan), errors='coerce')
    trips['departure_hour']=(dep//100).astype('Int64')

    # Aggregate direct adjacent OD, both directions.
    results=[]
    for _,r in sl.iterrows():
        a=str(r['tract_a_geoid'])
        b=str(r['tract_b_geoid'])
        m1=(trips['o_geoid'].astype(str).eq(a)&trips['d_geoid'].astype(str).eq(b))
        m2=(trips['o_geoid'].astype(str).eq(b)&trips['d_geoid'].astype(str).eq(a))
        tab=trips[m1|m2]
        flow=float(tab['vehicle_weight'].sum())
        a_to_b=float(tab[m1]['vehicle_weight'].sum())
        b_to_a=float(tab[m2]['vehicle_weight'].sum())
        out=r.to_dict()
        out.update({
            'model_direct_adjacent_od_vehicle_flow_daily': flow,
            'model_direct_a_to_b_vehicle_flow_daily': a_to_b,
            'model_direct_b_to_a_vehicle_flow_daily': b_to_a,
            'survey_records_direct_adjacent_od': int(len(tab)),
            'comparison_warning': 'direct adjacent OD only; undercounts through trips crossing the station tract boundary'
        })
        for obs in ['AADT_2017','AADT_2018','AAWDT_2017','AAWDT_2018']:
            val=pd.to_numeric(r.get(obs, np.nan), errors='coerce')
            if pd.notna(val) and val != 0:
                out[f'pct_error_vs_{obs}']=(flow-val)/val
        results.append(out)
    res=pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out,index=False)
    print(f'Wrote {args.out} ({len(res):,} rows)')

if __name__ == '__main__':
    main()

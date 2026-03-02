#!/usr/bin/env python3
"""Untuple synthesized household rows back into trip rows.

Input: synthesized-household-trips.parquet
Outputs:
- synthetic_trips.parquet
- sample_synthetic_trips.csv (all trips from 5040 randomly selected households)
"""

from __future__ import annotations

import argparse
import ast
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tabular_io import StringRowWriter, count_rows, get_fieldnames, iter_rows
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


HOUSEHOLD_IN_COLS = [
    "area_type",
    "food_delivered",
    "hh_income_detailed",
    "hhsize",
    "home_bmc_taz",
    "home_ownership",
    "home_fips",
    "home_state_puma",
    "home_tpb_taz",
    "home_type",
    "numbicycles",
    "numvehicles",
    "numworkers",
    "package_delivered",
    "service_delivered",
    "tdate_string",
    "wthhfin",
]

PERSON_IN_BASE_COLS = [
    "age",
    "disability",
    "employment_status",
    "gender",
    "job_count",
    "license",
    "person_tripcount",
    "race_ethnicity_hispanic",
    "race_ethnicity",
    "smartphone",
    "student_status",
    "td_shoptime",
    "td_telecommute_time",
    "volunteer_status",
    "walk_bike_loop_trips",
    "work_bmc_taz",
    "work_fips",
    "work_tpb_taz",
    "school_bmc_taz",
    "school_fips",
    "school_mode",
    "school_tpb_taz",
    "school_type",
]

TRIP_CHAIN_BASE_COLS = [
    "travel_mode",
    "travelers_hh",
    "travelers_nonhh",
    "vehicle_occupancy",
    "subway_access_mode",
    "subway_egress_mode",
    "subway_pay",
    "subway_station_board",
    "subway_station_exit",
    "subway_used",
    "toll_road_used",
    "transit_access_mode",
    "transit_egress_mode",
    "hov_used",
    "park_loc",
    "park_pay",
    "reported_travel_time",
    "vehicle",
]


@lru_cache(maxsize=200_000)
def split_fips11(v: str) -> Tuple[str, str]:
    s = (v or "").strip()
    if not s:
        return "", ""
    s = "".join(ch for ch in s if ch.isdigit())
    if len(s) < 11:
        s = s.zfill(11)
    elif len(s) > 11:
        s = s[:11]
    return s[:5], s[5:11]


@lru_cache(maxsize=500_000)
def parse_tuple_like(raw: str) -> Tuple[object, ...]:
    s = (raw or "").strip()
    if not s:
        return ()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, tuple):
            return val
        if isinstance(val, list):
            return tuple(val)
        return (val,)
    except Exception:
        pass

    # Fallback for malformed tuple strings.
    if "," in s:
        return tuple(part.strip().strip("\"'") for part in s.split(","))
    return (s,)


def as_text(v: object) -> str:
    if v is None:
        return ""
    return str(v)


def parse_minutes_value(v: object) -> Optional[int]:
    s = as_text(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    if x < 0:
        return None
    return int(round(x))


def minutes_to_hhmm(total_minutes: Optional[int]) -> str:
    if total_minutes is None:
        return ""
    m = int(total_minutes) % (24 * 60)
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"


def detect_p_slots(fieldnames: Sequence[str]) -> List[int]:
    out = []
    for n in range(1, 9):
        pfx = f"P{n}-"
        if any(c.startswith(pfx) for c in fieldnames):
            out.append(n)
    return out


def detect_j1_cols(fieldnames: Sequence[str]) -> List[str]:
    # derive from P1-* headers (same set exists across Pn)
    out = []
    pfx = "P1-j1_"
    for c in fieldnames:
        if c.startswith(pfx):
            out.append(c[len("P1-") :])
    return sorted(set(out))


def detect_subway_cols(fieldnames: Sequence[str]) -> List[str]:
    out = []
    pfx = "P1-subway_"
    for c in fieldnames:
        if c.startswith(pfx):
            out.append(c[len("P1-") :])
    return sorted(set(out))


def detect_od_cols(fieldnames: Sequence[str]) -> List[str]:
    out = []
    pfx = "P1-od_"
    for c in fieldnames:
        if c.startswith(pfx):
            out.append(c[len("P1-") :])
    return sorted(set(out))


def trip_count_from_person(
    row: Dict[str, object], pfx: str, trip_cols: Sequence[str], od_cols: Sequence[str]
) -> int:
    lengths: List[int] = []

    for c in trip_cols:
        vals = parse_tuple_like(as_text(row.get(pfx + c, "")))
        if vals:
            lengths.append(len(vals))

    for c in od_cols:
        vals = parse_tuple_like(as_text(row.get(pfx + c, "")))
        if vals:
            lengths.append(max(0, len(vals) - 1))

    return max(lengths) if lengths else 0


def get_seq_value(seq: Sequence[object], idx: int) -> str:
    if idx < 0 or idx >= len(seq):
        return ""
    return as_text(seq[idx]).strip()


def parse_vehicle_item(v: object) -> Tuple[str, str, str, str, str, str]:
    if isinstance(v, (list, tuple)):
        arr = list(v)
    else:
        s = as_text(v).strip()
        arr = parse_tuple_like(s) if s else []

    arr = [as_text(x).strip() for x in arr]
    if len(arr) < 6:
        arr.extend([""] * (6 - len(arr)))
    return arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]


def main() -> int:
    ap = argparse.ArgumentParser(description="Untuple synthesized-household-trips.parquet")
    ap.add_argument("--input", type=Path, default=Path("synthesized-household-trips.parquet"))
    ap.add_argument("--output", type=Path, default=Path("synthetic_trips.parquet"))
    ap.add_argument("--sample-output", type=Path, default=Path("sample_synthetic_trips.csv"))
    ap.add_argument("--sample-households", type=int, default=5040)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input file: {args.input}")

    total_households = count_rows(args.input)
    if total_households <= 0:
        raise ValueError("Input table has no rows.")

    sample_n = min(args.sample_households, total_households)
    rng = random.Random(args.seed)
    sampled_households = set(rng.sample(range(1, total_households + 1), sample_n))

    fieldnames = get_fieldnames(args.input)
    if not fieldnames:
        raise ValueError("Input table has no header row.")
    reader = iter_rows(args.input)

    p_slots = detect_p_slots(fieldnames)
    j1_cols = detect_j1_cols(fieldnames)
    subway_cols = detect_subway_cols(fieldnames)
    od_cols = detect_od_cols(fieldnames)

    trip_chain_cols = [
        "travel_mode",
        "travelers_hh",
        "travelers_nonhh",
        "vehicle_occupancy",
        *subway_cols,
        "toll_road_used",
        "transit_access_mode",
        "transit_egress_mode",
        "hov_used",
        "park_loc",
        "park_pay",
        "reported_travel_time",
        "vehicle",
        "departure_time_in_minutes_after_arrival",
    ]
    trip_chain_cols = [c for c in trip_chain_cols if f"P1-{c}" in fieldnames]

    out_cols = [
        "household_id",
        "person_id",
        "tripid",
        "persno",
        "tripno",
        "area_type",
        "food_delivered",
        "hh_income_detailed",
        "hhsize",
        "home_bmc_taz",
        "home_ownership",
        "home_state_puma",
        "home_tpb_taz",
        "home_type",
        "numbicycle",
        "numvehicle",
        "numworkers",
        "package_delivered",
        "service_delivered",
        "tdate_string",
        "wthhfin",
        "home_state_county_fips",
        "home_tract_fips",
        "home_fips",
        "age",
        "disability",
        "employment_status",
        "gender",
        *j1_cols,
        "jobs_count",
        "license",
        "person_tripcount",
        "race_ethnicity_hispanic",
        "race_ethnicity",
        "smartphone",
        "student_status",
        "td_shop_time",
        "td_telecommute_time",
        "volunteer_status",
        "walk_bike_loop_trips",
        "work_bmc_taz",
        "work_state_county_fips",
        "work_tract_fips",
        "work_fips",
        "work_tpb_taz",
        "school_bmc_taz",
        "school_state_county_fips",
        "school_tract_fips",
        "school_fips",
        "school_mode",
        "school_tpb_taz",
        "school_type",
        "travel_mode",
        "travelers_hh",
        "travelers_nonhh",
        "vehicle_occupancy",
        *subway_cols,
        "toll_road_used",
        "transit_access_mode",
        "transit_egress_mode",
        "hov_used",
        "park_loc",
        "park_pay",
        "reported_travel_time",
        "departure_time_in_minutes_after_arrival",
        "departure_time_hhmm",
        "arrival_time_hhmm",
        "fueltype",
        "bodytype",
        "year",
        "make",
        "model",
        "tolltransponder",
    ]
    for od in od_cols:
        postfix = od[len("od_") :]
        out_cols.extend([f"o_{postfix}", f"d_{postfix}"])
        if postfix == "fips":
            out_cols.extend(
                [
                    "o_state_county_fips",
                    "o_tract_fips",
                    "d_state_county_fips",
                    "d_tract_fips",
                ]
            )
    out_cols = list(dict.fromkeys(out_cols))

    with StringRowWriter(args.output, out_cols) as writer_all, StringRowWriter(
        args.sample_output, out_cols
    ) as writer_sample:
        new_household_id = 0
        new_person_id = 0
        new_tripid = 0

        for hh_row in tqdm(
            reader,
            total=total_households,
            desc="Untupling households",
            unit="hh",
        ):
            new_household_id += 1
            hh_id_str = str(new_household_id)
            write_sample = new_household_id in sampled_households

            household_values = {
                "area_type": as_text(hh_row.get("area_type", "")).strip(),
                "food_delivered": as_text(hh_row.get("food_delivered", "")).strip(),
                "hh_income_detailed": as_text(hh_row.get("hh_income_detailed", "")).strip(),
                "hhsize": as_text(hh_row.get("hhsize", "")).strip(),
                "home_bmc_taz": as_text(hh_row.get("home_bmc_taz", "")).strip(),
                "home_ownership": as_text(hh_row.get("home_ownership", "")).strip(),
                "home_state_puma": as_text(hh_row.get("home_state_puma", "")).strip(),
                "home_tpb_taz": as_text(hh_row.get("home_tpb_taz", "")).strip(),
                "home_type": as_text(hh_row.get("home_type", "")).strip(),
                "numbicycle": as_text(hh_row.get("numbicycles", "")).strip(),
                "numvehicle": as_text(hh_row.get("numvehicles", "")).strip(),
                "numworkers": as_text(hh_row.get("numworkers", "")).strip(),
                "package_delivered": as_text(hh_row.get("package_delivered", "")).strip(),
                "service_delivered": as_text(hh_row.get("service_delivered", "")).strip(),
                "tdate_string": as_text(hh_row.get("tdate_string", "")).strip(),
                "wthhfin": as_text(hh_row.get("wthhfin", "")).strip(),
                "home_fips": as_text(hh_row.get("home_fips", "")).strip(),
            }
            home_sc, home_tr = split_fips11(household_values["home_fips"])
            household_values["home_state_county_fips"] = home_sc
            household_values["home_tract_fips"] = home_tr

            for p in p_slots:
                pfx = f"P{p}-"
                if not any(
                    (
                        as_text(hh_row.get(pfx + c, "")).strip()
                        for c in PERSON_IN_BASE_COLS + trip_chain_cols + od_cols
                    )
                ):
                    continue

                trip_count = trip_count_from_person(hh_row, pfx, trip_chain_cols, od_cols)
                if trip_count <= 0:
                    continue

                new_person_id += 1
                person_values = {
                    "age": as_text(hh_row.get(pfx + "age", "")).strip(),
                    "disability": as_text(hh_row.get(pfx + "disability", "")).strip(),
                    "employment_status": as_text(hh_row.get(pfx + "employment_status", "")).strip(),
                    "gender": as_text(hh_row.get(pfx + "gender", "")).strip(),
                    "jobs_count": as_text(hh_row.get(pfx + "job_count", "")).strip(),
                    "license": as_text(hh_row.get(pfx + "license", "")).strip(),
                    "person_tripcount": as_text(hh_row.get(pfx + "person_tripcount", "")).strip(),
                    "race_ethnicity_hispanic": as_text(hh_row.get(pfx + "race_ethnicity_hispanic", "")).strip(),
                    "race_ethnicity": as_text(hh_row.get(pfx + "race_ethnicity", "")).strip(),
                    "smartphone": as_text(hh_row.get(pfx + "smartphone", "")).strip(),
                    "student_status": as_text(hh_row.get(pfx + "student_status", "")).strip(),
                    "td_shop_time": as_text(hh_row.get(pfx + "td_shoptime", "")).strip(),
                    "td_telecommute_time": as_text(hh_row.get(pfx + "td_telecommute_time", "")).strip(),
                    "volunteer_status": as_text(hh_row.get(pfx + "volunteer_status", "")).strip(),
                    "walk_bike_loop_trips": as_text(hh_row.get(pfx + "walk_bike_loop_trips", "")).strip(),
                    "work_bmc_taz": as_text(hh_row.get(pfx + "work_bmc_taz", "")).strip(),
                    "work_fips": as_text(hh_row.get(pfx + "work_fips", "")).strip(),
                    "work_tpb_taz": as_text(hh_row.get(pfx + "work_tpb_taz", "")).strip(),
                    "school_bmc_taz": as_text(hh_row.get(pfx + "school_bmc_taz", "")).strip(),
                    "school_fips": as_text(hh_row.get(pfx + "school_fips", "")).strip(),
                    "school_mode": as_text(hh_row.get(pfx + "school_mode", "")).strip(),
                    "school_tpb_taz": as_text(hh_row.get(pfx + "school_tpb_taz", "")).strip(),
                    "school_type": as_text(hh_row.get(pfx + "school_type", "")).strip(),
                }
                for j1 in j1_cols:
                    person_values[j1] = as_text(hh_row.get(pfx + j1, "")).strip()

                work_sc, work_tr = split_fips11(person_values["work_fips"])
                school_sc, school_tr = split_fips11(person_values["school_fips"])
                person_values["work_state_county_fips"] = work_sc
                person_values["work_tract_fips"] = work_tr
                person_values["school_state_county_fips"] = school_sc
                person_values["school_tract_fips"] = school_tr

                seq_cache: Dict[str, List[object]] = {}
                for tc in trip_chain_cols:
                    seq_cache[tc] = parse_tuple_like(as_text(hh_row.get(pfx + tc, "")))
                for od in od_cols:
                    seq_cache[od] = parse_tuple_like(as_text(hh_row.get(pfx + od, "")))

                dep_delta_seq = seq_cache.get("departure_time_in_minutes_after_arrival", [])
                rt_seq = seq_cache.get("reported_travel_time", [])
                dep_abs_seq: List[Optional[int]] = [None] * trip_count
                arr_abs_seq: List[Optional[int]] = [None] * trip_count
                if trip_count > 0:
                    dep0 = parse_minutes_value(dep_delta_seq[0]) if len(dep_delta_seq) > 0 else None
                    dep_abs_seq[0] = dep0
                    rt0 = parse_minutes_value(rt_seq[0]) if len(rt_seq) > 0 else None
                    arr_abs_seq[0] = (dep0 + rt0) if dep0 is not None and rt0 is not None else None
                    for k in range(1, trip_count):
                        delta_k = parse_minutes_value(dep_delta_seq[k]) if k < len(dep_delta_seq) else None
                        prev_arr = arr_abs_seq[k - 1]
                        dep_k = (prev_arr + delta_k) if prev_arr is not None and delta_k is not None else None
                        dep_abs_seq[k] = dep_k
                        rt_k = parse_minutes_value(rt_seq[k]) if k < len(rt_seq) else None
                        arr_abs_seq[k] = (dep_k + rt_k) if dep_k is not None and rt_k is not None else None

                base_out = {
                    "household_id": hh_id_str,
                    "person_id": str(new_person_id),
                    "persno": str(p),
                    **household_values,
                    **person_values,
                }

                for t in range(trip_count):
                    new_tripid += 1
                    out = dict(base_out)
                    out["tripid"] = str(new_tripid)
                    out["tripno"] = str(t + 1)

                    for tc in trip_chain_cols:
                        if tc == "vehicle":
                            vehicle_item = seq_cache[tc][t] if t < len(seq_cache[tc]) else ""
                            fuel, body, year, make, model, trans = parse_vehicle_item(vehicle_item)
                            out["fueltype"] = fuel
                            out["bodytype"] = body
                            out["year"] = year
                            out["make"] = make
                            out["model"] = model
                            out["tolltransponder"] = trans
                        else:
                            out[tc] = get_seq_value(seq_cache[tc], t)

                    out["departure_time_hhmm"] = minutes_to_hhmm(dep_abs_seq[t])
                    out["arrival_time_hhmm"] = minutes_to_hhmm(arr_abs_seq[t])

                    for od in od_cols:
                        postfix = od[len("od_") :]
                        seq = seq_cache[od]
                        out[f"o_{postfix}"] = get_seq_value(seq, t)
                        out[f"d_{postfix}"] = get_seq_value(seq, t + 1)
                        if postfix == "fips":
                            o_sc, o_tr = split_fips11(out.get("o_fips", ""))
                            d_sc, d_tr = split_fips11(out.get("d_fips", ""))
                            out["o_state_county_fips"] = o_sc
                            out["o_tract_fips"] = o_tr
                            out["d_state_county_fips"] = d_sc
                            out["d_tract_fips"] = d_tr

                    writer_all.write(out)
                    if write_sample:
                        writer_sample.write(out)

    print(f"Wrote {args.output}")
    print(
        f"Wrote {args.sample_output} using {sample_n} sampled households out of {total_households}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

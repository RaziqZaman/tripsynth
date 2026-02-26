#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def parse_hhmm_to_minutes(value: str) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            return None
        h, m = parts[0].strip(), parts[1].strip()
    else:
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            return None
        if len(digits) <= 2:
            h, m = "0", digits
        else:
            h, m = digits[:-2], digits[-2:]

    if not h.isdigit() or not m.isdigit():
        return None
    minute = int(m)
    if minute < 0 or minute > 59:
        return None

    return int(h) * 60 + minute


def tuple_repr(values: List[object]) -> str:
    return repr(tuple(values))


def zfill_or_empty(value: str, width: int) -> str:
    s = "" if value is None else str(value).strip()
    if not s:
        return ""
    return s.zfill(width)


def first_non_empty(rows: List[Dict[str, str]], col: str) -> str:
    for r in rows:
        v = str(r.get(col, "")).strip()
        if v:
            return v
    return ""


def to_float_or_neg_inf(value: str) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("-inf")


def trip_sort_key(row: Dict[str, str]) -> Tuple[float, str]:
    tripno_raw = str(row.get("tripno", "")).strip()
    try:
        tripno_num = float(tripno_raw)
    except Exception:
        tripno_num = float("inf")
    tripid = str(row.get("tripid", ""))
    return (tripno_num, tripid)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform combined-flat-survey.csv into tupled-survey.csv"
    )
    parser.add_argument("--input", default="combined-flat-survey.csv")
    parser.add_argument("--output", default="tupled-survey.csv")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_columns = list(reader.fieldnames or [])
        raw_rows = list(reader)

    # Discover FIPS pair prefixes
    state_county_cols = [c for c in original_columns if c.endswith("_state_county_fips")]
    prefixes = []
    for sc_col in state_county_cols:
        prefix = sc_col[: -len("_state_county_fips")]
        tract_col = f"{prefix}_tract_fips"
        if tract_col in original_columns:
            prefixes.append(prefix)

    # Transform rows: add {prefix}_fips and drop the paired source cols
    rows: List[Dict[str, str]] = []
    for src in raw_rows:
        r = dict(src)

        for prefix in prefixes:
            sc_col = f"{prefix}_state_county_fips"
            tract_col = f"{prefix}_tract_fips"
            out_col = f"{prefix}_fips"
            r[out_col] = zfill_or_empty(r.get(sc_col, ""), 5) + zfill_or_empty(
                r.get(tract_col, ""), 6
            )
            r.pop(sc_col, None)
            r.pop(tract_col, None)

        # Build trip-level vehicle tuple
        vehicle_parts = ["fueltype", "bodytype", "year", "make", "model", "tolltransponder"]
        vehicle_vals = [str(r.get(c, "")).strip() for c in vehicle_parts]
        r["vehicle"] = tuple_repr(vehicle_vals)

        rows.append(r)

    # Working column set after fips merge
    columns_after = set()
    for r in rows:
        columns_after.update(r.keys())

    j1_cols = sorted(c for c in columns_after if c.startswith("j1_"))
    subway_cols = sorted(c for c in columns_after if c.startswith("subway_"))

    o_cols = [c for c in columns_after if c.startswith("o_")]
    od_postfixes = sorted(c[2:] for c in o_cols if f"d_{c[2:]}" in columns_after)

    household_cols_out = [
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
    ]
    household_source_map = {
        "numbicycles": "numbicycle",
        "numvehicles": "numvehicle",
    }

    person_cols_out_base = [
        "age",
        "disability",
        "employment_status",
        "gender",
        *j1_cols,
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
    person_source_map = {
        "job_count": "jobs_count",
        "td_shoptime": "td_shop_time",
    }

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
    ]

    od_out_cols = [f"od_{p}" for p in od_postfixes]
    trip_special_cols = trip_chain_cols + od_out_cols + [
        "departure_time_in_minutes_after_arrival"
    ]

    output_columns = ["household_id"] + household_cols_out
    for n in range(1, 9):
        pfx = f"P{n}-"
        output_columns.extend(pfx + c for c in person_cols_out_base)
        output_columns.extend(pfx + c for c in trip_special_cols)
    output_columns.append("wthhfin")

    # Group by household
    households: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        households[str(r.get("household_id", ""))].append(r)

    out_rows: List[Dict[str, str]] = []

    for household_id, hh_rows in households.items():
        out = {c: "" for c in output_columns}
        out["household_id"] = household_id

        # Household-level fields
        for col_out in household_cols_out:
            src = household_source_map.get(col_out, col_out)
            out[col_out] = first_non_empty(hh_rows, src)
        out["wthhfin"] = first_non_empty(hh_rows, "wthhfin")

        # Group persons and rank by descending age
        persons: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for r in hh_rows:
            persons[str(r.get("person_id", ""))].append(r)

        ranked_people = []
        for pid, p_rows in persons.items():
            age_val = first_non_empty(p_rows, "age")
            if not age_val:
                age_val = first_non_empty(p_rows, "age_imp")
            ranked_people.append((pid, p_rows, to_float_or_neg_inf(age_val)))

        ranked_people.sort(key=lambda x: (-x[2], x[0]))

        for idx, (_, p_rows, _) in enumerate(ranked_people[:8], start=1):
            pfx = f"P{idx}-"

            # Person-level columns
            for col_out in person_cols_out_base:
                src = person_source_map.get(col_out, col_out)
                out[pfx + col_out] = first_non_empty(p_rows, src)

            # Sort trips for this person
            p_sorted = sorted(p_rows, key=trip_sort_key)

            # Sequential tuple columns
            for c in trip_chain_cols:
                if c == "vehicle":
                    vals = [
                        (
                            str(r.get("fueltype", "")).strip(),
                            str(r.get("bodytype", "")).strip(),
                            str(r.get("year", "")).strip(),
                            str(r.get("make", "")).strip(),
                            str(r.get("model", "")).strip(),
                            str(r.get("tolltransponder", "")).strip(),
                        )
                        for r in p_sorted
                    ]
                else:
                    vals = [str(r.get(c, "")).strip() for r in p_sorted]
                out[pfx + c] = tuple_repr(vals)

            # od_{postfix}: all origins + destination from last trip
            for postfix in od_postfixes:
                o_col = f"o_{postfix}"
                d_col = f"d_{postfix}"
                o_vals = [str(r.get(o_col, "")).strip() for r in p_sorted]
                last_d = str(p_sorted[-1].get(d_col, "")).strip() if p_sorted else ""
                out[pfx + f"od_{postfix}"] = tuple_repr(o_vals + [last_d])

            # departure_time_in_minutes_after_arrival
            dep_mins = [
                parse_hhmm_to_minutes(str(r.get("departure_time_hhmm", "")).strip())
                for r in p_sorted
            ]
            arr_mins = [
                parse_hhmm_to_minutes(str(r.get("arrival_time_hhmm", "")).strip())
                for r in p_sorted
            ]

            deltas: List[Optional[int]] = []
            if dep_mins:
                deltas.append(dep_mins[0])
                for k in range(1, len(dep_mins)):
                    cur_dep = dep_mins[k]
                    prev_arr = arr_mins[k - 1] if (k - 1) < len(arr_mins) else None
                    if cur_dep is None or prev_arr is None:
                        deltas.append(None)
                    else:
                        diff = cur_dep - prev_arr
                        if diff < 0:
                            diff += 24 * 60
                        deltas.append(diff)

            out[pfx + "departure_time_in_minutes_after_arrival"] = tuple_repr(deltas)

        out_rows.append(out)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()

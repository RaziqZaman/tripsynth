from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trip_synth.data.load import read_survey_csv
from trip_synth.data.schema import FeatureSchema, clean_fips_value
from trip_synth.utils.io import ensure_dir, load_yaml, read_json, write_json
from trip_synth.utils.progress import progress_iter

from .aadt_screenlines import build_screenlines, comparison_basis_for_field, load_tracts, temporal_label
from .metrics import geh, regression_slope_intercept, safe_corr
from .route_assignment import screenline_id


def compute_aadt_metrics(observed: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(synthetic)
    if mask.sum() == 0:
        return {"screenlines": 0}
    o = observed[mask]
    s = synthetic[mask]
    g = geh(o, s)
    log_o = np.log1p(o)
    log_s = np.log1p(s)
    slope, intercept = regression_slope_intercept(log_o, log_s)
    denom = np.maximum(np.abs(o), 1.0)
    ss_res = float(np.sum((s - o) ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    log_ss_res = float(np.sum((log_s - log_o) ** 2))
    log_ss_tot = float(np.sum((log_o - np.mean(log_o)) ** 2))
    return {
        "screenlines": int(mask.sum()),
        "pearson_raw": safe_corr(o, s, "pearson"),
        "spearman_raw": safe_corr(o, s, "spearman"),
        "pearson_log1p": safe_corr(log_o, log_s, "pearson"),
        "spearman_log1p": safe_corr(log_o, log_s, "spearman"),
        "rmse": float(np.sqrt(np.mean((s - o) ** 2))),
        "mae": float(np.mean(np.abs(s - o))),
        "mape_guarded": float(np.mean(np.abs(s - o) / denom)),
        "smape": float(np.mean(2 * np.abs(s - o) / np.maximum(np.abs(s) + np.abs(o), 1.0))),
        "median_ape": float(np.median(np.abs(s - o) / denom)),
        "bias_mean_error": float(np.mean(s - o)),
        "r2_raw": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "r2_log1p": float(1.0 - log_ss_res / log_ss_tot) if log_ss_tot > 0 else float("nan"),
        "geh_mean": float(np.mean(g)),
        "share_geh_lt_5": float(np.mean(g < 5)),
        "share_geh_lt_10": float(np.mean(g < 10)),
        "calibration_slope_log1p": slope,
        "calibration_intercept_log1p": intercept,
    }


def _clean_tract_series(series: pd.Series) -> pd.Series:
    return series.map(clean_fips_value).astype(str)


def _candidate_od_pairs(config: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    schema = FeatureSchema.from_yaml("configs/schema.yaml")
    real_df, _ = read_survey_csv(config["input_csv"], schema)
    frames = []
    if {"o_tract_fips", "d_tract_fips"}.issubset(real_df.columns):
        frames.append(real_df[["o_tract_fips", "d_tract_fips"]])
    for method in config.get("methods", []):
        sample_path = run_dir / "samples" / f"{method}_synthetic.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path, usecols=lambda c: c in {"o_tract_fips", "d_tract_fips"}, low_memory=False)
            if {"o_tract_fips", "d_tract_fips"}.issubset(df.columns):
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["o_tract_fips", "d_tract_fips"])
    od = pd.concat(frames, ignore_index=True)
    od["o_tract_fips"] = _clean_tract_series(od["o_tract_fips"])
    od["d_tract_fips"] = _clean_tract_series(od["d_tract_fips"])
    od = od[(od["o_tract_fips"] != "-1") & (od["d_tract_fips"] != "-1")]
    od = od.drop_duplicates()
    max_pairs = int(config.get("aadt_validation", {}).get("max_od_pairs", 250000))
    if len(od) > max_pairs:
        od = od.head(max_pairs)
    return od.reset_index(drop=True)


def _ordered_crossed_tracts(line, candidates, origin: str, dest: str) -> list[str]:
    rows = []
    for tract, geom in candidates[["GEOID", "geometry"]].itertuples(index=False):
        try:
            inter = geom.intersection(line)
        except Exception:
            continue
        if inter.is_empty:
            continue
        point = inter.representative_point()
        pos = float(line.project(point))
        rows.append((pos, str(tract)))
    rows.extend([(0.0, origin), (float(line.length), dest)])
    rows = sorted(rows, key=lambda x: x[0])
    ordered: list[str] = []
    for _, tract in rows:
        if not ordered or ordered[-1] != tract:
            ordered.append(tract)
    if ordered and ordered[0] != origin:
        ordered.insert(0, origin)
    if ordered and ordered[-1] != dest:
        ordered.append(dest)
    return ordered


def build_od_screenline_paths(config: dict[str, Any], run_dir: str | Path):
    import geopandas as gpd
    from shapely.geometry import LineString

    run_dir = Path(run_dir)
    out_path = run_dir / "geo" / "od_screenline_paths.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    screenline_path = run_dir / "geo" / "screenlines.parquet"
    if not screenline_path.exists():
        status = build_screenlines(config, run_dir)
        if status.get("status") != "ok":
            raise FileNotFoundError("screenlines.parquet is unavailable and could not be built")
    tracts = load_tracts(config, run_dir)
    screenlines = gpd.read_parquet(screenline_path)
    valid_screenlines = set(screenlines["screenline_id"].astype(str))
    od = _candidate_od_pairs(config, run_dir)
    if od.empty:
        empty = pd.DataFrame(columns=["o_tract_fips", "d_tract_fips", "path_position", "screenline_id"])
        empty.to_parquet(out_path)
        return empty
    tract_lookup = tracts.set_index("GEOID")
    sindex = tracts.sindex
    path_rows: list[dict[str, Any]] = []
    od_pairs = od[["o_tract_fips", "d_tract_fips"]].itertuples(index=False)
    for origin, dest in progress_iter(od_pairs, desc="AADT OD screenline paths", total=len(od), unit="pair"):
        if origin == dest or origin not in tract_lookup.index or dest not in tract_lookup.index:
            continue
        a = tract_lookup.loc[origin]
        b = tract_lookup.loc[dest]
        line = LineString([(float(a.rep_x), float(a.rep_y)), (float(b.rep_x), float(b.rep_y))])
        candidate_idx = list(sindex.query(line, predicate="intersects"))
        candidates = tracts.iloc[candidate_idx]
        ordered = _ordered_crossed_tracts(line, candidates, origin, dest)
        position = 0
        for left, right in zip(ordered[:-1], ordered[1:]):
            sid = screenline_id(left, right)
            if sid not in valid_screenlines:
                continue
            path_rows.append(
                {
                    "o_tract_fips": origin,
                    "d_tract_fips": dest,
                    "path_position": position,
                    "screenline_id": sid,
                }
            )
            position += 1
    paths = pd.DataFrame(path_rows)
    if paths.empty:
        paths = pd.DataFrame(columns=["o_tract_fips", "d_tract_fips", "path_position", "screenline_id"])
    paths.to_parquet(out_path)
    return paths


def _validation_mode(config: dict[str, Any]) -> str:
    return str(config.get("aadt_validation", {}).get("validation_mode", "exact_day"))


def trip_dates_from_week_dow(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Reconstruct trip dates from week offsets after Jan. 1, 2017 and DOW codes."""
    index = frame.index
    if "tdate_week" not in frame.columns or "tdate_dow" not in frame.columns:
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    cfg = config.get("aadt_validation", {})
    origin = pd.Timestamp(str(cfg.get("trip_week_origin_date", "2017-01-01")))
    encoding = str(cfg.get("tdate_dow_encoding", "iso_monday_1"))
    weeks = pd.to_numeric(frame["tdate_week"], errors="coerce")
    dow = pd.to_numeric(frame["tdate_dow"], errors="coerce")
    offsets = pd.Series(np.nan, index=index, dtype="float64")
    valid = weeks.notna() & dow.notna()
    if encoding in {"iso_monday_1", "monday_1"}:
        # Jan. 1, 2017 was Sunday. ISO Monday=1 therefore has offset 1 inside
        # each Sunday-start week bucket; Sunday=7 maps to offset 0.
        offsets.loc[valid] = dow.loc[valid] % 7
    elif encoding == "sunday_1":
        offsets.loc[valid] = dow.loc[valid] - 1
    else:
        raise ValueError(f"Unsupported tdate_dow_encoding: {encoding}")
    valid = valid & offsets.between(0, 6)
    dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    day_offsets = weeks.loc[valid].round().astype(int) * 7 + offsets.loc[valid].round().astype(int)
    dates.loc[valid] = origin + pd.to_timedelta(day_offsets, unit="D")
    return pd.to_datetime(dates).dt.normalize()


def _weekday_values(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.notna(), series.astype(str))


def _synthetic_average_screenline_counts(
    synthetic: pd.DataFrame,
    paths: pd.DataFrame,
    comparison_basis: str,
) -> pd.DataFrame:
    if paths.empty or synthetic.empty:
        return pd.DataFrame(columns=["screenline_id", "synthetic_count"])
    trips = synthetic[["o_tract_fips", "d_tract_fips"]].copy()
    trips["trip_id"] = np.arange(len(trips))
    trips["o_tract_fips"] = _clean_tract_series(trips["o_tract_fips"])
    trips["d_tract_fips"] = _clean_tract_series(trips["d_tract_fips"])
    if "tdate_dow" in synthetic.columns:
        trips["tdate_dow"] = _weekday_values(synthetic["tdate_dow"])
    expanded = trips.merge(paths, on=["o_tract_fips", "d_tract_fips"], how="inner")
    if expanded.empty:
        return pd.DataFrame(columns=["screenline_id", "synthetic_count"])
    if comparison_basis == "average_weekday" and "tdate_dow" in expanded.columns:
        dow = pd.to_numeric(expanded["tdate_dow"], errors="coerce")
        expanded = expanded[dow.between(1, 5)]
        if expanded.empty:
            return pd.DataFrame(columns=["screenline_id", "synthetic_count"])
        by_dow = expanded.groupby(["screenline_id", "tdate_dow"]).size().reset_index(name="daily_count")
        counts = by_dow.groupby("screenline_id")["daily_count"].mean().reset_index(name="synthetic_count")
    elif "tdate_dow" in expanded.columns:
        by_dow = expanded.groupby(["screenline_id", "tdate_dow"]).size().reset_index(name="daily_count")
        counts = by_dow.groupby("screenline_id")["daily_count"].mean().reset_index(name="synthetic_count")
    else:
        counts = expanded.groupby("screenline_id").size().reset_index(name="synthetic_count")
    return counts


def _synthetic_exact_day_screenline_counts(
    synthetic: pd.DataFrame,
    paths: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = ["screenline_id", "trip_date", "synthetic_count"]
    required = {"o_tract_fips", "d_tract_fips", "tdate_week", "tdate_dow"}
    if paths.empty or synthetic.empty or not required.issubset(synthetic.columns):
        return pd.DataFrame(columns=columns)
    trips = synthetic[["o_tract_fips", "d_tract_fips", "tdate_week", "tdate_dow"]].copy()
    trips["trip_id"] = np.arange(len(trips))
    trips["o_tract_fips"] = _clean_tract_series(trips["o_tract_fips"])
    trips["d_tract_fips"] = _clean_tract_series(trips["d_tract_fips"])
    trips["trip_date"] = trip_dates_from_week_dow(trips, config)
    trips = trips[trips["trip_date"].notna()].copy()
    expanded = trips.merge(paths, on=["o_tract_fips", "d_tract_fips"], how="inner")
    if expanded.empty:
        return pd.DataFrame(columns=columns)
    counts = expanded.groupby(["screenline_id", "trip_date"]).size().reset_index(name="synthetic_count")
    return counts[columns]


def _date_strata_expansion_enabled(config: dict[str, Any]) -> bool:
    cfg = config.get("aadt_validation", {})
    basis = str(cfg.get("synthetic_population_basis", "average_day")).lower()
    return bool(cfg.get("expand_date_strata_to_average_day", True)) and basis in {"average_day", "average_weekday"}


def _date_strata_expansion_from_synthetic(
    synthetic: pd.DataFrame,
    config: dict[str, Any],
    comparison_basis: str = "average_day",
) -> float:
    if not _date_strata_expansion_enabled(config) or not {"tdate_week", "tdate_dow"}.issubset(synthetic.columns):
        return 1.0
    dates = trip_dates_from_week_dow(synthetic, config).dropna()
    if comparison_basis == "average_weekday":
        dates = dates[dates.dt.weekday < 5]
    n_dates = int(dates.dt.normalize().nunique())
    return float(max(1, n_dates))


def _date_strata_expansion_from_counts(counts: pd.DataFrame, config: dict[str, Any]) -> float:
    if not _date_strata_expansion_enabled(config) or counts.empty or "trip_date" not in counts.columns:
        return 1.0
    dates = pd.to_datetime(counts["trip_date"], errors="coerce").dt.normalize().dropna()
    n_dates = int(dates.nunique())
    return float(max(1, n_dates))


def _apply_date_strata_expansion(counts: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = counts.copy()
    if not out.empty and "synthetic_count" in out.columns:
        out["synthetic_count"] = pd.to_numeric(out["synthetic_count"], errors="coerce").fillna(0.0) * factor
        out["temporal_expansion_factor"] = factor
    return out


def _sample_expansion_factor(run_dir: Path, method: str) -> float:
    meta = read_json(run_dir / "metrics" / f"{method}_sample_scaling.json", default={}) or {}
    try:
        factor = float(meta.get("sample_expansion_factor", 1.0))
    except (TypeError, ValueError):
        factor = 1.0
    return factor if np.isfinite(factor) and factor > 0 else 1.0


def _apply_sample_expansion(counts: pd.DataFrame, run_dir: Path, method: str) -> tuple[pd.DataFrame, float]:
    factor = _sample_expansion_factor(run_dir, method)
    out = counts.copy()
    if not out.empty and "synthetic_count" in out.columns:
        out["synthetic_count"] = pd.to_numeric(out["synthetic_count"], errors="coerce").fillna(0.0) * factor
        out["sample_expansion_factor"] = factor
    return out, factor


def _daily_count_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("aadt_validation", {})
    daily = dict(cfg.get("daily_counts", {}) or {})
    if "file" not in daily:
        file_value = cfg.get("daily_counts_file") or config.get("geo", {}).get("mdot_daily_counts_file")
        if file_value:
            daily["file"] = file_value
    daily.setdefault("station_id_column", "station_id")
    daily.setdefault("date_column", "date")
    daily.setdefault("count_column", "observed_count")
    return daily


def load_daily_station_counts(config: dict[str, Any]) -> pd.DataFrame:
    daily = _daily_count_config(config)
    path_value = daily.get("file")
    if not path_value:
        raise FileNotFoundError("Exact-day AADT validation requires aadt_validation.daily_counts.file")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Missing exact-day MDOT station count file: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, low_memory=False)
    station_col = str(daily["station_id_column"])
    date_col = str(daily["date_column"])
    count_col = str(daily["count_column"])
    missing = [c for c in [station_col, date_col, count_col] if c not in raw.columns]
    if missing:
        raise ValueError(f"Daily station count file is missing required columns: {missing}")
    out = raw[[station_col, date_col, count_col]].copy()
    out = out.rename(columns={station_col: "station_id", date_col: "trip_date", count_col: "observed_count"})
    out["station_id"] = out["station_id"].astype(str)
    out["trip_date"] = pd.to_datetime(out["trip_date"], errors="coerce").dt.normalize()
    out["observed_count"] = pd.to_numeric(out["observed_count"], errors="coerce")
    out = out[out["station_id"].ne("") & out["trip_date"].notna() & out["observed_count"].notna()].copy()
    return out.groupby(["station_id", "trip_date"], as_index=False)["observed_count"].sum()


def build_observed_exact_day_screenline_counts(config: dict[str, Any], run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    out_path = run_dir / "metrics" / "observed_exact_day_screenline_counts.parquet"
    station_map_path = run_dir / "geo" / "screenline_station_map.parquet"
    if not station_map_path.exists():
        raise FileNotFoundError("screenline_station_map.parquet is unavailable")
    station_map = pd.read_parquet(station_map_path)
    if station_map.empty:
        return pd.DataFrame(columns=["screenline_id", "trip_date", "observed_count", "station_count_observed"])
    daily = load_daily_station_counts(config)
    mapped = station_map[["screenline_id", "station_id"]].drop_duplicates().merge(daily, on="station_id", how="inner")
    if mapped.empty:
        return pd.DataFrame(columns=["screenline_id", "trip_date", "observed_count", "station_count_observed"])
    observed = (
        mapped.groupby(["screenline_id", "trip_date"])
        .agg(observed_count=("observed_count", "sum"), station_count_observed=("station_id", "nunique"))
        .reset_index()
    )
    observed.to_parquet(out_path)
    return observed


def _hourly_count_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("aadt_validation", {})
    hourly = dict(cfg.get("hourly_counts", {}) or {})
    if "file" not in hourly:
        file_value = cfg.get("hourly_counts_file") or config.get("geo", {}).get("fhwa_hourly_counts_file")
        if file_value:
            hourly["file"] = file_value
    hourly.setdefault("file", "data/external/mdot_daily_counts/station_hourly_counts.csv")
    hourly.setdefault("station_id_column", "station_id")
    hourly.setdefault("date_column", "date")
    hourly.setdefault("hour_column", "hour")
    hourly.setdefault("count_column", "observed_count")
    return hourly


def load_hourly_station_counts(config: dict[str, Any]) -> pd.DataFrame:
    hourly = _hourly_count_config(config)
    path = Path(str(hourly["file"]))
    if not path.exists():
        raise FileNotFoundError(f"Missing hourly FHWA TMAS station count file: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, low_memory=False)
    station_col = str(hourly["station_id_column"])
    date_col = str(hourly["date_column"])
    hour_col = str(hourly["hour_column"])
    count_col = str(hourly["count_column"])
    missing = [c for c in [station_col, date_col, hour_col, count_col] if c not in raw.columns]
    if missing:
        raise ValueError(f"Hourly station count file is missing required columns: {missing}")
    out = raw[[station_col, date_col, hour_col, count_col]].copy()
    out = out.rename(
        columns={station_col: "station_id", date_col: "trip_date", hour_col: "hour", count_col: "observed_count"}
    )
    out["station_id"] = out["station_id"].astype(str)
    out["trip_date"] = pd.to_datetime(out["trip_date"], errors="coerce").dt.normalize()
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out["observed_count"] = pd.to_numeric(out["observed_count"], errors="coerce")
    out = out[
        out["station_id"].ne("")
        & out["trip_date"].notna()
        & out["hour"].between(0, 23)
        & out["observed_count"].notna()
    ].copy()
    out["hour"] = out["hour"].astype(int)
    return out.groupby(["station_id", "trip_date", "hour"], as_index=False)["observed_count"].sum()


def _synthetic_long_term_average_screenline_counts(
    synthetic: pd.DataFrame,
    paths: pd.DataFrame,
    config: dict[str, Any],
    comparison_basis: str,
) -> pd.DataFrame:
    columns = ["screenline_id", "synthetic_count"]
    required = {"o_tract_fips", "d_tract_fips"}
    if paths.empty or synthetic.empty or not required.issubset(synthetic.columns):
        return pd.DataFrame(columns=columns)

    trips = synthetic[["o_tract_fips", "d_tract_fips"]].copy()
    trips["trip_id"] = np.arange(len(trips))
    trips["o_tract_fips"] = _clean_tract_series(trips["o_tract_fips"])
    trips["d_tract_fips"] = _clean_tract_series(trips["d_tract_fips"])
    if {"tdate_week", "tdate_dow"}.issubset(synthetic.columns):
        trips["trip_date"] = trip_dates_from_week_dow(synthetic, config)
    elif "tdate_dow" in synthetic.columns:
        trips["tdate_dow"] = _weekday_values(synthetic["tdate_dow"])

    expanded = trips.merge(paths, on=["o_tract_fips", "d_tract_fips"], how="inner")
    if expanded.empty:
        return pd.DataFrame(columns=columns)

    if "trip_date" in expanded.columns and expanded["trip_date"].notna().any():
        expanded = expanded[expanded["trip_date"].notna()].copy()
        if comparison_basis == "average_weekday":
            expanded = expanded[pd.to_datetime(expanded["trip_date"]).dt.weekday < 5].copy()
        dates = pd.to_datetime(expanded["trip_date"]).dt.normalize().dropna().unique()
        if len(dates) == 0:
            return pd.DataFrame(columns=columns)
        totals = expanded.groupby("screenline_id").size().reset_index(name="total_crossings")
        totals["synthetic_count"] = totals["total_crossings"] / float(len(dates))
        return totals[["screenline_id", "synthetic_count"]]

    return _synthetic_average_screenline_counts(synthetic, paths, comparison_basis)


def _synthetic_hourly_screenline_counts(
    synthetic: pd.DataFrame,
    paths: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = ["screenline_id", "trip_date", "hour", "synthetic_count"]
    required = {
        "o_tract_fips",
        "d_tract_fips",
        "tdate_week",
        "tdate_dow",
        "departure_time_minutes",
        "reported_travel_time",
    }
    if paths.empty or synthetic.empty or not required.issubset(synthetic.columns):
        return pd.DataFrame(columns=columns)

    trips = synthetic[list(required)].copy()
    trips["trip_id"] = np.arange(len(trips))
    trips["o_tract_fips"] = _clean_tract_series(trips["o_tract_fips"])
    trips["d_tract_fips"] = _clean_tract_series(trips["d_tract_fips"])
    trips["trip_date"] = trip_dates_from_week_dow(trips, config)
    trips["departure_time_minutes"] = pd.to_numeric(trips["departure_time_minutes"], errors="coerce")
    trips["reported_travel_time"] = pd.to_numeric(trips["reported_travel_time"], errors="coerce").clip(lower=0)
    trips = trips[trips["trip_date"].notna() & trips["departure_time_minutes"].notna() & trips["reported_travel_time"].notna()].copy()
    expanded = trips.merge(paths, on=["o_tract_fips", "d_tract_fips"], how="inner")
    if expanded.empty:
        return pd.DataFrame(columns=columns)

    expanded["path_position"] = pd.to_numeric(expanded["path_position"], errors="coerce").fillna(0).astype(int)
    path_counts = expanded.groupby("trip_id")["path_position"].transform("max") + 1
    expanded["path_fraction"] = (expanded["path_position"] + 1) / (path_counts + 1)
    crossing_minutes = expanded["departure_time_minutes"] + expanded["reported_travel_time"] * expanded["path_fraction"]
    day_offsets = np.floor(crossing_minutes / 1440.0).astype(int)
    minutes_in_day = np.mod(crossing_minutes, 1440.0)
    expanded["hour"] = np.floor(minutes_in_day / 60.0).astype(int).clip(0, 23)
    expanded["trip_date"] = pd.to_datetime(expanded["trip_date"]).dt.normalize() + pd.to_timedelta(day_offsets, unit="D")
    counts = expanded.groupby(["screenline_id", "trip_date", "hour"]).size().reset_index(name="synthetic_count")
    return counts[columns]


def build_observed_hourly_screenline_counts(config: dict[str, Any], run_dir: str | Path, screenlines: pd.DataFrame) -> pd.DataFrame:
    run_dir = Path(run_dir)
    out_path = run_dir / "metrics" / "observed_hourly_screenline_counts.parquet"
    station_map_path = run_dir / "geo" / "screenline_station_map.parquet"
    if not station_map_path.exists():
        raise FileNotFoundError("screenline_station_map.parquet is unavailable")
    station_map = pd.read_parquet(station_map_path)
    if station_map.empty or "observed_count" not in station_map.columns:
        return pd.DataFrame(columns=["screenline_id", "trip_date", "hour", "observed_count"])

    hourly = load_hourly_station_counts(config)
    annual = station_map[["screenline_id", "station_id", "observed_count"]].drop_duplicates().copy()
    annual = annual.rename(columns={"observed_count": "annual_station_count"})
    screen_totals = screenlines[["screenline_id", "observed_count"]].drop_duplicates().rename(
        columns={"observed_count": "annual_screenline_count"}
    )
    fallback_totals = annual.groupby("screenline_id", as_index=False)["annual_station_count"].sum().rename(
        columns={"annual_station_count": "fallback_annual_screenline_count"}
    )
    annual = annual.merge(screen_totals, on="screenline_id", how="left").merge(fallback_totals, on="screenline_id", how="left")
    annual["annual_screenline_count"] = annual["annual_screenline_count"].where(
        annual["annual_screenline_count"].gt(0), annual["fallback_annual_screenline_count"]
    )
    annual["annual_share"] = annual["annual_station_count"] / annual["annual_screenline_count"]
    annual = annual[annual["annual_share"].gt(0) & np.isfinite(annual["annual_share"])].copy()

    mapped = annual.merge(hourly, on="station_id", how="inner")
    if mapped.empty:
        return pd.DataFrame(columns=["screenline_id", "trip_date", "hour", "observed_count"])
    observed = (
        mapped.groupby(["screenline_id", "trip_date", "hour"])
        .agg(
            observed_tmas_count=("observed_count", "sum"),
            annual_share_observed=("annual_share", "sum"),
            tmas_station_count=("station_id", "nunique"),
            annual_station_count_observed=("annual_station_count", "sum"),
            annual_screenline_count=("annual_screenline_count", "first"),
        )
        .reset_index()
    )
    observed = observed[observed["annual_share_observed"].gt(0)].copy()
    observed["observed_count"] = observed["observed_tmas_count"] / observed["annual_share_observed"]
    observed["scaling_method"] = "fhwa_tmas_hourly_scaled_by_annual_screenline_share"
    observed.to_parquet(out_path)
    return observed


def build_virtual_screenline_counts(config: dict[str, Any], run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    paths = build_od_screenline_paths(config, run_dir)
    screenlines = pd.read_parquet(run_dir / "geo" / "screenlines.parquet")
    mode = _validation_mode(config)
    observed_field = str(screenlines.get("observed_count_field", pd.Series([""])).dropna().astype(str).iloc[0]) if len(screenlines) else ""
    if mode == "exact_day":
        comparison_basis = "exact_day"
    elif "comparison_basis" in screenlines.columns and screenlines["comparison_basis"].notna().any():
        comparison_basis = str(screenlines["comparison_basis"].dropna().astype(str).iloc[0])
    else:
        comparison_basis = comparison_basis_for_field(config, observed_field)
    outputs: dict[str, pd.DataFrame] = {}
    methods = list(config.get("methods", []))
    for method in progress_iter(methods, desc="AADT virtual counts", total=len(methods), unit="method"):
        sample_path = run_dir / "samples" / f"{method}_synthetic.csv"
        if not sample_path.exists():
            continue
        synthetic = pd.read_csv(sample_path, low_memory=False)
        if mode == "exact_day":
            counts = _synthetic_exact_day_screenline_counts(synthetic, paths, config)
            temporal_factor = _date_strata_expansion_from_counts(counts, config)
        else:
            counts = _synthetic_average_screenline_counts(synthetic, paths, comparison_basis)
            temporal_factor = _date_strata_expansion_from_synthetic(synthetic, config, comparison_basis)
        counts = _apply_date_strata_expansion(counts, temporal_factor)
        counts, expansion_factor = _apply_sample_expansion(counts, run_dir, str(method))
        counts["method"] = method
        counts["sample_expansion_factor"] = expansion_factor
        counts["temporal_expansion_factor"] = temporal_factor
        counts["comparison_basis"] = comparison_basis
        out_path = run_dir / "metrics" / f"{method}_virtual_screenline_counts.parquet"
        counts.to_parquet(out_path)
        outputs[str(method)] = counts
    return outputs


def _plot_aadt_outputs(run_dir: Path, comparisons: pd.DataFrame, screenlines) -> None:
    ensure_dir(run_dir / "figures" / "poster")
    if comparisons.empty:
        return
    plt.figure(figsize=(7, 6))
    for method, group in comparisons.groupby("method"):
        plt.scatter(np.log1p(group["observed_count"]), np.log1p(group["synthetic_count"]), s=18, alpha=0.55, label=method)
    plt.xlabel("log1p observed screenline counts")
    plt.ylabel("log1p synthetic virtual crossings")
    plt.title("Observed vs synthetic screenline volumes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "poster" / "aadt_observed_vs_synthetic_log_scatter.png", dpi=300)
    plt.savefig(run_dir / "figures" / "poster" / "08_aadt_observed_vs_synthetic_log_scatter.png", dpi=300)
    plt.close()

    leaderboard = comparisons.groupby("method").apply(
        lambda g: compute_aadt_metrics(g["observed_count"].to_numpy(), g["synthetic_count"].to_numpy()).get("pearson_log1p", np.nan),
        include_groups=False,
    ).reset_index(name="pearson_log1p")
    plt.figure(figsize=(8, 4.8))
    plt.bar(leaderboard["method"], leaderboard["pearson_log1p"].fillna(0.0))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Pearson correlation, log1p")
    plt.title("Traffic count validation leaderboard")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "poster" / "aadt_method_leaderboard.png", dpi=300)
    plt.savefig(run_dir / "figures" / "poster" / "09_aadt_method_leaderboard.png", dpi=300)
    plt.close()

    comparisons["geh"] = geh(comparisons["observed_count"].to_numpy(), comparisons["synthetic_count"].to_numpy())
    plt.figure(figsize=(8, 4.8))
    for method, group in comparisons.groupby("method"):
        plt.hist(group["geh"], bins=30, alpha=0.45, label=method)
    plt.xlabel("GEH")
    plt.ylabel("Screenlines")
    plt.title("GEH distribution by method")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "poster" / "geh_distribution_by_method.png", dpi=300)
    plt.savefig(run_dir / "figures" / "poster" / "10_geh_distribution_by_method.png", dpi=300)
    plt.close()

    contrastive = comparisons[comparisons["method"] == "contrastive_vae"]
    if not contrastive.empty and hasattr(screenlines, "plot"):
        import geopandas as gpd

        mapped = screenlines.merge(
            contrastive[["screenline_id", "residual"]], on="screenline_id", how="left"
        )
        mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=screenlines.crs)
        ax = mapped.dropna(subset=["residual"]).plot(
            column="residual", cmap="coolwarm", legend=True, figsize=(8, 8), linewidth=1.2
        )
        ax.set_axis_off()
        ax.set_title("Contrastive VAE screenline residuals")
        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "poster" / "aadt_residual_map_contrastive_vae.png", dpi=300)
        plt.close()


def _annual_comparison_basis(config: dict[str, Any], observed_field: str) -> str:
    cfg = config.get("aadt_validation", {})
    configured = str(cfg.get("annual_comparison_basis", ""))
    if configured:
        return configured
    basis = str(cfg.get("comparison_basis", ""))
    if basis and basis not in {"exact_day", "two_prong", "hourly"}:
        return basis
    return "average_weekday" if str(observed_field).upper().startswith("AAWDT") else "average_day"


def _run_annual_average_tier(
    config: dict[str, Any],
    run_dir: Path,
    usable: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    usable = usable[usable["observed_count"] > 0].copy()
    if usable.empty:
        status = {"status": "skipped", "reason": "No screenlines have positive annual-average observed counts"}
        return pd.DataFrame(), pd.DataFrame(), status

    paths = build_od_screenline_paths(config, run_dir)
    observed_field = str(usable["observed_count_field"].dropna().astype(str).iloc[0]) if "observed_count_field" in usable else "observed_count"
    comparison_basis = _annual_comparison_basis(config, observed_field)
    comparison_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    methods = list(config.get("methods", []))
    for method in progress_iter(methods, desc="AADT annual tier", total=len(methods), unit="method"):
        sample_path = run_dir / "samples" / f"{method}_synthetic.csv"
        if not sample_path.exists():
            continue
        synthetic = pd.read_csv(sample_path, low_memory=False)
        counts = _synthetic_long_term_average_screenline_counts(synthetic, paths, config, comparison_basis)
        temporal_factor = _date_strata_expansion_from_synthetic(synthetic, config, comparison_basis)
        counts = _apply_date_strata_expansion(counts, temporal_factor)
        counts, expansion_factor = _apply_sample_expansion(counts, run_dir, str(method))
        counts["comparison_basis"] = comparison_basis
        merged = usable[["screenline_id", "tract_a", "tract_b", "observed_count", "observed_count_field", "station_count"]].merge(
            counts[["screenline_id", "synthetic_count", "comparison_basis", "temporal_expansion_factor"]], on="screenline_id", how="left"
        )
        merged["synthetic_count"] = merged["synthetic_count"].fillna(0.0)
        merged["comparison_basis"] = merged["comparison_basis"].fillna(comparison_basis)
        merged["method"] = method
        merged["sample_expansion_factor"] = expansion_factor
        merged["temporal_expansion_factor"] = merged["temporal_expansion_factor"].fillna(temporal_factor)
        merged["validation_tier"] = "annual_average"
        merged["residual"] = merged["synthetic_count"] - merged["observed_count"]
        merged["residual_pct_guarded"] = merged["residual"] / merged["observed_count"].clip(lower=1.0)
        comparison_rows.append(merged)
        metrics = compute_aadt_metrics(merged["observed_count"].to_numpy(), merged["synthetic_count"].to_numpy())
        metrics["screenlines_compared"] = metrics.pop("screenlines", 0)
        metrics.update(
            {
                "method": method,
                "validation_tier": "annual_average",
                "comparison_basis": comparison_basis,
                "observed_count_field": observed_field,
                "observed_count_temporal_label": temporal_label(config, observed_field),
                "validation_mode": _validation_mode(config),
                "sample_expansion_factor": expansion_factor,
                "temporal_expansion_factor": temporal_factor,
                "effective_expansion_factor": expansion_factor * temporal_factor,
            }
        )
        summary_rows.append(metrics)

    if not comparison_rows:
        status = {"status": "skipped", "reason": "No synthetic samples were available for annual-average validation"}
        return pd.DataFrame(), pd.DataFrame(), status

    comparisons = pd.concat(comparison_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    comparisons.to_parquet(run_dir / "metrics" / "aadt_annual_screenline_comparisons.parquet")
    summary.to_csv(run_dir / "tables" / "aadt_annual_validation_summary.csv", index=False)
    residuals = comparisons.assign(abs_residual=lambda d: d["residual"].abs()).sort_values("abs_residual", ascending=False)
    residuals.head(20).to_csv(run_dir / "tables" / "aadt_annual_top_residual_screenlines.csv", index=False)
    status = {
        "status": "ok",
        "screenlines_compared": int(comparisons["screenline_id"].nunique()),
        "comparison_basis": comparison_basis,
    }
    return comparisons, summary, status


def _run_hourly_tmas_tier(
    config: dict[str, Any],
    run_dir: Path,
    usable: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        observed_hourly = build_observed_hourly_screenline_counts(config, run_dir, usable)
    except (FileNotFoundError, ValueError) as exc:
        status = {
            "status": "skipped",
            "reason": str(exc),
            "hourly_counts": _hourly_count_config(config),
        }
        return pd.DataFrame(), pd.DataFrame(), status
    if observed_hourly.empty:
        status = {"status": "skipped", "reason": "Hourly FHWA TMAS counts did not match any screenline station shares"}
        return pd.DataFrame(), pd.DataFrame(), status

    paths = build_od_screenline_paths(config, run_dir)
    comparison_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    methods = list(config.get("methods", []))
    for method in progress_iter(methods, desc="AADT hourly tier", total=len(methods), unit="method"):
        sample_path = run_dir / "samples" / f"{method}_synthetic.csv"
        if not sample_path.exists():
            continue
        synthetic = pd.read_csv(sample_path, low_memory=False)
        counts = _synthetic_hourly_screenline_counts(synthetic, paths, config)
        temporal_factor = _date_strata_expansion_from_counts(counts, config)
        counts = _apply_date_strata_expansion(counts, temporal_factor)
        counts, expansion_factor = _apply_sample_expansion(counts, run_dir, str(method))
        if counts.empty:
            continue
        counts.to_parquet(run_dir / "metrics" / f"{method}_hourly_virtual_screenline_counts.parquet")
        dates = pd.to_datetime(counts["trip_date"], errors="coerce").dt.normalize().dropna().unique()
        observed = observed_hourly[observed_hourly["trip_date"].isin(dates)].copy()
        if observed.empty:
            continue
        merged = observed.merge(
            counts[["screenline_id", "trip_date", "hour", "synthetic_count"]],
            on=["screenline_id", "trip_date", "hour"],
            how="left",
        )
        merged["synthetic_count"] = merged["synthetic_count"].fillna(0.0)
        merged = merged.merge(usable[["screenline_id", "tract_a", "tract_b", "station_count"]], on="screenline_id", how="left")
        merged["method"] = method
        merged["sample_expansion_factor"] = expansion_factor
        merged["temporal_expansion_factor"] = temporal_factor
        merged["validation_tier"] = "hourly_tmas_scaled"
        merged["comparison_basis"] = "hourly_exact_time_scaled_by_annual_share"
        merged["residual"] = merged["synthetic_count"] - merged["observed_count"]
        merged["residual_pct_guarded"] = merged["residual"] / merged["observed_count"].clip(lower=1.0)
        comparison_rows.append(merged)
        metrics = compute_aadt_metrics(merged["observed_count"].to_numpy(), merged["synthetic_count"].to_numpy())
        metrics["screenline_hours_compared"] = metrics.pop("screenlines", 0)
        metrics.update(
            {
                "method": method,
                "validation_tier": "hourly_tmas_scaled",
                "comparison_basis": "hourly_exact_time_scaled_by_annual_share",
                "observed_count_field": "FHWA_TMAS_hourly_scaled",
                "observed_count_temporal_label": "FHWA TMAS hourly counts scaled by annual station share of screenline volume",
                "validation_mode": _validation_mode(config),
                "unique_screenlines": int(merged["screenline_id"].nunique()),
                "unique_dates": int(pd.to_datetime(merged["trip_date"]).nunique()),
                "unique_hours": int(merged["hour"].nunique()),
                "mean_annual_share_observed": float(merged["annual_share_observed"].mean()) if "annual_share_observed" in merged else float("nan"),
                "sample_expansion_factor": expansion_factor,
                "temporal_expansion_factor": temporal_factor,
                "effective_expansion_factor": expansion_factor * temporal_factor,
            }
        )
        summary_rows.append(metrics)

    if not comparison_rows:
        status = {"status": "skipped", "reason": "No comparable synthetic hourly crossings and observed TMAS cells were available"}
        return pd.DataFrame(), pd.DataFrame(), status

    comparisons = pd.concat(comparison_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    comparisons.to_parquet(run_dir / "metrics" / "aadt_hourly_screenline_comparisons.parquet")
    summary.to_csv(run_dir / "tables" / "aadt_hourly_validation_summary.csv", index=False)
    residuals = comparisons.assign(abs_residual=lambda d: d["residual"].abs()).sort_values("abs_residual", ascending=False)
    residuals.head(20).to_csv(run_dir / "tables" / "aadt_hourly_top_residual_screenlines.csv", index=False)
    unique_cells = comparisons[["screenline_id", "trip_date", "hour"]].drop_duplicates()
    status = {
        "status": "ok",
        "screenlines_compared": int(comparisons["screenline_id"].nunique()),
        "screenline_hours_compared": int(len(unique_cells)),
        "hourly_counts": _hourly_count_config(config),
        "scaling_method": "FHWA station hourly count divided by covered annual station share of the screenline",
    }
    return comparisons, summary, status


def run_aadt_validation(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    import geopandas as gpd

    run_dir = Path(run_dir)
    ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir / "tables")
    mode = _validation_mode(config)
    screenlines_path = run_dir / "geo" / "screenlines.parquet"
    if not screenlines_path.exists():
        status = build_screenlines(config, run_dir)
        if status.get("status") != "ok":
            result = {
                "status": "skipped",
                "reason": "screenlines.parquet is unavailable",
                "screenline_status": status,
                "validation_mode": mode,
            }
            write_json(result, run_dir / "metrics" / "aadt_validation_skipped.json")
            return result
    screenlines = gpd.read_parquet(screenlines_path)
    usable = screenlines[screenlines["station_count"] > 0].copy()
    if usable.empty:
        result = {"status": "skipped", "reason": "No screenlines have mapped stations", "validation_mode": mode}
        write_json(result, run_dir / "metrics" / "aadt_validation_skipped.json")
        return result

    comparison_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    tier_status: dict[str, Any] = {}

    if mode in {"two_prong", "annual_average", "average_day", "average_weekday"}:
        annual_comparisons, annual_summary, annual_status = _run_annual_average_tier(config, run_dir, usable)
        tier_status["annual_average"] = annual_status
        if not annual_comparisons.empty:
            comparison_frames.append(annual_comparisons)
            summary_frames.append(annual_summary)
            _plot_aadt_outputs(run_dir, annual_comparisons, usable)

    if mode == "two_prong":
        hourly_comparisons, hourly_summary, hourly_status = _run_hourly_tmas_tier(config, run_dir, usable)
        tier_status["hourly_tmas_scaled"] = hourly_status
        if not hourly_comparisons.empty:
            comparison_frames.append(hourly_comparisons)
            summary_frames.append(hourly_summary)

    if mode == "exact_day":
        try:
            observed_daily = build_observed_exact_day_screenline_counts(config, run_dir)
        except (FileNotFoundError, ValueError) as exc:
            result = {
                "status": "skipped",
                "reason": str(exc),
                "validation_mode": mode,
                "required_daily_count_schema": {
                    "file": "aadt_validation.daily_counts.file",
                    "station_id_column": "station id matching MDOT LOCATION_ID by default",
                    "date_column": "exact station count date",
                    "count_column": "observed vehicle count for that station-date",
                },
            }
            write_json(result, run_dir / "metrics" / "aadt_validation_skipped.json")
            return result
        if observed_daily.empty:
            result = {"status": "skipped", "reason": "Daily station counts did not match any mapped screenline stations", "validation_mode": mode}
            write_json(result, run_dir / "metrics" / "aadt_validation_skipped.json")
            return result
        virtual = build_virtual_screenline_counts(config, run_dir)
        comparison_rows: list[pd.DataFrame] = []
        summary_rows: list[dict[str, Any]] = []
        virtual_items = list(virtual.items())
        for method, counts in progress_iter(virtual_items, desc="AADT exact-day tier", total=len(virtual_items), unit="method"):
            if counts.empty or "trip_date" not in counts.columns:
                continue
            counts = counts.copy()
            counts["trip_date"] = pd.to_datetime(counts["trip_date"], errors="coerce").dt.normalize()
            dates = counts["trip_date"].dropna().unique()
            observed = observed_daily[observed_daily["trip_date"].isin(dates)].copy()
            if observed.empty:
                continue
            merged = observed.merge(
                counts[["screenline_id", "trip_date", "synthetic_count", "comparison_basis"]],
                on=["screenline_id", "trip_date"],
                how="left",
            )
            merged["synthetic_count"] = merged["synthetic_count"].fillna(0.0)
            merged["comparison_basis"] = merged["comparison_basis"].fillna("exact_day")
            merged = merged.merge(usable[["screenline_id", "tract_a", "tract_b", "station_count"]], on="screenline_id", how="left")
            temporal_factor = float(counts["temporal_expansion_factor"].dropna().iloc[0]) if "temporal_expansion_factor" in counts and counts["temporal_expansion_factor"].notna().any() else _date_strata_expansion_from_counts(counts, config)
            merged["method"] = method
            merged["sample_expansion_factor"] = _sample_expansion_factor(run_dir, str(method))
            merged["temporal_expansion_factor"] = temporal_factor
            merged["validation_tier"] = "exact_day"
            merged["residual"] = merged["synthetic_count"] - merged["observed_count"]
            merged["residual_pct_guarded"] = merged["residual"] / merged["observed_count"].clip(lower=1.0)
            comparison_rows.append(merged)
            metrics = compute_aadt_metrics(merged["observed_count"].to_numpy(), merged["synthetic_count"].to_numpy())
            metrics["screenline_dates_compared"] = metrics.pop("screenlines", 0)
            metrics.update(
                {
                    "method": method,
                    "validation_tier": "exact_day",
                    "comparison_basis": "exact_day",
                    "observed_count_field": "daily_station_count",
                    "observed_count_temporal_label": "Exact station-day counts matched to reconstructed synthetic trip dates",
                    "validation_mode": mode,
                    "sample_expansion_factor": _sample_expansion_factor(run_dir, str(method)),
                    "temporal_expansion_factor": temporal_factor,
                    "effective_expansion_factor": _sample_expansion_factor(run_dir, str(method)) * temporal_factor,
                    "unique_screenlines": int(merged["screenline_id"].nunique()),
                    "unique_dates": int(pd.to_datetime(merged["trip_date"]).nunique()),
                }
            )
            summary_rows.append(metrics)
        if comparison_rows:
            exact_comparisons = pd.concat(comparison_rows, ignore_index=True)
            exact_summary = pd.DataFrame(summary_rows)
            comparison_frames.append(exact_comparisons)
            summary_frames.append(exact_summary)
            tier_status["exact_day"] = {"status": "ok", "screenlines_compared": int(exact_comparisons["screenline_id"].nunique())}
            _plot_aadt_outputs(run_dir, exact_comparisons, usable)
        else:
            tier_status["exact_day"] = {"status": "skipped", "reason": "No comparable synthetic and observed exact-day cells were available"}

    if not comparison_frames:
        result = {
            "status": "skipped",
            "reason": "No comparable synthetic and observed screenline counts were available",
            "validation_mode": mode,
            "tiers": tier_status,
        }
        write_json(result, run_dir / "metrics" / "aadt_validation_skipped.json")
        return result

    comparisons = pd.concat(comparison_frames, ignore_index=True, sort=False)
    summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    summary.to_csv(run_dir / "tables" / "aadt_validation_summary.csv", index=False)
    residuals = comparisons.assign(abs_residual=lambda d: d["residual"].abs()).sort_values("abs_residual", ascending=False)
    residuals.head(20).to_csv(run_dir / "tables" / "aadt_top_residual_screenlines.csv", index=False)
    comparisons.to_parquet(run_dir / "metrics" / "aadt_screenline_comparisons.parquet")
    result = {
        "status": "ok",
        "validation_mode": mode,
        "tiers": tier_status,
        "screenlines_compared": int(comparisons["screenline_id"].nunique()),
        "methods": sorted(comparisons["method"].dropna().astype(str).unique()),
        "temporal_framing": "Two-prong annual-average plus FHWA TMAS hourly validation." if mode == "two_prong" else "AADT screenline validation.",
        "trip_week_origin_date": config.get("aadt_validation", {}).get("trip_week_origin_date", "2017-01-01"),
        "tdate_dow_encoding": config.get("aadt_validation", {}).get("tdate_dow_encoding", "iso_monday_1"),
        "daily_counts": _daily_count_config(config) if mode == "exact_day" else None,
        "hourly_counts": _hourly_count_config(config) if mode == "two_prong" else None,
        "survey_period": config.get("aadt_validation", {}).get("survey_period", {}),
        "synthetic_population_basis": config.get("aadt_validation", {}).get("synthetic_population_basis", "average_day"),
        "expand_date_strata_to_average_day": config.get("aadt_validation", {}).get("expand_date_strata_to_average_day", True),
        "assignment_note": "OD-to-screenline paths use centroid-line tract crossings as a geometric proxy, not true route assignment.",
    }
    write_json(result, run_dir / "metrics" / "aadt_validation_summary.json")
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/manual_aadt")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(run_aadt_validation(config, args.run_dir))


if __name__ == "__main__":
    main()

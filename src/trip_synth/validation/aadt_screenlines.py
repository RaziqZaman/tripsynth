from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trip_synth.utils.io import ensure_dir, load_yaml, read_json, write_json

from .route_assignment import screenline_id


def geospatial_available() -> bool:
    try:
        import geopandas  # noqa: F401
        import shapely  # noqa: F401

        return True
    except Exception:
        return False


def _require_geo() -> None:
    if not geospatial_available():
        raise RuntimeError(
            "GeoPandas/Shapely are required for AADT screenline validation. "
            "Install optional dependencies with: .venv/bin/python -m pip install geopandas shapely pyproj pyogrio rtree"
        )


def _clean_geoid(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(11)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("N/A", "", regex=False),
        errors="coerce",
    )


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    existing = set(columns)
    for name in candidates:
        if name in existing:
            return name
    return None


def _file_signature(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    signature: dict[str, Any] = {"path": str(p)}
    if p.exists():
        stat = p.stat()
        signature.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    else:
        signature["missing"] = True
    return signature


def _data_source_preferences() -> dict[str, Any]:
    path = Path("configs/data_sources.yaml")
    if path.exists():
        return load_yaml(path).get("mdot_aadt", {}).get("prefer_fields", {})
    return {}


def observed_count_field_priority(config: dict[str, Any]) -> list[str]:
    aadt_cfg = config.get("aadt_validation", {})
    configured = aadt_cfg.get("observed_count_field_priority")
    if configured:
        return [str(v) for v in configured]
    pref = config.get("mdot_aadt", {}).get("prefer_fields", {}) or _data_source_preferences()
    return [str(v) for v in pref.get("observed_count_priority", ["AAWDT", "AADT"])]


def choose_observed_count_field(columns: list[str], config: dict[str, Any]) -> str | None:
    return _first_existing(columns, observed_count_field_priority(config))


def comparison_basis_for_field(config: dict[str, Any], field: str) -> str:
    configured = config.get("aadt_validation", {}).get("comparison_basis")
    if configured:
        return str(configured)
    return "average_weekday" if str(field).upper().startswith("AAWDT") else "average_day"


def temporal_label(config: dict[str, Any], field: str) -> str:
    return str(
        config.get("aadt_validation", {}).get(
            "observed_count_temporal_label",
            f"Observed MDOT field {field}",
        )
    )


def _cached_points_match(points, config: dict[str, Any]) -> bool:
    if "observed_count_field" not in points.columns:
        return False
    desired = choose_observed_count_field(list(points.columns), config)
    if desired is None:
        return False
    fields = set(points["observed_count_field"].dropna().astype(str).unique())
    return fields == {desired}


def _data_limited_tract_geoids(config: dict[str, Any]) -> set[str]:
    geo = config.get("geo", {})
    filter_cfg = geo.get("tract_filter", {}) or {}
    if not filter_cfg.get("enabled", False):
        return set()
    input_csv = Path(str(config.get("input_csv", "")))
    if not input_csv.exists():
        return set()
    columns = [
        str(c)
        for c in filter_cfg.get(
            "columns",
            ["o_tract_fips", "d_tract_fips", "home_tract_fips", "work_tract_fips"],
        )
    ]
    limited_states = {str(s).zfill(2) for s in filter_cfg.get("data_limited_state_fips", [])}
    if not limited_states:
        return set()
    df = pd.read_csv(input_csv, usecols=lambda c: c in columns, dtype=str, low_memory=False)
    geoids: set[str] = set()
    for col in df.columns:
        vals = df[col].dropna().map(_clean_geoid).astype(str)
        geoids.update(v for v in vals if len(v) == 11 and v[:2] in limited_states)
    return geoids


def _tract_filter_signature(config: dict[str, Any]) -> dict[str, Any]:
    geo = config.get("geo", {})
    filter_cfg = geo.get("tract_filter", {}) or {}
    input_csv = Path(str(config.get("input_csv", "")))
    signature: dict[str, Any] = {"tract_filter": filter_cfg}
    if filter_cfg.get("enabled", False) and input_csv.exists():
        signature["input_csv"] = str(input_csv)
        signature["input_csv_size"] = input_csv.stat().st_size
        signature["input_csv_mtime_ns"] = input_csv.stat().st_mtime_ns
    return signature


def load_tracts(config: dict[str, Any], run_dir: str | Path):
    _require_geo()
    import geopandas as gpd

    run_dir = Path(run_dir)
    geo = config.get("geo", {})
    tract_files = geo.get("tract_files", [])
    signature = {
        "tract_files": [str(p) for p in tract_files],
        "crs_projected": str(geo.get("crs_projected", "EPSG:26985")),
        "tract_vintage": geo.get("tract_vintage"),
        "point_fields": "representative_and_centroid_v2",
        **_tract_filter_signature(config),
    }
    out_path = run_dir / "geo" / "tracts.parquet"
    meta_path = run_dir / "geo" / "tracts_metadata.json"
    if out_path.exists() and read_json(meta_path, {}) == signature:
        return gpd.read_parquet(out_path)

    if not tract_files:
        raise FileNotFoundError("No tract files configured under geo.tract_files")
    frames = []
    for path in tract_files:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing tract file: {p}")
        frames.append(gpd.read_file(p))
    tracts = pd.concat(frames, ignore_index=True)
    tracts = gpd.GeoDataFrame(tracts, geometry="geometry", crs=frames[0].crs)
    tracts["GEOID"] = tracts["GEOID"].map(_clean_geoid)
    tracts = tracts.drop_duplicates("GEOID").copy()
    filter_cfg = geo.get("tract_filter", {}) or {}
    if filter_cfg.get("enabled", False):
        limited_states = {str(s).zfill(2) for s in filter_cfg.get("data_limited_state_fips", [])}
        if limited_states:
            data_geoids = _data_limited_tract_geoids(config)
            state = tracts["GEOID"].astype(str).str[:2]
            tracts = tracts[(~state.isin(limited_states)) | tracts["GEOID"].isin(data_geoids)].copy()
    keep = [c for c in ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "NAMELSAD", "geometry"] if c in tracts.columns]
    tracts = tracts[keep]
    tracts = tracts.to_crs(geo.get("crs_projected", "EPSG:26985"))
    reps = tracts.geometry.representative_point()
    tracts["rep_x"] = reps.x
    tracts["rep_y"] = reps.y
    centroids = tracts.geometry.centroid
    tracts["centroid_x"] = centroids.x
    tracts["centroid_y"] = centroids.y
    ensure_dir(out_path.parent)
    tracts.to_parquet(out_path)
    write_json(signature, meta_path)
    return tracts


def load_aadt_points(config: dict[str, Any], run_dir: str | Path):
    _require_geo()
    import geopandas as gpd

    run_dir = Path(run_dir)
    out_path = run_dir / "geo" / "aadt_points.parquet"
    meta_path = run_dir / "geo" / "aadt_points_metadata.json"
    geo = config.get("geo", {})
    p = Path(geo.get("aadt_points_file", ""))
    signature = {
        "cache_version": 1,
        "aadt_points_file": _file_signature(p),
        "crs_projected": str(geo.get("crs_projected", "EPSG:26985")),
        "observed_count_field_priority": observed_count_field_priority(config),
    }
    if out_path.exists() and read_json(meta_path, {}) == signature:
        cached = gpd.read_parquet(out_path)
        if _cached_points_match(cached, config):
            return cached

    if not p.exists():
        raise FileNotFoundError(f"Missing AADT points file: {p}")
    points = gpd.read_file(p)
    if points.empty:
        raise ValueError(f"AADT points file is empty: {p}")
    if points.crs is None:
        points = points.set_crs("EPSG:4326")
    points = points.to_crs(geo.get("crs_projected", "EPSG:26985"))
    pref = config.get("mdot_aadt", {}).get("prefer_fields", {}) or _data_source_preferences()
    id_col = _first_existing(list(points.columns), pref.get("id_priority", ["LOCATION_ID", "OBJECTID"]))
    if id_col is None:
        id_col = "OBJECTID" if "OBJECTID" in points.columns else points.columns[0]
    observed_col = choose_observed_count_field(list(points.columns), config)
    if observed_col is None:
        raise ValueError(
            "AADT points do not include any configured observed-count field from "
            f"{observed_count_field_priority(config)}"
        )
    points["station_id"] = points[id_col].astype(str)
    points["observed_count"] = _safe_numeric(points[observed_col])
    points["observed_count_field"] = observed_col
    points["observed_count_temporal_label"] = temporal_label(config, observed_col)
    points["comparison_basis"] = comparison_basis_for_field(config, observed_col)
    for count_col in [c for c in observed_count_field_priority(config) if c in points.columns]:
        points[count_col] = _safe_numeric(points[count_col])
    if "AAWDT" in points.columns:
        points["AAWDT"] = _safe_numeric(points["AAWDT"])
    if "AADT" in points.columns:
        points["AADT"] = _safe_numeric(points["AADT"])
    points = points[points.geometry.notna() & points["observed_count"].notna()].copy()
    points.to_parquet(out_path)
    write_json(signature, meta_path)
    return points


def build_tract_adjacency(config: dict[str, Any], run_dir: str | Path):
    _require_geo()
    import geopandas as gpd

    run_dir = Path(run_dir)
    out_path = run_dir / "geo" / "tract_adjacency.parquet"
    meta_path = run_dir / "geo" / "tract_adjacency_metadata.json"
    tracts = load_tracts(config, run_dir)
    min_shared = float(config.get("aadt_validation", {}).get("min_shared_boundary_m", 1.0))
    signature = {
        "cache_version": 1,
        "tracts_metadata": read_json(run_dir / "geo" / "tracts_metadata.json", {}),
        "min_shared_boundary_m": min_shared,
    }
    if out_path.exists() and read_json(meta_path, {}) == signature:
        return gpd.read_parquet(out_path)
    left = tracts[["GEOID", "geometry"]].rename(columns={"GEOID": "tract_a"})
    right = tracts[["GEOID", "geometry"]].rename(columns={"GEOID": "tract_b"})
    joined = gpd.sjoin(left, right, how="inner", predicate="touches")
    rows: list[dict[str, Any]] = []
    right_geom = right.set_index("tract_b")["geometry"]
    left_geom = left.set_index("tract_a")["geometry"]
    for tract_a, tract_b in joined[["tract_a", "tract_b"]].itertuples(index=False):
        if tract_a >= tract_b:
            continue
        geom_a = left_geom.loc[tract_a]
        geom_b = right_geom.loc[tract_b]
        shared = geom_a.boundary.intersection(geom_b.boundary)
        length = float(shared.length) if not shared.is_empty else 0.0
        if length <= min_shared:
            continue
        rows.append(
            {
                "screenline_id": screenline_id(tract_a, tract_b),
                "tract_a": tract_a,
                "tract_b": tract_b,
                "shared_boundary_m": length,
                "geometry": shared,
            }
        )
    adjacency = gpd.GeoDataFrame(rows, geometry="geometry", crs=tracts.crs)
    adjacency.to_parquet(out_path)
    write_json(signature, meta_path)
    return adjacency


def _count_fields(points) -> tuple[str, str | None]:
    if "observed_count_field" in points.columns and points["observed_count_field"].notna().any():
        observed = str(points["observed_count_field"].dropna().astype(str).iloc[0])
    else:
        observed = "observed_count"
    fallback = "AADT" if observed.upper().startswith("AAWDT") and "AADT" in points.columns else None
    return observed, fallback


def _station_boundary_membership(
    station_geom,
    boundary_geom,
    tract_a_centroid,
    tract_b_centroid,
    ratio_threshold: float,
) -> tuple[bool, float, float, float]:
    boundary_distance = float(station_geom.distance(boundary_geom))
    nearest_centroid_distance = float(
        min(station_geom.distance(tract_a_centroid), station_geom.distance(tract_b_centroid))
    )
    ratio = boundary_distance / nearest_centroid_distance if nearest_centroid_distance > 0 else np.inf
    return ratio < ratio_threshold, boundary_distance, nearest_centroid_distance, ratio


def build_screenlines(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "geo")
    geo = config.get("geo", {})
    missing = [p for p in geo.get("tract_files", []) if not Path(p).exists()]
    points = Path(geo.get("aadt_points_file", ""))
    if str(points) and not points.exists():
        missing.append(str(points))
    if missing:
        result = {
            "status": "skipped",
            "reason": "Missing external geospatial files",
            "missing": missing,
            "method_note": "Screenline assignment is a geometric proxy, not true route assignment.",
        }
        write_json(result, run_dir / "geo" / "screenlines_skipped.json")
        return result
    if not geospatial_available():
        result = {
            "status": "skipped",
            "reason": "GeoPandas/Shapely dependencies are not installed",
            "install_hint": "Install optional dependencies with pip install -e '.[geo]'.",
            "method_note": "Screenline assignment is a geometric proxy, not true route assignment.",
        }
        write_json(result, run_dir / "geo" / "screenlines_skipped.json")
        return result

    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracts = load_tracts(config, run_dir)
    adjacency = build_tract_adjacency(config, run_dir)
    points_gdf = load_aadt_points(config, run_dir)
    if adjacency.empty:
        result = {"status": "skipped", "reason": "No tract adjacencies with shared boundaries were found"}
        write_json(result, run_dir / "geo" / "screenlines_skipped.json")
        return result

    aadt_cfg = config.get("aadt_validation", {})
    buffers = aadt_cfg.get("screenline_buffer_m", [100, 250, 500])
    if isinstance(buffers, (int, float)):
        buffers = [float(buffers)]
    buffers = [float(v) for v in buffers]
    total_adjacent_pairs = int(len(adjacency))
    candidate_scope = str(aadt_cfg.get("screenline_candidate_scope", "all"))
    membership_rule = str(
        aadt_cfg.get(
            "station_membership_rule",
            "boundary_closer_than_nearest_tract_centroid",
        )
    )
    boundary_ratio_threshold = float(aadt_cfg.get("boundary_distance_ratio_threshold", 1.0))
    screenline_signature = {
        "cache_version": 1,
        "tracts_metadata": read_json(run_dir / "geo" / "tracts_metadata.json", {}),
        "tract_adjacency_metadata": read_json(run_dir / "geo" / "tract_adjacency_metadata.json", {}),
        "aadt_points_metadata": read_json(run_dir / "geo" / "aadt_points_metadata.json", {}),
        "screenline_buffer_m": buffers,
        "screenline_candidate_scope": candidate_scope,
        "station_membership_rule": membership_rule,
        "boundary_distance_ratio_threshold": boundary_ratio_threshold,
        "observed_count_field_priority": observed_count_field_priority(config),
    }
    screenline_path = run_dir / "geo" / "screenlines.parquet"
    station_map_path = run_dir / "geo" / "screenline_station_map.parquet"
    summary_path = run_dir / "geo" / "screenline_summary.json"
    meta_path = run_dir / "geo" / "screenlines_metadata.json"
    if (
        screenline_path.exists()
        and station_map_path.exists()
        and summary_path.exists()
        and read_json(meta_path, {}) == screenline_signature
    ):
        return read_json(summary_path, {})

    if candidate_scope == "observed_station_buffer" and not adjacency.empty and not points_gdf.empty:
        max_buffer = max(buffers) if buffers else 0.0
        adjacency_sindex = adjacency.sindex
        candidate_idx: set[int] = set()
        for geom in points_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            candidate_idx.update(adjacency_sindex.query(geom.buffer(max_buffer), predicate="intersects"))
        adjacency = adjacency.iloc[sorted(candidate_idx)].copy()
    elif candidate_scope != "all":
        raise ValueError(f"Unsupported screenline_candidate_scope: {candidate_scope}")
    observed_field, fallback_field = _count_fields(points_gdf)
    sindex = points_gdf.sindex
    tract_centroids = tracts.set_index("GEOID")[["centroid_x", "centroid_y"]]
    screen_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []

    for row in adjacency.itertuples(index=False):
        selected_idx: list[int] = []
        used_buffer = np.nan
        boundary = row.geometry
        for buffer_m in buffers:
            area = boundary.buffer(buffer_m)
            candidates = list(sindex.query(area, predicate="intersects"))
            if candidates:
                selected_idx = candidates
                used_buffer = buffer_m
                break
        stations = points_gdf.iloc[selected_idx].copy() if selected_idx else points_gdf.iloc[[]].copy()
        if not stations.empty:
            stations = stations.drop_duplicates("station_id")
            if membership_rule == "boundary_closer_than_nearest_tract_centroid":
                from shapely.geometry import Point

                tract_a_centroid = Point(
                    float(tract_centroids.loc[row.tract_a, "centroid_x"]),
                    float(tract_centroids.loc[row.tract_a, "centroid_y"]),
                )
                tract_b_centroid = Point(
                    float(tract_centroids.loc[row.tract_b, "centroid_x"]),
                    float(tract_centroids.loc[row.tract_b, "centroid_y"]),
                )
                metrics = stations.geometry.apply(
                    lambda geom: _station_boundary_membership(
                        geom,
                        boundary,
                        tract_a_centroid,
                        tract_b_centroid,
                        boundary_ratio_threshold,
                    )
                )
                stations["boundary_distance_m"] = [item[1] for item in metrics]
                stations["nearest_tract_centroid_distance_m"] = [item[2] for item in metrics]
                stations["boundary_centroid_distance_ratio"] = [item[3] for item in metrics]
                stations["boundary_centroid_margin_m"] = (
                    stations["nearest_tract_centroid_distance_m"] * boundary_ratio_threshold
                    - stations["boundary_distance_m"]
                )
                stations = stations[[item[0] for item in metrics]].copy()
            elif membership_rule == "buffer_only":
                stations["boundary_distance_m"] = stations.geometry.distance(boundary)
                stations["nearest_tract_centroid_distance_m"] = np.nan
                stations["boundary_centroid_margin_m"] = np.nan
                stations["boundary_centroid_distance_ratio"] = np.nan
            else:
                raise ValueError(f"Unsupported station_membership_rule: {membership_rule}")
        observed_values = stations[observed_field] if observed_field in stations.columns else stations["observed_count"]
        if not stations.empty and fallback_field and observed_values.notna().sum() == 0:
            observed_values = stations[fallback_field]
            count_field = fallback_field
        else:
            count_field = observed_field
        total_observed = float(pd.to_numeric(observed_values, errors="coerce").dropna().sum()) if not stations.empty else 0.0
        screen_rows.append(
            {
                "screenline_id": row.screenline_id,
                "tract_a": row.tract_a,
                "tract_b": row.tract_b,
                "station_count": int(len(stations)),
                "observed_count": total_observed,
                "observed_count_field": count_field,
                "observed_count_temporal_label": temporal_label(config, count_field),
                "comparison_basis": comparison_basis_for_field(config, count_field),
                "buffer_m": used_buffer,
                "shared_boundary_m": float(row.shared_boundary_m),
                "station_membership_rule": membership_rule,
                "boundary_distance_ratio_threshold": boundary_ratio_threshold,
                "geometry_confidence": "boundary_buffer_station" if len(stations) else "no_station_in_buffer",
                "method_note": "geometric proxy, not true route assignment",
                "geometry": boundary,
            }
        )
        for station in stations.itertuples(index=False):
            value = getattr(station, count_field) if hasattr(station, count_field) else np.nan
            station_rows.append(
                {
                    "screenline_id": row.screenline_id,
                    "station_id": str(station.station_id),
                    "observed_count": float(value) if pd.notna(value) else np.nan,
                    "observed_count_field": count_field,
                    "observed_count_temporal_label": temporal_label(config, count_field),
                    "comparison_basis": comparison_basis_for_field(config, count_field),
                    "buffer_m": used_buffer,
                    "boundary_distance_m": float(getattr(station, "boundary_distance_m", np.nan)),
                    "nearest_tract_centroid_distance_m": float(
                        getattr(station, "nearest_tract_centroid_distance_m", np.nan)
                    ),
                    "boundary_centroid_margin_m": float(getattr(station, "boundary_centroid_margin_m", np.nan)),
                    "boundary_centroid_distance_ratio": float(
                        getattr(station, "boundary_centroid_distance_ratio", np.nan)
                    ),
                    "station_membership_rule": membership_rule,
                    "boundary_distance_ratio_threshold": boundary_ratio_threshold,
                }
            )

    screenlines = gpd.GeoDataFrame(screen_rows, geometry="geometry", crs=tracts.crs)
    station_map = pd.DataFrame(station_rows)
    screenlines.to_parquet(run_dir / "geo" / "screenlines.parquet")
    station_map.to_parquet(run_dir / "geo" / "screenline_station_map.parquet")

    ensure_dir(run_dir / "figures" / "poster")
    plt.figure(figsize=(8, 4.5))
    screenlines["station_count"].plot.hist(bins=30)
    plt.xlabel("Stations per screenline")
    plt.ylabel("Screenlines")
    plt.title("AADT station count per tract-boundary screenline")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "poster" / "screenline_station_count_distribution.png", dpi=250)
    plt.close()

    example = screenlines.sort_values("station_count", ascending=False).head(1)
    if not example.empty:
        ax = tracts[tracts["GEOID"].isin([example.iloc[0]["tract_a"], example.iloc[0]["tract_b"]])].plot(
            figsize=(7, 7), facecolor="none", edgecolor="#555555"
        )
        example.plot(ax=ax, color="#de2d26", linewidth=3)
        nearby = station_map[station_map["screenline_id"] == example.iloc[0]["screenline_id"]]
        if not nearby.empty:
            points_gdf[points_gdf["station_id"].isin(nearby["station_id"])].plot(ax=ax, color="#2c7fb8", markersize=20)
        ax.set_axis_off()
        ax.set_title("Example tract-boundary screenline")
        plt.tight_layout()
        plt.savefig(run_dir / "figures" / "poster" / "screenline_map_example.png", dpi=250)
        plt.close()

    result = {
        "status": "ok",
        "tracts": int(len(tracts)),
        "adjacent_pairs": int(total_adjacent_pairs),
        "candidate_screenlines": int(len(adjacency)),
        "screenlines": int(len(screenlines)),
        "screenlines_with_stations": int((screenlines["station_count"] > 0).sum()),
        "stations_mapped": int(len(station_map)),
        "observed_count_field": observed_field,
        "observed_count_temporal_label": temporal_label(config, observed_field),
        "comparison_basis": comparison_basis_for_field(config, observed_field),
        "station_membership_rule": membership_rule,
        "boundary_distance_ratio_threshold": boundary_ratio_threshold,
        "screenline_candidate_scope": candidate_scope,
        "survey_period": config.get("aadt_validation", {}).get("survey_period", {}),
        "segments_note": "MDOT segment layer was optional; current downloaded layer may be empty.",
        "method_note": "Screenline assignment is a geometric proxy, not true route assignment.",
    }
    write_json(result, run_dir / "geo" / "screenline_summary.json")
    write_json(screenline_signature, run_dir / "geo" / "screenlines_metadata.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/manual_screenlines")
    args = parser.parse_args()
    config = load_yaml(args.config)
    result = build_screenlines(config, args.run_dir)
    print(result)

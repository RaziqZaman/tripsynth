from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trip_synth.utils.io import ensure_dir, load_yaml, write_json

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


def _data_source_preferences() -> dict[str, Any]:
    path = Path("configs/data_sources.yaml")
    if path.exists():
        return load_yaml(path).get("mdot_aadt", {}).get("prefer_fields", {})
    return {}


def load_tracts(config: dict[str, Any], run_dir: str | Path):
    _require_geo()
    import geopandas as gpd

    run_dir = Path(run_dir)
    out_path = run_dir / "geo" / "tracts.parquet"
    if out_path.exists():
        return gpd.read_parquet(out_path)

    geo = config.get("geo", {})
    tract_files = geo.get("tract_files", [])
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
    keep = [c for c in ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "NAMELSAD", "geometry"] if c in tracts.columns]
    tracts = tracts[keep]
    tracts = tracts.to_crs(geo.get("crs_projected", "EPSG:26985"))
    reps = tracts.geometry.representative_point()
    tracts["rep_x"] = reps.x
    tracts["rep_y"] = reps.y
    ensure_dir(out_path.parent)
    tracts.to_parquet(out_path)
    return tracts


def load_aadt_points(config: dict[str, Any], run_dir: str | Path):
    _require_geo()
    import geopandas as gpd

    run_dir = Path(run_dir)
    out_path = run_dir / "geo" / "aadt_points.parquet"
    if out_path.exists():
        return gpd.read_parquet(out_path)

    geo = config.get("geo", {})
    p = Path(geo.get("aadt_points_file", ""))
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
    observed_col = _first_existing(
        list(points.columns), pref.get("observed_count_priority", ["AAWDT", "AADT"])
    )
    if observed_col is None:
        raise ValueError("AADT points do not include any configured observed-count field")
    points["station_id"] = points[id_col].astype(str)
    points["observed_count"] = _safe_numeric(points[observed_col])
    points["observed_count_field"] = observed_col
    if "AAWDT" in points.columns:
        points["AAWDT"] = _safe_numeric(points["AAWDT"])
    if "AADT" in points.columns:
        points["AADT"] = _safe_numeric(points["AADT"])
    points = points[points.geometry.notna() & points["observed_count"].notna()].copy()
    points.to_parquet(out_path)
    return points


def build_tract_adjacency(config: dict[str, Any], run_dir: str | Path):
    _require_geo()
    import geopandas as gpd

    run_dir = Path(run_dir)
    out_path = run_dir / "geo" / "tract_adjacency.parquet"
    if out_path.exists():
        return gpd.read_parquet(out_path)
    tracts = load_tracts(config, run_dir)
    left = tracts[["GEOID", "geometry"]].rename(columns={"GEOID": "tract_a"})
    right = tracts[["GEOID", "geometry"]].rename(columns={"GEOID": "tract_b"})
    joined = gpd.sjoin(left, right, how="inner", predicate="touches")
    rows: list[dict[str, Any]] = []
    right_geom = right.set_index("tract_b")["geometry"]
    left_geom = left.set_index("tract_a")["geometry"]
    min_shared = float(config.get("aadt_validation", {}).get("min_shared_boundary_m", 1.0))
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
    return adjacency


def _count_fields(points) -> tuple[str, str | None]:
    observed = "AAWDT" if "AAWDT" in points.columns and points["AAWDT"].notna().any() else "observed_count"
    fallback = "AADT" if observed == "AAWDT" and "AADT" in points.columns else None
    return observed, fallback


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

    buffers = config.get("aadt_validation", {}).get("screenline_buffer_m", [100, 250, 500])
    if isinstance(buffers, (int, float)):
        buffers = [float(buffers)]
    buffers = [float(v) for v in buffers]
    observed_field, fallback_field = _count_fields(points_gdf)
    sindex = points_gdf.sindex
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
        observed_values = stations[observed_field] if observed_field in stations.columns else stations["observed_count"]
        if fallback_field and observed_values.notna().sum() == 0:
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
                "buffer_m": used_buffer,
                "shared_boundary_m": float(row.shared_boundary_m),
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
                    "buffer_m": used_buffer,
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
        "adjacent_pairs": int(len(adjacency)),
        "screenlines": int(len(screenlines)),
        "screenlines_with_stations": int((screenlines["station_count"] > 0).sum()),
        "stations_mapped": int(len(station_map)),
        "observed_count_field": observed_field,
        "segments_note": "MDOT segment layer was optional; current downloaded layer may be empty.",
        "method_note": "Screenline assignment is a geometric proxy, not true route assignment.",
    }
    write_json(result, run_dir / "geo" / "screenline_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--run-dir", default="outputs/runs/manual_screenlines")
    args = parser.parse_args()
    config = load_yaml(args.config)
    result = build_screenlines(config, args.run_dir)
    print(result)

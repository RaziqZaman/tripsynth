#!/usr/bin/env python3
"""Generate reproducible OD-pair tract-screenline example maps.

The maps use the frozen paper run.  OD pairs are sampled from observed survey
support, while the map itself reconstructs every shared tract boundary crossed
by the same representative-point line used by the validation pipeline.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points

from trip_synth.validation.aadt_screenlines import screenline_id
from trip_synth.validation.aadt_validation import _ordered_crossed_tracts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs" / "runs" / "paper_wctr_1m_vae_sweep"
DEFAULT_OUTPUT = ROOT / "screenline_examples"
DEFAULT_SURVEY = ROOT / "05_transformed_survey.csv"
DEFAULT_FHWA_METADATA = (
    ROOT / "data" / "external" / "mdot_daily_counts" / "fhwa_tmas_station_metadata_2017_2019.csv"
)
DEFAULT_FHWA_HOURLY = ROOT / "data" / "external" / "mdot_daily_counts" / "station_hourly_counts.csv"

ORIGIN_FACE = "#f4a261"
ORIGIN_EDGE = "#9c421f"
DEST_FACE = "#72b7d2"
DEST_EDGE = "#155f92"
CONTEXT_EDGE = "#c7c9c4"
PATH_FACE = "#e9f1f5"
UNUSED_MDOT = "#868686"
UNUSED_FHWA = "#5f5f5f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY)
    parser.add_argument("--fhwa-metadata", type=Path, default=DEFAULT_FHWA_METADATA)
    parser.add_argument("--fhwa-hourly", type=Path, default=DEFAULT_FHWA_HOURLY)
    parser.add_argument("--n-pairs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--selection-profile",
        choices=["observed_nonempty", "exactly_two_stationed"],
        default="observed_nonempty",
        help=(
            "OD eligibility rule. exactly_two_stationed requires exactly two reconstructed shared "
            "boundaries, at least one mapped AADT station on both, and multiple stations on one or both."
        ),
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=None,
        help="Optional prior manifest.csv whose directed OD pairs must be excluded.",
    )
    parser.add_argument("--background-color", default="#FFFFFF")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _clean_geoids(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return values.str.zfill(11)


def _check_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    owned = list(output_dir.glob("screenline_*.png"))
    owned += [
        path
        for path in [
            output_dir / "manifest.csv",
            output_dir / "screenlines.csv",
            output_dir / "station_assignments.csv",
            output_dir / "README.md",
            output_dir / "generation_summary.json",
        ]
        if path.exists()
    ]
    contact_dir = output_dir / "contact_sheets"
    if contact_dir.exists():
        owned.extend(contact_dir.glob("contact_sheet_*.png"))
    if owned and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains generated files; pass --overwrite to replace only generator-owned files"
        )
    if overwrite:
        for path in owned:
            path.unlink()
    contact_dir.mkdir(parents=True, exist_ok=True)


def _load_fhwa_points(metadata_path: Path, hourly_path: Path, target_crs: Any) -> gpd.GeoDataFrame:
    metadata = pd.read_csv(metadata_path, dtype={"station_id": "string"}, low_memory=False)
    required = {"station_id", "latitude", "longitude"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"FHWA metadata lacks {sorted(required - set(metadata.columns))}")
    metadata["station_id"] = metadata["station_id"].astype(str).str.strip()
    metadata["latitude"] = pd.to_numeric(metadata["latitude"], errors="coerce")
    metadata["longitude"] = pd.to_numeric(metadata["longitude"], errors="coerce")
    metadata = metadata.dropna(subset=["latitude", "longitude"])
    stations = (
        metadata.groupby("station_id", as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        .sort_values("station_id")
    )
    hourly_ids = set(
        pd.read_csv(hourly_path, usecols=["station_id"], dtype={"station_id": "string"})["station_id"]
        .astype(str)
        .str.strip()
    )
    points = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs="EPSG:4326",
    ).to_crs(target_crs)
    points["has_retained_hourly_counts"] = points["station_id"].isin(hourly_ids)
    return points


def _sampling_universe(
    survey_path: Path,
    tracts: gpd.GeoDataFrame,
    cached_paths: pd.DataFrame,
    station_map: pd.DataFrame,
) -> pd.DataFrame:
    observed = pd.read_csv(
        survey_path,
        usecols=["o_tract_fips", "d_tract_fips"],
        dtype={"o_tract_fips": "string", "d_tract_fips": "string"},
        low_memory=False,
    )
    observed["o_tract_fips"] = _clean_geoids(observed["o_tract_fips"])
    observed["d_tract_fips"] = _clean_geoids(observed["d_tract_fips"])
    valid_geoids = set(tracts["GEOID"].astype(str))
    observed = observed[
        observed["o_tract_fips"].isin(valid_geoids)
        & observed["d_tract_fips"].isin(valid_geoids)
        & observed["o_tract_fips"].ne(observed["d_tract_fips"])
    ][["o_tract_fips", "d_tract_fips"]].drop_duplicates()

    path_pairs = cached_paths[["o_tract_fips", "d_tract_fips"]].drop_duplicates()
    mapped_ids = set(station_map["screenline_id"].astype(str))
    mapped_pairs = cached_paths[cached_paths["screenline_id"].astype(str).isin(mapped_ids)][
        ["o_tract_fips", "d_tract_fips"]
    ].drop_duplicates()
    universe = observed.merge(path_pairs, how="inner").merge(mapped_pairs, how="inner")
    return universe.sort_values(["o_tract_fips", "d_tract_fips"]).reset_index(drop=True)


def _sample_pairs(
    universe: pd.DataFrame,
    n_pairs: int,
    seed: int,
    avoid_reverse_duplicates: bool = False,
) -> pd.DataFrame:
    if n_pairs <= 0:
        raise ValueError("--n-pairs must be positive")
    if len(universe) < n_pairs:
        raise ValueError(f"Requested {n_pairs} pairs from a universe of only {len(universe)}")
    rng = np.random.default_rng(seed)
    if avoid_reverse_duplicates:
        indices = []
        seen_unordered: set[tuple[str, str]] = set()
        for index in rng.permutation(len(universe)):
            row = universe.iloc[int(index)]
            key = tuple(sorted((str(row["o_tract_fips"]), str(row["d_tract_fips"]))))
            if key in seen_unordered:
                continue
            seen_unordered.add(key)
            indices.append(int(index))
            if len(indices) == n_pairs:
                break
        if len(indices) < n_pairs:
            raise ValueError(
                f"Requested {n_pairs} visually distinct pairs, but only {len(indices)} were available"
            )
    else:
        indices = rng.choice(len(universe), size=n_pairs, replace=False)
    sampled = universe.iloc[indices].reset_index(drop=True).copy()
    sampled.insert(0, "example_id", np.arange(1, len(sampled) + 1))
    return sampled


def _full_path(
    origin: str,
    destination: str,
    tracts: gpd.GeoDataFrame,
    tract_lookup: pd.DataFrame,
    tract_sindex: Any,
    adjacency_lookup: dict[str, Any],
) -> tuple[LineString, pd.DataFrame, list[str]]:
    origin_row = tract_lookup.loc[origin]
    destination_row = tract_lookup.loc[destination]
    od_line = LineString(
        [
            (float(origin_row["rep_x"]), float(origin_row["rep_y"])),
            (float(destination_row["rep_x"]), float(destination_row["rep_y"])),
        ]
    )
    candidate_idx = list(tract_sindex.query(od_line, predicate="intersects"))
    candidates = tracts.iloc[candidate_idx]
    ordered_tracts = _ordered_crossed_tracts(od_line, candidates, origin, destination)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: dict[str, int] = {}
    for raw_position, (left, right) in enumerate(zip(ordered_tracts[:-1], ordered_tracts[1:]), start=1):
        sid = screenline_id(str(left), str(right))
        if sid not in adjacency_lookup:
            missing.append(sid)
            continue
        if sid in seen:
            rows[seen[sid]]["path_occurrences"] += 1
            continue
        seen[sid] = len(rows)
        adjacency = adjacency_lookup[sid]
        rows.append(
            {
                "path_order": len(rows) + 1,
                "first_raw_position": raw_position,
                "path_occurrences": 1,
                "screenline_id": sid,
                "tract_a": str(adjacency.tract_a),
                "tract_b": str(adjacency.tract_b),
                "shared_boundary_m": float(adjacency.shared_boundary_m),
                "geometry": adjacency.geometry,
            }
        )
    return od_line, pd.DataFrame(rows), missing


def _filter_exactly_two_stationed(
    universe: pd.DataFrame,
    cached_paths: pd.DataFrame,
    station_map: pd.DataFrame,
    tracts: gpd.GeoDataFrame,
    tract_lookup: pd.DataFrame,
    tract_sindex: Any,
    adjacency_lookup: dict[str, Any],
    exclude_manifest: Path | None,
) -> pd.DataFrame:
    """Apply the exact two-screenline/station criteria before random sampling."""
    filtered = universe.copy()
    if exclude_manifest is not None:
        exclude_manifest = exclude_manifest.resolve()
        if not exclude_manifest.exists():
            raise FileNotFoundError(f"Missing exclusion manifest: {exclude_manifest}")
        excluded = pd.read_csv(
            exclude_manifest,
            usecols=["o_tract_fips", "d_tract_fips"],
            dtype={"o_tract_fips": "string", "d_tract_fips": "string"},
        )
        excluded["o_tract_fips"] = _clean_geoids(excluded["o_tract_fips"])
        excluded["d_tract_fips"] = _clean_geoids(excluded["d_tract_fips"])
        reversed_excluded = excluded.rename(
            columns={
                "o_tract_fips": "d_tract_fips",
                "d_tract_fips": "o_tract_fips",
            }
        )[["o_tract_fips", "d_tract_fips"]]
        excluded = pd.concat([excluded, reversed_excluded], ignore_index=True).drop_duplicates()
        filtered = (
            filtered.merge(
                excluded.drop_duplicates().assign(_excluded=True),
                on=["o_tract_fips", "d_tract_fips"],
                how="left",
            )
            .loc[lambda frame: frame["_excluded"].isna(), ["o_tract_fips", "d_tract_fips"]]
            .reset_index(drop=True)
        )

    station_counts = station_map.groupby("screenline_id")["station_id"].nunique()
    candidate_paths = cached_paths.merge(
        filtered,
        on=["o_tract_fips", "d_tract_fips"],
        how="inner",
    )[["o_tract_fips", "d_tract_fips", "screenline_id"]].drop_duplicates()
    candidate_paths["mapped_station_count"] = (
        candidate_paths["screenline_id"].map(station_counts).fillna(0).astype(int)
    )
    cheap_stats = (
        candidate_paths.groupby(["o_tract_fips", "d_tract_fips"], as_index=False)
        .agg(
            cached_screenlines=("screenline_id", "nunique"),
            minimum_station_count=("mapped_station_count", "min"),
            maximum_station_count=("mapped_station_count", "max"),
        )
    )
    cheap_eligible = cheap_stats[
        cheap_stats["cached_screenlines"].eq(2)
        & cheap_stats["minimum_station_count"].ge(1)
        & cheap_stats["maximum_station_count"].ge(2)
    ][["o_tract_fips", "d_tract_fips"]].sort_values(["o_tract_fips", "d_tract_fips"])

    exact_rows: list[dict[str, str]] = []
    for position, row in enumerate(cheap_eligible.itertuples(index=False), start=1):
        origin = str(row.o_tract_fips)
        destination = str(row.d_tract_fips)
        _, full_path, _ = _full_path(
            origin,
            destination,
            tracts,
            tract_lookup,
            tract_sindex,
            adjacency_lookup,
        )
        if len(full_path) != 2:
            continue
        counts = full_path["screenline_id"].map(station_counts).fillna(0).astype(int)
        if counts.ge(1).all() and counts.max() >= 2:
            exact_rows.append({"o_tract_fips": origin, "d_tract_fips": destination})
        if position % 1_000 == 0:
            print(
                f"Eligibility audit {position:,}/{len(cheap_eligible):,}: "
                f"{len(exact_rows):,} exact matches",
                flush=True,
            )
    return (
        pd.DataFrame(exact_rows, columns=["o_tract_fips", "d_tract_fips"])
        .sort_values(["o_tract_fips", "d_tract_fips"])
        .reset_index(drop=True)
    )


def _screenline_colors(n: int) -> list[str]:
    categorical: list[Any] = []
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    for name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(name)
        categorical.extend([cmap.colors[index] for index in order])
    colors = [matplotlib.colors.to_hex(color) for color in categorical[:n]]
    golden = 0.6180339887498949
    for index in range(len(colors), n):
        hue = (0.11 + golden * index) % 1.0
        saturation = 0.62 + 0.18 * (index % 3) / 2.0
        value = 0.74 + 0.18 * (index % 2)
        colors.append(matplotlib.colors.to_hex(colorsys.hsv_to_rgb(hue, saturation, value)))
    return colors


def _resolve_station_assignments(
    path_rows: pd.DataFrame,
    station_map: pd.DataFrame,
    aadt_points: gpd.GeoDataFrame,
    fhwa_ids: set[str],
) -> pd.DataFrame:
    if path_rows.empty:
        return pd.DataFrame()
    lookup = path_rows.set_index("screenline_id")
    selected = station_map[station_map["screenline_id"].isin(lookup.index)].copy()
    selected = selected.drop_duplicates(["screenline_id", "station_id"])
    if selected.empty:
        return selected
    selected["path_order"] = selected["screenline_id"].map(lookup["path_order"])
    selected["screenline_color"] = selected["screenline_id"].map(lookup["screenline_color"])
    points_by_id = {str(key): frame for key, frame in aadt_points.groupby("station_id", sort=False)}

    resolved: list[dict[str, Any]] = []
    for row in selected.sort_values(["path_order", "station_id"]).itertuples(index=False):
        sid = str(row.screenline_id)
        station_id_value = str(row.station_id)
        candidates = points_by_id.get(station_id_value)
        if candidates is None or candidates.empty:
            continue
        boundary = lookup.loc[sid, "geometry"]
        distances = candidates.geometry.distance(boundary)
        target = float(getattr(row, "boundary_distance_m", np.nan))
        if np.isfinite(target):
            selected_index = (distances - target).abs().idxmin()
        else:
            selected_index = distances.idxmin()
        point = candidates.loc[selected_index, "geometry"]
        record = row._asdict()
        record.update(
            {
                "true_x": float(point.x),
                "true_y": float(point.y),
                "is_fhwa_station": station_id_value in fhwa_ids,
            }
        )
        resolved.append(record)
    assignments = pd.DataFrame(resolved)
    if assignments.empty:
        return assignments

    assignments["location_key"] = (
        assignments["station_id"].astype(str)
        + "@"
        + assignments["true_x"].round(2).astype(str)
        + ","
        + assignments["true_y"].round(2).astype(str)
    )
    assignments["plot_x"] = assignments["true_x"]
    assignments["plot_y"] = assignments["true_y"]
    return assignments


def _apply_assignment_offsets(assignments: pd.DataFrame, map_span: float) -> pd.DataFrame:
    if assignments.empty:
        return assignments
    assignments = assignments.copy()
    radius = max(55.0, min(180.0, map_span * 0.003))
    for _, indices in assignments.groupby("location_key", sort=False).groups.items():
        indices = list(indices)
        if len(indices) == 1:
            continue
        station_id_value = str(assignments.loc[indices[0], "station_id"])
        digest = hashlib.sha1(station_id_value.encode("utf-8")).hexdigest()
        phase = int(digest[:8], 16) / 0xFFFFFFFF * 2.0 * math.pi
        angles = phase + np.linspace(0.0, 2.0 * math.pi, len(indices), endpoint=False)
        for index, angle in zip(indices, angles):
            assignments.loc[index, "plot_x"] += radius * math.cos(float(angle))
            assignments.loc[index, "plot_y"] += radius * math.sin(float(angle))
    return assignments


def _nearest_boundary_point(boundary: Any, x: float, y: float) -> Point | None:
    try:
        return nearest_points(boundary, Point(float(x), float(y)))[0]
    except Exception:
        return None


def _label_point(boundary: Any, od_line: LineString) -> Point:
    try:
        intersection = boundary.intersection(od_line)
        if not intersection.is_empty:
            return intersection.representative_point()
        return nearest_points(boundary, od_line)[0]
    except Exception:
        return boundary.representative_point()


def _nice_scale_length(width_m: float) -> float:
    target = max(width_m * 0.12, 1.0)
    exponent = 10.0 ** math.floor(math.log10(target))
    candidates = [1.0, 2.0, 5.0, 10.0]
    return max(value * exponent for value in candidates if value * exponent <= target)


def _draw_scale_bar(ax: Any, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]
    length = _nice_scale_length(width)
    x0 = xlim[1] - 0.055 * width - length
    y0 = ylim[0] + 0.045 * height
    ax.plot([x0, x0 + length], [y0, y0], color="#222222", linewidth=3.0, zorder=30)
    ax.plot([x0, x0], [y0 - 0.006 * height, y0 + 0.006 * height], color="#222222", linewidth=1.2, zorder=30)
    ax.plot(
        [x0 + length, x0 + length],
        [y0 - 0.006 * height, y0 + 0.006 * height],
        color="#222222",
        linewidth=1.2,
        zorder=30,
    )
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.text(
        x0 + length / 2,
        y0 + 0.012 * height,
        label,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#222222",
        path_effects=[path_effects.withStroke(linewidth=2.0, foreground="white")],
        zorder=31,
    )


def _draw_north_arrow(ax: Any) -> None:
    ax.annotate(
        "N",
        xy=(0.965, 0.91),
        xytext=(0.965, 0.82),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=1.5),
        zorder=31,
    )


def _visible(gdf: gpd.GeoDataFrame, extent: Any) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    indices = list(gdf.sindex.query(extent, predicate="intersects"))
    return gdf.iloc[indices].copy()


def _plot_map(
    output_path: Path,
    example_id: int,
    origin: str,
    destination: str,
    od_line: LineString,
    path_rows: pd.DataFrame,
    ordered_tract_ids: set[str],
    tracts: gpd.GeoDataFrame,
    tract_lookup: pd.DataFrame,
    aadt_points: gpd.GeoDataFrame,
    fhwa_points: gpd.GeoDataFrame,
    assignments: pd.DataFrame,
    dpi: int,
    background_color: str,
    map_only: bool = False,
) -> dict[str, Any]:
    origin_geometry = tract_lookup.loc[origin, "geometry"]
    destination_geometry = tract_lookup.loc[destination, "geometry"]

    bound_values = [origin_geometry.bounds, destination_geometry.bounds, od_line.bounds]
    bound_values.extend(geometry.bounds for geometry in path_rows["geometry"])
    if not assignments.empty:
        bound_values.extend(
            (float(row.true_x), float(row.true_y), float(row.true_x), float(row.true_y))
            for row in assignments.itertuples(index=False)
        )
    xmin = min(value[0] for value in bound_values)
    ymin = min(value[1] for value in bound_values)
    xmax = max(value[2] for value in bound_values)
    ymax = max(value[3] for value in bound_values)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    padding = max(2_000.0, min(12_000.0, 0.08 * span))
    xlim = (xmin - padding, xmax + padding)
    ylim = (ymin - padding, ymax + padding)
    extent = box(xlim[0], ylim[0], xlim[1], ylim[1])

    context = _visible(tracts, extent)
    visible_mdot = _visible(aadt_points, extent)
    visible_fhwa = _visible(fhwa_points, extent)
    assignments = _apply_assignment_offsets(assignments, max(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    origin_frame = tracts[tracts["GEOID"] == origin]
    destination_frame = tracts[tracts["GEOID"] == destination]
    path_tracts = tracts[tracts["GEOID"].isin(ordered_tract_ids)]

    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    if not context.empty:
        context.plot(
            ax=ax,
            facecolor=background_color,
            edgecolor=CONTEXT_EDGE,
            linewidth=0.35,
            zorder=1,
        )
    if not path_tracts.empty:
        path_tracts.plot(ax=ax, facecolor=PATH_FACE, edgecolor="#8da1ac", linewidth=0.65, alpha=0.78, zorder=2)
    origin_frame.plot(ax=ax, facecolor=ORIGIN_FACE, edgecolor=ORIGIN_EDGE, linewidth=1.2, alpha=0.78, zorder=3)
    destination_frame.plot(ax=ax, facecolor=DEST_FACE, edgecolor=DEST_EDGE, linewidth=1.2, alpha=0.78, zorder=3)

    arrow = FancyArrowPatch(
        od_line.coords[0],
        od_line.coords[-1],
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.45,
        linestyle=(0, (4, 3)),
        color="#262626",
        alpha=0.78,
        shrinkA=5,
        shrinkB=5,
        zorder=4,
    )
    ax.add_patch(arrow)

    for row in path_rows.sort_values("path_order").itertuples(index=False):
        gpd.GeoSeries([row.geometry], crs=tracts.crs).plot(
            ax=ax,
            color=row.screenline_color,
            linewidth=3.8,
            alpha=0.98,
            zorder=7,
        )
        point = _label_point(row.geometry, od_line)
        label = ax.text(
            point.x,
            point.y,
            str(int(row.path_order)),
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="circle,pad=0.20",
                facecolor=row.screenline_color,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.98,
            ),
            zorder=18,
        )
        label.set_path_effects([path_effects.withStroke(linewidth=0.65, foreground="#222222")])

    if not visible_mdot.empty:
        ax.scatter(
            visible_mdot.geometry.x,
            visible_mdot.geometry.y,
            s=11,
            marker="o",
            facecolor=UNUSED_MDOT,
            edgecolor="white",
            linewidth=0.18,
            alpha=0.52,
            zorder=8,
        )
    if not visible_fhwa.empty:
        active = visible_fhwa[visible_fhwa["has_retained_hourly_counts"]]
        inactive = visible_fhwa[~visible_fhwa["has_retained_hourly_counts"]]
        for frame, alpha, linewidth in [(inactive, 0.58, 0.75), (active, 0.90, 1.15)]:
            if frame.empty:
                continue
            ax.scatter(
                frame.geometry.x,
                frame.geometry.y,
                s=36,
                marker="^",
                facecolor="none",
                edgecolor=UNUSED_FHWA,
                linewidth=linewidth,
                alpha=alpha,
                zorder=9,
            )

    if not assignments.empty:
        boundary_lookup = path_rows.set_index("screenline_id")["geometry"]
        for row in assignments.itertuples(index=False):
            boundary = boundary_lookup.loc[str(row.screenline_id)]
            target = _nearest_boundary_point(boundary, float(row.true_x), float(row.true_y))
            if target is not None:
                ax.plot(
                    [float(row.plot_x), target.x],
                    [float(row.plot_y), target.y],
                    color=row.screenline_color,
                    linewidth=0.75,
                    alpha=0.38,
                    zorder=6,
                )
            ax.scatter(
                [float(row.plot_x)],
                [float(row.plot_y)],
                s=43,
                marker="o",
                facecolor=row.screenline_color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.98,
                zorder=13,
            )
            if bool(row.is_fhwa_station):
                ax.scatter(
                    [float(row.plot_x)],
                    [float(row.plot_y)],
                    s=75,
                    marker="^",
                    facecolor="none",
                    edgecolor=row.screenline_color,
                    linewidth=1.8,
                    alpha=1.0,
                    zorder=14,
                )

    origin_xy = od_line.coords[0]
    destination_xy = od_line.coords[-1]
    ax.scatter(
        [origin_xy[0], destination_xy[0]],
        [origin_xy[1], destination_xy[1]],
        s=66,
        color=[ORIGIN_EDGE, DEST_EDGE],
        edgecolor="white",
        linewidth=0.9,
        zorder=20,
    )
    for label_text, coordinates, horizontal in [
        ("O", origin_xy, "right"),
        ("D", destination_xy, "left"),
    ]:
        label = ax.text(
            coordinates[0],
            coordinates[1],
            f" {label_text} ",
            ha=horizontal,
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#202020",
            zorder=21,
        )
        label.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground="white")])

    legend_handles = [
        Patch(facecolor=ORIGIN_FACE, edgecolor=ORIGIN_EDGE, label="Origin tract"),
        Patch(facecolor=DEST_FACE, edgecolor=DEST_EDGE, label="Destination tract"),
        Line2D([0], [0], color="#262626", linestyle="--", linewidth=1.5, label="O–D representative-point proxy"),
        Line2D(
            [0],
            [0],
            color="#d62728",
            linewidth=3.2,
            marker="o",
            markersize=5.5,
            markerfacecolor="#d62728",
            markeredgecolor="white",
            label="Boundary + its mapped MDOT station",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=4.5,
            markerfacecolor=UNUSED_MDOT,
            markeredgecolor="white",
            label="Nearby unused MDOT AADT station",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="^",
            markersize=6.5,
            markerfacecolor="none",
            markeredgecolor=UNUSED_FHWA,
            label="FHWA/TMAS station (triangle)",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.018),
        fontsize=7.1,
        ncol=2,
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="#bbbbbb",
        handlelength=2.0,
        columnspacing=1.0,
        labelspacing=0.45,
        borderpad=0.65,
    )
    legend.set_zorder(40)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    _draw_north_arrow(ax)
    _draw_scale_bar(ax, xlim, ylim)

    mapped_station_count = int(assignments["station_id"].nunique()) if not assignments.empty else 0
    mapped_fhwa_count = (
        int(assignments.loc[assignments["is_fhwa_station"], "station_id"].nunique()) if not assignments.empty else 0
    )
    fig.suptitle(
        f"Screenline example {example_id:03d}: O {origin} → D {destination}",
        x=0.5,
        y=0.978,
        fontsize=14,
        fontweight="bold",
        color="#1f1f1f",
    )
    fig.text(
        0.5,
        0.942,
        (
            f"{len(path_rows)} shared tract-boundary screenlines  •  "
            f"{len(visible_mdot):,} MDOT AADT stations in view  •  "
            f"{len(visible_fhwa):,} FHWA/TMAS stations in view  •  "
            f"{mapped_station_count} mapped stations ({mapped_fhwa_count} FHWA)"
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#424242",
    )
    fig.text(
        0.5,
        0.018,
        (
            "Colored stations inherit the color of their assigned boundary; gray stations are visible but unused. "
            "Straight representative-point proxy—not roadway routing. Number labels give O→D boundary order."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    if map_only:
        legend.remove()
        for text_artist in list(fig.texts):
            text_artist.remove()
        fig.subplots_adjust(left=0.006, right=0.994, bottom=0.006, top=0.994)
    else:
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.055, top=0.925)
    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        metadata={
            "Title": f"OD tract screenline example {example_id:03d}",
            "Description": "Tract-boundary screenlines with MDOT AADT and FHWA/TMAS station context",
        },
        pil_kwargs={"optimize": True},
    )
    plt.close(fig)

    return {
        "nearby_mdot_station_points": int(len(visible_mdot)),
        "nearby_mdot_unique_station_ids": int(visible_mdot["station_id"].nunique()),
        "nearby_fhwa_station_ids": int(visible_fhwa["station_id"].nunique()),
        "nearby_fhwa_active_station_ids": int(
            visible_fhwa.loc[visible_fhwa["has_retained_hourly_counts"], "station_id"].nunique()
        ),
        "mapped_station_ids": mapped_station_count,
        "mapped_fhwa_station_ids": mapped_fhwa_count,
        "mapped_station_screenline_links": int(len(assignments)),
        "x_min": float(xlim[0]),
        "y_min": float(ylim[0]),
        "x_max": float(xlim[1]),
        "y_max": float(ylim[1]),
    }


def _make_contact_sheets(image_paths: list[Path], output_dir: Path) -> list[Path]:
    columns = 5
    rows = 6
    thumb_size = (300, 194)
    label_height = 22
    margin = 12
    per_sheet = columns * rows
    font = ImageFont.load_default()
    outputs: list[Path] = []
    for sheet_number, start in enumerate(range(0, len(image_paths), per_sheet), start=1):
        chunk = image_paths[start : start + per_sheet]
        sheet_width = margin + columns * (thumb_size[0] + margin)
        sheet_height = margin + rows * (thumb_size[1] + label_height + margin)
        sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
        draw = ImageDraw.Draw(sheet)
        for position, path in enumerate(chunk):
            row, column = divmod(position, columns)
            x = margin + column * (thumb_size[0] + margin)
            y = margin + row * (thumb_size[1] + label_height + margin)
            with Image.open(path) as source:
                thumb = ImageOps.fit(source.convert("RGB"), thumb_size, method=Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y))
            draw.text((x + 2, y + thumb_size[1] + 4), path.stem, fill="#222222", font=font)
        output = output_dir / f"contact_sheet_{sheet_number:02d}.png"
        sheet.save(output, optimize=True)
        outputs.append(output)
    return outputs


def _write_readme(
    output_dir: Path,
    n_pairs: int,
    seed: int,
    universe_size: int,
    summary: dict[str, Any],
) -> None:
    if summary["selection_profile"] == "exactly_two_stationed":
        selection_note = """
Every sampled pair satisfies all three requested constraints:

1. exactly two unique >1 m shared tract-boundary screenlines;
2. at least one mapped MDOT AADT station on each screenline; and
3. two or more mapped MDOT AADT stations on at least one of the two screenlines.

Pairs from the prior manifest, including their reverse directions, were excluded so this is a
visually distinct additional batch.
"""
    else:
        selection_note = ""
    text = f"""# Screenline examples

This folder contains **{n_pairs} reproducibly sampled OD census-tract maps** from the frozen
`paper_wctr_1m_vae_sweep` run. The random seed is `{seed}` and the eligible observed-survey
universe contains {universe_size:,} unique directed OD pairs.
{selection_note}
The figure and map background is exactly `{summary['background_color']}`.

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

- `screenline_*.png`: the {n_pairs} requested maps.
- `manifest.csv`: one row per sampled OD pair, with map and station counts and projected extent.
- `screenlines.csv`: every displayed boundary, its exact color, order, and mapped-station summary.
- `station_assignments.csv`: every colored station–boundary link used in the maps.
- `contact_sheets/`: four visual indexes for quick review.
- `generation_summary.json`: source paths, seed, CRS, and QA totals.

Generated with `scripts/generate_screenline_examples.py`. All coordinates and distance fields are in
EPSG:26985 (NAD83 / Maryland, meters). The generation summary reports
{summary['screenlines_displayed']:,} displayed boundaries and {summary['station_screenline_links']:,}
colored station–boundary links across the map set.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        background_color = matplotlib.colors.to_hex(args.background_color, keep_alpha=False).upper()
    except ValueError as exc:
        raise ValueError(f"Invalid --background-color: {args.background_color}") from exc
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    geo_dir = run_dir / "geo"
    input_paths = [
        geo_dir / "tracts.parquet",
        geo_dir / "tract_adjacency.parquet",
        geo_dir / "screenlines.parquet",
        geo_dir / "screenline_station_map.parquet",
        geo_dir / "aadt_points.parquet",
        geo_dir / "od_screenline_paths.parquet",
        args.survey.resolve(),
        args.fhwa_metadata.resolve(),
        args.fhwa_hourly.resolve(),
    ]
    _check_inputs(input_paths)
    _prepare_output(output_dir, args.overwrite)

    tracts = gpd.read_parquet(geo_dir / "tracts.parquet")
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    adjacency = gpd.read_parquet(geo_dir / "tract_adjacency.parquet")
    candidates = gpd.read_parquet(geo_dir / "screenlines.parquet")
    station_map = pd.read_parquet(geo_dir / "screenline_station_map.parquet")
    aadt_points = gpd.read_parquet(geo_dir / "aadt_points.parquet")
    cached_paths = pd.read_parquet(geo_dir / "od_screenline_paths.parquet")

    station_map["screenline_id"] = station_map["screenline_id"].astype(str)
    station_map["station_id"] = station_map["station_id"].astype(str)
    aadt_points["station_id"] = aadt_points["station_id"].astype(str)
    cached_paths["o_tract_fips"] = _clean_geoids(cached_paths["o_tract_fips"])
    cached_paths["d_tract_fips"] = _clean_geoids(cached_paths["d_tract_fips"])
    cached_paths["screenline_id"] = cached_paths["screenline_id"].astype(str)
    fhwa_points = _load_fhwa_points(args.fhwa_metadata.resolve(), args.fhwa_hourly.resolve(), tracts.crs)
    fhwa_ids = set(fhwa_points["station_id"].astype(str))

    tract_lookup = tracts.set_index("GEOID", drop=False)
    tract_sindex = tracts.sindex
    adjacency_lookup = {str(row.screenline_id): row for row in adjacency.itertuples(index=False)}
    candidate_ids = set(candidates["screenline_id"].astype(str))
    base_universe = _sampling_universe(args.survey.resolve(), tracts, cached_paths, station_map)
    if args.selection_profile == "exactly_two_stationed":
        universe = _filter_exactly_two_stationed(
            base_universe,
            cached_paths,
            station_map,
            tracts,
            tract_lookup,
            tract_sindex,
            adjacency_lookup,
            args.exclude_manifest,
        )
    else:
        universe = base_universe
        if args.exclude_manifest is not None:
            excluded = pd.read_csv(
                args.exclude_manifest.resolve(),
                usecols=["o_tract_fips", "d_tract_fips"],
                dtype={"o_tract_fips": "string", "d_tract_fips": "string"},
            )
            excluded["o_tract_fips"] = _clean_geoids(excluded["o_tract_fips"])
            excluded["d_tract_fips"] = _clean_geoids(excluded["d_tract_fips"])
            universe = (
                universe.merge(
                    excluded.drop_duplicates().assign(_excluded=True),
                    on=["o_tract_fips", "d_tract_fips"],
                    how="left",
                )
                .loc[lambda frame: frame["_excluded"].isna(), ["o_tract_fips", "d_tract_fips"]]
                .reset_index(drop=True)
            )
    print(
        f"Eligible OD universe for {args.selection_profile}: {len(universe):,} pairs",
        flush=True,
    )
    sampled = _sample_pairs(
        universe,
        args.n_pairs,
        args.seed,
        avoid_reverse_duplicates=args.selection_profile == "exactly_two_stationed",
    )

    cached_pair_groups = {
        key: frame.sort_values("path_position")
        for key, frame in cached_paths[cached_paths.set_index(["o_tract_fips", "d_tract_fips"]).index.isin(
            pd.MultiIndex.from_frame(sampled[["o_tract_fips", "d_tract_fips"]])
        )].groupby(["o_tract_fips", "d_tract_fips"], sort=False)
    }

    manifest_rows: list[dict[str, Any]] = []
    screenline_rows_out: list[dict[str, Any]] = []
    assignment_rows_out: list[dict[str, Any]] = []
    image_paths: list[Path] = []

    for sampled_row in sampled.itertuples(index=False):
        example_id = int(sampled_row.example_id)
        origin = str(sampled_row.o_tract_fips)
        destination = str(sampled_row.d_tract_fips)
        od_line, full_path, missing_adjacencies = _full_path(
            origin,
            destination,
            tracts,
            tract_lookup,
            tract_sindex,
            adjacency_lookup,
        )
        if full_path.empty:
            raise RuntimeError(f"No shared tract-boundary path could be reconstructed for {origin} -> {destination}")
        full_path["screenline_color"] = _screenline_colors(len(full_path))
        full_path["is_validation_candidate"] = full_path["screenline_id"].isin(candidate_ids)
        cached = cached_pair_groups[(origin, destination)]
        cached_ids = set(cached["screenline_id"].astype(str))
        full_path["is_in_cached_validation_path"] = full_path["screenline_id"].isin(cached_ids)

        assignments = _resolve_station_assignments(full_path, station_map, aadt_points, fhwa_ids)
        mapped_counts_by_screenline = (
            assignments.groupby("screenline_id")["station_id"].nunique()
            if not assignments.empty
            else pd.Series(dtype=int)
        )
        mapped_counts_on_path = (
            full_path["screenline_id"].map(mapped_counts_by_screenline).fillna(0).astype(int)
        )
        if args.selection_profile == "exactly_two_stationed":
            if not (
                len(full_path) == 2
                and mapped_counts_on_path.ge(1).all()
                and mapped_counts_on_path.max() >= 2
            ):
                raise AssertionError(
                    f"Filtered pair failed exact criteria during rendering: {origin} -> {destination}"
                )
        ordered_tract_ids = {origin, destination}
        ordered_tract_ids.update(full_path["tract_a"].astype(str))
        ordered_tract_ids.update(full_path["tract_b"].astype(str))
        filename = f"screenline_{example_id:03d}_O_{origin}_D_{destination}.png"
        output_path = output_dir / filename
        map_stats = _plot_map(
            output_path,
            example_id,
            origin,
            destination,
            od_line,
            full_path,
            ordered_tract_ids,
            tracts,
            tract_lookup,
            aadt_points,
            fhwa_points,
            assignments,
            args.dpi,
            background_color,
        )
        image_paths.append(output_path)

        manifest_rows.append(
            {
                "example_id": example_id,
                "filename": filename,
                "o_tract_fips": origin,
                "d_tract_fips": destination,
                "o_state_fips": origin[:2],
                "d_state_fips": destination[:2],
                "proxy_distance_km": float(od_line.length / 1000.0),
                "displayed_screenlines": int(len(full_path)),
                "cached_validation_path_rows": int(len(cached)),
                "cached_unique_screenlines": int(cached["screenline_id"].nunique()),
                "validation_candidate_screenlines": int(full_path["is_validation_candidate"].sum()),
                "minimum_mapped_aadt_stations_per_screenline": int(mapped_counts_on_path.min()),
                "maximum_mapped_aadt_stations_per_screenline": int(mapped_counts_on_path.max()),
                "missing_point_touch_adjacencies": int(len(missing_adjacencies)),
                **map_stats,
            }
        )

        station_lists = (
            assignments.groupby("screenline_id")["station_id"].agg(lambda values: ";".join(sorted(set(map(str, values)))))
            if not assignments.empty
            else pd.Series(dtype=str)
        )
        fhwa_lists = (
            assignments[assignments["is_fhwa_station"]]
            .groupby("screenline_id")["station_id"]
            .agg(lambda values: ";".join(sorted(set(map(str, values)))))
            if not assignments.empty
            else pd.Series(dtype=str)
        )
        for row in full_path.itertuples(index=False):
            screenline_rows_out.append(
                {
                    "example_id": example_id,
                    "o_tract_fips": origin,
                    "d_tract_fips": destination,
                    "path_order": int(row.path_order),
                    "first_raw_position": int(row.first_raw_position),
                    "path_occurrences": int(row.path_occurrences),
                    "screenline_id": str(row.screenline_id),
                    "tract_a": str(row.tract_a),
                    "tract_b": str(row.tract_b),
                    "screenline_color": str(row.screenline_color),
                    "shared_boundary_m": float(row.shared_boundary_m),
                    "is_validation_candidate": bool(row.is_validation_candidate),
                    "is_in_cached_validation_path": bool(row.is_in_cached_validation_path),
                    "mapped_mdot_station_ids": station_lists.get(str(row.screenline_id), ""),
                    "mapped_fhwa_station_ids": fhwa_lists.get(str(row.screenline_id), ""),
                }
            )
        if not assignments.empty:
            for row in assignments.itertuples(index=False):
                assignment_rows_out.append(
                    {
                        "example_id": example_id,
                        "o_tract_fips": origin,
                        "d_tract_fips": destination,
                        "path_order": int(row.path_order),
                        "screenline_id": str(row.screenline_id),
                        "screenline_color": str(row.screenline_color),
                        "station_id": str(row.station_id),
                        "is_fhwa_station": bool(row.is_fhwa_station),
                        "boundary_distance_m": float(row.boundary_distance_m),
                        "boundary_centroid_distance_ratio": float(row.boundary_centroid_distance_ratio),
                        "station_x": float(row.true_x),
                        "station_y": float(row.true_y),
                    }
                )
        if example_id == 1 or example_id % 10 == 0 or example_id == args.n_pairs:
            print(f"Generated {example_id:3d}/{args.n_pairs}: {filename}", flush=True)

    manifest = pd.DataFrame(manifest_rows).sort_values("example_id")
    screenline_table = pd.DataFrame(screenline_rows_out).sort_values(["example_id", "path_order"])
    assignments_table = pd.DataFrame(assignment_rows_out).sort_values(
        ["example_id", "path_order", "station_id"]
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    screenline_table.to_csv(output_dir / "screenlines.csv", index=False)
    assignments_table.to_csv(output_dir / "station_assignments.csv", index=False)
    contact_sheets = _make_contact_sheets(image_paths, output_dir / "contact_sheets")

    summary = {
        "status": "ok",
        "seed": int(args.seed),
        "requested_pairs": int(args.n_pairs),
        "eligible_observed_od_pairs": int(len(universe)),
        "eligible_unordered_od_geometries": int(
            universe.apply(
                lambda row: "__".join(sorted((row["o_tract_fips"], row["d_tract_fips"]))),
                axis=1,
            ).nunique()
        ),
        "base_eligible_observed_od_pairs": int(len(base_universe)),
        "selection_profile": str(args.selection_profile),
        "reverse_direction_duplicates_avoided": bool(
            args.selection_profile == "exactly_two_stationed"
        ),
        "background_color": background_color,
        "excluded_prior_manifest": str(args.exclude_manifest.resolve()) if args.exclude_manifest else None,
        "png_maps": int(len(image_paths)),
        "unique_directed_od_pairs": int(manifest[["o_tract_fips", "d_tract_fips"]].drop_duplicates().shape[0]),
        "screenlines_displayed": int(len(screenline_table)),
        "validation_candidate_screenlines_displayed": int(screenline_table["is_validation_candidate"].sum()),
        "station_screenline_links": int(len(assignments_table)),
        "unique_mdot_station_ids_mapped_across_maps": int(assignments_table["station_id"].nunique()),
        "unique_fhwa_station_ids_mapped_across_maps": int(
            assignments_table.loc[assignments_table["is_fhwa_station"], "station_id"].nunique()
        ),
        "contact_sheets": int(len(contact_sheets)),
        "crs": str(tracts.crs),
        "sampling_frame": (
            (
                "unique directed nonintrazonal observed-survey OD pairs with exactly two reconstructed "
                "shared tract-boundary screenlines; every boundary has at least one mapped AADT station; "
                "at least one boundary has multiple mapped AADT stations"
            )
            if args.selection_profile == "exactly_two_stationed"
            else (
                "unique directed nonintrazonal observed-survey OD pairs with both endpoints in the frozen "
                "study-area tract layer, a nonempty cached validation path, and at least one mapped station"
            )
        ),
        "path_method": (
            "straight tract representative-point line; ordered intersected tracts; all consecutive >1 m "
            "shared boundaries from tract_adjacency.parquet"
        ),
        "source_files": [str(path.relative_to(ROOT)) for path in input_paths],
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_readme(output_dir, args.n_pairs, args.seed, len(universe), summary)

    assert len(image_paths) == args.n_pairs
    assert manifest["filename"].map(lambda name: (output_dir / name).exists()).all()
    assert manifest[["o_tract_fips", "d_tract_fips"]].drop_duplicates().shape[0] == args.n_pairs
    print(
        f"Completed {args.n_pairs} maps, {len(screenline_table)} screenlines, and "
        f"{len(assignments_table)} colored station links in {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

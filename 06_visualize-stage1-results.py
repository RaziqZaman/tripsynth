#!/usr/bin/env python3
"""Build Stage 1 synthetic-vs-real MATSim comparison tables and charts."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_DIR = Path("06x_stage1_md_internal")
NETWORK_DIR = Path("06x_stage1_osm")
OUTPUT_DIR = SCENARIO_DIR / "stage1_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def scenario_log_path(scenario_dir: Path, scenario: str) -> Path:
    suffix = "real" if scenario == "real" else "synthetic"
    return scenario_dir / f"matsim_{suffix}.log"


def scenario_output_dir(scenario_dir: Path, scenario: str) -> Path:
    suffix = "real" if scenario == "real" else "synthetic"
    return scenario_dir / f"matsim_{suffix}"


def scenario_events_path(scenario_dir: Path, scenario: str) -> Path:
    return scenario_output_dir(scenario_dir, scenario) / "ITERS" / "it.0" / "0.events.xml.gz"


def count_gzip_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as input_file:
        return sum(1 for _ in input_file)


def parse_elapsed(text: str) -> str:
    matches = re.findall(r"matsim elapsed:\s*([0-9:.\sA-Za-z]+)", text)
    return matches[-1].strip() if matches else ""


def elapsed_minutes(value: str) -> float:
    value = value.strip()
    if not value:
        return float("nan")
    if "h" in value:
        match = re.match(r"(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)", value)
        if match:
            return (
                int(match.group("hours")) * 60
                + int(match.group("minutes"))
                + int(match.group("seconds")) / 60
            )
    parts = value.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) + float(parts[1]) / 60
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + float(parts[2]) / 60
    except ValueError:
        return float("nan")
    return float("nan")


def run_status_rows(scenario_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in ["real", "synthetic"]:
        log_path = scenario_log_path(scenario_dir, scenario)
        events_path = scenario_events_path(scenario_dir, scenario)
        persons_path = scenario_output_dir(scenario_dir, scenario) / "output_persons.csv.gz"
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        failure_reason = ""
        if "OutOfMemoryError" in text:
            failure_reason = "java_heap_out_of_memory"
        elif "BUILD FAILURE" in text or "ERROR --- MATSim unexpectedly terminated" in text:
            failure_reason = "matsim_failed"
        elif not events_path.exists():
            failure_reason = "missing_events_file"

        event_bytes = events_path.stat().st_size if events_path.exists() else 0
        event_lines = count_gzip_lines(events_path) if event_bytes and event_bytes < 5_000_000 else None
        empty_events = bool(event_lines is not None and event_lines <= 3)
        if not failure_reason and empty_events:
            failure_reason = "empty_events_file"
        status = "completed" if not failure_reason else "failed"

        rows.append(
            {
                "scenario": scenario,
                "status": status,
                "failure_reason": failure_reason,
                "elapsed": parse_elapsed(text),
                "elapsed_minutes": elapsed_minutes(parse_elapsed(text)),
                "events_file": str(events_path),
                "events_file_bytes": event_bytes,
                "events_line_count_if_small": event_lines if event_lines is not None else "",
                "events_empty": empty_events,
                "output_persons_file": str(persons_path),
                "output_persons_exists": persons_path.exists(),
            }
        )
    return rows


def load_vehicle_trips(scenario_dir: Path) -> pd.DataFrame:
    frames = []
    for scenario in ["real", "synthetic"]:
        path = scenario_dir / f"{scenario}_vehicle_trips.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["scenario"] = scenario
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    trips = pd.concat(frames, ignore_index=True)
    for column in ["vehicle_weight", "departure_time_min", "vehicle_occupancy"]:
        if column in trips:
            trips[column] = pd.to_numeric(trips[column], errors="coerce")
    trips["departure_hour"] = (trips["departure_time_min"] // 60).clip(0, 30).astype("Int64")
    trips["activity_pair"] = (
        trips.get("o_activity", "").astype(str) + " -> " + trips.get("d_activity", "").astype(str)
    )
    return trips


def weighted_summary(trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in trips.groupby("scenario", sort=False):
        weights = group["vehicle_weight"].fillna(0)
        departures = group["departure_time_min"].fillna(0)
        total_weight = float(weights.sum())
        rows.append(
            {
                "scenario": scenario,
                "rows": len(group),
                "vehicle_trips": total_weight,
                "mean_vehicle_weight": float(weights.mean()),
                "median_vehicle_weight": float(weights.median()),
                "weighted_mean_departure_min": float((departures * weights).sum() / total_weight)
                if total_weight
                else float("nan"),
                "weighted_mean_departure_hour": float((departures * weights).sum() / total_weight / 60)
                if total_weight
                else float("nan"),
                "mean_vehicle_occupancy": float(group["vehicle_occupancy"].mean())
                if "vehicle_occupancy" in group
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def weighted_distribution(
    trips: pd.DataFrame,
    column: str,
    out_column: str | None = None,
) -> pd.DataFrame:
    if column not in trips:
        return pd.DataFrame()
    out_column = out_column or column
    data = (
        trips.groupby(["scenario", column], dropna=False)["vehicle_weight"]
        .sum()
        .reset_index(name="vehicle_trips")
    )
    totals = data.groupby("scenario")["vehicle_trips"].transform("sum")
    data["share"] = np.where(totals > 0, data["vehicle_trips"] / totals, np.nan)
    if out_column != column:
        data = data.rename(columns={column: out_column})
    return data.sort_values(["scenario", "vehicle_trips"], ascending=[True, False])


def distribution_overlap(distribution: pd.DataFrame, value_column: str) -> float:
    if distribution.empty:
        return float("nan")
    pivot = distribution.pivot_table(
        index=value_column,
        columns="scenario",
        values="share",
        aggfunc="sum",
        fill_value=0.0,
    )
    if "real" not in pivot or "synthetic" not in pivot:
        return float("nan")
    return float(np.minimum(pivot["real"], pivot["synthetic"]).sum())


def overlap_rows(distributions: dict[str, tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    rows = []
    for metric, (distribution, value_column) in distributions.items():
        rows.append(
            {
                "distribution": metric,
                "overlap": distribution_overlap(distribution, value_column),
            }
        )
    return pd.DataFrame(rows)


def observed_count_summary(network_dir: Path) -> pd.DataFrame:
    observed = read_csv_if_exists(network_dir / "observed_counts.csv")
    if observed.empty:
        return pd.DataFrame(
            [{"metric": "matched_count_points", "value": 0.0}]
        )
    observed["observed_volume"] = pd.to_numeric(observed["observed_volume"], errors="coerce")
    observed["match_distance_m"] = pd.to_numeric(observed["match_distance_m"], errors="coerce")
    metrics = {
        "matched_count_points": len(observed),
        "observed_volume_total": observed["observed_volume"].sum(),
        "observed_volume_mean": observed["observed_volume"].mean(),
        "observed_volume_median": observed["observed_volume"].median(),
        "observed_volume_p90": observed["observed_volume"].quantile(0.9),
        "match_distance_mean_m": observed["match_distance_m"].mean(),
        "match_distance_median_m": observed["match_distance_m"].median(),
        "match_distance_max_m": observed["match_distance_m"].max(),
    }
    return pd.DataFrame(
        [{"metric": metric, "value": value} for metric, value in metrics.items()]
    )


def count_validation_summary(scenario_dir: Path) -> pd.DataFrame:
    summary_path = scenario_dir / "count_validation_summary.csv"
    validation_path = scenario_dir / "count_validation.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        if not validation.empty:
            return validation.groupby("scenario").agg(
                matched_counts=("link_id", "nunique"),
                observed_total=("observed_volume", "sum"),
                model_total=("model_volume", "sum"),
                mae=("absolute_error", "mean"),
                rmse=("error", lambda x: math.sqrt(float(np.mean(np.square(x))))),
                geh_lt_5_share=("geh", lambda x: float((x < 5).mean())),
            ).reset_index()
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "status": "not_available",
                "reason": "MATSim event files are empty or postprocess has not produced count_validation.csv",
            }
            for scenario in ["real", "synthetic"]
        ]
    )


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hourly(hourly: pd.DataFrame, path: Path) -> None:
    if hourly.empty:
        return
    pivot = hourly.pivot_table(
        index="departure_hour",
        columns="scenario",
        values="vehicle_trips",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    plt.figure(figsize=(10, 5))
    for scenario, color in [("real", "#2f6f9f"), ("synthetic", "#d8892b")]:
        if scenario in pivot:
            plt.plot(pivot.index, pivot[scenario], marker="o", linewidth=2, label=scenario, color=color)
    plt.xlabel("Departure hour")
    plt.ylabel("Weighted vehicle trips")
    plt.title("Weighted Vehicle Trips By Departure Hour")
    plt.legend()
    savefig(path)


def plot_overlap(overlaps: pd.DataFrame, path: Path) -> None:
    if overlaps.empty:
        return
    data = overlaps.sort_values("overlap", ascending=True)
    plt.figure(figsize=(8, 5))
    colors = ["#6f8f72" if value >= 0.75 else "#c77c4a" for value in data["overlap"]]
    plt.barh(data["distribution"], data["overlap"], color=colors)
    plt.xlim(0, 1)
    plt.xlabel("Histogram overlap")
    plt.title("Synthetic vs Real Weighted Demand Overlap")
    savefig(path)


def plot_top_categories(
    distribution: pd.DataFrame,
    value_column: str,
    title: str,
    path: Path,
    top_n: int = 15,
) -> None:
    if distribution.empty:
        return
    totals = (
        distribution.groupby(value_column)["vehicle_trips"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    data = distribution[distribution[value_column].isin(totals)].copy()
    data[value_column] = data[value_column].astype(str)
    pivot = data.pivot_table(
        index=value_column,
        columns="scenario",
        values="share",
        aggfunc="sum",
        fill_value=0.0,
    ).loc[[str(value) for value in totals if str(value) in set(data[value_column])]]
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)
    ax = pivot.plot(kind="bar", figsize=(10, 5), color=["#2f6f9f", "#d8892b"])
    ax.set_ylabel("Share of weighted vehicle trips")
    ax.set_xlabel(value_column)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    savefig(path)


def plot_observed_counts(network_dir: Path, out_dir: Path) -> None:
    observed = read_csv_if_exists(network_dir / "observed_counts.csv")
    if observed.empty:
        return
    observed["observed_volume"] = pd.to_numeric(observed["observed_volume"], errors="coerce")
    observed["match_distance_m"] = pd.to_numeric(observed["match_distance_m"], errors="coerce")

    plt.figure(figsize=(8, 5))
    plt.hist(observed["observed_volume"].dropna(), bins=50, color="#5a7d7c")
    plt.xlabel("Observed AADT")
    plt.ylabel("Count sites")
    plt.title("Observed Count Volume Distribution")
    savefig(out_dir / "observed_count_volume_histogram.png")

    plt.figure(figsize=(8, 5))
    plt.hist(observed["match_distance_m"].dropna(), bins=40, color="#876c99")
    plt.xlabel("Nearest OSM link match distance (m)")
    plt.ylabel("Count sites")
    plt.title("MDOT Count-To-OSM Link Match Distance")
    savefig(out_dir / "count_match_distance_histogram.png")


def plot_run_status(run_status: pd.DataFrame, path: Path) -> None:
    if run_status.empty:
        return
    plt.figure(figsize=(8, 4))
    labels = run_status["scenario"].tolist()
    elapsed = run_status["elapsed_minutes"].fillna(0).to_numpy()
    colors = ["#a5483d" if status == "failed" else "#4f8f5b" for status in run_status["status"]]
    plt.bar(labels, elapsed, color=colors)
    plt.ylabel("Runtime before exit (minutes)")
    plt.title("MATSim Run Status")
    for index, row in run_status.iterrows():
        label = row["failure_reason"] or row["status"]
        plt.text(index, elapsed[index] + max(elapsed.max() * 0.02, 1), label, ha="center", va="bottom")
    savefig(path)


def plot_validation_if_available(scenario_dir: Path, out_dir: Path) -> None:
    validation_path = scenario_dir / "count_validation.csv"
    if not validation_path.exists():
        (out_dir / "count_validation_not_available.txt").write_text(
            "count_validation.csv is not available because the MATSim event files are empty or postprocess has not run after a successful simulation.\\n"
        )
        return
    validation = pd.read_csv(validation_path)
    if validation.empty:
        (out_dir / "count_validation_not_available.txt").write_text(
            "count_validation.csv exists but has no matched assigned link volumes.\\n"
        )
        return
    for column in ["observed_volume", "model_volume", "absolute_error", "geh"]:
        validation[column] = pd.to_numeric(validation[column], errors="coerce")

    plt.figure(figsize=(7, 7))
    for scenario, color in [("real", "#2f6f9f"), ("synthetic", "#d8892b")]:
        group = validation[validation["scenario"] == scenario]
        plt.scatter(
            group["observed_volume"],
            group["model_volume"],
            s=10,
            alpha=0.45,
            label=scenario,
            color=color,
        )
    limit = float(np.nanmax(validation[["observed_volume", "model_volume"]].to_numpy()))
    plt.plot([0, limit], [0, limit], color="#444444", linewidth=1)
    plt.xlabel("Observed scaled volume")
    plt.ylabel("Modeled volume")
    plt.title("Assigned vs Observed Count Volumes")
    plt.legend()
    savefig(out_dir / "assigned_vs_observed_scatter.png")

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, min(50, validation["geh"].quantile(0.99)), 50)
    for scenario, color in [("real", "#2f6f9f"), ("synthetic", "#d8892b")]:
        group = validation[validation["scenario"] == scenario]
        plt.hist(group["geh"].dropna(), bins=bins, alpha=0.55, label=scenario, color=color)
    plt.xlabel("GEH")
    plt.ylabel("Count links")
    plt.title("GEH Distribution")
    plt.legend()
    savefig(out_dir / "geh_distribution.png")


def write_outputs(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = args.out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    setup_plot()

    run_status = pd.DataFrame(run_status_rows(args.scenario_dir))
    run_status.to_csv(args.out_dir / "run_status.csv", index=False)
    plot_run_status(run_status, charts_dir / "run_status.png")

    scenario_totals = read_csv_if_exists(args.scenario_dir / "scenario_totals.csv")
    if not scenario_totals.empty:
        scenario_totals.to_csv(args.out_dir / "scenario_totals.csv", index=False)

    trips = load_vehicle_trips(args.scenario_dir)
    if not trips.empty:
        demand_summary = weighted_summary(trips)
        demand_summary.to_csv(args.out_dir / "demand_summary.csv", index=False)

        hourly = weighted_distribution(trips, "departure_hour")
        hourly.to_csv(args.out_dir / "demand_hourly.csv", index=False)
        plot_hourly(hourly, charts_dir / "demand_hourly_vehicle_trips.png")

        mode = weighted_distribution(trips, "travel_mode")
        mode.to_csv(args.out_dir / "demand_travel_mode.csv", index=False)
        plot_top_categories(
            mode,
            "travel_mode",
            "Travel Mode Distribution",
            charts_dir / "demand_travel_mode.png",
        )

        origin_activity = weighted_distribution(trips, "o_activity")
        origin_activity.to_csv(args.out_dir / "demand_origin_activity.csv", index=False)
        destination_activity = weighted_distribution(trips, "d_activity")
        destination_activity.to_csv(args.out_dir / "demand_destination_activity.csv", index=False)
        activity_pair = weighted_distribution(trips, "activity_pair")
        activity_pair.to_csv(args.out_dir / "demand_activity_pair.csv", index=False)
        plot_top_categories(
            activity_pair,
            "activity_pair",
            "Top Origin-Destination Activity Pairs",
            charts_dir / "demand_activity_pairs.png",
            top_n=20,
        )

        origin_taz = weighted_distribution(trips, "o_tpb_taz")
        destination_taz = weighted_distribution(trips, "d_tpb_taz")
        overlaps = overlap_rows(
            {
                "departure_hour": (hourly, "departure_hour"),
                "travel_mode": (mode, "travel_mode"),
                "origin_activity": (origin_activity, "o_activity"),
                "destination_activity": (destination_activity, "d_activity"),
                "activity_pair": (activity_pair, "activity_pair"),
                "origin_taz": (origin_taz, "o_tpb_taz"),
                "destination_taz": (destination_taz, "d_tpb_taz"),
            }
        )
        overlaps.to_csv(args.out_dir / "synthetic_real_demand_overlaps.csv", index=False)
        plot_overlap(overlaps, charts_dir / "synthetic_real_demand_overlaps.png")

    observed_summary = observed_count_summary(args.network_dir)
    observed_summary.to_csv(args.out_dir / "observed_count_summary.csv", index=False)
    plot_observed_counts(args.network_dir, charts_dir)

    validation_summary = count_validation_summary(args.scenario_dir)
    validation_summary.to_csv(args.out_dir / "count_validation_summary.csv", index=False)
    plot_validation_if_available(args.scenario_dir, charts_dir)

    manifest = pd.DataFrame(
        [
            {"path": str(path.relative_to(args.out_dir)), "bytes": path.stat().st_size}
            for path in sorted(args.out_dir.rglob("*"))
            if path.is_file()
        ]
    )
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    print(f"wrote comparison outputs to {args.out_dir}")


def main() -> int:
    args = parse_args()
    write_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

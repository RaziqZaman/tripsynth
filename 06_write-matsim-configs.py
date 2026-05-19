#!/usr/bin/env python3
"""Write MATSim config XML files for the Stage 1 Maryland-internal scenarios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape


STAGE1_DIR = Path("06x_stage1")
SCENARIO_DIR = Path("06x_stage1_md_internal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, default=STAGE1_DIR)
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--last-iteration", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20260519)
    return parser.parse_args()


def activity_types(*vehicle_trip_csvs: Path) -> list[str]:
    values = set()
    for path in vehicle_trip_csvs:
        with path.open(newline="") as input_file:
            for row in csv.DictReader(input_file):
                for column in ["o_activity", "d_activity"]:
                    value = row.get(column, "").strip()
                    if value:
                        values.add(f"act_{value}")
    return sorted(values)


def path_text(path: Path) -> str:
    return escape(str(path.resolve()))


def write_config(
    path: Path,
    network_file: Path,
    population_file: Path,
    output_dir: Path,
    activities: list[str],
    last_iteration: int,
    threads: int,
    random_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    activity_params = "\n".join(
        f'''    <parameterset type="activityParams">
      <param name="activityType" value="{escape(activity)}" />
      <param name="typicalDuration" value="12:00:00" />
      <param name="openingTime" value="00:00:00" />
      <param name="closingTime" value="30:00:00" />
    </parameterset>'''
        for activity in activities
    )

    path.write_text(
        f'''<?xml version="1.0" encoding="UTF-8"?>
<config>
  <module name="global">
    <param name="randomSeed" value="{random_seed}" />
    <param name="coordinateSystem" value="EPSG:4326" />
    <param name="numberOfThreads" value="{threads}" />
  </module>

  <module name="network">
    <param name="inputNetworkFile" value="{path_text(network_file)}" />
  </module>

  <module name="plans">
    <param name="inputPlansFile" value="{path_text(population_file)}" />
  </module>

  <module name="controler">
    <param name="outputDirectory" value="{path_text(output_dir)}" />
    <param name="firstIteration" value="0" />
    <param name="lastIteration" value="{last_iteration}" />
    <param name="overwriteFiles" value="deleteDirectoryIfExists" />
    <param name="writeEventsInterval" value="1" />
    <param name="writePlansInterval" value="1" />
    <param name="createGraphs" value="false" />
  </module>

  <module name="qsim">
    <param name="startTime" value="00:00:00" />
    <param name="endTime" value="30:00:00" />
    <param name="flowCapFactor" value="1.0" />
    <param name="storageCapFactor" value="1.0" />
    <param name="numberOfThreads" value="{threads}" />
    <param name="trafficDynamics" value="queue" />
    <param name="vehiclesSource" value="defaultVehicle" />
    <param name="removeStuckVehicles" value="true" />
    <param name="stuckTime" value="00:30:00" />
  </module>

  <module name="routing">
    <param name="networkModes" value="car" />
  </module>

  <module name="scoring">
    <parameterset type="modeParams">
      <param name="mode" value="car" />
      <param name="constant" value="0.0" />
      <param name="marginalUtilityOfTraveling_util_hr" value="-6.0" />
      <param name="marginalUtilityOfDistance_util_m" value="0.0" />
      <param name="monetaryDistanceRate" value="0.0" />
    </parameterset>
{activity_params}
  </module>
</config>
'''
    )


def main() -> int:
    args = parse_args()
    real_vehicle_trips = args.scenario_dir / "real_vehicle_trips.csv"
    synthetic_vehicle_trips = args.scenario_dir / "synthetic_vehicle_trips.csv"
    activities = activity_types(real_vehicle_trips, synthetic_vehicle_trips)

    write_config(
        args.scenario_dir / "config_real.xml",
        args.stage1_dir / "network.xml.gz",
        args.scenario_dir / "real_population.xml.gz",
        args.scenario_dir / "matsim_real",
        activities,
        args.last_iteration,
        args.threads,
        args.random_seed,
    )
    write_config(
        args.scenario_dir / "config_synthetic.xml",
        args.stage1_dir / "network.xml.gz",
        args.scenario_dir / "synthetic_population.xml.gz",
        args.scenario_dir / "matsim_synthetic",
        activities,
        args.last_iteration,
        args.threads,
        args.random_seed,
    )

    print(f"wrote {args.scenario_dir / 'config_real.xml'}")
    print(f"wrote {args.scenario_dir / 'config_synthetic.xml'}")
    print(f"activity types: {len(activities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

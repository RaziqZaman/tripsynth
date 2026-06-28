"""Common routing interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import geopandas as gpd
import pandas as pd


@dataclass(frozen=True)
class RoutingResult:
    method: str
    routed_volumes: gpd.GeoDataFrame
    metadata: dict[str, Any]
    route_table: pd.DataFrame | None = None
    failed_routes: pd.DataFrame | None = None


class Router(Protocol):
    method: str

    def route(self, trips) -> RoutingResult:
        ...

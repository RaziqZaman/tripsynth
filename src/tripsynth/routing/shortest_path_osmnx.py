"""OSMnx shortest-path routing adapter placeholder."""

from __future__ import annotations


class ShortestPathOSMnxRouter:
    method = "shortest_path_osmnx"

    def route(self, trips):
        raise NotImplementedError("OSMnx routing needs a configured or downloaded road network.")

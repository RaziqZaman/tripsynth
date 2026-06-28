"""Transparent tract-border-crossing routing baseline placeholder."""

from __future__ import annotations


class TractBorderCrossingRouter:
    method = "tract_border_crossing"

    def route(self, trips):
        raise NotImplementedError("Tract-border-crossing routing needs tract geometries.")

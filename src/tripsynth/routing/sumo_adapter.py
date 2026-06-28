"""SUMO export/import adapter placeholder."""

from __future__ import annotations


class SUMOAdapter:
    method = "sumo_or_aequilibrae_adapter"

    def export(self, trips, output_dir):
        raise NotImplementedError("SUMO export is scaffolded for a later pipeline phase.")

    def import_link_volumes(self, path):
        raise NotImplementedError("SUMO link-volume import is scaffolded for a later phase.")

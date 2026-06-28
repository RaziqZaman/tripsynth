"""MATSim export/import adapter placeholder."""

from __future__ import annotations


class MATSimAdapter:
    method = "matsim_adapter"

    def export(self, trips, output_dir):
        raise NotImplementedError("MATSim export is scaffolded for a later pipeline phase.")

    def import_link_volumes(self, path):
        raise NotImplementedError("MATSim link-volume import is scaffolded for a later phase.")

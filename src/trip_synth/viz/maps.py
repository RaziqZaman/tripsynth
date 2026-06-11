from __future__ import annotations


def map_dependencies_available() -> bool:
    try:
        import geopandas  # noqa: F401

        return True
    except Exception:
        return False

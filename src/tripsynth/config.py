"""Configuration loading and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML config and attach lightweight internal path metadata."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    config["_config_path"] = str(config_path.resolve())
    config["_config_dir"] = str(config_path.resolve().parent)
    return config


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    """Resolve paths relative to cwd first, then relative to the config file."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists() or str(value).startswith(("data/", "reports/", "outputs/")):
        return cwd_candidate
    config_dir = Path(config.get("_config_dir", "."))
    return config_dir / path


def ensure_standard_directories() -> None:
    """Create the standard output directories used by first-deliverable commands."""
    for path in [
        "data/raw/survey",
        "data/raw/observed_counts",
        "data/processed/survey",
        "data/processed/observed_counts",
        "data/metadata",
        "reports/tables",
        "reports/figures",
        "reports/maps",
        "outputs",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def flatten_study_counties(config: dict[str, Any]) -> set[str]:
    counties = config.get("study_area", {}).get("counties", {})
    out: set[str] = set()
    for values in counties.values():
        out.update(str(value).zfill(5) for value in values)
    return out


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)

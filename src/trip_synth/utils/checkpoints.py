from __future__ import annotations

from pathlib import Path

from .io import atomic_stage_marker, ensure_dir


def stage_done(run_dir: str | Path, stage: str) -> bool:
    return atomic_stage_marker(run_dir, stage).exists()


def mark_stage_done(run_dir: str | Path, stage: str) -> None:
    marker = atomic_stage_marker(run_dir, stage)
    ensure_dir(marker.parent)
    marker.write_text("done\n")


def latest_checkpoint(checkpoint_dir: str | Path, prefix: str = "epoch_") -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob(f"{prefix}*.pt"))
    return candidates[-1] if candidates else None

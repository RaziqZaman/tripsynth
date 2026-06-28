"""Progress-bar helpers with a quiet fallback."""

from __future__ import annotations

import os
from typing import Any, Iterable, Iterator

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - exercised only when tqdm is absent.
    _tqdm = None


_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _default_disable() -> bool:
    value = os.environ.get("TRIPSYNTH_PROGRESS", "").strip().lower()
    if value in _FALSE_VALUES:
        return True
    if value in _TRUE_VALUES:
        return False
    return "PYTEST_CURRENT_TEST" in os.environ


class _NullProgress:
    def __init__(self, iterable: Iterable[Any] | None = None, total: int | None = None) -> None:
        self.iterable = iterable
        self.total = total
        self.n = 0

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def close(self) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def set_description(self, *args, **kwargs) -> None:
        return None


def progress(
    iterable: Iterable[Any] | None = None,
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str | None = None,
    disable: bool | None = None,
    leave: bool = True,
    **kwargs: Any,
):
    """Return a tqdm progress bar, or a no-op stand-in when disabled/unavailable."""
    if disable is None:
        disable = _default_disable()
    if _tqdm is None or disable:
        return _NullProgress(iterable=iterable, total=total)
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=float(kwargs.pop("mininterval", 1.0)),
        smoothing=float(kwargs.pop("smoothing", 0.1)),
        **kwargs,
    )

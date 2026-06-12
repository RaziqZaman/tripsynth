from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

T = TypeVar("T")


def _progress_enabled(disable: bool | None = None) -> bool:
    if disable is not None:
        return not disable
    return os.environ.get("TRIP_SYNTH_PROGRESS", "1").lower() not in {"0", "false", "no", "off"}


class _BasicProgress:
    def __init__(
        self,
        total: int | None = None,
        desc: str = "",
        unit: str = "it",
        disable: bool | None = None,
        leave: bool = True,
    ) -> None:
        self.total = int(total) if total is not None else None
        self.desc = desc or "progress"
        self.unit = unit
        self.enabled = _progress_enabled(disable)
        self.leave = leave
        self.n = 0
        self._postfix: dict[str, Any] = {}
        self._start = time.time()
        self._last_emit = 0.0
        self._next_count = 0
        if self.enabled:
            self._emit(force=True)

    def __enter__(self) -> "_BasicProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _message(self) -> str:
        if self.total:
            pct = 100.0 * min(self.n, self.total) / self.total
            core = f"{self.desc}: {min(self.n, self.total)}/{self.total} {self.unit} ({pct:.1f}%)"
        else:
            core = f"{self.desc}: {self.n} {self.unit}"
        if self._postfix:
            extra = ", ".join(f"{k}={v}" for k, v in self._postfix.items())
            core = f"{core} | {extra}"
        return core

    def _emit(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last_emit < 5.0 and (self.total is None or self.n < self._next_count):
            return
        elapsed = now - self._start
        sys.stderr.write(f"[progress] {self._message()} elapsed={elapsed:.1f}s\n")
        sys.stderr.flush()
        self._last_emit = now
        if self.total:
            step = max(1, self.total // 20)
            self._next_count = self.n + step

    def update(self, n: int = 1) -> None:
        self.n += int(n)
        self._emit()

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any) -> None:
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self._postfix = values
        if refresh:
            self._emit(force=True)

    def close(self) -> None:
        if self.enabled and self.leave:
            if self.total and self.n < self.total:
                self.n = self.total
            self._emit(force=True)


def _tqdm_kwargs(desc: str, unit: str, disable: bool | None, leave: bool) -> dict[str, Any]:
    return {
        "desc": desc,
        "unit": unit,
        "dynamic_ncols": True,
        "disable": not _progress_enabled(disable),
        "leave": leave,
    }


def progress_bar(
    total: int | None,
    desc: str,
    unit: str = "it",
    disable: bool | None = None,
    leave: bool = True,
):
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, **_tqdm_kwargs(desc, unit, disable, leave))
    except Exception:
        return _BasicProgress(total=total, desc=desc, unit=unit, disable=disable, leave=leave)


def progress_iter(
    iterable: Iterable[T],
    desc: str,
    total: int | None = None,
    unit: str = "it",
    disable: bool | None = None,
    leave: bool = True,
) -> Iterator[T]:
    try:
        from tqdm.auto import tqdm

        yield from tqdm(iterable, total=total, **_tqdm_kwargs(desc, unit, disable, leave))
    except Exception:
        bar = _BasicProgress(total=total, desc=desc, unit=unit, disable=disable, leave=leave)
        try:
            for item in iterable:
                yield item
                bar.update()
        finally:
            bar.close()

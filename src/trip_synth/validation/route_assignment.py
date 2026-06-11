from __future__ import annotations

from typing import Iterable


def screenline_id(a: str, b: str) -> str:
    aa = str(a)
    bb = str(b)
    return "__".join(sorted([aa, bb]))


def adjacent_pairs_from_ordered_tracts(tracts: Iterable[str]) -> list[tuple[str, str]]:
    ordered = [str(t) for t in tracts]
    return [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1) if ordered[i] != ordered[i + 1]]


class RouteAssignmentNotAvailable(RuntimeError):
    """Raised when optional network route assignment dependencies are missing."""

"""Interval-isolation backend (rigorous). Phase 3 — needs C++ mp_interval. Placeholder."""

from typing import List

from ..formula import Number


def find_all(func, t_min: Number, t_max: Number, precision: int, **_) -> List[Number]:
    """Rigorously enclose all roots of g (needs the C++ mp_interval type, phase 3)."""
    raise NotImplementedError("interval backend requires the C++ mp_interval type")

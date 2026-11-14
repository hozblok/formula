"""Homogeneous world coordinates for scalar and jet surface calculations."""

from collections.abc import Sequence

from ...backend import MP_TYPES, round_up_precision
from ...formula import NumberContext
from .jets import _Jet


def surface_coordinates(point: Sequence, precision: int) -> tuple:
    """Prepare xyz as backend values or jets of one algebra, without changing a value's precision."""
    x, y, z = point
    precision = round_up_precision(precision)
    coordinates = (x, y, z)
    if any(not isinstance(v, _Jet) and getattr(v, "precision", precision) != precision
           for v in coordinates):
        raise ValueError("surface and coordinate precision must agree")
    jets = [v for v in coordinates if isinstance(v, _Jet)]
    if jets:
        alg = jets[0].alg
        if any(v.alg is not alg for v in jets):
            raise ValueError("jets must belong to the same algebra")
        if round_up_precision(alg.precision) != precision:
            raise ValueError("surface and jet precision must agree")
    N = NumberContext(precision)
    coordinates = tuple(v if isinstance(v, (*MP_TYPES, _Jet)) else N(v)
                        for v in coordinates)
    if jets:
        return tuple(v if isinstance(v, _Jet) else alg.constant(v) for v in coordinates)
    return coordinates

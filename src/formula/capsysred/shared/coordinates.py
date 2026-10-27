"""Homogeneous world coordinates for scalar and jet surface calculations."""

from collections.abc import Sequence

from ...backend import round_up_precision
from ...formula import Number
from .jets import _Jet


def surface_coordinates(
    point: Sequence[int | float | Number | _Jet], precision: int,
) -> tuple[Number, ...] | tuple[_Jet, ...]:
    """Prepare xyz as Numbers or jets of one algebra, without changing Number precision."""
    x, y, z = point
    precision = round_up_precision(precision)
    coordinates = (x, y, z)
    if any(isinstance(v, Number) and v.precision != precision for v in coordinates):
        raise ValueError("surface and coordinate precision must agree")
    jets = [v for v in coordinates if isinstance(v, _Jet)]
    if jets:
        alg = jets[0].alg
        if any(v.alg is not alg for v in jets):
            raise ValueError("jets must belong to the same algebra")
        if round_up_precision(alg.precision) != precision:
            raise ValueError("surface and jet precision must agree")
    coordinates = tuple(v if isinstance(v, (Number, _Jet)) else Number(v, precision)
                        for v in coordinates)
    if jets:
        return tuple(v if isinstance(v, _Jet) else alg.constant(v) for v in coordinates)
    return coordinates

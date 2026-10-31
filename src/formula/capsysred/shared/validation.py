"""Validation of numerical inputs."""

from collections.abc import Iterable, Mapping, Set
import math


def finite_vector(value: Iterable[object], size: int, name: str) -> tuple[float, ...]:
    """Convert a flat, ordered iterable to exactly size finite float coordinates."""
    try:
        if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
            raise ValueError
        coordinates = tuple(value)
        if len(coordinates) != size or any(getattr(v, "ndim", 0) != 0 for v in coordinates):
            raise ValueError
        result = tuple(map(float, coordinates))
        if not all(math.isfinite(v) for v in result):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must contain {size} finite coordinates") from None
    return result

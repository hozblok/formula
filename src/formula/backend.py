"""Bridge to the compiled mp_real/mp_complex backend classes.

The single Python place that encodes the mp_real_<P> / mp_complex_<P> naming.
"""

# pylint: disable=no-name-in-module, import-error
from . import _formula

_REAL_PREFIX = "mp_real_"
_COMPLEX_PREFIX = "mp_complex_"


def mp_class(precision: int, is_complex: bool = False) -> type:
    """Return the mp_real_<P> / mp_complex_<P> backend class for this precision."""
    prefix = _COMPLEX_PREFIX if is_complex else _REAL_PREFIX
    return getattr(_formula, f"{prefix}{precision}")


# Backend classes by kind, for isinstance() checks.
REAL_TYPES = tuple(
    getattr(_formula, n) for n in dir(_formula) if n.startswith(_REAL_PREFIX)
)
COMPLEX_TYPES = tuple(
    getattr(_formula, n) for n in dir(_formula) if n.startswith(_COMPLEX_PREFIX)
)

# Engine precision ceiling (csconstants::max_precision), independent of
# which mp_real_<P> wrappers are bound.
MAX_PRECISION = _formula.MAX_PRECISION

# The engine's rounding rule (single source of truth): smallest supported
# precision >= the request, i.e. what a value is stored at (0 if over the max).
round_up_precision = _formula.round_up_precision

MP_TYPES = REAL_TYPES + COMPLEX_TYPES

def mp_precision(value) -> int:
    if not isinstance(value, MP_TYPES):
        raise TypeError(
            f"expected mp_real_* or mp_complex_*, got {type(value).__name__}"
        )
    name = type(value).__name__
    if name.startswith(_REAL_PREFIX):
        suffix = name[len(_REAL_PREFIX):]
    elif name.startswith(_COMPLEX_PREFIX):
        suffix = name[len(_COMPLEX_PREFIX):]
    else:
        raise TypeError(f"unexpected mp class name: {name!r}")
    try:
        return int(suffix)
    except ValueError as exc:
        raise TypeError(f"invalid precision in mp class name: {name!r}") from exc

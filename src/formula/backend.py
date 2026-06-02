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

# Largest precision the C++ backend was built with (mirrors AllowedPrecisions).
MAX_PRECISION = max(
    int(n[len(_REAL_PREFIX):]) for n in dir(_formula) if n.startswith(_REAL_PREFIX)
)

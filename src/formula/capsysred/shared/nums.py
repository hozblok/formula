"""Scalar and 3-vector kit: fast literal lift, value text, cached solvers.

Precision travels inside the values (never as a parameter downstream): `lift` and
`text` are the entries that take one, and callers derive it from a value they hold.
"""

from ..._formula import FmtFlags
from ...backend import MP_TYPES, mp_number, round_up_precision
from ...formula import Solver

_SOLVERS: dict = {}


def lift(value, precision: int):
    """float/int/str literal -> backend value at the storage precision (hot-path constructor)."""
    s = value if isinstance(value, str) else repr(value)
    return mp_number(round_up_precision(precision), s)


def text(value, precision: int) -> str:
    """Value text at the storage precision's significant digits; other scalars as str()."""
    if isinstance(value, MP_TYPES):
        return value.str(round_up_precision(precision), FmtFlags.default)
    return str(value)


def solver(expression: str, precision: int) -> Solver:
    """Cached Solver for a fixed expression at a precision."""
    key = (expression, round_up_precision(precision))
    if key not in _SOLVERS:
        _SOLVERS[key] = Solver(expression, key[1])
    return _SOLVERS[key]


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vnorm(a):
    return vdot(a, a).sqrt()


def vunit(a):
    n = vnorm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


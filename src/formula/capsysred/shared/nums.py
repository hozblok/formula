"""Scalar and 3-vector kit on Number: fast float lift, cached solvers.

Precision travels inside Numbers (never as a parameter downstream): `lift` is the
only entry that takes one, and callers derive it from a Number they already hold.
"""

from ...backend import mp_class, round_up_precision
from ...formula import Number, Solver

_SOLVERS: dict = {}


def lift(value, precision: int) -> Number:
    """float/int/str -> Number without formula parsing (hot-path constructor)."""
    s = value if isinstance(value, str) else repr(value)
    return Number(mp_class(round_up_precision(precision))(s))


def raw(number: Number):
    """Number -> its backend mp value (the C++ boundary takes these)."""
    return number._value


def solver(expression: str, precision: int) -> Solver:
    """Cached Solver for a fixed expression at a precision."""
    key = (expression, round_up_precision(precision))
    if key not in _SOLVERS:
        _SOLVERS[key] = Solver(expression, key[1])
    return _SOLVERS[key]


def exp_i(x: Number) -> Number:
    """exp(i*x) for a real Number x (unit-modulus phase factor)."""
    return solver("exp(i*x)", x.precision).number({"x": str(x)})


def sqrt(x: Number) -> Number:
    return x.sqrt()


def vdot(a, b) -> Number:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s: Number):
    return (a[0] * s, a[1] * s, a[2] * s)


def vnorm(a) -> Number:
    return sqrt(vdot(a, a))


def vunit(a):
    n = vnorm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def conj(z: Number) -> Number:
    return z.real - Number("i", z.precision) * z.imag

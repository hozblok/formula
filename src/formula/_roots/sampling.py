"""Reference backend: sample, bracket sign changes, refine (safeguarded Newton).

Not rigorous — even-multiplicity roots and sub-sample features can be missed.
Serves as the baseline/oracle that the rigorous backends are checked against.
"""

from typing import List

from ..formula import Number


def _tol(precision: int) -> Number:
    return Number("1e-{}".format(max(precision - 2, 1)), precision)


def _sign(value: Number) -> int:
    if value._is_complex:
        raise NotImplementedError(
            "sampling backend handles real surfaces only; use sturm/chebyshev"
        )
    zero = Number(0, value._precision)
    if value > zero:
        return 1
    if value < zero:
        return -1
    return 0


def _rtsafe(func, a: Number, b: Number, fa: Number, fb: Number, xacc: Number, maxit=200) -> Number:
    """Newton step bracketed by bisection; converges on the root in [a, b]."""
    if _sign(fa) < 0:
        lo, hi = a, b
    else:
        lo, hi = b, a
    t = (a + b) * Number("0.5", a._precision)
    dt_old = abs(b - a)
    dt = dt_old
    g = func.g(t)
    gp = func.gprime(t)
    zero = Number(0, a._precision)
    for _ in range(maxit):
        out_of_range = ((t - hi) * gp - g) * ((t - lo) * gp - g) > zero
        newton_step = gp == zero or out_of_range or abs(g + g) > abs(dt_old * gp)
        dt_old = dt
        if newton_step:
            dt = (hi - lo) * Number("0.5", a._precision)
            t = lo + dt
        else:
            dt = g / gp
            t = t - dt
        if abs(dt) < xacc:
            return t
        g = func.g(t)
        gp = func.gprime(t)
        if _sign(g) < 0:
            lo = t
        else:
            hi = t
    return t


def find_all(func, t_min: Number, t_max: Number, precision: int, samples: int = 256, **_) -> List[Number]:
    xacc = _tol(precision)
    step = (t_max - t_min) / Number(samples, precision)
    roots: List[Number] = []
    t_prev = t_min
    g_prev = func.g(t_prev)
    if _sign(g_prev) == 0:
        roots.append(t_prev)
    for i in range(1, samples + 1):
        t_cur = t_min + step * Number(i, precision)
        g_cur = func.g(t_cur)
        s_cur = _sign(g_cur)
        if s_cur == 0:
            roots.append(t_cur)
        elif _sign(g_prev) * s_cur < 0:
            roots.append(_rtsafe(func, t_prev, t_cur, g_prev, g_cur, xacc))
        t_prev, g_prev = t_cur, g_cur
    return _dedupe(roots, xacc)


def _dedupe(roots: List[Number], xacc: Number) -> List[Number]:
    roots = sorted(roots)
    out: List[Number] = []
    for t in roots:
        if not out or abs(t - out[-1]) > xacc:
            out.append(t)
    return out

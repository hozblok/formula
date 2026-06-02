"""Chebyshev-proxy backend (general analytic surfaces: sin/exp/log).

g is approximated on [t_min, t_max] by a Chebyshev interpolant (Chebyshev-Gauss
nodes), converted to a monomial proxy, whose real roots are isolated by the same
Sturm machinery and then polished on the true g. Captures all roots of smooth g,
including the even-multiplicity ones sampling misses.
"""

from typing import List

from ..formula import Number, Solver
from . import _isolate, _poly


def _cheb_t(k: int, x: Number) -> Number:
    """T_k(x) by recurrence."""
    if k == 0:
        return Number(1, x.precision)
    t0, t1 = Number(1, x.precision), x
    for _ in range(2, k + 1):
        t0, t1 = t1, x * t1 * Number(2, x.precision) - t0
    return t1


def _cheb_fit(func, mid, span, m, prec):
    """Chebyshev-Gauss coefficients a_k of g on the mapped interval."""
    pi = Number("4*atan(1)", prec)
    cos_solver = Solver("cos(a)", prec)
    half = Number("0.5", prec)
    xs, fs = [], []
    for j in range(m):
        theta = pi * (Number(j, prec) + half) / Number(m, prec)
        x = Number.wrap(cos_solver.evaluate({"a": str(theta)}), prec)
        xs.append(x)
        fs.append(Number(func.g(mid + span * x).parts()[0], prec))
    two_over_m = Number(2, prec) / Number(m, prec)
    return [sum((f * _cheb_t(k, x) for x, f in zip(xs, fs)), Number(0, prec)) * two_over_m
            for k in range(m)]


def _cheb_to_monomial(coeffs, prec):
    """Sum_k coeffs[k]*T_k(x) - coeffs[0]/2 as monomial coefficients."""
    half = Number("0.5", prec)
    t_prev = [Number(1, prec)]                    # T_0
    t_cur = [Number(0, prec), Number(1, prec)]    # T_1
    poly = [coeffs[0] * half]                      # a0*T_0 - a0/2 = a0/2
    if len(coeffs) > 1:
        poly = _padd(poly, [c * coeffs[1] for c in t_cur])
    for k in range(2, len(coeffs)):
        t_next = _psub(_pmul_x2(t_cur), t_prev)    # T_{k+1}=2x T_k - T_{k-1}
        poly = _padd(poly, [c * coeffs[k] for c in t_next])
        t_prev, t_cur = t_cur, t_next
    return poly


def _padd(a, b):
    if len(a) < len(b):
        a, b = b, a
    out = list(a)
    for i, bi in enumerate(b):
        out[i] = out[i] + bi
    return out


def _psub(a, b):
    return _padd(a, [-bi for bi in b])


def _pmul_x2(p):
    prec = p[0].precision
    return [Number(0, prec)] + [c * Number(2, prec) for c in p]


def _newton(func, t: Number, xacc: Number, maxit: int = 100) -> Number:
    """Polish a simple root of g starting from a good guess."""
    zero = Number(0, t.precision)
    for _ in range(maxit):
        gp = func.gprime(t)
        if gp == zero:
            return t
        step = func.g(t) / gp
        t = t - step
        if abs(step) < xacc:
            break
    return t


def _proxy_roots(func, q, transform, tol, xacc):
    """Isolate roots of the monomial proxy q on [-1,1], map to t, polish simple ones."""
    mid, span = transform
    chain = _isolate.sturm_chain(q, tol)
    one = Number(1, mid.precision)
    roots = []
    for xl, xr in _isolate.isolate(chain, -one, one):
        t_root = mid + span * _isolate.bisect(q, xl, xr, xacc)
        if abs(func.gprime(t_root)) > tol:  # simple root: polish on true g
            t_root = _newton(func, t_root, xacc)
        roots.append(t_root)
    return roots


def find_all(
    func, t_min: Number, t_max: Number, precision: int, cheb_degree: int = 32, **_
) -> List[Number]:
    """All real roots of g in [t_min, t_max] via a Chebyshev polynomial proxy."""
    if func.g(t_min).is_complex:
        raise NotImplementedError("chebyshev backend handles real surfaces only")
    tol = Number(f"1e-{max(precision // 3, 4)}", precision)
    xacc = Number(f"1e-{max(precision - 2, 1)}", precision)
    half = Number("0.5", precision)
    mid, span = (t_min + t_max) * half, (t_max - t_min) * half
    coeffs = _cheb_fit(func, mid, span, cheb_degree, precision)
    q = _poly.square_free(_poly.trim(_cheb_to_monomial(coeffs, precision), tol), tol)
    if _poly.deg(q, tol) == 0:
        return []
    roots = _proxy_roots(func, q, (mid, span), tol, xacc)
    return sorted(r for r in roots if t_min - xacc <= r <= t_max + xacc)

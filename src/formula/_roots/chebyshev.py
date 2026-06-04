"""Chebyshev-proxy backend (general analytic surfaces: sin/exp/log).

g is approximated on [t_min, t_max] by a Chebyshev interpolant (Chebyshev-Gauss
nodes), converted to a monomial proxy, whose real roots are isolated by the same
Sturm machinery and then polished on the true g. Captures all roots of smooth g,
including the even-multiplicity ones sampling misses.
"""

from typing import List

from ..formula import Number, Solver
from . import _isolate, _poly

_MAX_DEGREE = 256


def _gauss_samples(func, mid, span, m, prec):
    """Chebyshev-Gauss nodes x_j and g values f_j on the mapped interval."""
    pi = Number("4*atan(1)", prec)
    cos_solver = Solver("cos(a)", prec)
    half = Number("0.5", prec)
    xs, fs = [], []
    for j in range(m):
        theta = pi * (Number(j, prec) + half) / Number(m, prec)
        x = Number.wrap(cos_solver.evaluate({"a": str(theta)}), prec)
        xs.append(x)
        fs.append(Number(func.g(mid + span * x).parts()[0], prec))
    return xs, fs


def _cheb_fit(func, mid, span, m, prec):
    """Chebyshev-Gauss coefficients a_k of g on the mapped interval (O(m^2))."""
    xs, fs = _gauss_samples(func, mid, span, m, prec)
    two = Number(2, prec)
    coeffs = [Number(0, prec) for _ in range(m)]
    for x, f in zip(xs, fs):
        t0, t1 = Number(1, prec), x  # T_0(x), T_1(x)
        coeffs[0] = coeffs[0] + f
        for k in range(1, m):
            coeffs[k] = coeffs[k] + f * t1
            t0, t1 = t1, two * x * t1 - t0
    inv = two / Number(m, prec)
    return [c * inv for c in coeffs]


def _tail_small(coeffs, tol) -> bool:
    """True when the high-order Chebyshev coefficients have decayed below tol."""
    prec = coeffs[0].precision
    scale = Number(0, prec)
    for c in coeffs:
        scale = max(scale, abs(c))
    if scale == Number(0, prec):
        return True
    tail = Number(0, prec)
    for c in coeffs[-max(3, len(coeffs) // 8):]:
        tail = max(tail, abs(c))
    return tail < scale * tol


def _converged_fit(func, transform, prec, tol, start_degree):
    """Self-validate: grow the degree until the spectral tail converges."""
    mid, span = transform
    degree = start_degree
    coeffs = _cheb_fit(func, mid, span, degree, prec)
    while degree < _MAX_DEGREE and not _tail_small(coeffs, tol):
        degree = min(degree * 2, _MAX_DEGREE)
        coeffs = _cheb_fit(func, mid, span, degree, prec)
    return coeffs


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
    coeffs = _converged_fit(func, (mid, span), precision, tol, cheb_degree)
    q = _poly.square_free(_poly.trim(_cheb_to_monomial(coeffs, precision), tol), tol)
    if _poly.deg(q, tol) == 0:
        return []
    roots = _proxy_roots(func, q, (mid, span), tol, xacc)
    return sorted(r for r in roots if t_min - xacc <= r <= t_max + xacc)

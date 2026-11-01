"""Chebyshev-proxy backend (general analytic surfaces: sin/exp/log).

g is approximated on [t_min, t_max] by a Chebyshev interpolant (Chebyshev-Gauss
nodes), converted to a monomial proxy, whose real roots are isolated by the same
Sturm machinery and then polished on the true g. Captures all roots of smooth g,
including the even-multiplicity ones sampling misses.
"""

from ..formula import Number, NumberContext, Solver
from . import _isolate, _poly

_MAX_DEGREE = 256


def _gauss_samples(func, mid, span, m, N: NumberContext):
    """Chebyshev-Gauss nodes x_j and g values f_j on the mapped interval."""
    pi = Number("4*atan(1)", N.precision)
    cos_solver = Solver("cos(a)", N.precision)
    xs, fs = [], []
    for j in range(m):
        theta = pi * (j + 0.5) / m
        x = cos_solver.number({"a": str(theta)})
        xs.append(x)
        fs.append(func.g(mid + span * x).real)
    return xs, fs


def _cheb_fit(func, mid, span, m, N: NumberContext):
    """Chebyshev-Gauss coefficients a_k of g on the mapped interval (O(m^2))."""
    xs, fs = _gauss_samples(func, mid, span, m, N)
    two = N(2)
    coeffs = [0] * m
    for x, f in zip(xs, fs):
        t0, t1 = 1, x  # T_0(x), T_1(x)
        coeffs[0] = coeffs[0] + f
        for k in range(1, m):
            coeffs[k] = coeffs[k] + f * t1
            t0, t1 = t1, two * x * t1 - t0
    inv = two / m  # mp division; 2 / m in Python is a float
    return [c * inv for c in coeffs]


def _tail_small(coeffs, tol) -> bool:
    """True when the high-order Chebyshev coefficients have decayed below tol."""
    scale = 0
    for c in coeffs:
        scale = max(scale, abs(c))
    if scale == 0:
        return True
    tail = 0
    for c in coeffs[-max(3, len(coeffs) // 8):]:
        tail = max(tail, abs(c))
    return tail < scale * tol


def _converged_fit(func, transform, N: NumberContext, tol, start_degree):
    """Self-validate: grow the degree until the spectral tail converges."""
    mid, span = transform
    degree = start_degree
    coeffs = _cheb_fit(func, mid, span, degree, N)
    while degree < _MAX_DEGREE and not _tail_small(coeffs, tol):
        degree = min(degree * 2, _MAX_DEGREE)
        coeffs = _cheb_fit(func, mid, span, degree, N)
    return coeffs


def _cheb_to_monomial(coeffs, N: NumberContext):
    """Sum_k coeffs[k]*T_k(x) - coeffs[0]/2 as monomial coefficients."""
    t_prev = [N(1)]                                # T_0
    t_cur = [N(0), N(1)]                           # T_1
    poly = [coeffs[0] * 0.5]                       # a0*T_0 - a0/2 = a0/2
    if len(coeffs) > 1:
        poly = _padd(poly, [c * coeffs[1] for c in t_cur])
    for k in range(2, len(coeffs)):
        t_next = _psub(_pmul_x2(t_cur, N), t_prev)  # T_{k+1}=2x T_k - T_{k-1}
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


def _pmul_x2(p, N: NumberContext):
    return [N(0)] + [c * 2 for c in p]


def _newton(func, t, xacc, maxit: int = 100):
    """Polish a simple root of g starting from a good guess."""
    for _ in range(maxit):
        gp = func.gprime(t)
        if gp == 0:
            return t
        step = func.g(t) / gp
        t = t - step
        if abs(step) < xacc:
            break
    return t


def _proxy_roots(func, q, transform, tol, xacc, N: NumberContext):
    """Isolate roots of the monomial proxy q on [-1,1], map to t, polish simple ones."""
    mid, span = transform
    chain = _isolate.sturm_chain(q, tol, N)
    one = N(1)
    roots = []
    for x in _isolate.isolate_roots(q, chain, -one, one, xacc, tol):
        t_root = mid + span * x
        if abs(func.gprime(t_root)) > tol:  # simple root: polish on true g
            t_root = _newton(func, t_root, xacc)
        roots.append(t_root)
    return roots


def find_all(
    func, t_min, t_max, precision: int, cheb_degree: int = 32, **_
) -> list:
    """All real roots of g in [t_min, t_max] via a Chebyshev polynomial proxy."""
    if func.g(t_min).is_complex:
        raise NotImplementedError("chebyshev backend handles real surfaces only")
    N = NumberContext(precision)
    tol = N(f"1e-{max(precision // 3, 4)}")
    xacc = N(f"1e-{max(precision - 2, 1)}")
    mid, span = (t_min + t_max) * 0.5, (t_max - t_min) * 0.5
    coeffs = _converged_fit(func, (mid, span), N, tol, cheb_degree)
    q = _poly.square_free(_poly.trim(_cheb_to_monomial(coeffs, N), tol), tol, N)
    if _poly.deg(q, tol) == 0:
        return []
    roots = _proxy_roots(func, q, (mid, span), tol, xacc, N)
    return sorted(r for r in roots if t_min - xacc <= r <= t_max + xacc)

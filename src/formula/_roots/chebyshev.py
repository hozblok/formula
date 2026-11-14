"""Chebyshev-proxy backend (general analytic surfaces: sin/exp/log).

g is approximated on [t_min, t_max] by a Chebyshev interpolant (Chebyshev-Gauss
nodes), converted to a monomial proxy, whose real roots are isolated by the same
Sturm machinery and then polished on the true g. Captures all roots of smooth g,
including the even-multiplicity ones sampling misses.
"""

from .._formula import FmtFlags
from ..formula import NumberContext
from . import _isolate, _poly
from .utils import ZeroFunctionError, finite, tol_full, tol_third

_MAX_DEGREE = 256


def _gauss_samples(func, mid, span, m, N: NumberContext):
    """Chebyshev-Gauss nodes x_j and g values f_j on the mapped interval."""
    pi = N.eval("4*atan(1)")
    xs, fs = [], []
    for j in range(m):
        theta = pi * (j + 0.5) / m
        # cos of the p-significant-digit text of theta
        x = N.eval("cos(a)", a=theta.str(N.precision, FmtFlags.default))
        xs.append(x)
        f = func.g(mid + span * x).real
        if not finite(f):
            raise ValueError(f"non-finite g at t = {mid + span * x}")
        fs.append(f)
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
        if not finite(gp) or gp == 0:  # a kink or a turning point: the guess stands
            return t
        g = func.g(t)
        if not finite(g):
            raise ValueError(f"non-finite g at t = {t}")
        step = g / gp
        t = t - step
        if abs(step) < xacc:
            break
    return t


def _proxy_noise(proxy, cut, coeffs, N: NumberContext):
    """Bound on |proxy - g/scale| near x: rounding, the monomial terms trim cut (cut[0] at
    the power len(proxy)) and the fit's truncation (len(coeffs) times its last coefficients)."""
    rounding = _isolate.noise(proxy, N)
    mags = [abs(c) for c in cut]
    tail = max(abs(c) for c in coeffs[-max(3, len(coeffs) // 8):]) * len(coeffs)

    def bound(x):
        ax = abs(x)
        dropped = _poly.peval(mags, ax) * ax ** len(proxy) if mags else 0
        return rounding(x) + dropped + tail
    return bound


def _proxy_roots(func, proxy, q, transform, tol, gtol, xacc, noise, N: NumberContext):
    """Isolate roots of q, the square-free part of the monomial proxy, on [-1,1], judge
    them on the proxy itself within noise, map to t, polish the simple ones (|g'| above
    gtol) on g."""
    mid, span = transform
    chain = _isolate.sturm_chain(q, tol, N)
    one = N(1)
    roots = []
    xs = _isolate.isolate_roots(q, chain, -one, one, xacc, tol)
    kept = _isolate.resolved(proxy, xs, -one, one, xacc, noise,
                             where=lambda u: mid + span * u,
                             fused=_isolate.fused_distinct(proxy, q, tol, N) if xs else 0)
    for x in kept:
        t_root = mid + span * x
        gp = func.gprime(t_root)
        if finite(gp) and abs(gp) > gtol:
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
    tol, xacc = tol_third(N), tol_full(N)
    mid, span = (t_min + t_max) * 0.5, (t_max - t_min) * 0.5
    coeffs, scale = _poly.unit_scaled(_converged_fit(func, (mid, span), N, tol, cheb_degree))
    if not scale:
        raise ZeroFunctionError(
            "g is zero at every Chebyshev node: the ray lies in the surface "
            "within tolerance; no isolated-root list exists")
    full = _cheb_to_monomial(coeffs, N)
    proxy = _poly.trim(full, tol)
    q = _poly.square_free(proxy, tol, N)
    if _poly.deg(q, tol) == 0:
        return []
    noise = _proxy_noise(proxy, full[len(proxy):], coeffs, N)
    roots = _proxy_roots(func, proxy, q, (mid, span), tol, tol * scale, xacc, noise, N)
    return sorted(r for r in roots if t_min - xacc <= r <= t_max + xacc)

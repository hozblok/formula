"""Polynomial backend: exact all-roots via Sturm sequences.

g(t)=F(O+t*d) is recovered as a univariate polynomial by interpolation in the
scaled coordinate u in [-1, 1] (t = mid + span*u), made square-free, and its
real roots are counted/isolated by a Sturm chain, then bisected to precision.
Complete for algebraic surfaces; the geometry is real-only.

The u-scaling keeps the interpolated coefficients comparable in size, so the
degree cutoff (scale*tol) cannot mistake a true leading term for noise on
badly scaled polynomials (a micrometre torus quartic spans ~16 orders in raw t).
"""

from ..formula import NumberContext
from . import _isolate, _poly
from .utils import ZeroFunctionError, tol_full, tol_half


def _nodes(t_min, t_max, count: int):
    step = (t_max - t_min) / (count - 1)
    return [t_min + step * i for i in range(count)]


def _poly_from_samples(func, us, ts, tol, max_degree, N: NumberContext):
    """Interpolate g over nodes us (g evaluated at the mapped ts) to a polynomial, g taken
    in units of its largest sampled magnitude, so the absolute tol acts on any scale."""
    samples, scale = _poly.unit_scaled([func.g(t) for t in ts])
    if not scale:
        raise ZeroFunctionError(
            "g is zero at every interpolation node: the ray lies in the surface "
            "within tolerance; no isolated-root list exists")
    return _poly.interpolate(us, samples, tol, max_degree, N)


def find_all(
    func, t_min, t_max, precision: int, max_degree: int = 16, **_
) -> list:
    """All real roots of g in [t_min, t_max] via Sturm isolation."""
    if max_degree < 2:
        raise ValueError("max_degree must be >= 2 (need >= 3 interpolation nodes)")
    N = NumberContext(precision)
    tol, xacc = tol_half(N), tol_full(N)
    if func.g(t_min).is_complex:
        raise NotImplementedError("the geometry is real-only: a complex g has no "
                                  "sign and no isolated real roots")
    one = N(1)
    mid, span = (t_min + t_max) * 0.5, (t_max - t_min) * 0.5
    # bisection runs in u; dividing by span keeps the t-units accuracy at xacc
    xacc = xacc / max(span, one)
    # max_degree+2 nodes: one guard node above the cap so an over-degree surface
    # is detected and rejected rather than silently under-fit.
    us = _nodes(-one, one, max_degree + 2)
    p = _poly_from_samples(func, us, [mid + span * u for u in us], tol, max_degree, N)
    if _poly.deg(p, tol) == 0:
        return []
    q = _poly.square_free(p, tol, N)
    if _poly.deg(q, tol) == 0:
        return []
    chain = _isolate.sturm_chain(q, tol, N)
    roots = _isolate.isolate_roots(q, chain, -one, one, xacc, tol)
    # the square-free gcd is approximate: judge the roots on p itself
    roots = _isolate.resolved(p, roots, -one, one, xacc, _isolate.noise(p, N),
                              where=lambda u: mid + span * u)
    t_acc = tol_full(N)
    return sorted(t for t in (mid + span * u for u in roots)
                  if t_min - t_acc <= t <= t_max + t_acc)

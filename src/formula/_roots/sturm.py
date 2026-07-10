"""Polynomial backend: exact all-roots via Sturm sequences.

g(t)=F(O+t*d) is recovered as a univariate polynomial by interpolation in the
scaled coordinate u in [-1, 1] (t = mid + span*u), made square-free, and its
real roots are counted/isolated by a Sturm chain, then bisected to precision.
Complete for algebraic surfaces. Complex surfaces: real intersections are the
common roots of Re g and Im g (their polynomial gcd).

The u-scaling keeps the interpolated coefficients comparable in size, so the
degree cutoff (scale*tol) cannot mistake a true leading term for noise on
badly scaled polynomials (a micrometre torus quartic spans ~16 orders in raw t).
"""

from typing import List

from ..formula import Number
from . import _isolate, _poly


def _nodes(t_min: Number, t_max: Number, count: int, prec: int) -> List[Number]:
    step = (t_max - t_min) / Number(count - 1, prec)
    return [t_min + step * Number(i, prec) for i in range(count)]


def _poly_from_samples(func, us, ts, complex_surface, tol, max_degree):
    """Interpolate g over nodes us (g evaluated at the mapped ts) to a polynomial."""
    reals, imags = [], []
    for t in ts:
        g = func.g(t)
        reals.append(g.real)
        imags.append(g.imag)
    pr = _poly.interpolate(us, reals, tol, max_degree)
    if not complex_surface:
        return pr
    pi = _poly.interpolate(us, imags, tol, max_degree)
    return _poly.pgcd(pr, pi, tol)  # common roots of Re g and Im g


def find_all(
    func, t_min: Number, t_max: Number, precision: int, max_degree: int = 16, **_
) -> List[Number]:
    """All real roots of g in [t_min, t_max] via Sturm isolation."""
    if max_degree < 2:
        raise ValueError("max_degree must be >= 2 (need >= 3 interpolation nodes)")
    tol = Number(f"1e-{max(precision // 2, 4)}", precision)
    xacc = Number(f"1e-{max(precision - 2, 1)}", precision)
    complex_surface = func.g(t_min).is_complex
    half = Number("0.5", precision)
    one = Number(1, precision)
    mid, span = (t_min + t_max) * half, (t_max - t_min) * half
    # bisection runs in u; dividing by span keeps the t-units accuracy at xacc
    xacc = xacc / max(span, one)
    # max_degree+2 nodes: one guard node above the cap so an over-degree surface
    # is detected and rejected rather than silently under-fit.
    us = _nodes(-one, one, max_degree + 2, precision)
    p = _poly_from_samples(func, us, [mid + span * u for u in us],
                           complex_surface, tol, max_degree)
    if _poly.deg(p, tol) == 0:
        return []
    q = _poly.square_free(p, tol)
    if _poly.deg(q, tol) == 0:
        return []
    chain = _isolate.sturm_chain(q, tol)
    roots = _isolate.isolate_roots(q, chain, -one, one, xacc, tol)
    return sorted(mid + span * u for u in roots)

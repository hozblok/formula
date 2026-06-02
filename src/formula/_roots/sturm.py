"""Polynomial backend: exact all-roots via Sturm sequences.

g(t)=F(O+t*d) is recovered as a univariate polynomial by interpolation, made
square-free, and its real roots in [t_min, t_max] are counted/isolated by a
Sturm chain, then bisected to precision. Complete for algebraic surfaces.
Complex surfaces: real intersections are the common roots of Re g and Im g
(their polynomial gcd).
"""

from typing import List

from ..formula import Number
from . import _isolate, _poly


def _nodes(t_min: Number, t_max: Number, count: int, prec: int) -> List[Number]:
    step = (t_max - t_min) / Number(count - 1, prec)
    return [t_min + step * Number(i, prec) for i in range(count)]


def _poly_from_samples(func, xs, complex_surface, tol, max_degree):
    """Interpolate g (real part, and imag part if complex) to a polynomial."""
    prec = func.precision
    reals, imags = [], []
    for x in xs:
        rstr, istr = func.g(x).parts()
        reals.append(Number(rstr, prec))
        imags.append(Number(istr, prec))
    pr = _poly.interpolate(xs, reals, tol, max_degree)
    if not complex_surface:
        return pr
    pi = _poly.interpolate(xs, imags, tol, max_degree)
    return _poly.pgcd(pr, pi, tol)  # common roots of Re g and Im g


def find_all(
    func, t_min: Number, t_max: Number, precision: int, max_degree: int = 16, **_
) -> List[Number]:
    """All real roots of g in [t_min, t_max] via Sturm isolation."""
    tol = Number(f"1e-{max(precision // 2, 4)}", precision)
    xacc = Number(f"1e-{max(precision - 2, 1)}", precision)
    complex_surface = func.g(t_min).is_complex
    xs = _nodes(t_min, t_max, max_degree + 1, precision)
    p = _poly_from_samples(func, xs, complex_surface, tol, max_degree)
    if _poly.deg(p, tol) == 0:
        return []
    q = _poly.square_free(p, tol)
    if _poly.deg(q, tol) == 0:
        return []
    chain = _isolate.sturm_chain(q, tol)
    roots = [_isolate.bisect(q, lo, hi, xacc) for lo, hi in _isolate.isolate(chain, t_min, t_max)]
    return sorted(roots)

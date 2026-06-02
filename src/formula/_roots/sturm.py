"""Polynomial backend: exact all-roots via Sturm sequences.

g(t)=F(O+t*d) is recovered as a univariate polynomial by interpolation, made
square-free, and its real roots in [t_min, t_max] are counted/isolated by a
Sturm chain, then bisected to precision. Complete for algebraic surfaces.
Complex surfaces: real intersections are the common roots of Re g and Im g
(their polynomial gcd).
"""

from typing import List

from ..formula import Number
from . import _poly


def _sign(x: Number) -> int:
    zero = Number(0, x._precision)
    if x > zero:
        return 1
    if x < zero:
        return -1
    return 0


def _nodes(t_min: Number, t_max: Number, count: int, prec: int) -> List[Number]:
    step = (t_max - t_min) / Number(count - 1, prec)
    return [t_min + step * Number(i, prec) for i in range(count)]


def _poly_from_samples(func, xs, complex_surface, tol, max_degree):
    reals, imags = [], []
    for x in xs:
        rstr, istr = func.g(x)._pair()
        prec = func.precision
        reals.append(Number(rstr, prec))
        imags.append(Number(istr, prec))
    pr = _poly.interpolate(xs, reals, tol, max_degree)
    if not complex_surface:
        return pr
    pi = _poly.interpolate(xs, imags, tol, max_degree)
    common = _poly.pgcd(pr, pi, tol)
    return common  # real intersections lie at common roots of Re g and Im g


def _sturm_chain(p, tol):
    chain = [p, _poly.pderiv(p)]
    while _poly.deg(chain[-1], tol) > 0:
        _, r = _poly.pdivmod(chain[-2], chain[-1], tol)
        chain.append([-c for c in r])
    return chain


def _variations(chain, t):
    signs = [s for s in (_sign(_poly.peval(p, t)) for p in chain) if s != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _isolate(chain, a, b, tol, max_depth=200):
    roots = []
    stack = [(a, b, _variations(chain, a) - _variations(chain, b), 0)]
    while stack:
        lo, hi, count, depth = stack.pop()
        if count <= 0:
            continue
        if count == 1 or depth >= max_depth:
            roots.append((lo, hi))
            continue
        mid = (lo + hi) * Number("0.5", lo._precision)
        vmid = _variations(chain, mid)
        stack.append((lo, mid, _variations(chain, lo) - vmid, depth + 1))
        stack.append((mid, hi, vmid - _variations(chain, hi), depth + 1))
    return roots


def _bisect(q, lo, hi, xacc):
    half = Number("0.5", lo._precision)
    flo = _poly.peval(q, lo)
    for _ in range(10 * lo._precision):
        mid = (lo + hi) * half
        if hi - lo < xacc:
            return mid
        fmid = _poly.peval(q, mid)
        if _sign(fmid) == 0:
            return mid
        if _sign(flo) * _sign(fmid) < 0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return (lo + hi) * half


def find_all(func, t_min: Number, t_max: Number, precision: int, max_degree: int = 16, **_) -> List[Number]:
    tol = Number("1e-{}".format(max(precision // 2, 4)), precision)
    xacc = Number("1e-{}".format(max(precision - 2, 1)), precision)
    complex_surface = func.g(t_min)._is_complex
    xs = _nodes(t_min, t_max, max_degree + 1, precision)
    p = _poly_from_samples(func, xs, complex_surface, tol, max_degree)
    if _poly.deg(p, tol) == 0:
        return []
    q = _poly.square_free(p, tol)
    if _poly.deg(q, tol) == 0:
        return []
    chain = _sturm_chain(q, tol)
    roots = [_bisect(q, lo, hi, xacc) for lo, hi in _isolate(chain, t_min, t_max, tol)]
    return sorted(roots)

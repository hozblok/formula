"""Multiprecision univariate polynomial utilities over Number.

A polynomial is a list of Number coefficients, index = power:
p[0] + p[1]*t + ... + p[n]*t**n. Used by the Sturm backend.
"""

from typing import List, Tuple

from ..formula import Number

Poly = List[Number]


def _zero(prec: int) -> Number:
    return Number(0, prec)


def _is_zero(x: Number, tol: Number) -> bool:
    return abs(x) <= tol


def deg(p: Poly, tol: Number) -> int:
    """Degree ignoring trailing (high-order) coefficients below tol."""
    i = len(p) - 1
    while i > 0 and _is_zero(p[i], tol):
        i -= 1
    return i


def trim(p: Poly, tol: Number) -> Poly:
    """Drop high-order coefficients below tol."""
    return p[: deg(p, tol) + 1]


def peval(p: Poly, t: Number) -> Number:
    """Evaluate p at t (Horner)."""
    acc = p[-1]
    for c in reversed(p[:-1]):
        acc = acc * t + c
    return acc


def pderiv(p: Poly) -> Poly:
    """Derivative polynomial."""
    prec = p[0].precision
    if len(p) == 1:
        return [_zero(prec)]
    return [p[k] * Number(k, prec) for k in range(1, len(p))]


def pmul(a: Poly, b: Poly) -> Poly:
    """Polynomial product."""
    prec = a[0].precision
    out = [_zero(prec) for _ in range(len(a) + len(b) - 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = out[i + j] + ai * bj
    return out


def pdivmod(u: Poly, v: Poly, tol: Number) -> Tuple[Poly, Poly]:
    """Polynomial long division: u = q*v + r, deg(r) < deg(v)."""
    prec = u[0].precision
    r = list(u)
    dv = deg(v, tol)
    lead = v[dv]
    q = [_zero(prec) for _ in range(max(len(u) - dv, 1))]
    dr = deg(r, tol)
    while dr >= dv and not (dr == 0 and _is_zero(r[0], tol)):
        shift = dr - dv
        factor = r[dr] / lead
        q[shift] = factor
        for i in range(dv + 1):
            r[i + shift] = r[i + shift] - factor * v[i]
        r = r[:dr] or [_zero(prec)]  # drop the now-zero leading term
        dr = deg(r, tol)
    return q, trim(r, tol)


def pgcd(a: Poly, b: Poly, tol: Number) -> Poly:
    """Monic gcd of a and b (Euclidean remainder chain)."""
    a, b = trim(a, tol), trim(b, tol)
    while deg(b, tol) > 0 or not _is_zero(b[0], tol):
        _, r = pdivmod(a, b, tol)
        a, b = b, r
    return _monic(a, tol)


def _monic(p: Poly, tol: Number) -> Poly:
    """Scale so the leading coefficient is 1."""
    d = deg(p, tol)
    lead = p[d]
    return [p[i] / lead for i in range(d + 1)]


def square_free(p: Poly, tol: Number) -> Poly:
    """p divided by gcd(p, p'): same roots, all simple."""
    g = pgcd(p, pderiv(p), tol)
    if deg(g, tol) == 0:
        return _monic(p, tol)
    q, _ = pdivmod(p, g, tol)
    return _monic(trim(q, tol), tol)


def interpolate(xs: List[Number], ys: List[Number], tol: Number, max_degree: int) -> Poly:
    """Newton divided differences -> monomial coefficients; detects true degree."""
    prec = ys[0].precision
    c = list(ys)
    n = len(xs)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            c[i] = (c[i] - c[i - 1]) / (xs[i] - xs[i - j])
    scale = _zero(prec)
    for ck in c:
        scale = max(scale, abs(ck))
    cutoff = scale * tol if scale > _zero(prec) else tol
    degree = 0
    for k in range(n - 1, -1, -1):
        if abs(c[k]) > cutoff:
            degree = k
            break
    # n = max_degree+1 nodes uniquely fix a polynomial up to degree max_degree.
    if degree > max_degree:
        raise ValueError("polynomial degree exceeds max_degree; not a low-degree polynomial")
    poly = [c[degree]]
    for k in range(degree - 1, -1, -1):
        poly = pmul(poly, [-xs[k], Number(1, prec)])
        poly[0] = poly[0] + c[k]
    return poly

"""Real-root isolation of a monomial polynomial via Sturm sequences.

Shared by the sturm and chebyshev backends: build the Sturm chain, count/isolate
roots by sign variations, and bisect a square-free polynomial to precision.
"""

from ..formula import Number
from . import _poly


def sign(x: Number) -> int:
    """-1, 0 or +1."""
    zero = Number(0, x.precision)
    if x > zero:
        return 1
    if x < zero:
        return -1
    return 0


def sturm_chain(p, tol):
    """Sturm chain p0=p, p1=p', p_{k+1}=-rem(p_{k-1}, p_k)."""
    chain = [p, _poly.pderiv(p)]
    while _poly.deg(chain[-1], tol) > 0:
        _, r = _poly.pdivmod(chain[-2], chain[-1], tol)
        chain.append([-c for c in r])
    return chain


def variations(chain, t):
    """Sign variations of the chain evaluated at t."""
    signs = [s for s in (sign(_poly.peval(p, t)) for p in chain) if s != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def isolate(chain, a, b, max_depth=200):
    """Subdivide [a, b] until each kept subinterval brackets exactly one root."""
    roots = []
    stack = [(a, b, variations(chain, a) - variations(chain, b), 0)]
    while stack:
        lo, hi, count, depth = stack.pop()
        if count <= 0:
            continue
        if count == 1 or depth >= max_depth:
            roots.append((lo, hi))
            continue
        mid = (lo + hi) * Number("0.5", lo.precision)
        vmid = variations(chain, mid)
        stack.append((lo, mid, variations(chain, lo) - vmid, depth + 1))
        stack.append((mid, hi, vmid - variations(chain, hi), depth + 1))
    return roots


def rtsafe(func, a, b, xacc, maxit=200):
    """Newton step bracketed by bisection; converges on the single root of g in [a, b]."""
    prec = a.precision
    zero, half = Number(0, prec), Number("0.5", prec)
    lo, hi = (a, b) if sign(func.g(a)) < 0 else (b, a)
    t, step = (a + b) * half, abs(b - a)
    g, gp = func.g(t), func.gprime(t)
    for _ in range(maxit):
        secure = (gp == zero
                  or ((t - hi) * gp - g) * ((t - lo) * gp - g) > zero
                  or abs(g + g) > abs(step * gp))
        step = (hi - lo) * half if secure else g / gp
        t = lo + step if secure else t - step
        if abs(step) < xacc:
            return t
        g, gp = func.g(t), func.gprime(t)
        lo, hi = (t, hi) if sign(g) < 0 else (lo, t)
    return t


def isolate_roots(q, chain, lo, hi, xacc, tol):
    """All roots of square-free q in the closed [lo, hi], endpoints included.

    Sturm's V(lo)-V(hi) counts the half-open (lo, hi]: a root at lo is dropped
    (peel it, then isolate the open interior) while a root at hi is already
    counted and bisect returns it exactly.
    """
    roots = []
    a = lo
    if abs(_poly.peval(q, lo)) <= tol:
        roots.append(lo)
        a = lo + xacc
    if a < hi:
        roots.extend(bisect(q, l, r, xacc) for l, r in isolate(chain, a, hi))
    return roots


def bisect(q, lo, hi, xacc):
    """Bisect a square-free polynomial q on a single-root bracket [lo, hi]."""
    half = Number("0.5", lo.precision)
    flo = _poly.peval(q, lo)
    if sign(flo) == 0:
        return lo
    if sign(_poly.peval(q, hi)) == 0:
        return hi
    for _ in range(10 * lo.precision):
        mid = (lo + hi) * half
        if hi - lo < xacc:
            return mid
        fmid = _poly.peval(q, mid)
        if sign(fmid) == 0:
            return mid
        if sign(flo) * sign(fmid) < 0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return (lo + hi) * half

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


def bisect(q, lo, hi, xacc):
    """Bisect a square-free polynomial q on a single-root bracket [lo, hi]."""
    half = Number("0.5", lo.precision)
    flo = _poly.peval(q, lo)
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

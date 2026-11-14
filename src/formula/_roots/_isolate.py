"""Real-root isolation of a monomial polynomial via Sturm sequences.

Shared by the sturm and chebyshev backends: build the Sturm chain, count/isolate
roots by sign variations, and bisect a square-free polynomial to precision.
"""

from ..formula import NumberContext
from . import _poly
from .utils import UnresolvedRootError, sign, tol_full


def sturm_chain(p, tol, N: NumberContext):
    """Sturm chain p0=p, p1=p', p_{k+1}=-rem(p_{k-1}, p_k)."""
    chain = [p, _poly.pderiv(p, N)]
    while _poly.deg(chain[-1], tol) > 0:
        _, r = _poly.pdivmod(chain[-2], chain[-1], tol, N)
        chain.append([-c for c in r])
    return chain


def variations(chain, t):
    """Sign variations of the chain evaluated at t."""
    signs = [s for s in (sign(_poly.peval(p, t)) for p in chain) if s != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def isolate(chain, a, b, max_depth=200):
    """Subdivide [a, b] until each kept subinterval brackets exactly one root."""
    roots = []
    stack = [(a, b, variations(chain, a), variations(chain, b), 0)]
    while stack:
        lo, hi, vlo, vhi, depth = stack.pop()
        count = vlo - vhi
        if count <= 0:
            continue
        if count == 1 or depth >= max_depth:
            roots.append((lo, hi))
            continue
        mid = (lo + hi) * 0.5
        vmid = variations(chain, mid)
        stack.append((lo, mid, vlo, vmid, depth + 1))
        stack.append((mid, hi, vmid, vhi, depth + 1))
    return roots


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
    roots.sort()
    # candidates within a few xacc are one root at the bisection resolution (an endpoint
    # can come in twice: its own check and an isolating interval peeled by xacc)
    return [r for i, r in enumerate(roots) if not i or r - roots[i - 1] > 4 * xacc]


def fused_distinct(p, q, tol, N: NumberContext):
    """Roots the tol-level square-free part q fused beyond what the rounding-level gcd
    fuses: distinct-at-resolution roots hidden under one representative of q. Counted
    only for small algebraic p (degree <= 16, the interpolation range): the degree of a
    long spectral proxy is fit structure, not a root count."""
    dp = _poly.deg(p, tol)
    if dp > 16 or _poly.deg(q, tol) >= dp:
        return 0
    # 4 orders above the rounding scale: interpolation noise of an exact multiple root
    # stays "repeated", a cluster with gaps above ~10^(3-p/2) stays distinct.
    strict = tol_full(N) * N(10) ** 4
    return max(0, _poly.deg(_poly.square_free(p, strict, N), strict) - _poly.deg(q, tol))


def noise(p, N: NumberContext):
    """Bound on the rounding of p at x: 100 * 1e-(prec-2) * sum |p_k| |x|^k."""
    scale = N(100) * N(f"1e-{N.precision - 2}")
    mags = [abs(c) for c in p]
    return lambda x: scale * _poly.peval(mags, abs(x))


def _beside(p, r, edge, v, xacc, rounding):
    """A point between r and edge where p has the decided sign opposite to v = p(r):
    probes at r + xacc, r + 2 xacc, ... toward edge; None if there is none."""
    step = xacc if edge > r else -xacc
    while abs(step) < abs(edge - r):
        x = r + step
        px = _poly.peval(p, x)
        if abs(px) > rounding(x) and (px < 0) != (v < 0):
            return x
        step = step + step
    return None


def _decided_point(p, r, bound, xacc, rounding):
    """First probe from r toward bound where |p| clears its rounding band: (x, sign);
    (None, None) when the band reaches the bound. Probes r +- xacc, 2 xacc, 4 xacc, ..."""
    step = xacc if bound > r else -xacc
    while abs(step) < abs(bound - r):
        x = r + step
        px = _poly.peval(p, x)
        if abs(px) > rounding(x):
            return x, (-1 if px < 0 else 1)
        step = step + step
    return None, None


def resolved(p, roots, lo, hi, xacc, rounding, where=None, fused=0):
    """Candidate roots of an approximate square-free part, judged on p itself.

    An exactly zero p(r) is a root. Otherwise the first decided-|p| probes on both sides
    bracket the candidate's noise basin (at a window end the probe may step past the
    closed end: sign information only). Opposite signs prove an odd root there; the
    square-free part's own root is kept as its position — deflation conditions a multiple
    root far better than p. Equal signs with the residual inside the band leave a
    tangency and a near-miss indistinguishable — UnresolvedRootError (the strict contract
    of 2026-11-02). Equal signs with the residual above the band mean any real root
    nearby announces itself by a sign dip toward a neighbour (a fused close pair) — each
    side with a dip is bisected on p, no dip means no root. `where` maps a backend
    coordinate to the caller's t for the error message.

    `fused` is the count of distinct-at-resolution roots the square-free step merged
    (fused_distinct). They are accounted for by extra dip roots (one per dipped
    candidate is the candidate itself) or by dismissed candidates (a dismissal's decided
    no-root evidence vouches for one fused complex pair); any remainder may hide inside
    a kept candidate's opaque basin — UnresolvedRootError rather than a partial list.
    """
    ordered = []
    for r in sorted(roots):
        if ordered and _decided_point(p, ordered[-1], r, xacc, rounding)[0] is None:
            # no decided |p| anywhere between them: one noise basin, one root
            if abs(_poly.peval(p, r)) < abs(_poly.peval(p, ordered[-1])):
                ordered[-1] = r
            continue
        ordered.append(r)
    out = []
    dips = dip_candidates = dismissed = 0
    margin = (hi - lo) * 0.5
    for k, r in enumerate(ordered):
        v = _poly.peval(p, r)
        if v == 0:
            out.append(r)
            continue
        left = ordered[k - 1] if k else lo - margin
        right = ordered[k + 1] if k + 1 < len(ordered) else hi + margin
        xl, sl = _decided_point(p, r, left, xacc, rounding)
        xr, sr = _decided_point(p, r, right, xacc, rounding)
        if sl and sr and sl != sr:
            out.append(r)
            continue
        if sl is None or sr is None or abs(v) <= rounding(r):
            raise UnresolvedRootError(
                f"a root candidate at t = {where(r) if where else r} cannot be resolved "
                "at this precision: a tangency and a near-miss lie inside the same "
                "noise band")
        found = 0
        for edge in (left, right):
            x = _beside(p, r, edge, v, xacc, rounding)
            if x is not None:
                out.append(bisect(p, min(x, r), max(x, r), xacc))
                found += 1
        dips += found
        dip_candidates += found > 0
        dismissed += found == 0
    if fused > (dips - dip_candidates) + 2 * dismissed:
        r = ordered[0] if ordered else lo
        raise UnresolvedRootError(
            f"{fused} root(s) fused below the working tolerance near "
            f"t = {where(r) if where else r} cannot all be recovered: the root count "
            "in the cluster is unproven")
    return out


def bisect(q, lo, hi, xacc):
    """Bisect a square-free polynomial q on a single-root bracket [lo, hi]."""
    flo = _poly.peval(q, lo)
    if sign(flo) == 0:
        return lo
    if sign(_poly.peval(q, hi)) == 0:
        return hi
    for _ in range(10 * lo.precision):
        mid = (lo + hi) * 0.5
        if hi - lo < xacc:
            return mid
        fmid = _poly.peval(q, mid)
        if sign(fmid) == 0:
            return mid
        if sign(flo) * sign(fmid) < 0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return (lo + hi) * 0.5

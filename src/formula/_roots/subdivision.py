"""Subdivision backend: adaptive exclusion with derivative/range bounds.

A pure-Python middle ground between sampling (fast, can miss roots) and a full
interval type. On each sub-interval a Taylor bound excludes regions that
provably cannot contain a root:

    |g(t) - g(m)| <= |g'(m)|*h + (M2/2)*h^2,   m = midpoint, h = half-width,

so a root needs |g(m)| <= |g'(m)|*h + (M2/2)*h^2. M2 is an estimate of max|g''|
over the search interval (sampled from g', inflated). Practically reliable, not
a formal proof — the guarantee is only as good as the M2 bound.
"""

from typing import List, Optional

from ..formula import Number
from .utils import merge_close_roots, rtsafe, sign


def _estimate_m2(func, t_min: Number, t_max: Number, prec: int, samples: int) -> Number:
    """Inflated estimate of max|g''| via the Lipschitz constant of g' on a grid."""
    step = (t_max - t_min) / Number(samples, prec)
    gp_prev = func.gprime(t_min)
    m2 = Number(0, prec)
    for i in range(1, samples + 1):
        gp = func.gprime(t_min + step * Number(i, prec))
        m2 = max(m2, abs(gp - gp_prev) / step)
        gp_prev = gp
    return m2 * Number(2, prec)


def _candidates(func, t_min, t_max, m2, region_tol):
    """Sub-intervals (width < region_tol) the exclusion test cannot rule out."""
    half = Number("0.5", t_min.precision)
    stack = [(t_min, t_max)]
    out = []
    while stack:
        a, b = stack.pop()
        m, h = (a + b) * half, (b - a) * half
        if abs(func.g(m)) > abs(func.gprime(m)) * h + m2 * h * h * half:
            continue
        if b - a < region_tol:
            out.append((a, b))
        else:
            stack.append((a, m))
            stack.append((m, b))
    return sorted(out)


def _merge(regions, region_tol):
    """Coalesce touching/adjacent candidate intervals."""
    merged = []
    for a, b in regions:
        if merged and a - merged[-1][1] <= region_tol:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _bisect_gprime(func, a, b, xacc):
    """Locate g'=0 in [a, b] (a turning point) by bisection."""
    half = Number("0.5", a.precision)
    sa = sign(func.gprime(a))
    for _ in range(10 * a.precision):
        m = (a + b) * half
        if b - a < xacc:
            return m
        if sign(func.gprime(m)) == 0:
            return m
        if sa * sign(func.gprime(m)) < 0:
            b = m
        else:
            a = m
    return (a + b) * half


def _refine(func, a, b, xacc, gtol) -> Optional[Number]:
    """Resolve one candidate region to a root, or None if it is a near-miss."""
    if sign(func.g(a)) * sign(func.g(b)) < 0:
        return rtsafe(func, a, b, xacc)  # simple root
    if sign(func.gprime(a)) * sign(func.gprime(b)) < 0:  # tangency: g'=0
        t = _bisect_gprime(func, a, b, xacc)
        return t if abs(func.g(t)) < gtol else None
    m = (a + b) * Number("0.5", a.precision)
    return m if abs(func.g(m)) < gtol else None


def find_all(
    func, t_min: Number, t_max: Number, precision: int, m2_samples: int = 200, **opts
) -> List[Number]:
    """All real roots of g in [t_min, t_max] via adaptive exclusion subdivision."""
    if func.g(t_min).is_complex:
        raise NotImplementedError("subdivision backend handles real surfaces only")
    xacc = Number(f"1e-{max(precision - 2, 1)}", precision)
    gtol = Number(f"1e-{max(precision // 3, 4)}", precision)
    region_tol = opts.get("region_tol") or (t_max - t_min) * Number("1e-6", precision)
    m2 = _estimate_m2(func, t_min, t_max, precision, m2_samples)
    regions = _merge(_candidates(func, t_min, t_max, m2, region_tol), region_tol)
    roots = [r for r in (_refine(func, a, b, xacc, gtol) for a, b in regions) if r is not None]
    return merge_close_roots(roots, xacc)

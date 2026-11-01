"""Subdivision backend: adaptive exclusion with derivative/range bounds.

A pure-Python middle ground between sampling (fast, can miss roots) and a full
interval type. On each sub-interval a Taylor bound excludes regions that
provably cannot contain a root:

    |g(t) - g(m)| <= |g'(m)|*h + (M2/2)*h^2,   m = midpoint, h = half-width,

so a root needs |g(m)| <= |g'(m)|*h + (M2/2)*h^2. M2 is an estimate of max|g''|
over the search interval (sampled from g', inflated). Practically reliable, not
a formal proof — the guarantee is only as good as the M2 bound.
"""

from ..formula import NumberContext
from .utils import finite, finite_sign, merge_close_roots, rtsafe, sign, tol_full, tol_third


def _estimate_m2(func, t_min, t_max, samples: int):
    """Inflated estimate of max|g''| via the Lipschitz constant of g' on a grid."""
    step = (t_max - t_min) / samples
    t = t_min
    gp_prev = func.gprime(t)
    if not finite(gp_prev):
        raise ValueError(f"non-finite g' at t = {t}")
    m2 = 0
    for _ in range(samples):
        t = t + step
        gp = func.gprime(t)
        if not finite(gp):
            raise ValueError(f"non-finite g' at t = {t}")
        m2 = max(m2, abs(gp - gp_prev) / step)
        gp_prev = gp
    return m2 * 2


def _candidates(func, t_min, t_max, m2, region_tol):
    """Sub-intervals (width < region_tol) the exclusion test cannot rule out."""
    stack = [(t_min, t_max)]
    out = []
    while stack:
        a, b = stack.pop()
        m, h = (a + b) * 0.5, (b - a) * 0.5
        g, gp = func.g(m), func.gprime(m)
        if not finite(g):
            raise ValueError(f"non-finite g at t = {m}")
        # A non-finite g' cannot exclude the interval: it keeps subdividing.
        if finite(gp) and abs(g) > abs(gp) * h + m2 * h * h * 0.5:
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
    """Locate g'=0 in [a, b] (a turning point, or a kink where g' is undefined) by bisection."""
    sa = finite_sign(func.gprime(a))
    if sa is None:
        return a
    for _ in range(10 * a.precision):
        m = (a + b) * 0.5
        if b - a < xacc:
            return m
        sm = finite_sign(func.gprime(m))
        if sm is None or sm == 0:
            return m
        if sa * sm < 0:
            b = m
        else:
            a = m
    return (a + b) * 0.5


def _refine(func, a, b, xacc, gtol):
    """Resolve one candidate region to a root, or None if it is a near-miss."""
    if sign(func.g(a)) * sign(func.g(b)) < 0:
        return rtsafe(func, a, b, xacc)  # simple root
    spa, spb = finite_sign(func.gprime(a)), finite_sign(func.gprime(b))
    if spa is None or spb is None or spa * spb < 0:  # tangency: g'=0, or a kink
        t = _bisect_gprime(func, a, b, xacc)
        return t if abs(func.g(t)) < gtol else None
    m = (a + b) * 0.5
    return m if abs(func.g(m)) < gtol else None


def find_all(
    func, t_min, t_max, precision: int, m2_samples: int = 200, **opts
) -> list:
    """All real roots of g in [t_min, t_max] via adaptive exclusion subdivision."""
    if func.g(t_min).is_complex:
        raise NotImplementedError("subdivision backend handles real surfaces only")
    N = NumberContext(precision)
    xacc, gtol = tol_full(N), tol_third(N)
    region_tol = opts.get("region_tol") or (t_max - t_min) * 1e-6
    m2 = _estimate_m2(func, t_min, t_max, m2_samples)
    regions = _merge(_candidates(func, t_min, t_max, m2, region_tol), region_tol)
    roots = [r for r in (_refine(func, a, b, xacc, gtol) for a, b in regions) if r is not None]
    return merge_close_roots(roots, xacc)

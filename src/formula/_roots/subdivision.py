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
from .utils import (UnresolvedRootError, ZeroFunctionError, finite, finite_sign,
                    merge_close_roots, rtsafe, sign, tol_full)


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
    """Sub-intervals (width < region_tol) the exclusion test cannot rule out, and the
    largest |g| met along the way — the scale of g for the noise band."""
    stack = [(t_min, t_max)]
    out = []
    gmax = abs(func.g(t_min))
    while stack:
        a, b = stack.pop()
        m, h = (a + b) * 0.5, (b - a) * 0.5
        g, gp = func.g(m), func.gprime(m)
        if not finite(g):
            raise ValueError(f"non-finite g at t = {m}")
        gmax = max(gmax, abs(g))
        # A non-finite g' cannot exclude the interval: it keeps subdividing.
        if finite(gp) and abs(g) > abs(gp) * h + m2 * h * h * 0.5:
            continue
        if b - a < region_tol or not a < m < b:   # small enough, or no longer divisible
            out.append((a, b))
        else:
            stack.append((a, m))
            stack.append((m, b))
    return sorted(out), gmax


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


def _refine(func, a, b, xacc, band):
    """Resolve one candidate region: a root, or None for a proven near-miss.

    A sign change is a root (rtsafe). Otherwise the closest approach to zero decides:
    |g| above the noise band proves a near-miss; |g| inside it leaves a tangency and a
    near-miss indistinguishable — UnresolvedRootError (the strict contract of 2026-11-02).
    """
    if sign(func.g(a)) * sign(func.g(b)) < 0:
        return rtsafe(func, a, b, xacc)  # simple root
    spa, spb = finite_sign(func.gprime(a)), finite_sign(func.gprime(b))
    if spa is None or spb is None or spa * spb < 0:  # tangency: g'=0, or a kink
        t = _bisect_gprime(func, a, b, xacc)
    else:
        t = (a + b) * 0.5
    v = abs(func.g(t))
    if v > band:
        return None
    raise UnresolvedRootError(
        f"|g| = {v} at t = {t} is inside the noise band {band}: a tangency and a "
        "near-miss cannot be told apart at this precision")


def find_all(
    func, t_min, t_max, precision: int, m2_samples: int = 200, **opts
) -> list:
    """All real roots of g in [t_min, t_max] via adaptive exclusion subdivision."""
    if func.g(t_min).is_complex:
        raise NotImplementedError("subdivision backend handles real surfaces only")
    N = NumberContext(precision)
    xacc = tol_full(N)
    if not t_min < t_max:
        g0 = func.g(t_min)
        if g0 == 0:
            return [t_min]
        gp0 = func.gprime(t_min)
        # the Newton step bounds |t - root|; a kink (non-finite g') proves nothing
        return [t_min] if finite(gp0) and abs(g0) <= xacc * abs(gp0) else []
    region_tol = opts.get("region_tol")
    if region_tol is None:
        region_tol = (t_max - t_min) * 1e-6
    elif not region_tol > 0:
        raise ValueError(f"region_tol must be positive, got {region_tol}")
    m2 = _estimate_m2(func, t_min, t_max, m2_samples)
    candidates, gmax = _candidates(func, t_min, t_max, m2, region_tol)
    regions = _merge(candidates, region_tol)
    if len(regions) == 1 and regions[0] == (t_min, t_max):
        quarter = (t_max - t_min) * 0.25
        if all(func.g(t_min + quarter * k) == 0 for k in range(5)):
            raise ZeroFunctionError(
                f"g is zero at every probe of [{t_min}, {t_max}] and nowhere excludable: "
                "the ray lies in the surface within tolerance; no isolated-root list exists")
    band = N(100) * xacc * gmax
    roots = [r for r in (_refine(func, a, b, xacc, band) for a, b in regions) if r is not None]
    return merge_close_roots(roots, xacc)

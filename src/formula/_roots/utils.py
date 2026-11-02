"""Shared helpers for root-finding backends."""

from typing import Protocol

# The native tracer's quartic throws the same class; one type for both worlds.
from .._formula import UnresolvedRootError
from ..formula import NumberContext


class ZeroFunctionError(ValueError):
    """g is zero, within the working tolerance, at every probe of the interval: the ray
    lies in the surface, and no finite list of isolated roots exists."""


class _RootFunc(Protocol):
    """A surface restricted to one ray; what every backend consumes."""

    def g(self, t):
        """g(t) = F(O + t*d), the surface value at ray parameter t."""

    def gprime(self, t):
        """g'(t) = grad F . d, the derivative of g along the ray."""


def tol_full(N: NumberContext):
    """1e-(p-2): accuracy of a root parameter t, two digits inside the storage precision."""
    return N(f"1e-{N.precision - 2}")


def tol_half(N: NumberContext):
    """1e-(p//2): roots closer than this are one root; polynomial coefficients below it are zero."""
    return N(f"1e-{N.precision // 2}")


def tol_third(N: NumberContext):
    """1e-(p//3): a value of g this small counts as zero; the Chebyshev proxy tolerance."""
    return N(f"1e-{N.precision // 3}")


def sign(x) -> int:
    """-1, 0 or +1; a nan raises instead of posing as 0."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    if x == 0:
        return 0
    raise ValueError("non-finite value in a sign test")


def finite(value) -> bool:
    """False for inf and nan: two operations, no text conversion."""
    return value - value == 0


def finite_sign(value):
    """sign(value), or None where value is non-finite (inf/nan)."""
    if value.is_complex:
        raise NotImplementedError("real surfaces only; use sturm/chebyshev for complex")
    return sign(value) if finite(value) else None


def rtsafe(func: _RootFunc, a, b, xacc, maxit: int = 200):
    """Refine one root of g in a sign-change bracket [a, b]; None when there is none.

    Safeguarded Newton: Newton when the iterate stays bracketed, bisection
    otherwise; when maxit steps leave |step| >= xacc, plain bisection narrows the
    bracket below xacc, and a bracket that cannot shrink any further at this
    precision raises UnresolvedRootError. A root's residual vanishes with the
    bracket; a pole's grows and a jump's stays, so a residual not below tol_third
    of the bracket-end values means no root. A non-finite g inside the bracket is
    an error: g has no value there.

    Recipe — Press, Teukolsky, Vetterling & Flannery, *Numerical Recipes*
    (Cambridge Univ. Press), Ch. 9 §9.4 "Newton-Raphson Method Using
    Derivative", routine ``rtsafe``: 2nd ed. *Numerical Recipes in C* (1992),
    pp. 366-368 (ISBN 0-521-43108-5); 3rd ed. *The Art of Scientific
    Computing* (2007), pp. 456-459 (ISBN 978-0-521-88068-8).

    func — callable surface restriction with .g(t) and .gprime(t).
    a, b — bracket endpoints; require g(a) and g(b) of opposite sign.
    xacc — absolute tolerance on t (NR name; independent variable is t here);
            stop when |step| < xacc.
    maxit — Newton-phase cap; the bisection tail after it is bounded by the bracket.
    """
    g_a, g_b = func.g(a), func.g(b)
    if not (finite(g_a) and finite(g_b)):
        raise ValueError("non-finite g at a bracket end")
    scale = max(abs(g_a), abs(g_b))
    lo, hi = (a, b) if sign(g_a) < 0 else (b, a)
    t, step = (a + b) * 0.5, abs(b - a)
    g, gp = func.g(t), func.gprime(t)
    for _ in range(maxit):
        if not finite(g):
            raise ValueError(f"non-finite g inside the bracket at t = {t}")
        # A non-finite g' (a kink) takes the bisection step, as g' = 0 does.
        secure = (not finite(gp) or gp == 0
                  or ((t - hi) * gp - g) * ((t - lo) * gp - g) > 0
                  or abs(g + g) > abs(step * gp))
        step = (hi - lo) * 0.5 if secure else g / gp
        t = lo + step if secure else t - step
        if abs(step) < xacc:
            break
        g, gp = func.g(t), func.gprime(t)
        lo, hi = (t, hi) if sign(g) < 0 else (lo, t)
    if abs(hi - lo) >= xacc:
        # a sign change across t -+ xacc certifies |t - root| < xacc; otherwise the probes
        # narrow the bracket and plain bisection finishes it
        probes = [(x, func.g(x)) for x in (t - xacc, t + xacc)]
        if any(not finite(gx) for _, gx in probes):
            raise ValueError(f"non-finite g inside the bracket near t = {t}")
        if sign(probes[0][1]) * sign(probes[1][1]) > 0:
            for x, gx in probes:
                if min(lo, hi) < x < max(lo, hi):
                    lo, hi = (x, hi) if sign(gx) < 0 else (lo, x)
            while abs(hi - lo) >= xacc:
                t = (lo + hi) * 0.5
                if t == lo or t == hi:
                    raise UnresolvedRootError(
                        f"rtsafe: the bracket near t = {t} cannot shrink below "
                        f"xacc = {xacc} at this precision")
                g = func.g(t)
                if not finite(g):
                    raise ValueError(f"non-finite g inside the bracket at t = {t}")
                lo, hi = (t, hi) if sign(g) < 0 else (lo, t)
            t = (lo + hi) * 0.5
    g = func.g(t)
    return t if finite(g) and abs(g) <= scale * tol_third(NumberContext(xacc.precision)) else None


def merge_close_roots(roots, xacc) -> list:
    """One t per cluster after numerical root finding.

    roots — candidate ray parameters t with g(t)≈0; may repeat the same root.
    xacc — merge radius: |t1-t2| ≤ xacc counts as one root (leftmost kept).
    """
    roots = list(roots)
    if not all(finite(t) for t in roots):
        raise ValueError("non-finite root candidate")
    out = []
    for t in sorted(roots):
        if not out or abs(t - out[-1]) > xacc:
            out.append(t)
    return out

"""Shared helpers for root-finding backends."""

from typing import Protocol


class _RootFunc(Protocol):
    """A surface restricted to one ray; what every backend consumes."""

    def g(self, t):
        """g(t) = F(O + t*d), the surface value at ray parameter t."""

    def gprime(self, t):
        """g'(t) = grad F . d, the derivative of g along the ray."""


def sign(x) -> int:
    """-1, 0 or +1."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def real_text(value, precision: int) -> str:
    """Real part of a Number or backend value, formatted at precision."""
    return getattr(value, "_value", value).real.str(precision)


def finite_sign(value):
    """sign(value), or None where value is non-finite (inf/nan)."""
    if value.is_complex:
        raise NotImplementedError("real surfaces only; use sturm/chebyshev for complex")
    if str(value) in ("inf", "-inf", "nan"):
        return None
    return sign(value)


def rtsafe(func: _RootFunc, a, b, xacc, maxit: int = 200):
    """Refine one root of g in a sign-change bracket [a, b].

    Safeguarded Newton: Newton when the iterate stays bracketed, bisection
    otherwise.

    Recipe — Press, Teukolsky, Vetterling & Flannery, *Numerical Recipes*
    (Cambridge Univ. Press), Ch. 9 §9.4 "Newton-Raphson Method Using
    Derivative", routine ``rtsafe``: 2nd ed. *Numerical Recipes in C* (1992),
    pp. 366-368 (ISBN 0-521-43108-5); 3rd ed. *The Art of Scientific
    Computing* (2007), pp. 456-459 (ISBN 978-0-521-88068-8).

    func — callable surface restriction with .g(t) and .gprime(t).
    a, b — bracket endpoints; require g(a) and g(b) of opposite sign.
    xacc — absolute tolerance on t (NR name; independent variable is t here);
            stop when |step| < xacc.
    maxit — iteration cap if xacc is not reached.
    """
    lo, hi = (a, b) if sign(func.g(a)) < 0 else (b, a)
    t, step = (a + b) * 0.5, abs(b - a)
    g, gp = func.g(t), func.gprime(t)
    for _ in range(maxit):
        secure = (gp == 0
                  or ((t - hi) * gp - g) * ((t - lo) * gp - g) > 0
                  or abs(g + g) > abs(step * gp))
        step = (hi - lo) * 0.5 if secure else g / gp
        t = lo + step if secure else t - step
        if abs(step) < xacc:
            return t
        g, gp = func.g(t), func.gprime(t)
        lo, hi = (t, hi) if sign(g) < 0 else (lo, t)
    return t


def merge_close_roots(roots, xacc) -> list:
    """One t per cluster after numerical root finding.

    roots — candidate ray parameters t with g(t)≈0; may repeat the same root.
    xacc — merge radius: |t1-t2| ≤ xacc counts as one root (leftmost kept).
    """
    out = []
    for t in sorted(roots):
        if not out or abs(t - out[-1]) > xacc:
            out.append(t)
    return out

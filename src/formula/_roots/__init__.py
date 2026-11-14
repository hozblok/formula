"""Root-finding backends for RaySurface. Each exposes find_all(func, t_min, t_max, precision)."""

import importlib
import re
from typing import Callable

from ..formula import NumberContext
from .utils import UnresolvedRootError, ZeroFunctionError, merge_close_roots, tol_half

_BACKENDS = {
    "sampling": "sampling",
    "sturm": "sturm",
    "chebyshev": "chebyshev",
    "subdivision": "subdivision",
}

# Transcendental tokens that rule out the exact polynomial (Sturm) path.
_NON_POLY = ("sin", "cos", "tan", "asin", "acos", "atan", "log", "exp", "sqrt")

# A '^'/'**' exponent that is negative, fractional, or a variable -> not a
# polynomial (e.g. x^-2, x^0.5, x^x). Natural powers (x^2, x**3) stay polynomial.
_NONNATURAL_POWER = re.compile(r"(?:\*\*|\^)\s*\(?\s*(?:-|\.\d|[a-z]|\d+\.)")


def _denominator(expr: str, slash: int) -> str:
    """The operand right after a '/': a parenthesized group or a bare token."""
    j = slash + 1
    while j < len(expr) and expr[j] == " ":
        j += 1
    if j < len(expr) and expr[j] == "(":
        depth, start = 0, j
        while j < len(expr):
            depth += (expr[j] == "(") - (expr[j] == ")")
            if depth == 0:
                return expr[start : j + 1]
            j += 1
        return expr[start:]
    start = j
    while j < len(expr) and expr[j] not in "+-*/^() ":
        j += 1
    return expr[start:j]


def is_polynomial(surface) -> bool:
    """Cheap hint: no transcendentals, no non-natural powers, no variable denominator."""
    expr = surface.expression.lower()
    if any(fn in expr for fn in _NON_POLY) or _NONNATURAL_POWER.search(expr):
        return False
    variables = {v.lower() for v in surface.variables()}
    # Reject division whose denominator operand contains a variable, e.g. 1/(x-1).
    for i, ch in enumerate(expr):
        if ch == "/" and any(v in _denominator(expr, i) for v in variables):
            return False
    return True


def _load(name: str) -> Callable:
    module = importlib.import_module(f"._roots.{_BACKENDS[name]}", "formula")
    return module.find_all


def _union(lists, precision: int) -> list:
    """Merge root lists, treating values within a tolerance as the same root."""
    tol = tol_half(NumberContext(precision))
    return merge_close_roots((r for lst in lists for r in lst), tol)


def _general(func, t_min, t_max, precision, **opts) -> list:
    """Chebyshev reconciled with subdivision; either may fail numerically and
    drop out, but NotImplementedError (e.g. complex surface) still propagates.
    A degeneracy verdict (zero function, unresolved roots) is not a failure:
    it propagates instead of silently falling back to a guessing backend."""
    lists, numeric_error = [], None
    for name in ("chebyshev", "subdivision"):
        try:
            lists.append(_load(name)(func, t_min, t_max, precision, **opts))
        except (ZeroFunctionError, UnresolvedRootError):
            raise
        except (ArithmeticError, ValueError) as exc:
            numeric_error = exc
    if not lists and numeric_error is not None:
        raise numeric_error
    return _union(lists, precision)


def _auto(func, t_min, t_max, precision, **opts) -> list:
    """Sturm for algebraic surfaces; else Chebyshev backed up by subdivision.

    A misclassified or over-degree surface that makes Sturm raise falls back to
    the general backends instead of failing the whole call; a degeneracy verdict
    propagates.
    """
    if is_polynomial(func.surface):
        try:
            return _load("sturm")(func, t_min, t_max, precision, **opts)
        except (ZeroFunctionError, UnresolvedRootError):
            raise
        except (ArithmeticError, ValueError):
            pass
    return _general(func, t_min, t_max, precision, **opts)


def get_backend(method: str) -> Callable:
    """Resolve a method name to its find_all backend; 'auto' picks per surface."""
    if method == "auto":
        return _auto
    if method not in _BACKENDS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_BACKENDS)}")
    return _load(method)

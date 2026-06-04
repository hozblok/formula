"""Root-finding backends for RaySurface. Each exposes find_all(func, t_min, t_max, precision)."""

import importlib
from typing import Callable

from ..formula import Number

_BACKENDS = {
    "sampling": "sampling",
    "sturm": "sturm",
    "chebyshev": "chebyshev",
    "subdivision": "subdivision",
    "interval": "interval",
}

# Transcendental tokens that rule out the exact polynomial (Sturm) path.
_NON_POLY = ("sin", "cos", "tan", "asin", "acos", "atan", "log", "exp", "sqrt")


def is_polynomial(surface) -> bool:
    """Cheap hint: no transcendental functions and no variable in a denominator."""
    expr = surface.expression.lower()
    if any(fn in expr for fn in _NON_POLY):
        return False
    variables = {v.lower() for v in surface.variables()}
    # Reject division by anything containing a variable, e.g. 1/(x-1).
    for i, ch in enumerate(expr):
        if ch == "/":
            tail = expr[i + 1:]
            if any(v in tail for v in variables):
                return False
    return True


def _load(name: str) -> Callable:
    module = importlib.import_module(f"._roots.{_BACKENDS[name]}", "formula")
    return module.find_all


def _union(lists, precision: int) -> list:
    """Merge root lists, treating values within a tolerance as the same root."""
    tol = Number(f"1e-{max(precision // 2, 6)}", precision)
    out = []
    for t in sorted(r for lst in lists for r in lst):
        if not out or abs(t - out[-1]) > tol:
            out.append(t)
    return out


def _auto(func, t_min, t_max, precision, **opts) -> list:
    """Sturm for algebraic surfaces; else Chebyshev backed up by subdivision."""
    if is_polynomial(func.surface):
        return _load("sturm")(func, t_min, t_max, precision, **opts)
    cheb = _load("chebyshev")(func, t_min, t_max, precision, **opts)
    sub = _load("subdivision")(func, t_min, t_max, precision, **opts)
    return _union((cheb, sub), precision)


def get_backend(method: str) -> Callable:
    """Resolve a method name to its find_all backend; 'auto' picks per surface."""
    if method == "auto":
        return _auto
    if method not in _BACKENDS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_BACKENDS)}")
    return _load(method)

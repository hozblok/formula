"""Root-finding backends for RaySurface. Each exposes find_all(func, t_min, t_max, precision)."""

import importlib
from typing import Callable

_BACKENDS = {
    "sampling": "sampling",
    "sturm": "sturm",
    "chebyshev": "chebyshev",
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


def get_backend(method: str, surface) -> Callable:
    """Resolve a method name to its find_all backend; 'auto' picks by surface."""
    if method == "auto":
        method = "sturm" if is_polynomial(surface) else "sampling"
    if method not in _BACKENDS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_BACKENDS)}")
    return _load(method)

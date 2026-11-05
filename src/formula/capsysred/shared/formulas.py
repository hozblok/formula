"""Cached Formula instances shared by CAPSYSred calculations."""

from functools import lru_cache

from ... import Formula


@lru_cache(maxsize=64)
def cached_formula(expression, precision):
    """Return a shared Formula instance; callers must not mutate it."""
    return Formula(expression, precision=precision)

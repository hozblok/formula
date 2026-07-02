"""Arbitrary-precision formula parser and solver."""

__version__ = "6.0.0"

# pylint: disable=no-name-in-module, import-error
from .constants import DEFAULT_CASE_INSENSITIVE, DEFAULT_IMAGINARY_UNIT
from ._formula import FmtFlags, Formula
from .backend import COMPLEX_TYPES, MAX_PRECISION, REAL_TYPES, mp_class
from .formula import Number, Solver
from .intersect import RaySurface, RaySurfaceFunction

__all__ = [
    "DEFAULT_CASE_INSENSITIVE",
    "DEFAULT_IMAGINARY_UNIT",
    "FmtFlags",
    "Formula",
    "COMPLEX_TYPES",
    "MAX_PRECISION",
    "REAL_TYPES",
    "mp_class",
    "Number",
    "Solver",
    "RaySurface",
    "RaySurfaceFunction",
]

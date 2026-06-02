"""Arbitrary-precision formula parser and solver."""

__version__ = "4.0.3"

# pylint: disable=no-name-in-module, import-error
from ._formula import FmtFlags, Formula
from .backend import COMPLEX_TYPES, MAX_PRECISION, REAL_TYPES, mp_class
from .formula import Number, Solver

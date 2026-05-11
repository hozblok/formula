"""Arbitrary-precision formula parser and solver."""

__version__ = "4.0.3"

# pylint: disable=no-name-in-module, import-error
from ._formula import FmtFlags, Formula
from .formula import Solver, Number

MAX_PRECISION = 8192

__all__ = ["FmtFlags", "Formula", "Solver", "Number", "MAX_PRECISION"]

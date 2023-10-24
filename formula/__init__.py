"""Arbitrary-precision formula parser and solver."""

# pylint: disable=no-name-in-module, import-error
from ._formula import FmtFlags, Formula
from .formula import Solver, Number

MAX_PRECISION = 8192

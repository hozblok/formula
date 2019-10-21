"""Arbitrary-precision formula parser and solver."""

from ._formula import FmtFlags, Formula # pylint: disable=no-name-in-module
from .formula import Solver

MAX_PRECISION = 8192

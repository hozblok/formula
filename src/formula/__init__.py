"""Multiprecision formula parser and solver."""

__version__ = "6.0.0"

# pylint: disable=no-name-in-module, import-error
from .constants import DEFAULT_CASE_INSENSITIVE, DEFAULT_IMAGINARY_UNIT
from ._formula import FmtFlags, Formula, UnresolvedRootError
from .backend import MAX_PRECISION, MP_TYPES
from .formula import NumberContext, Solver
from ._roots.utils import ZeroFunctionError
from .intersect import RayPath, RaySurface, RaySurfaceFunction, Reflection

from .xray import (
    FUSED_SILICA,
    SILICON,
    GlassMaterial,
    ReflectionEvent,
    reflect_amplitude,
    reflect_ray,
    reflectivity,
    wavelength_angstrom,
)

__all__ = [
    "DEFAULT_CASE_INSENSITIVE",
    "DEFAULT_IMAGINARY_UNIT",
    "FmtFlags",
    "Formula",
    "MAX_PRECISION",
    "MP_TYPES",
    "NumberContext",
    "Solver",
    "RayPath",
    "RaySurface",
    "RaySurfaceFunction",
    "Reflection",
    "UnresolvedRootError",
    "ZeroFunctionError",
]

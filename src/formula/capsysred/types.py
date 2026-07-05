"""Shared geometry primitives for CAPSYSred."""

from ..formula import Number

Vec3 = tuple[Number, Number, Number]

# Forward-step floor for the ray parameter t (m): rejects the t~0 root at the
# reflection origin (on-wall point); far below any physical chord.
_EPS_T = 1e-12

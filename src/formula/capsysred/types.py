"""Shared geometry primitives for CAPSYSred."""

from ..formula import Number

Vec3 = tuple[Number, Number, Number]

# Forward-step floor for the ray parameter t (m): rejects the t~0 root at the
# reflection origin (on-wall point); far below any physical chord.
_EPS_T = 1e-12

# Bore-membership nudge along the ray (m): far below any bounce spacing, far
# above the on-wall inside() tolerance band.
_EPS_LOC = 1e-7

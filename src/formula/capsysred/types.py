"""Shared geometry primitives for CAPSYSred."""

from ..formula import Number

Vec3 = tuple[Number, Number, Number]

# Forward-step floor for the ray parameter t (m): rejects the t~0 root at the
# reflection origin (on-wall point); far below any physical chord.
_EPS_T = 1e-12

# Bore-membership nudge along the ray (m): far below any bounce spacing, far
# above the on-wall inside() tolerance band.
_EPS_LOC = 1e-7

# Metres -> micrometres for expr_um. Geometry lives in SI, but the RaySurface
# cross-check equation is written in micrometre coordinates: in metres the
# coefficients are ~a^2 ~ 1e-12 and the subdivision root finder is badly
# conditioned; in micrometres they are ~1. Linear terms scale by the factor,
# the quadratic ones by its square; engine_hit_t applies the same scale to the
# ray origin and divides the returned t back. Lifted to precision p at use
# site so coefficient strings never pass through double (1e6 is exact).
_M_TO_UM = 1e6

# Relative widening of the float root-search cap past t_exit in hit(). The
# cap is only float(t_exit), so a wall hit landing within rounding of the exit
# plane could fall just outside the window and be lost — the ray would
# silently pass through the wall. Widened, the boundary root is still found,
# and the caller's full-precision t_exit <= t comparison (next_event) makes
# the actual pass/reflect call; a genuinely-past-exit root it lets through is
# discarded there, so the slack can only recover hits, never invent them.
_TCAP_TOL = 1e-9

# Relative slack on r^2 in inside(): the probed point is a reflection point,
# i.e. exactly ON the wall, so after Number->double conversion dx^2+dy^2 lands
# on either side of a^2 within a few ulp (~1e-15 relative); a strict < would
# randomly classify on-wall points as outside and _locate would absorb the
# ray. 1e-9 covers double roundoff with ~6 orders of margin, yet is ~fm in
# radius for a um-scale bore and stays far below the _EPS_LOC nudge, which
# makes the actual inside/outside decision.
_INSIDE_TOL = 1e-9

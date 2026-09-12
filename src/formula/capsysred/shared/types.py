"""Shared geometry primitives for CAPSYSred."""

from collections import namedtuple
from enum import StrEnum

from ...formula import Number

Vec3 = tuple[Number, Number, Number]


class HitMethod(StrEnum):
    """Every ray-hit method: the closed forms plus the RaySurface backends."""
    PYTHON_CLOSED_FORM = "python-closed-form"
    CPP_CLOSED_FORM = "cpp-closed-form"
    STURM = "sturm"
    SUBDIVISION = "subdivision"
    CHEBYSHEV = "chebyshev"   # @experimental
    SAMPLING = "sampling"     # @experimental

# One traced ray as every estimator consumes it (protocol: new_mode ->
# add_ray(rec, amps) -> fold_mode -> finalize). Pure geometry — amplitudes
# are physics and stay a separate add_ray argument. opl/sins/point/direction
# are Number; float estimators convert at the point of use.
RayRecord = namedtuple("RayRecord", [
    "mode", "ray", "fate", "pixel",   # fate is post-amplitude_min
    "point", "direction",             # arrival on the screen plane
    "opl", "sins",                    # optical path, grazing sines per bounce
    "refl",                           # reflection points (Gamma tensor input)
])


def ray_record(tr, screen, mode: int, ray: int, fate: str) -> RayRecord:
    """TraceResult -> RayRecord; fate comes in separately because the
    amplitude_min threshold may demote "screen" to "absorbed"."""
    pixel = screen.pixel(tr.point) if fate == "screen" else None
    return RayRecord(mode, ray, fate, pixel, tr.point, tr.direction, tr.opl,
                     tuple(s for _, s in tr.reflections),
                     tuple(p for p, _ in tr.reflections))

# Forward-step floor for the ray parameter t (m): rejects the t~0 root at the
# reflection origin (on-wall point); far below any physical chord.
_EPS_T = 1e-12

# On-wall root filter: discard hits below this multiple of eps, the t~0 root
# sitting on the reflection origin. Shared by ImplicitWall.hit and stage 9.
_ONWALL_TOL = 1.5

# Bore-membership nudge along the ray (m): far below any bounce spacing, far
# above the on-wall inside() tolerance band.
_EPS_LOC = 1e-7

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

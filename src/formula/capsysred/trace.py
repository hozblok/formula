"""The one multi-bounce tracer behind every stage.

Free space, Lloyd wall and capillaries run through the same loop — the optic
only supplies wall events. Pure geometry: exact Number positions, grazing
sines per bounce, optical path length from the source point. Energy enters
later — Fresnel amplitudes are computed from the recorded sines.
"""

from collections import namedtuple

from ..formula import Number
from .nums import const, vadd, vdot, vscale, vsub
from .types import Vec3

# fate: "screen" | "absorbed" | "lost". reflections: [(point, sin_grazing), ...]
TraceResult = namedtuple(
    "TraceResult", ["fate", "point", "opl", "reflections"]
)


def trace_ray(origin: Vec3, direction: Vec3, optic, screen_z: Number,
              max_bounces: int) -> TraceResult:
    """Trace one ray from the source point to the screen plane z=screen_z.

    `direction` must be unit (Number-normalized) so parameters are path lengths.
    """
    p = origin[0].precision
    two = const("2", p)
    opl = const("0", p)
    reflections = []
    O, d = origin, direction

    if optic is not None:
        for _ in range(max_bounces):
            event = optic.next_event(O, d)
            kind = event[0]
            if kind == "exit":
                break
            if kind == "pass":
                t = event[1]
                O = vadd(O, vscale(d, t))
                opl = opl + t
                continue
            if kind == "absorb":
                t = event[1]
                return TraceResult("absorbed", vadd(O, vscale(d, t)),
                                   opl + t, reflections)
            _, t, P, normal = event
            opl = opl + t
            dot = vdot(d, normal)
            d = vsub(d, vscale(normal, two * dot))
            reflections.append((P, abs(dot)))
            O = P
        else:
            return TraceResult("lost", O, opl, reflections)

    if float(d[2]) <= 0.0:
        return TraceResult("lost", O, opl, reflections)
    t = (screen_z - O[2]) / d[2]
    if float(t) < 0.0:
        return TraceResult("lost", O, opl, reflections)
    P = vadd(O, vscale(d, t))
    return TraceResult("screen", P, opl + t, reflections)

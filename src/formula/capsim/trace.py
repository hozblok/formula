"""The one multi-bounce tracer behind every stage.

Free space, Lloyd wall and capillaries run through the same loop — the optic
only supplies wall events. Pure geometry: exact Number positions, grazing
sines per bounce, optical path length from the source point. Energy enters
later — Fresnel amplitudes are computed from the recorded sines.
"""

from collections import namedtuple

from .nums import const, vadd, vdot, vscale, vsub

# fate: "screen" | "absorbed" | "lost". reflections: [(point, sin_grazing), ...]
TraceResult = namedtuple(
    "TraceResult", ["fate", "point", "opl", "reflections"]
)


class FresnelAmplitude:
    """Complex r(sin theta), s-pol, with 2*delta and 2*beta fixed at the energy.

    Same formula as xray.reflect_amplitude (cross-checked per run); constants are
    precomputed once so the per-bounce cost is pure Number arithmetic.
    """

    def __init__(self, material, energy_kev):
        p = energy_kev.precision
        two = const("2", p)
        self.d2 = material.delta(energy_kev, p) * two
        self.b2i = material.beta(energy_kev, p) * two * const("i", p)
        self._half = const("0.5", p)
        self._one = const("1", p)

    def __call__(self, sin_theta):
        root = (sin_theta * sin_theta - self.d2 - self.b2i) ** self._half
        return (sin_theta - root) / (sin_theta + root)

    def product(self, sins):
        amp = self._one
        for s in sins:
            amp = amp * self(s)
        return amp


def trace_ray(origin, direction, optic, screen_z, max_bounces):
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

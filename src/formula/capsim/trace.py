"""The one multi-bounce tracer behind every stage.

Free space, Lloyd wall and capillaries run through the same loop — the optic
only supplies wall events. Per ray: exact Number geometry, complex Fresnel
amplitude product (phase kept), optical path length from the source point.
"""

from collections import namedtuple

from .nums import const, vadd, vdot, vscale, vsub

# fate: "screen" | "absorbed" | "lost". reflections: [(point, sin_grazing, r), ...]
TraceResult = namedtuple(
    "TraceResult", ["fate", "point", "amplitude", "opl", "reflections"]
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

    def __call__(self, sin_theta):
        root = (sin_theta * sin_theta - self.d2 - self.b2i) ** self._half
        return (sin_theta - root) / (sin_theta + root)


def trace_ray(origin, direction, optic, screen_z, fresnel, max_bounces,
              amplitude_min):
    """Trace one ray from the source point to the screen plane z=screen_z.

    `direction` must be unit (Number-normalized) so parameters are path lengths.
    """
    p = origin[0].precision
    two = const("2", p)
    amplitude = const("1", p)
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
                                   amplitude, opl + t, reflections)
            _, t, P, normal = event
            opl = opl + t
            dot = vdot(d, normal)
            sin_g = abs(dot)
            r = fresnel(sin_g)
            amplitude = amplitude * r
            d = vsub(d, vscale(normal, two * dot))
            reflections.append((P, sin_g, r))
            O = P
            if float(abs(amplitude)) < amplitude_min:
                return TraceResult("absorbed", P, amplitude, opl, reflections)
        else:
            return TraceResult("lost", O, amplitude, opl, reflections)

    if float(d[2]) <= 0.0:
        return TraceResult("lost", O, amplitude, opl, reflections)
    t = (screen_z - O[2]) / d[2]
    if float(t) < 0.0:
        return TraceResult("lost", O, amplitude, opl, reflections)
    P = vadd(O, vscale(d, t))
    return TraceResult("screen", P, amplitude, opl + t, reflections)

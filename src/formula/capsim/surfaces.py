"""Reflecting optics as event generators over exact Number geometry.

One event protocol for every optic (the Lloyd wall is literally a capillary-wall
surface): ("reflect", t, point, normal) | ("pass", t) | ("absorb", t) |
("exit", None). Hits are solved analytically in Number arithmetic at full
precision; `engine_hit_t` reproduces them through the RaySurface root-finding
engine for the per-run cross-check.
"""

from .nums import const, lift, sqrt, vadd, vscale

# Forward-step floor (m): far below any physical chord, far above 1e-30 root noise.
_EPS_T = 1e-12


class Mirror:
    """Glass half-space x<0 with the reflecting face x=0 over z in [z0, z1]."""

    def __init__(self, z0, z1):
        self.z0, self.z1 = z0, z1
        self._z0f, self._z1f = float(z0), float(z1)
        p = z0.precision
        self._normal = (const("1", p), const("0", p), const("0", p))

    expr_um = "x"  # surface F=0 for the engine cross-check (scale-free)

    def next_event(self, O, d):
        if float(O[0]) <= 0.0 or float(d[0]) >= 0.0:
            return ("exit", None)
        t = -O[0] / d[0]
        zf = float(O[2]) + float(t) * float(d[2])
        if zf > self._z1f:
            return ("exit", None)          # crosses the plane beyond the mirror
        if zf < self._z0f:                 # runs under the leading edge -> front face
            return ("absorb", (self.z0 - O[2]) / d[2])
        return ("reflect", t, vadd(O, vscale(d, t)), self._normal)


class CapillaryBundle:
    """Parallel hollow cylinders along z in [z0, z1]; rays reflect off bore walls."""

    def __init__(self, bores, z0, z1):
        # bores: [{"center": (cx, cy), "radius": a}] as Numbers
        self.bores, self.z0, self.z1 = bores, z0, z1
        self._z0f, self._z1f = float(z0), float(z1)
        p = z0.precision
        self._two = const("2", p)
        self._zero = const("0", p)
        self._eps = _EPS_T
        self._boresf = [
            (float(b["center"][0]), float(b["center"][1]), float(b["radius"]))
            for b in bores
        ]

    def _locate(self, O):
        """Bore containing the point; slack keeps on-wall points (post-reflection)."""
        xf, yf = float(O[0]), float(O[1])
        for bore, (cx, cy, a) in zip(self.bores, self._boresf):
            if (xf - cx) ** 2 + (yf - cy) ** 2 < a * a * (1.0 + 1e-9):
                return bore
        return None

    def _wall_hit(self, O, d, bore):
        """Forward parameter to the bore wall from inside (exact quadratic)."""
        rx, ry = O[0] - bore["center"][0], O[1] - bore["center"][1]
        A = d[0] * d[0] + d[1] * d[1]
        if float(A) < 1e-30:
            return None
        B = self._two * (rx * d[0] + ry * d[1])
        C = rx * rx + ry * ry - bore["radius"] * bore["radius"]
        disc = B * B - self._two * self._two * A * C
        if float(disc) < 0.0:
            return None
        root = sqrt(disc)
        for t in ((root - B) / (self._two * A), (self._zero - root - B) / (self._two * A)):
            if float(t) > self._eps:
                return t
        return None

    def next_event(self, O, d):
        if float(d[2]) <= 0.0:
            return ("absorb", self._zero)          # backward ray: lost in the assembly
        zf = float(O[2])
        if zf < self._z0f - self._eps:
            return ("pass", (self.z0 - O[2]) / d[2])
        if zf >= self._z1f - self._eps:
            return ("exit", None)
        bore = self._locate(O)
        if bore is None:
            return ("absorb", self._zero)          # entrance face / web between bores
        t_wall = self._wall_hit(O, d, bore)
        t_exit = (self.z1 - O[2]) / d[2]
        if t_wall is None or t_exit <= t_wall:
            return ("pass", t_exit)
        P = vadd(O, vscale(d, t_wall))
        a = bore["radius"]
        normal = ((P[0] - bore["center"][0]) / a, (P[1] - bore["center"][1]) / a,
                  self._zero)
        return ("reflect", t_wall, P, normal)


def engine_hit_t(surface_expr_um: str, O, d, t_max_m: float):
    """First hit via the RaySurface root-finding engine; returns t in metres.

    Coordinates are scaled to micrometres for backend conditioning; t scales back.
    """
    from ..intersect import RaySurface
    p = O[0].precision
    scale = lift(1e6, p)
    rs = RaySurface(surface_expr_um, p)
    ts = rs.intersect(tuple(c * scale for c in O), tuple(d),
                      t_max=t_max_m * 1e6, method="subdivision")
    return ts[0] / scale if ts else None

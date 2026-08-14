"""Reflecting optics as event generators over exact Number geometry.

One event protocol for every optic (the flat mirror uses the same wall-event
contract): ("reflect", t, point, normal) | ("pass", t) | ("absorb", t) |
("exit", None). The reference wall is ImplicitWall: an arbitrary implicit
F(x,y,z)=0 traced by the RaySurface root-finding engine at full precision —
exact but slow. The EXPERIMENTAL closed-form fast paths (cylinder/revolution
quadratics, polygon plane fans, torus quartics) live in wall_cylinder.py /
wall_revolution.py / wall_polygon.py / wall_torus.py and must stay
bit-equivalent to the engine:
every run cross-checks the first hit via expr_um (`engine_hit_t`).
"""

import math

from ..formula import Number
from .nums import lift, vadd, vscale
from .types import (_EPS_LOC, _EPS_T, _INSIDE_TOL, _ONWALL_TOL, _TCAP_TOL,
                    HitMethod)
from .units import m_to_um
from .wall_cylinder import CylinderWall
from .wall_funnel import FunnelWall
from .wall_polygon import PolygonWall
from .wall_revolution import RevolutionWall
from .wall_torus import TorusWall


class Mirror:
    """Glass half-space x<0 with the reflecting face x=0 over z in [z0, z1]."""

    def __init__(self, z0, z1):
        self.z0, self.z1 = z0, z1
        self._z0f, self._z1f = float(z0), float(z1)
        p = z0.precision
        self._normal = (Number("1", p), Number("0", p), Number("0", p))

    expr_um = "x"  # surface F=0 for engine cross-checks (scale-free)

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


class ImplicitWall:
    """Arbitrary implicit wall F(x,y,z)=0 in micrometre coordinates, F<0 inside.

    Hits and normals come from the RaySurface root-finding engine at full
    precision — orders of magnitude slower than the closed-form walls; meant
    for prototypes and small ray budgets, not overnight maps.
    """

    def __init__(self, expr, center, aim_radius, method=HitMethod.SUBDIVISION):
        from ..intersect import RaySurface
        self.kind = "implicit"
        self.center = center
        # float center cache: the wall protocol read by gamma._center
        self._cxf, self._cyf = float(center[0]), float(center[1])
        p = center[0].precision
        self.rs = RaySurface(expr, p)
        self.method = method
        self._scale = lift(m_to_um(1), p)
        self.expr_um = None            # the engine IS the hit path here
        self.probe_xy = (1.0, 0.0)

    def _f(self, xf, yf, zf) -> float:
        val = self.rs.surface.evaluate(
            {"x": repr(m_to_um(xf)), "y": repr(m_to_um(yf)),
             "z": repr(m_to_um(zf))})
        return float(Number(val, self.rs.precision))

    def inside(self, xf, yf, zf):
        # slack vs the bore-center depth keeps on-wall points (F rounds to ±eps)
        depth = abs(self._f(float(self.center[0]), float(self.center[1]), zf))
        return self._f(xf, yf, zf) < _INSIDE_TOL * depth

    def hit(self, O, d, t_exit):
        eps_um = m_to_um(_EPS_T)
        Oum = tuple(x * self._scale for x in O)
        ts = self.rs.intersect(Oum, d,
                               t_max=m_to_um(t_exit) * (1.0 + _TCAP_TOL),
                               t_min=eps_um, method=self.method)
        ts = [t for t in ts if float(t) > _ONWALL_TOL * eps_um]
        if not ts:
            return None
        t = ts[0] / self._scale
        return t, vadd(O, vscale(d, t)), self.rs.function(Oum, d).normal_at(ts[0])


def entrance_disk(bore: dict, z0f: float):
    """(cx, cy, r) floats of the entrance aperture — the source-aiming target."""
    kind = bore.get("kind", "cylinder")
    cxf, cyf = float(bore["center"][0]), float(bore["center"][1])
    if kind == "revolution":
        c0, c1, c2 = (float(v) for v in bore["r2_poly"])
        rf = math.sqrt(max(c0 + z0f * (c1 + z0f * c2), 0.0))
    elif kind == "polygon":
        rf = float(bore["radius"]) / math.cos(math.pi / int(bore["sides"]))
    elif kind == "implicit":
        rf = float(bore["aim_radius"])
    else:
        rf = float(bore["radius"])
    return cxf, cyf, rf


def _make_wall(bore: dict, z0):
    """Bore spec -> wall object."""
    kind = bore.get("kind", "cylinder")
    center = bore["center"]
    if kind == "implicit":
        return ImplicitWall(bore["surface"], center, bore["aim_radius"],
                            method=bore["engine_method"])
    if kind == "cylinder":
        return CylinderWall(center, bore["radius"])
    if kind == "revolution":
        return RevolutionWall("revolution", center, bore["r2_poly"])
    if kind == "polygon":
        return PolygonWall(center, bore["radius"], bore["sides"], bore["rotation"])
    if kind == "torus":
        return TorusWall(center, bore["radius"], bore["bend"]["radius"],
                         bore["bend"]["toward"], z0)
    if kind == "funnel":
        return FunnelWall(center, bore["radius"], bore["g"], bore["f"], z0)
    raise ValueError(f"unknown bore kind: {kind!r}")


class CapillaryBundle:
    """Parallel bores along z in [z0, z1]; rays reflect off per-bore walls."""

    def __init__(self, bores, z0, z1):
        self.bores, self.z0, self.z1 = bores, z0, z1
        self._z0f, self._z1f = float(z0), float(z1)
        self._zero = Number("0", z0.precision)
        self.walls = []
        for bore in bores:
            wall = _make_wall(bore, z0)
            wall.aim = entrance_disk(bore, self._z0f)
            self.walls.append(wall)

    def _locate(self, O, d):
        """Wall of the bore containing the point nudged ahead along d: a
        reflection at a bore-bore tangency is on two walls at once."""
        xf, yf, zf = (float(c) for c in O)
        dxf, dyf, dzf = (float(c) for c in d)
        xf, yf, zf = xf + _EPS_LOC * dxf, yf + _EPS_LOC * dyf, zf + _EPS_LOC * dzf
        for wall in self.walls:
            if wall.inside(xf, yf, zf):
                return wall
        return None

    def next_event(self, O, d):
        if float(d[2]) <= 0.0:
            return ("absorb", self._zero)          # backward ray: lost in the assembly
        zf = float(O[2])
        if zf < self._z0f - _EPS_T:
            return ("pass", (self.z0 - O[2]) / d[2])
        if zf >= self._z1f - _EPS_T:
            return ("exit", None)
        wall = self._locate(O, d)
        if wall is None:
            return ("absorb", self._zero)          # entrance face / web between bores
        t_exit = (self.z1 - O[2]) / d[2]
        hit = wall.hit(O, d, t_exit)
        if hit is None or t_exit <= hit[0]:
            return ("pass", t_exit)
        return ("reflect",) + hit


def engine_hit_t(surface_expr_um: str, O, d, t_max_m: float,
                 method=HitMethod.SUBDIVISION):
    """First hit via the RaySurface root-finding engine; returns t in metres.

    Coordinates are scaled to micrometres for backend conditioning; t scales back.
    """
    from ..intersect import RaySurface
    p = O[0].precision
    scale = lift(m_to_um(1), p)
    rs = RaySurface(surface_expr_um, p)
    ts = rs.intersect(tuple(c * scale for c in O), tuple(d),
                      t_max=m_to_um(t_max_m), method=method)
    return ts[0] / scale if ts else None

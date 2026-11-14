# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t), and the NN-symb twin configs A/B full maps against it
# (as one implicit string a polygon is the product of its face planes).
"""Polygon bore: flat faces, face k normal at rotation + 2πk/n and plane
(p - c)·n_k = a_k; equal apothems give the regular polygon. Hits are plane
intersections (exact linear)."""

import math

from ...formula import Number
from ..shared.nums import lift, solver, vadd, vscale
from ..shared.types import _EPS_T, _INSIDE_TOL
from ..shared.units import m_to_um


def polygon_vertices(apothems, sides, rotation):
    """Vertices (x, y) relative to the centre as floats: face k meets face k + 1."""
    n = int(sides)
    a = [float(v) for v in apothems]
    rot = float(rotation)
    out = []
    for k in range(n):
        p0, p1 = rot + 2.0 * math.pi * k / n, rot + 2.0 * math.pi * (k + 1) / n
        c0, s0, c1, s1 = math.cos(p0), math.sin(p0), math.cos(p1), math.sin(p1)
        det = c0 * s1 - s0 * c1
        a0, a1 = a[k], a[(k + 1) % n]
        out.append(((a0 * s1 - a1 * s0) / det, (c0 * a1 - c1 * a0) / det))
    return out


class PolygonWall:
    def __init__(self, center, apothems, sides, rotation):
        self.kind, self.center = "polygon", center
        self.apothems = tuple(apothems) if isinstance(apothems, (tuple, list)) else (apothems,) * int(sides)
        self.apothem = self.apothems[0]                   # face 0: expr_um and older consumers
        p = center[0].precision
        self._zero = Number("0", p)
        n = int(sides)
        cos_s, sin_s = solver("cos(x)", p), solver("sin(x)", p)
        two_pi = Number("pi", p) * Number("2", p)
        self.faces = []
        for k in range(n):
            phi = str(rotation + two_pi * lift(k, p) / lift(n, p))
            self.faces.append((cos_s.number({"x": phi}), sin_s.number({"x": phi})))
        self._facesf = [(float(mx), float(my)) for mx, my in self.faces]
        self._afs = [float(a) for a in self.apothems]
        self._cxf, self._cyf, self._af = float(center[0]), float(center[1]), self._afs[0]
        um = lift(m_to_um(1), p)
        m0x, m0y = self.faces[0]
        self.expr_um = (f"(x-({center[0] * um}))*({m0x})"
                        f"+(y-({center[1] * um}))*({m0y})-({self.apothem * um})")
        self.probe_xy = self._facesf[0]

    def inside(self, xf, yf, zf):
        dx, dy = xf - self._cxf, yf - self._cyf
        return all(mx * dx + my * dy < af * (1.0 + _INSIDE_TOL)
                   for (mx, my), af in zip(self._facesf, self._afs))

    def hit(self, O, d, t_exit):
        rxf, ryf = float(O[0]) - self._cxf, float(O[1]) - self._cyf
        dxf, dyf = float(d[0]), float(d[1])
        best = None
        for i, (mx, my) in enumerate(self._facesf):
            md = mx * dxf + my * dyf
            if md <= 1e-30:
                continue
            tf = (self._afs[i] - (mx * rxf + my * ryf)) / md
            if tf > _EPS_T and (best is None or tf < best[0]):
                best = (tf, i)
        if best is None:
            return None
        mx, my = self.faces[best[1]]
        rx, ry = O[0] - self.center[0], O[1] - self.center[1]
        t = (self.apothems[best[1]] - (mx * rx + my * ry)) / (mx * d[0] + my * d[1])
        return t, vadd(O, vscale(d, t)), (mx, my, self._zero)

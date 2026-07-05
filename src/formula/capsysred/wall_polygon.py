# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t), and the NN-symb twin configs A/B full maps against it
# (as one implicit string a polygon is the product of its face planes).
"""Regular-polygon bore: flat faces; `radius` is the apothem, face k normal
sits at rotation + 2πk/n. Hits are plane intersections (exact linear)."""

from ..formula import Number
from .nums import lift, solver, vadd, vscale
from .surfaces import _EPS_T


class PolygonWall:
    def __init__(self, center, apothem, sides, rotation):
        self.kind, self.center, self.apothem = "polygon", center, apothem
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
        self._cxf, self._cyf, self._af = float(center[0]), float(center[1]), float(apothem)
        um = lift(1e6, p)
        m0x, m0y = self.faces[0]
        self.expr_um = (f"(x-({center[0] * um}))*({m0x})"
                        f"+(y-({center[1] * um}))*({m0y})-({apothem * um})")
        self.probe_xy = self._facesf[0]

    def inside(self, xf, yf, zf):
        dx, dy = xf - self._cxf, yf - self._cyf
        lim = self._af * (1.0 + 1e-9)
        return all(mx * dx + my * dy < lim for mx, my in self._facesf)

    def hit(self, O, d, t_exit):
        rxf, ryf = float(O[0]) - self._cxf, float(O[1]) - self._cyf
        dxf, dyf = float(d[0]), float(d[1])
        best = None
        for i, (mx, my) in enumerate(self._facesf):
            md = mx * dxf + my * dyf
            if md <= 1e-30:
                continue
            tf = (self._af - (mx * rxf + my * ryf)) / md
            if tf > _EPS_T and (best is None or tf < best[0]):
                best = (tf, i)
        if best is None:
            return None
        mx, my = self.faces[best[1]]
        rx, ry = O[0] - self.center[0], O[1] - self.center[1]
        t = (self.apothem - (mx * rx + my * ry)) / (mx * d[0] + my * d[1])
        return t, vadd(O, vscale(d, t)), (mx, my, self._zero)

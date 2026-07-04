# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t), and the NN-symb twin configs A/B full maps against it.
"""Surface of revolution x'²+y'² = r²(z), r² = c0+c1·z+c2·z² — cone,
ellipsoid, paraboloid in one form; the ray intersection is an exact quadratic.
The straight cylinder is served by its own wall_cylinder.CylinderWall."""

from ..formula import Number
from .nums import const, lift, sqrt, vadd, vscale, vunit
from .surfaces import _EPS_T

Vec3 = tuple[Number, Number, Number]


class RevolutionWall:
    def __init__(self, kind: str, center: tuple[Number, Number], r2: Vec3,
                 eps: float = 1e-30):
        # eps: zero threshold for the quadratic coefficients (degenerate -> linear/no hit)
        self.kind, self.center, self.eps = kind, center, eps
        self.c0, self.c1, self.c2 = r2
        p = center[0].precision
        self._two, self._four = const("2", p), const("4", p)
        self._zero, self._half = const("0", p), const("0.5", p)
        self._cxf, self._cyf = float(center[0]), float(center[1])
        self._c0f, self._c1f, self._c2f = (float(v) for v in r2)
        um = lift(1e6, p)
        self.expr_um = (f"(x-({center[0] * um}))^2+(y-({center[1] * um}))^2"
                        f"-(({self.c0 * um * um})+({self.c1 * um})*z+({self.c2})*z^2)")
        self.probe_xy = (1.0, 0.0)

    def r2f(self, zf: float) -> float:
        return self._c0f + zf * (self._c1f + zf * self._c2f)

    def inside(self, xf: float, yf: float, zf: float) -> bool:
        dx, dy = xf - self._cxf, yf - self._cyf
        return dx * dx + dy * dy < self.r2f(zf) * (1.0 + 1e-9)

    def hit(self, O: Vec3, d: Vec3, t_exit: Number):
        """First forward parameter to the wall (exact quadratic), with point+normal."""
        rx, ry = O[0] - self.center[0], O[1] - self.center[1]
        A = d[0] * d[0] + d[1] * d[1] - self.c2 * d[2] * d[2]
        B = (self._two * (rx * d[0] + ry * d[1])
             - d[2] * (self.c1 + self._two * self.c2 * O[2]))
        C = rx * rx + ry * ry - (self.c0 + O[2] * (self.c1 + self.c2 * O[2]))
        if abs(float(A)) < self.eps:
            if abs(float(B)) < self.eps:
                return None
            ts = ((self._zero - C) / B,)
        else:
            disc = B * B - self._four * A * C
            if float(disc) < 0.0:
                return None
            root = sqrt(disc)
            ts = ((self._zero - B - root) / (self._two * A),
                  (self._zero - B + root) / (self._two * A))
        t = min((tt for tt in ts if float(tt) > _EPS_T), key=float, default=None)
        if t is None:
            return None
        P = vadd(O, vscale(d, t))
        n = (P[0] - self.center[0], P[1] - self.center[1],
             self._zero - (self.c1 * self._half + self.c2 * P[2]))
        return t, P, vunit(n)

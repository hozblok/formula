# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t), and the NN-symb twin configs A/B full maps against it.
"""Straight cylinder x'²+y'² = a²; the ray intersection is an exact quadratic
and the normal is radial with zero z-component."""

from ..formula import Number
from .nums import lift, sqrt, vadd, vscale, vunit
from .types import _EPS_T, Vec3


class CylinderWall:
    def __init__(self, center: tuple[Number, Number], radius: Number,
                 eps: float = 1e-30):
        self.kind, self.center, self.radius = "cylinder", center, radius
        # zero threshold for the quadratic coefficients (degenerate -> linear/no hit)
        self.eps = eps
        self._a2 = radius * radius
        p = center[0].precision
        self._two, self._four = Number("2", p), Number("4", p)
        self._zero = Number("0", p)
        self._cxf, self._cyf = float(center[0]), float(center[1])
        self._a2f = float(self._a2)
        um = lift(1e6, p)
        self.expr_um = (f"(x-({center[0] * um}))^2+(y-({center[1] * um}))^2"
                        f"-({self._a2 * um * um})")
        self.probe_xy = (1.0, 0.0)

    def inside(self, xf: float, yf: float, zf: float) -> bool:
        dx, dy = xf - self._cxf, yf - self._cyf
        return dx * dx + dy * dy < self._a2f * (1.0 + 1e-9)

    def hit(self, O: Vec3, d: Vec3, t_exit: Number):
        """First forward parameter to the wall (exact quadratic), with point+normal."""
        rx, ry = O[0] - self.center[0], O[1] - self.center[1]
        A = d[0] * d[0] + d[1] * d[1]
        B = self._two * (rx * d[0] + ry * d[1])
        C = rx * rx + ry * ry - self._a2
        if abs(float(A)) < self.eps:   # degenerate quadratic: same branch as RevolutionWall
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
        n = (P[0] - self.center[0], P[1] - self.center[1], self._zero)
        return t, P, vunit(n)

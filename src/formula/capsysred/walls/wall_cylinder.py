# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t), and the NN-symb twin configs A/B full maps against it.
"""Straight cylinder x'²+y'² = a²; the ray intersection is an exact quadratic
and the normal is radial with zero z-component."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from ...formula import Number
from ..shared.nums import lift, sqrt, vadd, vscale, vunit
from ..shared.types import _EPS_T, _INSIDE_TOL, Vec3
from ..shared.units import m_to_um

if TYPE_CHECKING:
    from ..stages._ray_jets import _Jet

_Coordinate = TypeVar("_Coordinate")


class CylinderWall:
    @staticmethod
    def _local_gradient_hessian(
        dx: _Coordinate, dy: _Coordinate,
    ) -> tuple[tuple[_Coordinate, _Coordinate, float],
               tuple[float, float, float, float, float, float]]:
        """Derivatives at offsets from the axis; Hessian: xx, xy, xz, yy, yz, zz."""
        return (dx, dy, 0.0), (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def __init__(self, center: tuple[Number, Number], radius: Number,
                 eps: float = 1e-30):
        self.kind, self.center, self.radius = "cylinder", center, radius
        # zero threshold for the quadratic coefficients (degenerate -> linear/no hit)
        self.eps = eps
        self._a2 = radius * radius
        self._p = center[0].precision
        self._cxf, self._cyf = float(center[0]), float(center[1])
        self._a2f = float(self._a2)
        self._rf = float(radius)
        um = lift(m_to_um(1), self._p)
        self.expr_um = (f"(x-({center[0] * um}))^2+(y-({center[1] * um}))^2"
                        f"-({self._a2 * um * um})")
        self.probe_xy = (1.0, 0.0)

    @property
    def precision(self) -> int:
        return self._p

    def value(self, point: Sequence[int | float | Number | _Jet]) -> Number | _Jet:
        """F = ((x-cx)^2 + (y-cy)^2 - radius^2)/2 in world metres."""
        dx, dy, _ = self.gradient(point)
        return (dx * dx + dy * dy - self._a2) / 2

    def gradient(self, point: Sequence[int | float | Number | _Jet]) -> tuple[Number | _Jet, Number | _Jet, float]:
        """Unnormalized gradient of F at scalar or jet world coordinates in metres."""
        x, y, _ = point
        return self._local_gradient_hessian(x - self.center[0], y - self.center[1])[0]

    def derivatives_at(self, x, y, z):
        """World-metre derivatives; Hessian order: xx, xy, xz, yy, yz, zz."""
        return self._local_gradient_hessian(x - self._cxf, y - self._cyf)

    def surface_error_at(self, x, y, z):
        """Nonnegative boundary residual in metres, scalar or batched."""
        dx, dy = x - self._cxf, y - self._cyf
        return abs((dx * dx + dy * dy) ** 0.5 - self._rf)

    def inside(self, xf: float, yf: float, zf: float) -> bool:
        dx, dy = xf - self._cxf, yf - self._cyf
        return dx * dx + dy * dy < self._a2f * (1.0 + _INSIDE_TOL)

    def hit(self, O: Vec3, d: Vec3, t_exit: Number):
        """First forward parameter to the wall (exact quadratic), with point+normal."""
        rx, ry = O[0] - self.center[0], O[1] - self.center[1]
        A = d[0] * d[0] + d[1] * d[1]
        B = 2 * (rx * d[0] + ry * d[1])
        C = rx * rx + ry * ry - self._a2
        if abs(float(A)) < self.eps:   # degenerate quadratic: same branch as RevolutionWall
            if abs(float(B)) < self.eps:
                return None
            ts = ((0 - C) / B,)
        else:
            disc = B * B - 4 * A * C
            if float(disc) < 0.0:
                return None
            root = sqrt(disc)
            ts = ((0 - B - root) / (2 * A),
                  (0 - B + root) / (2 * A))
        t = min((tt for tt in ts if float(tt) > _EPS_T), key=float, default=None)
        if t is None:
            return None
        P = vadd(O, vscale(d, t))
        return t, P, vunit(self.gradient(P))

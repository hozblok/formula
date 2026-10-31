# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t), and the NN-symb twin configs A/B full maps against it
# (as one implicit string a polygon is the product of its face planes).
"""Polygon bore: flat faces, face k normal at rotation + 2πk/n and plane
(p - c)·n_k = a_k; equal apothems give the regular polygon. Hits are plane
intersections (exact linear)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...formula import Number
from ..shared.coordinates import surface_coordinates
from ..shared.nums import lift, solver, vadd, vscale
from ..shared.types import _EPS_T, _INSIDE_TOL, _WALL_SURFACE_TOL_M
from ..shared.units import m_to_um

if TYPE_CHECKING:
    from ..shared.jets import _Jet


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


@dataclass(frozen=True)
class PolygonFace:
    """One fixed smooth face; value and gradient describe its supporting plane."""

    wall: PolygonWall
    index: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("polygon face index must be an integer")
        if not 0 <= self.index < len(self.wall.faces):
            raise IndexError("polygon face index out of range")

    @property
    def precision(self) -> int:
        return self.wall.precision

    def value(self, point: Sequence[int | float | Number | _Jet]) -> Number | _Jet:
        """Signed plane residual at world coordinates in metres."""
        x, y, _ = surface_coordinates(point, self.precision)
        nx, ny = self.wall.faces[self.index]
        return ((x - self.wall.center[0]) * nx + (y - self.wall.center[1]) * ny
                - self.wall.apothems[self.index])

    def gradient(self, point: Sequence[int | float | Number | _Jet]) -> tuple[Number, Number, Number]:
        """Constant world gradient, retaining the original Number precision."""
        surface_coordinates(point, self.precision)
        nx, ny = self.wall.faces[self.index]
        return nx, ny, self.wall._zero

    def validate(
        self, point: Sequence[int | float | Number], *,
        origin: Sequence[int | float | Number] | None = None,
        direction: Sequence[int | float | Number] | None = None,
    ) -> None:
        """Require a point on this finite face, separated from polygon edges."""
        if self.wall.surface_at(point).index != self.index:
            raise ValueError("hit is not on the selected polygon face")


class PolygonWall:
    @staticmethod
    def gradient_hessian(nx, ny, scale=1.0):
        """Face gradient and Hessian (xx, xy, xz, yy, yz, zz)."""
        return (scale * nx, scale * ny, 0.0), (0.0,) * 6

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

    @property
    def precision(self) -> int:
        return self.center[0].precision

    def face(self, index: int) -> PolygonFace:
        """Fix one smooth face by its zero-based index."""
        return PolygonFace(self, index)

    def surface_at(self, point: Sequence[int | float | Number]) -> PolygonFace:
        """Select one finite face at a nominal world point; reject edge ambiguity."""
        residuals = [self.face(i).value(point) for i in range(len(self.faces))]
        index = max(range(len(residuals)), key=residuals.__getitem__)
        if abs(residuals[index]) >= _WALL_SURFACE_TOL_M:
            raise ValueError("hit is not on the polygon boundary")
        if any(value >= -_WALL_SURFACE_TOL_M
               for i, value in enumerate(residuals) if i != index):
            raise ValueError("polygon hit is too close to an edge or corner")
        return self.face(index)

    def _face_at(self, x, y):
        """Selected face normal and signed boundary residual; ties retain the first."""
        dx, dy = x - self._cxf, y - self._cyf
        nx, ny = self._facesf[0]
        best = nx * dx + ny * dy - self._afs[0]
        for (mx, my), apothem in zip(self._facesf[1:], self._afs[1:]):
            value = mx * dx + my * dy - apothem
            select = value > best
            keep = 1 - select
            nx, ny = keep * nx + select * mx, keep * ny + select * my
            best = keep * best + select * value
        return nx, ny, best

    def derivatives_at(self, x, y, z):
        """Selected face's derivatives in world metres; ties retain the first face."""
        nx, ny, _ = self._face_at(x, y)
        return self.gradient_hessian(nx, ny)

    def surface_error_at(self, x, y, z):
        """Nonnegative maximum-face residual in metres, scalar or batched."""
        return abs(self._face_at(x, y)[2])

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

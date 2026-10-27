# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t).
"""Conformal-bundle bore (the Liu wall): axis center*g(z), radius r0*f(z),
f and g quadratic in z-z0. The ray hit is an exact Number polynomial —
degree 4, dropping to 2 on structural zeros (a straight cylinder is the
f = g = 1 degeneracy) — solved by the torus two-tier machinery: float
Durand-Kerner seeds -> Newton polish at full precision, exact-sign bisection
as the fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from ...formula import Number
from ..shared.nums import lift, vadd, vscale, vunit
from ..shared.types import _EPS_T, _INSIDE_TOL, _TCAP_TOL
from ..shared.units import m_to_um
from .wall_torus import _quartic_first

if TYPE_CHECKING:
    from ..stages._ray_jets import _Jet

_Coordinate = TypeVar("_Coordinate")


def _poly_first(cs, t_capf):
    """Smallest real root in (_EPS_T, t_capf] of the degree-<=4 Number
    polynomial (leading first). Exact-zero leading coefficients (structural
    wall degeneracies: 0-products, never near-zeros — magnitude truncation is
    the sturm-backend failure mode) shift out and the tail pads with zeros:
    q(t)*t^k keeps every t>0 root, the padded roots at t=0 sit below _EPS_T,
    and the monic quartic runs the torus two-tier machinery unchanged."""
    p = cs[0].precision
    zero = Number("0", p)
    lead = 0
    while lead < len(cs) - 1 and cs[lead] == zero:
        lead += 1
    if lead == len(cs) - 1:
        return None
    c = tuple(v / cs[lead] for v in cs[lead:]) + (zero,) * lead
    return _quartic_first(c, t_capf)


class FunnelWall:
    """Bore whose axis scales as center*g(z) and radius as r0*f(z), with
    g = 1 + ag*(z-z0) + bg*(z-z0)^2 and f of the same form (SI units)."""

    @staticmethod
    def _local_terms(
        x: _Coordinate, y: _Coordinate, z: _Coordinate,
        cx: float | Number, cy: float | Number, ag: float | Number,
        bg: float | Number, af: float | Number, bf: float | Number,
    ) -> tuple[_Coordinate, _Coordinate, _Coordinate, _Coordinate, _Coordinate, _Coordinate]:
        """Axis offsets, center projection, radius profile and its two profile slopes."""
        gg = (z * bg + ag) * z + 1
        ff = (z * bf + af) * z + 1
        gp, fp = z * (2 * bg) + ag, z * (2 * bf) + af
        uu, vv = x - gg * cx, y - gg * cy
        return uu, vv, uu * cx + vv * cy, ff, gp, fp

    @staticmethod
    def _local_gradient(
        uu: _Coordinate, vv: _Coordinate, cn: _Coordinate, ff: _Coordinate,
        gp: _Coordinate, fp: _Coordinate, r02: float | Number,
    ) -> tuple[_Coordinate, _Coordinate, _Coordinate]:
        return uu, vv, (cn * gp + ff * fp * r02) * -1

    @staticmethod
    def gradient_hessian(x, y, z, cx, cy, r0, ag, bg, af, bf):
        """Derivatives of (|xy - center*g(z)|^2 - r0^2*f(z)^2)/2; z is local."""
        uu, vv, cn, ff, gp, fp = FunnelWall._local_terms(x, y, z, cx, cy, ag, bg, af, bf)
        r02 = r0 ** 2
        hzz = ((cx * cx + cy * cy) * gp * gp - 2.0 * cn * bg
               - r02 * (fp * fp + 2.0 * ff * bf))
        return (FunnelWall._local_gradient(uu, vv, cn, ff, gp, fp, r02),
                (1.0, 0.0, -cx * gp, 1.0, -cy * gp, hzz))

    def __init__(self, center, r0, g, f, z0):
        self.kind = "funnel"
        p = center[0].precision
        self._one = Number("1", p)
        self._zero = Number("0", p)
        self.center, self.r0, self.z0 = center, r0, z0
        self.ag, self.bg = g
        self.af, self.bf = f
        self.r02 = r0 * r0
        self._cxf, self._cyf = float(center[0]), float(center[1])
        self._r0f, self._z0f = float(r0), float(z0)
        self._agf, self._bgf = float(g[0]), float(g[1])
        self._aff, self._bff = float(f[0]), float(f[1])
        self.probe_xy = (1.0, 0.0)
        um = lift(m_to_um(1), p)
        zr = f"(z-({self.z0 * um}))"
        G = f"(1+({self.ag / um})*{zr}+({self.bg / (um * um)})*{zr}^2)"
        F = f"(1+({self.af / um})*{zr}+({self.bf / (um * um)})*{zr}^2)"
        self.expr_um = (f"(x-({center[0] * um})*{G})^2"
                        f"+(y-({center[1] * um})*{G})^2"
                        f"-({r0 * um})^2*{F}^2")

    @property
    def precision(self) -> int:
        return self.center[0].precision

    def value(self, point: Sequence[int | float | Number | _Jet]) -> Number | _Jet:
        """F=(|xy-center*g(z)|^2-r0^2*f(z)^2)/2; jet points are all-jet."""
        x, y, z = point
        uu, vv, _, ff, _, _ = self._local_terms(
            x, y, z - self.z0, *self.center, self.ag, self.bg, self.af, self.bf)
        return (uu * uu + vv * vv - ff * ff * self.r02) / 2

    def gradient(
        self, point: Sequence[int | float | Number | _Jet],
    ) -> tuple[Number | _Jet, Number | _Jet, Number | _Jet]:
        """Unnormalized gradient at scalar or all-jet world coordinates in metres."""
        x, y, z = point
        terms = self._local_terms(x, y, z - self.z0, *self.center,
                                  self.ag, self.bg, self.af, self.bf)
        return self._local_gradient(*terms, self.r02)

    def derivatives_at(self, x, y, z):
        """World-metre derivatives; Hessian order: xx, xy, xz, yy, yz, zz."""
        return self.gradient_hessian(x, y, z - self._z0f, self._cxf, self._cyf,
                                     self._r0f, self._agf, self._bgf, self._aff, self._bff)

    def _gf(self, zrf):
        gg = 1.0 + zrf * (self._agf + zrf * self._bgf)
        ff = 1.0 + zrf * (self._aff + zrf * self._bff)
        return gg, ff

    def inside(self, xf, yf, zf):
        gg, ff = self._gf(zf - self._z0f)
        u = xf - self._cxf * gg
        v = yf - self._cyf * gg
        r = self._r0f * ff
        return u * u + v * v < r * r * (1.0 + _INSIDE_TOL)

    def hit(self, O, d, t_exit):
        """First wall hit: exact ray∩funnel polynomial at full precision."""
        one, zero = self._one, self._zero
        cx, cy, r0 = self.center[0], self.center[1], self.r0
        zr0 = O[2] - self.z0
        dz = d[2]
        G0 = one + zr0 * (self.ag + zr0 * self.bg)
        G1 = (self.ag + 2 * self.bg * zr0) * dz
        G2 = self.bg * dz * dz
        F0 = one + zr0 * (self.af + zr0 * self.bf)
        F1 = (self.af + 2 * self.bf * zr0) * dz
        F2 = self.bf * dz * dz
        u0, u1, u2 = O[0] - cx * G0, d[0] - cx * G1, zero - cx * G2
        v0, v1, v2 = O[1] - cy * G0, d[1] - cy * G1, zero - cy * G2
        w0, w1, w2 = r0 * F0, r0 * F1, r0 * F2
        cs = [u2 * u2 + v2 * v2 - w2 * w2,
              2 * (u1 * u2 + v1 * v2 - w1 * w2),
              u1 * u1 + v1 * v1 - w1 * w1 + 2 * (u0 * u2 + v0 * v2 - w0 * w2),
              2 * (u0 * u1 + v0 * v1 - w0 * w1),
              u0 * u0 + v0 * v0 - w0 * w0]
        t = _poly_first(cs, float(t_exit) * (1.0 + _TCAP_TOL) + _EPS_T)
        if t is None:
            return None
        P = vadd(O, vscale(d, t))
        zr = P[2] - self.z0
        gg = one + zr * (self.ag + zr * self.bg)
        gp = self.ag + 2 * self.bg * zr
        ff = one + zr * (self.af + zr * self.bf)
        fp = self.af + 2 * self.bf * zr
        u = P[0] - cx * gg
        v = P[1] - cy * gg
        n = (u, v, zero - (u * cx + v * cy) * gp - self.r02 * ff * fp)
        return t, P, vunit(n)

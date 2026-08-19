# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t). Plan: doc/2026-07-14-funnel-wall-plan.ru.md.
"""Conformal-bundle bore (the Liu wall): axis center*g(z), radius r0*f(z),
f and g quadratic in z-z0. The ray hit is an exact Number polynomial —
degree 4, dropping to 2 on structural zeros (a straight cylinder is the
f = g = 1 degeneracy) — solved by the torus two-tier machinery: float
Durand-Kerner seeds -> Newton polish at full precision, exact-sign bisection
as the fallback."""

from ..formula import Number
from .nums import lift, vadd, vscale, vunit
from .types import _EPS_T, _INSIDE_TOL, _TCAP_TOL
from .shared.units import m_to_um
from .wall_torus import _quartic_first


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

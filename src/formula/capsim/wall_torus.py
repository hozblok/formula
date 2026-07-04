# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t), and the NN-symb twin configs A/B full maps against it.
"""Bent bore: the axis is an arc of radius R (torus segment). The wall hit is
the exact ray∩torus quartic in Number: float Durand-Kerner seeds -> Newton
polish at full precision, exact-sign bisection as the fallback."""

import math

from .nums import const, lift, sqrt, vadd, vdot, vnorm, vscale, vsub, vunit
from .surfaces import _EPS_T


def _horner(cs, t):
    v = cs[0]
    for c in cs[1:]:
        v = v * t + c
    return v


def _dk_roots(cf):
    """All roots of a float polynomial (Durand-Kerner); seeds for Number Newton."""
    scale = max((abs(v) ** (1.0 / (i + 1)) for i, v in enumerate(cf[1:]) if v),
                default=1.0) or 1.0
    roots = [complex(0.4, 0.9) ** k * scale for k in range(1, len(cf))]
    for _ in range(80):
        moved = 0.0
        for i, r in enumerate(roots):
            num = cf[0]
            for c in cf[1:]:
                num = num * r + c
            den = cf[0]
            for j, s in enumerate(roots):
                if j != i:
                    den *= r - s
            step = num / den if den else 0.0
            roots[i] = r - step
            moved = max(moved, abs(step))
        if moved < 3e-15 * scale:
            break
    return roots


def _quartic_first(c, t_capf):
    """Smallest real root of the monic Number quartic in (_EPS_T, t_capf].

    Float Durand-Kerner seeds -> Newton in Number; exact-sign bisection is the
    fallback for roots Newton cannot polish (near-double, stalled seeds).
    """
    p = c[0].precision
    dc = (c[0] * const("4", p), c[1] * const("3", p), c[2] * const("2", p), c[3])
    cand = sorted(r.real for r in _dk_roots([float(v) for v in c])
                  if abs(r.imag) <= 1e-6 * max(1.0, abs(r.real))
                  and _EPS_T / 2 < r.real <= t_capf)
    best = None
    for seed in cand:
        t = lift(seed, p)
        for _ in range(12):
            gp = _horner(dc, t)
            if float(abs(gp)) == 0.0:
                break
            step = _horner(c, t) / gp
            t = t - step
            if float(abs(step)) <= 1e-24 * max(abs(float(t)), 1e-9):
                if float(t) > _EPS_T and (best is None or float(t) < float(best)):
                    best = t
                break
    if best is not None:
        return best
    # sign-change scan (geometric grid) + bisection, all on the exact quartic
    lo = lift(_EPS_T, p)
    flo = float(_horner(c, lo))
    half = const("0.5", p)
    for k in range(1, 49):
        hi = lift(_EPS_T * (t_capf / _EPS_T) ** (k / 48.0), p)
        fhi = float(_horner(c, hi))
        if (flo < 0.0) != (fhi < 0.0):
            for _ in range(90):
                mid = (lo + hi) * half
                if (flo < 0.0) != (float(_horner(c, mid)) < 0.0):
                    hi = mid
                else:
                    lo, flo = mid, float(_horner(c, mid))
            return (lo + hi) * half
        lo, flo = hi, fhi
    return None


class TorusWall:
    """Torus segment leaving z0 along z and curving toward the unit
    xy-direction `toward` (ring center at z=z0)."""

    def __init__(self, center, a, R, toward, z0):
        self.kind = "torus"
        p = center[0].precision
        self._one, self._two = const("1", p), const("2", p)
        self._zero, self._four = const("0", p), const("4", p)
        norm = sqrt(toward[0] * toward[0] + toward[1] * toward[1])
        ux, uy = toward[0] / norm, toward[1] / norm
        self.C = (center[0] + R * ux, center[1] + R * uy, z0)
        self.nhat = (self._zero - uy, ux, self._zero)     # ring-plane normal
        self.R, self.a = R, a
        self.K = R * R - a * a
        self.fourR2 = self._four * R * R
        self._Cf = tuple(float(c) for c in self.C)
        self._nf = (float(self.nhat[0]), float(self.nhat[1]))
        self._Rf, self._af = float(R), float(a)
        # float rho-R cancellation noise grows with R: widen the on-wall slack
        self._in2 = (self._af * self._af * (1.0 + 1e-9)
                     + 64.0 * 2.2e-16 * self._Rf * self._af)
        um = lift(1e6, p)
        cx, cy, cz = (str(c * um) for c in self.C)
        w2 = f"(x-({cx}))^2+(y-({cy}))^2+(z-({cz}))^2"
        s = f"((x-({cx}))*({self.nhat[0]})+(y-({cy}))*({self.nhat[1]}))"
        self.expr_um = (f"({w2}+({self.K * um * um}))^2"
                        f"-({self.fourR2 * um * um})*({w2}-{s}^2)")
        self.probe_xy = (float(ux), float(uy))

    def inside(self, xf, yf, zf):
        wx, wy, wz = xf - self._Cf[0], yf - self._Cf[1], zf - self._Cf[2]
        s = wx * self._nf[0] + wy * self._nf[1]
        rho = math.sqrt(max(wx * wx + wy * wy + wz * wz - s * s, 0.0))
        dr = rho - self._Rf
        return dr * dr + s * s < self._in2

    def hit(self, O, d, t_exit):
        """First wall hit: exact quartic (ray ∩ torus) solved to full precision."""
        w = vsub(O, self.C)
        ws, wd = vdot(w, w), vdot(w, d)
        s0, sd = vdot(w, self.nhat), vdot(d, self.nhat)
        u1, u0 = self._two * wd, ws + self.K
        c = (self._one,
             self._two * u1,
             u1 * u1 + self._two * u0 - self.fourR2 * (self._one - sd * sd),
             self._two * u1 * u0 - self.fourR2 * self._two * (wd - s0 * sd),
             u0 * u0 - self.fourR2 * (ws - s0 * s0))
        t = _quartic_first(c, float(t_exit) * (1.0 + 1e-9) + _EPS_T)
        if t is None:
            return None
        P = vadd(O, vscale(d, t))
        w = vsub(P, self.C)
        s = vdot(w, self.nhat)
        q = vsub(w, vscale(self.nhat, s))
        rho = vnorm(q)
        n = vadd(vscale(q, (rho - self.R) / rho), vscale(self.nhat, s))
        return t, P, vunit(n)

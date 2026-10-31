# EXPERIMENTAL: typed fast-path wall. The reference is the `surface:`
# ImplicitWall in surfaces.py (RaySurface engine); this closed form must stay
# bit-equivalent to it — every run cross-checks the first hit via expr_um
# (engine_hit_t), and the NN-symb twin configs A/B full maps against it.
"""Bent bore: the axis is an arc of radius R (torus segment). The wall hit is
the exact ray∩torus quartic in Number: float Durand-Kerner seeds -> Newton
polish at full precision, exact-sign bisection as the fallback."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...formula import Number
from ..shared.coordinates import surface_coordinates
from ..shared.nums import lift, sqrt, vadd, vdot, vnorm, vscale, vsub, vunit
from ..shared.types import _EPS_T, _INSIDE_TOL, _TCAP_TOL
from ..shared.units import m_to_um

if TYPE_CHECKING:
    from ..shared.jets import _Jet

# Newton polish stop, in digits: quit once the step falls below
# 10^-max(24, p//2) of the root. Newton doubles digits per step, so the final
# accepted step squares the error past 10^-p; targeting p//2 keeps the
# threshold ~p/2 digits above the Horner roundoff floor ~10^-p, which a
# straight 10^-p target would chase forever (never firing — every ray would
# drop to the slow bisection fallback). The 24-digit floor keeps the p=32
# behavior. The step is compared as Number: at p ~ 1024 it is ~1e-500 and
# would underflow double.
_NEWTON_MIN_DIGITS = 24

# Scale floor (m) in that stop: for roots below ~1 nm a purely relative test
# keeps tightening toward zero and may never fire; under the floor the
# threshold goes absolute (10^-digits * 1e-9 m).
_NEWTON_TFLOOR = 1e-9


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


def _float_seeds(c, t_capf):
    """Plausible real roots of the quartic from float Durand-Kerner, ascending.

    Filters the 4 complex DK roots down to Newton seeds:
    - |Im| <= 1e-6*max(1, |Re|): keeps roots whose imaginary part is DK noise
      (a grazing ray yields the near-double pair 3.0000001 +/- 2e-7j — kept)
      and drops genuinely complex pairs (0.2 +/- 0.9j of a branch the ray
      never crosses — dropped).
    - _EPS_T/2 < Re <= t_capf: only roots inside the search window; the
      half-_EPS_T lower edge admits a seed that float noise pushed slightly
      below the floor, letting Newton polish it back above it.
    """
    return sorted(r.real for r in _dk_roots([float(v) for v in c])
                  if abs(r.imag) <= 1e-6 * max(1.0, abs(r.real))
                  and _EPS_T / 2 < r.real <= t_capf)


def _newton_polish(c, dc, seed, rtol):
    """Newton on the exact quartic from a float seed: Number root or None.

    Digits double per step, so 12 iterations lift a ~13-digit double seed to
    ~13*2^12 digits — enough headroom for any realistic precision; the loop
    exits early via the rtol stop long before that. Example at p=64
    (rtol=1e-32): |step| goes ~1e-13 -> 1e-26 -> 1e-52, the third step fires
    the stop, and the returned root is off by ~step^2 ~ 1e-104. None means
    the seed stalled: at a (near-)double root f' -> 0, steps stop shrinking
    and the budget runs out — the caller falls back to exact bisection.
    """
    t = lift(seed, c[0].precision)
    for _ in range(12):
        gp = _horner(dc, t)
        if float(abs(gp)) == 0.0:
            break
        step = _horner(c, t) / gp
        t = t - step
        if abs(step) <= rtol * max(abs(float(t)), _NEWTON_TFLOOR):
            return t
    return None


def _bisect_first(c, t_capf):
    """First sign change of the quartic on a geometric grid, then bisection.

    Rescue path for roots Newton cannot polish (near-double roots, stalled
    seeds); everything runs on the exact quartic. The grid spans
    (_EPS_T, t_capf] in 48 geometric steps — for t_capf=0.1 the probes are
    1e-12, ~1.7e-12, ..., 0.1 — so an exact double root with NO sign change
    (a tangent graze, not a crossing) stays invisible, which is the intended
    miss. Signs are Number comparisons, not float(): near the root the
    residual underflows double (reads 0.0, i.e. "not negative") and at
    p ~ 1024 that used to corrupt the bracket. 8 + p*log2(10) halvings
    shrink the bracket below 10^-p of its span.
    """
    p = c[0].precision
    lo = lift(_EPS_T, p)
    slo = _horner(c, lo) < 0
    half = Number("0.5", p)
    halvings = 8 + int(p * math.log2(10))
    for k in range(1, 49):
        hi = lift(_EPS_T * (t_capf / _EPS_T) ** (k / 48.0), p)
        shi = _horner(c, hi) < 0
        if slo != shi:
            for _ in range(halvings):
                mid = (lo + hi) * half
                if slo != (_horner(c, mid) < 0):
                    hi = mid
                else:
                    lo = mid
            return (lo + hi) * half
        lo, slo = hi, shi
    return None


def _quartic_first(c, t_capf):
    """Smallest real root of the monic Number quartic in (_EPS_T, t_capf].

    Two tiers: float Durand-Kerner seeds polished by Newton at full precision
    (fast path, simple roots), exact-sign bisection when no seed polishes.
    """
    p = c[0].precision
    rtol = lift(f"1e-{max(_NEWTON_MIN_DIGITS, p // 2)}", p)
    dc = (c[0] * Number("4", p), c[1] * Number("3", p), c[2] * Number("2", p), c[3])
    roots = (_newton_polish(c, dc, seed, rtol) for seed in _float_seeds(c, t_capf))
    best = min((t for t in roots if t is not None and float(t) > _EPS_T),
               key=float, default=None)
    return best if best is not None else _bisect_first(c, t_capf)


class TorusWall:
    """Torus segment leaving z0 along z and curving toward the unit
    xy-direction `toward` (ring center at z=z0)."""

    @staticmethod
    def _local_gradient(qx, qy, qz, s, nx, ny, ratio):
        """Gradient from ring-plane coordinates; ratio=(rho-R)/rho."""
        return ratio * qx + s * nx, ratio * qy + s * ny, ratio * qz

    @staticmethod
    def gradient_hessian(qx, qy, qz, s, nx, ny, rho, ratio):
        """q lies in the ring plane, (nx, ny, 0) is its unit normal; ratio=(rho-R)/rho."""
        grad = TorusWall._local_gradient(qx, qy, qz, s, nx, ny, ratio)
        mx, my, mz = qx / rho, qy / rho, qz / rho
        w = 1.0 - ratio
        hess = (ratio + w * (mx * mx + nx * nx), w * (mx * my + nx * ny),
                w * mx * mz, ratio + w * (my * my + ny * ny), w * my * mz,
                ratio + w * mz * mz)
        return grad, hess

    def __init__(self, center: tuple[Number, Number], a: Number, R: Number,
                 toward: tuple[Number, Number], z0: Number) -> None:
        self.kind, self.center = "torus", center
        p = center[0].precision
        self._one = Number("1", p)
        self._zero = Number("0", p)
        norm = sqrt(toward[0] * toward[0] + toward[1] * toward[1])
        ux, uy = toward[0] / norm, toward[1] / norm
        self._toward = (ux, uy)
        self.C = (center[0] + R * ux, center[1] + R * uy, z0)
        self.nhat = (self._zero - uy, ux, self._zero)     # ring-plane normal
        self.R, self.a = R, a
        self.K = R * R - a * a
        self.fourR2 = 4 * R * R
        self._Cf = tuple(float(c) for c in self.C)
        self._nf = (float(self.nhat[0]), float(self.nhat[1]))
        self._Rf, self._af = float(R), float(a)
        self._cxf, self._cyf, self._z0f = float(center[0]), float(center[1]), float(z0)
        self._tf = (float(ux), float(uy))
        # float rho-R cancellation noise grows with R: widen the on-wall slack
        self._in2 = (self._af * self._af * (1.0 + _INSIDE_TOL)
                     + 64.0 * 2.2e-16 * self._Rf * self._af)
        um = lift(m_to_um(1), p)
        cx, cy, cz = (str(c * um) for c in self.C)
        w2 = f"(x-({cx}))^2+(y-({cy}))^2+(z-({cz}))^2"
        s = f"((x-({cx}))*({self.nhat[0]})+(y-({cy}))*({self.nhat[1]}))"
        self.expr_um = (f"({w2}+({self.K * um * um}))^2"
                        f"-({self.fourR2 * um * um})*({w2}-{s}^2)")
        self.probe_xy = (float(ux), float(uy))

    @property
    def precision(self) -> int:
        return self.center[0].precision

    def _tube_coordinates(
        self, point: Sequence[int | float | Number | _Jet],
    ) -> tuple[Number | _Jet, Number | _Jet, Number | _Jet, Number | _Jet, Number | _Jet]:
        """Entrance coordinates and stable rho-R, retaining Number parameter precision."""
        x, y, z = surface_coordinates(point, self.precision)
        dx, dy, zr = x - self.center[0], y - self.center[1], z - self.C[2]
        tx, ty = self._toward
        nx, ny, _ = self.nhat
        xi, eta = dx * tx + dy * ty, dx * nx + dy * ny
        radial = xi - self.R
        rho = sqrt(radial * radial + zr * zr)
        dr = (xi * (xi - 2 * self.R) + zr * zr) / (rho + self.R)
        return xi, eta, zr, rho, dr

    def value(self, point: Sequence[int | float | Number | _Jet]) -> Number | _Jet:
        """F=((rho-R)^2+eta^2-a^2)/2 at scalar or mixed jet world coordinates in metres."""
        _, eta, _, _, dr = self._tube_coordinates(point)
        return (dr * dr + eta * eta - self.a * self.a) / 2

    def gradient(
        self, point: Sequence[int | float | Number | _Jet],
    ) -> tuple[Number | _Jet, Number | _Jet, Number | _Jet]:
        """Unnormalized gradient at scalar or mixed jet world coordinates; requires rho>0."""
        xi, eta, zr, rho, dr = self._tube_coordinates(point)
        tx, ty = self._toward
        nx, ny, _ = self.nhat
        radial = xi - self.R
        return self._local_gradient(radial * tx, radial * ty, zr, eta, nx, ny, dr / rho)

    def _tube_coordinates_at(self, x, y, z):
        """Entrance-frame coordinates, ring-plane radius and stable rho-R."""
        dx, dy, zr = x - self._cxf, y - self._cyf, z - self._z0f
        tx, ty = self._tf
        nx, ny = self._nf
        xi, eta = dx * tx + dy * ty, dx * nx + dy * ny
        rho = ((self._Rf - xi) ** 2 + zr * zr) ** 0.5
        dr = (-2 * self._Rf * xi + xi * xi + zr * zr) / (rho + self._Rf)
        return xi, eta, zr, rho, dr

    def derivatives_at(self, x, y, z):
        """World-metre derivatives using entrance coordinates and stable rho-R."""
        xi, eta, zr, rho, dr = self._tube_coordinates_at(x, y, z)
        tx, ty = self._tf
        nx, ny = self._nf
        return self.gradient_hessian((xi - self._Rf) * tx, (xi - self._Rf) * ty,
                                     zr, eta, nx, ny, rho, dr / rho)

    def surface_error_at(self, x, y, z):
        """Nonnegative tube-boundary residual in metres, scalar or batched."""
        _, eta, _, _, dr = self._tube_coordinates_at(x, y, z)
        return abs((dr * dr + eta * eta) ** 0.5 - self._af)

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
        u1, u0 = 2 * wd, ws + self.K
        c = (self._one,
             2 * u1,
             u1 * u1 + 2 * u0 - self.fourR2 * (self._one - sd * sd),
             2 * u1 * u0 - self.fourR2 * 2 * (wd - s0 * sd),
             u0 * u0 - self.fourR2 * (ws - s0 * s0))
        t = _quartic_first(c, float(t_exit) * (1.0 + _TCAP_TOL) + _EPS_T)
        if t is None:
            return None
        P = vadd(O, vscale(d, t))
        w = vsub(P, self.C)
        s = vdot(w, self.nhat)
        q = vsub(w, vscale(self.nhat, s))
        rho = vnorm(q)
        n = vadd(vscale(q, (rho - self.R) / rho), vscale(self.nhat, s))
        return t, P, vunit(n)

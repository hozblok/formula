"""fb_integrator cf4 of the funnel bores of Stage 16 provider fb: the commutator-free step of order 4."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg.blas import dgemm as _gemm, zaxpy as _axpy
from scipy.special import jv

from ..fb import CHEB_DAMP, CHEB_MARGIN
from .constants import _C1, _C2

if TYPE_CHECKING:
    from ..funnel import FunnelChannel


# CF4 weights
_WP = 0.25 + math.sqrt(3.0) / 6.0
_WM = 0.25 - math.sqrt(3.0) / 6.0


class CF4Step:
    """fb_integrator cf4 of a FunnelChannel: the lens and the axis term as Galerkin blocks, a step as two Chebyshev
    exponentials of their combinations. degree: Chebyshev degrees summed over the steps."""

    def __init__(self, *, ch: FunnelChannel, tol: float) -> None:
        """ch: the bore; tol: Chebyshev tolerance."""
        self.ch, self.tol, self.degree = ch, tol, 0

    def operator(self, za: float, zb: float, wa: float, wb: float,
                 scale: float = 1.0) -> tuple[Callable[..., np.ndarray], tuple[float, float], float]:
        """A = scale (wa H(za) + wb H(zb)) on the paired layout: (apply(z, alpha, shift) = alpha (A - shift) z, spectral
        bounds of the Hermitian part of A, max damping). za, zb: distances from the entrance [m]; wa, wb: weights of H
        at them; scale: common factor of A."""
        ch, b, d = self.ch, self.ch.b, self.ch.b.disc
        qa, qb = ch.coeffs(za), ch.coeffs(zb)
        q2, q1, q0 = (scale * (wa * x + wb * y) for x, y in zip(qa, qb))
        f = q1 * ch.rho0                                  # 2 c0.rho = rho0 r (e^{i phi} + e^{-i phi}) in the turned frame
        hd = scale * (wa * b.h(za) + wb * b.h(zb) - 1j * (wa + wb) * b.damp) + q0
        R, S, T, off, cnt, M = b.R, b.S, b.T, b.off, b.cnt, d.M

        def apply(z: np.ndarray, alpha: float = 1.0, shift: float = 0.0) -> np.ndarray:
            y = (alpha * (hd - shift))[:, None, None] * z
            x = z.view(np.float64).reshape(z.shape[0], -1)
            yr = y.view(np.float64).reshape(z.shape[0], -1)
            a2, af = alpha * q2, alpha * f
            for m in range(M + 1):
                o, n = off[m], cnt[m]
                yo = yr[o:o + n].T
                _gemm(a2, x[o:o + n].T, R[m].T, 1.0, yo, overwrite_c=1)
                if f != 0.0:
                    if m > 0:
                        _gemm(af, x[off[m - 1]:off[m - 1] + cnt[m - 1]].T, S[m - 1].T, 1.0, yo, overwrite_c=1)
                    if m < M:
                        _gemm(af, x[off[m + 1]:off[m + 1] + cnt[m + 1]].T, T[m].T, 1.0, yo, overwrite_c=1)
            if f != 0.0 and M > 0:                             # m = 0 takes T_0 (c_{+1} + c_{-1}) in both slots
                sw = np.ascontiguousarray(z[off[1]:off[1] + cnt[1], ::-1])
                _gemm(af, sw.view(np.float64).reshape(cnt[1], -1).T, T[0].T, 1.0, yr[:cnt[0]].T, overwrite_c=1)
            return y
        v = abs(q2) * d.aw ** 2 + 2.0 * abs(f) * d.aw
        return apply, (min(0.0, float(hd.real.min())) - v, float(hd.real.max()) + v), float(np.max(-hd.imag))

    @staticmethod
    def _chebyshev(apply: Callable[..., np.ndarray], bounds: tuple[float, float], c: np.ndarray, L: float,
                   tol: float) -> tuple[np.ndarray, int]:
        """exp(-i A L) c for apply(z, alpha, shift) = alpha (A - shift) z with the spectrum of A inside bounds
        (Tal-Ezer & Kosloff): T_{i+1} = 2 X T_i - T_{i-1}, X = (A - ec) / dE. c: paired coefficients; L: length of
        the exponent [m]; tol: the degree n grows until |J_n(dE L)| <= tol."""
        emin, emax = bounds
        dE, ec = 0.5 * (emax - emin) * CHEB_MARGIN, 0.5 * (emax + emin)
        t = dE * L
        n = int(t) + 8
        while abs(jv(n, t)) > tol:
            n += 8
        p0 = c
        p1 = apply(c, 1.0 / dE, ec)
        res = jv(0, t) * p0 + (-2j * jv(1, t)) * p1
        for i in range(2, n + 1):
            p2 = apply(p1, 2.0 / dE, ec)
            p2 -= p0
            _axpy(p2.reshape(-1), res.reshape(-1), a=2.0 * (-1j) ** i * jv(i, t))
            p0, p1 = p1, p2
        res *= np.exp(-1j * ec * L)
        return res, n

    def step(self, z: np.ndarray, z0: float, dz: float) -> np.ndarray:
        """One CF4 step: exp(-i dz (w+ H1 + w- H2)) then exp(-i dz (w- H1 + w+ H2)), H1, H2 at the Gauss points,
        w+- = 1/4 +- sqrt(3)/6; each as exp(-i (dz/2) (2 A)) in sub-steps that keep the damping small on the scaled
        spectrum (2 g_max (dz/2) / n_sub <= CHEB_DAMP). z: paired coefficients at z0; z0: start of the step, from the
        entrance [m]; dz: step [m]."""
        z1, z2 = z0 + _C1 * dz, z0 + _C2 * dz
        for wa, wb in ((_WP, _WM), (_WM, _WP)):
            apply, bounds, g_max = self.operator(z1, z2, wa, wb, scale=2.0)
            n_sub = max(1, int(math.ceil(0.5 * g_max * dz / CHEB_DAMP)))
            for _ in range(n_sub):
                z, n = self._chebyshev(apply, bounds, z, 0.5 * dz / n_sub, self.tol)
                self.degree += n
        return z

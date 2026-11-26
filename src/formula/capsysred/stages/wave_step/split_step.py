"""fb_integrator split-step of the funnel bores of Stage 16 provider fb: the lens on the (r, phi) grid."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from scipy import fft as sfft

from .constants import _C1, _C2

if TYPE_CHECKING:
    from ..wave_funnel import FunnelChannel


class SplitStep:
    """fb_integrator split-step of a FunnelChannel: the lens and the axis term as one multiplication on the (r, phi)
    grid. lens_rad: largest lens phase gradient of a step [rad per disc radius]; projection_loss: power fraction
    dropped by the projection, summed over the steps, per node."""

    def __init__(self, *, ch: FunnelChannel, L: float, n: int) -> None:
        """ch: the bore; L: bore length [m]; n: number of steps."""
        b = ch.b
        self.ch = ch
        self.lens_rad = self.lens_phase(L, n)
        self.grid = g = b.grid(self.lens_rad)
        self.K = b.a / b.A0 * g.r ** 2 + 2.0 * ch.rho0 * np.outer(g.cos, g.r)
        # power weights of the paired layout: m = 0 sits in both slots, counted once
        self.pw = (np.concatenate(g.norm)[:, None] * np.stack([np.ones(b.mr.size), b.mr > 0], axis=1))[:, :, None]
        self.projection_loss = 0.0

    def lens_phase(self, L: float, n: int) -> float:
        """Largest radial gradient of the lens phase of one of n steps over L, times the disc radius [rad].
        L: bore length [m]; n: number of steps."""
        b, dz = self.ch.b, L / n
        S = max(b.scale(i * dz)[0] for i in range(n + 1))
        return dz * 2.0 * b.A0 * abs(b.k * b.g[1]) * S * (self.ch.rho0 + b.a)

    def lens(self, z: np.ndarray, q: float, chunk: int = 8) -> np.ndarray:
        """Paired coefficients of exp(-i q K) w, K = (a / A0) r^2 + 2 rho0 r cos(phi): synthesis on the (r, phi) grid,
        the product, projection. z: paired coefficients of w; q: factor of K [1/m^2]; chunk: source nodes per pass."""
        b, g, nt = self.ch.b, self.grid, os.cpu_count() or 1
        e = (g.nphi * np.exp(-1j * q * self.K))[:, :, None]
        out = np.empty_like(z)
        for j in range(0, z.shape[2], chunk):
            x = np.ascontiguousarray(z[:, :, j:j + chunk]).view(np.float64)    # real and imaginary parts as columns
            F = np.zeros((g.nphi, g.r.size, x.shape[2] // 2), complex)
            for m in range(b.disc.M + 1):
                o, n = b.off[m], b.cnt[m]
                F[m] = (g.B[m] @ x[o:o + n, 0]).view(complex)
                if m:
                    F[-m] = (g.B[m] @ x[o:o + n, 1]).view(complex)
            F = sfft.ifft(F, axis=0, overwrite_x=True, workers=nt)
            F *= e
            F = sfft.fft(F, axis=0, overwrite_x=True, workers=nt)
            F *= g.wr[None, :, None] / g.nphi
            y = np.empty((z.shape[0], 2, F.shape[2]), complex)
            yr = y.view(np.float64)
            for m in range(b.disc.M + 1):
                o, n = b.off[m], b.cnt[m]
                yr[o:o + n, 0] = (g.B[m].T @ F[m].view(np.float64)) / g.norm[m][:, None]
                yr[o:o + n, 1] = (g.B[m].T @ F[-m].view(np.float64)) / g.norm[m][:, None] if m else yr[o:o + n, 0]
            out[:, :, j:j + chunk] = y
        return out

    def step(self, z: np.ndarray, z0: float, dz: float) -> np.ndarray:
        """One split step exp(-i dz/2 D) exp(-i dz nu K) exp(-i dz/2 D): D the eigenvalues, the damping and the constant,
        nu = k g1 S the factor of the lens; both the means over the Gauss points of the step. The power that the product
        puts above jmax is dropped by the projection; its fraction is summed in projection_loss.
        z, z0, dz: as in CF4Step.step."""
        ch, b = self.ch, self.ch.b
        z1, z2 = z0 + _C1 * dz, z0 + _C2 * dz
        (_, n1, c1), (_, n2, c2) = ch.coeffs(z1), ch.coeffs(z2)
        e = np.exp(-0.5j * dz * (0.5 * (b.h(z1) + b.h(z2) + c1 + c2) - 1j * b.damp))[:, None, None]
        z = e * z
        y = self.lens(z, 0.5 * dz * (n1 + n2))
        self.projection_loss += 1.0 - (self.pw * np.abs(y) ** 2).sum((0, 1)) / (self.pw * np.abs(z) ** 2).sum((0, 1))
        return e * y

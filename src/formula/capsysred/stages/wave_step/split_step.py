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
    grid. phase_gradient_rad: largest lens phase gradient of a step [rad per disc radius]; projection_loss: power fraction
    dropped by the projection, summed over the steps, per node."""

    def __init__(self, *, ch: FunnelChannel, L: float, n: int) -> None:
        """ch: the bore; L: bore length [m]; n: number of steps."""
        b = ch.b
        self.ch = ch
        self.phase_gradient_rad = self.phase_gradient(L, n)
        self.grid = g = b.grid(self.phase_gradient_rad)
        self.K = b.a / b.A0 * g.r ** 2 + 2.0 * ch.rho0 * np.outer(g.cos, g.r)
        # power weights of the paired layout: m = 0 sits in both slots, counted once
        self.pw = (np.concatenate(g.norm)[:, None] * np.stack([np.ones(b.mr.size), b.mr > 0], axis=1))[:, :, None]
        self.projection_loss = 0.0

    def phase_gradient(self, L: float, n: int) -> float:
        """Largest radial gradient of the lens phase of one of n steps over L, times the disc radius [rad].
        L: bore length [m]; n: number of steps."""
        b, dz = self.ch.b, L / n
        S = max(b.scale(i * dz)[0] for i in range(n + 1))
        return dz * 2.0 * b.A0 * abs(b.k * b.g[1]) * S * (self.ch.rho0 + b.a)

    def to_grid(self, z: np.ndarray) -> np.ndarray:
        """Field of the paired coefficients z at the (phi, r) nodes, divided by nphi: (nphi, nr, B).
        z: paired coefficients, (N, 2, B)."""
        b, g = self.ch.b, self.grid
        x = np.ascontiguousarray(z).view(np.float64)           # real and imaginary parts as columns
        F = np.zeros((g.nphi, g.r.size, x.shape[2] // 2), complex)
        for m in range(b.disc.M + 1):
            o, n = b.off[m], b.cnt[m]
            F[m] = (g.B[m] @ x[o:o + n, 0]).view(complex)
            if m:
                F[-m] = (g.B[m] @ x[o:o + n, 1]).view(complex)
        return sfft.ifft(F, axis=0, overwrite_x=True, workers=os.cpu_count() or 1)

    def to_modes(self, F: np.ndarray) -> np.ndarray:
        """Paired coefficients (N, 2, B) of the field F at the (phi, r) nodes.
        F: field, (nphi, nr, B), overwritten."""
        b, g = self.ch.b, self.grid
        F = sfft.fft(F, axis=0, overwrite_x=True, workers=os.cpu_count() or 1)
        F *= g.wr[None, :, None] / g.nphi
        y = np.empty((b.mr.size, 2, F.shape[2]), complex)
        yr = y.view(np.float64)
        for m in range(b.disc.M + 1):
            o, n = b.off[m], b.cnt[m]
            yr[o:o + n, 0] = (g.B[m].T @ F[m].view(np.float64)) / g.norm[m][:, None]
            yr[o:o + n, 1] = (g.B[m].T @ F[-m].view(np.float64)) / g.norm[m][:, None] if m else yr[o:o + n, 0]
        return y

    def apply_K(self, z: np.ndarray, q: float, chunk: int = 8) -> np.ndarray:
        """Paired coefficients of exp(-i q K) w, K = (a / A0) r^2 + 2 rho0 r cos(phi): the product at the (phi, r) nodes.
        z: paired coefficients of w; q: factor of K [1/m^2]; chunk: source nodes per pass."""
        e = (self.grid.nphi * np.exp(-1j * q * self.K))[:, :, None]         # nphi: to_grid gives the field over nphi
        out = np.empty_like(z)
        for j in range(0, z.shape[2], chunk):
            F = self.to_grid(z[:, :, j:j + chunk])
            F *= e
            out[:, :, j:j + chunk] = self.to_modes(F)
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
        y = self.apply_K(z, 0.5 * dz * (n1 + n2))
        self.projection_loss += 1.0 - (self.pw * np.abs(y) ** 2).sum((0, 1)) / (self.pw * np.abs(z) ** 2).sum((0, 1))
        return e * y

"""One funnel bore of the provider fb: FunnelChannel with the entrance modes, the propagation and the exit field."""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import fft as sfft
from scipy.special import jv

from ..fb.basis import radial_nodes
from ..fb.channel import CHEB_TOL
from .basis import FunnelBasis
from .step.cf4_step import CF4Step
from .step.split_step import SplitStep


class FunnelChannel:
    """One funnel bore of entrance centre c0 on a shared FunnelBasis. The propagation frame is turned by the polar angle
    of c0, so the axis term is real: c_m -> c_m exp(i m alpha) on entry and back on exit."""

    def __init__(self, *, basis: FunnelBasis, c0: Sequence[float] | np.ndarray,
                 integrator: Literal["cf4", "split-step"]) -> None:
        """basis: shared FunnelBasis; c0: bore centre at the entrance (x, y) [m]; integrator: cf4 or split-step."""
        self.b, self.c0, self.integrator = basis, np.asarray(c0, float), integrator
        self.rho0 = float(np.hypot(*self.c0))
        self.alpha = float(np.arctan2(self.c0[1], self.c0[0]))

    def coeffs(self, z: float) -> tuple[float, float, float]:
        """Factors of |rho|^2 and of 2 c0.rho, and the constant: (k/2) S S'', (k/2) S G'', (k/2) G G'' rho0^2.
        z: distance from the entrance [m]."""
        b = self.b
        kg = b.k * b.g[1]                                      # k G'' / 2
        S = b.scale(z)[0]
        return kg * b.a / b.A0 * S, kg * S, kg * b.G(z) * self.rho0 ** 2

    def propagate(self, c: np.ndarray, L: float, dz_max: float, tol: float = CHEB_TOL) -> tuple[np.ndarray, int, int]:
        """Exit coefficients (signed layout) of the entrance coefficients c, steps of at most dz_max; with the power
        fraction left in the absorbing layer at the exit, per node. c: (2M+1, nmax, B); L: bore length [m];
        dz_max: largest step [m]; tol: Chebyshev tolerance of cf4. The integrator of the run is kept as stepper."""
        n_steps = max(1, int(math.ceil(L / dz_max - 1e-9)))
        dz, split = L / n_steps, self.integrator == "split-step"
        self.stepper = st = SplitStep(ch=self, L=L, n=n_steps) if split else CF4Step(ch=self, tol=tol)
        rot = np.exp(1j * np.array(self.b.disc.ms) * self.alpha)[:, None, None]
        z = self.b.pair(c * rot)
        for i in range(n_steps):
            z = st.step(z, i * dz, dz)
        cL = self.b.unpair(z) / rot
        d = self.b.disc
        p = np.abs(cL) ** 2 * d.mnorm[:, :, None]
        top = (d.zp[d.am] > (1.0 - self.b.layer) * d.jmax)[:, :, None]
        self.layer_power = np.where(top, p, 0.0).sum((0, 1)) / p.sum((0, 1))
        return cL, n_steps, 0 if split else st.degree

    def entrance_modes(self, nodes: Sequence[Sequence[float]] | np.ndarray, D: float,
                       nr: int | None = None) -> np.ndarray:
        """w(rho, 0) of paraxial point sources at (node, -D) through the geometric aperture a: (2M+1, nmax, B).
        c_mn = int_0^a int u J_|m|(j r / A0) e^{-i m phi} r dr dphi / (pi A0^2 J_{|m|+1}(j)^2), Gauss-Legendre on [0, a]
        (the aperture edge is the end of the interval), FFT over phi. nodes: source points (x, y) [m]; D: distance from
        the source plane to the entrance [m]; nr: radial nodes on [0, a] (None: radial_nodes(jmax))."""
        b, d = self.b, self.b.disc
        n_r = nr or radial_nodes(d.jmax)
        tq, wq = leggauss(n_r)
        r = 0.5 * b.a * (tq + 1.0)
        wr = 0.5 * b.a * wq
        nphi = d.nphi
        phi = 2.0 * math.pi * np.arange(nphi) / nphi
        dx, dy = r[:, None] * np.cos(phi)[None, :], r[:, None] * np.sin(phi)[None, :]
        X, Y = self.c0[0] + dx, self.c0[1] + dy
        S, ph = self._frame(0.0, dx, dy)
        lam = 2.0 * math.pi / b.k
        gauge = S * np.exp(-1j * ph)
        F = np.empty((len(nodes), n_r, nphi), complex)
        for i, (sx, sy) in enumerate(nodes):
            E = np.exp(1j * b.k * ((X - sx) ** 2 + (Y - sy) ** 2) / (2.0 * D)) / (1j * lam * D)
            F[i] = sfft.fft(gauge * E, axis=1) / nphi
        out = np.zeros((2 * d.M + 1, d.nmax, len(nodes)), complex)
        for mm, zz in enumerate(d.zeros):
            P = jv(mm, np.outer(r, zz / d.aw)) * (wr * r)[:, None] / (0.5 * d.aw ** 2 * jv(mm + 1, zz) ** 2)[None, :]
            for m in ({mm, -mm}):
                out[m + d.M, :zz.size] = (F[:, :, m % nphi] @ P).T
        return out

    def _frame(self, z: float, dx: float | np.ndarray, dy: float | np.ndarray) -> tuple[float, float | np.ndarray]:
        """Scale S of the bore frame at z and the phase of E / w at the lab offsets (dx, dy) from the bore axis.
        z: distance from the entrance [m]; dx, dy: [m]."""
        b = self.b
        S, dS = b.scale(z)
        G, dG = b.G(z), b.dG(z)
        return S, b.k * (dS * (dx * dx + dy * dy) / (2.0 * S) + dG * (self.c0[0] * dx + self.c0[1] * dy)
                         + 0.5 * G * dG * self.rho0 ** 2)

    def centre_values(self, c: np.ndarray, z: float) -> np.ndarray:
        """E at the bore centre G(z) c0 for every node: only m = 0 is non-zero at rho = 0. (B,)
        c: coefficients at z, (2M+1, nmax, B); z: distance from the entrance [m]."""
        S, ph = self._frame(z, 0.0, 0.0)
        w0 = c[self.b.disc.M].sum(axis=0)                       # J_0(0) = 1
        return w0 / S * np.exp(1j * ph)

    def points(self, c: np.ndarray, z: float, X: np.ndarray, Y: np.ndarray, dr: float) -> tuple[np.ndarray, np.ndarray]:
        """E at lab points (X, Y) inside the geometric core a G(z) for every node: (n_points, B) and the inside mask.
        Radial functions by cubic Hermite interpolation on r = 0, dr, ... (relative error ~ (2 pi dr jmax / A0)^4 / 384).
        c: coefficients at z, (2M+1, nmax, B); z: distance from the entrance [m]; X, Y: [m]; dr: radial step [m]."""
        b, d = self.b, self.b.disc
        G = b.G(z)
        dx, dy = X - G * self.c0[0], Y - G * self.c0[1]
        inside = np.hypot(dx, dy) < b.a * G
        dx, dy = dx[inside], dy[inside]
        S, ph = self._frame(z, dx, dy)
        rho, phi = np.hypot(dx, dy) / S, np.arctan2(dy, dx)
        b.prefetch(dr)
        i = np.minimum((rho / dr).astype(int), int(d.aw / dr))
        t = rho / dr - i
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        E = np.zeros((rho.size, c.shape[2]), complex)
        for mm in range(d.M + 1):
            J, dJ = b.radial_tables(mm, dr)
            n = J.shape[1]
            for m in ({mm, -mm}):
                cm = c[m + d.M, :n]
                R, dR = J @ cm, dJ @ cm
                val = (h00[:, None] * R[i] + h10[:, None] * dr * dR[i] + h01[:, None] * R[i + 1] + h11[:, None] * dr * dR[i + 1])
                E += val * np.exp(1j * m * phi)[:, None]
        return E * (np.exp(1j * ph) / S)[:, None], inside

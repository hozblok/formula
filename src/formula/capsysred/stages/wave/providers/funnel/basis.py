"""Basis of the provider fb for funnel bores: FunnelBasis with the Galerkin blocks and the radial tables of its disc."""

import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import fft as sfft
from scipy.special import jv, jvp

from ..fb.basis import Disc, newton_kappa, radial_nodes, wall_offset


def _galerkin(d, nr):
    """Galerkin blocks of the disc d per |m| on its own Gauss-Legendre radial quadrature (nr points):
    R_m = P_m r^2 B_m (n_m x n_m), S_m = P_{m+1} r B_m (n_{m+1} x n_m), T_m = P_m r B_{m+1} (n_m x n_{m+1})."""
    t, w = leggauss(nr)
    r = 0.5 * d.aw * (t + 1.0)
    wr = 0.5 * d.aw * w
    Bs = [jv(mm, np.outer(r, z / d.aw)) for mm, z in enumerate(d.zeros)]
    ns = [0.5 * d.aw ** 2 * jv(mm + 1, z) ** 2 for mm, z in enumerate(d.zeros)]
    R = [np.ascontiguousarray((B.T * (wr * r ** 3)) @ B / n[:, None]) for B, n in zip(Bs, ns)]
    S = [np.ascontiguousarray((Bs[m + 1].T * (wr * r ** 2)) @ Bs[m] / ns[m + 1][:, None]) for m in range(d.M)]
    T = [np.ascontiguousarray((Bs[m].T * (wr * r ** 2)) @ Bs[m + 1] / ns[m][:, None]) for m in range(d.M)]
    return R, S, T


class FunnelBasis:
    """What all bores of one radius share at one photon energy: the disc A0 = a + Re ell with every zero below jmax,
    the Galerkin blocks, the profile G, the absorbing top layer and the mode eigenvalues h(G) (cached by G).
    Propagation works on the paired layout (N, 2, B): rows (|m|, n) over |m| = 0..M, slots +|m| and -|m|."""

    def __init__(self, *, a: float, g: Sequence[float], k: float, delta: float, beta: float, jmax: float,
                 nr: int | None = None, layer: float = 0.0, gamma_max: float = 5.0e2, blocks: bool = True,
                 wall: str = "exact-roots") -> None:
        """a: bore radius at the entrance [m]; g: (g0 [1/m], g1 [1/m^2]) of G; k: wavenumber [1/m]; delta, beta: wall
        index n = 1 - delta + i beta; jmax: cut of the zeros, j_mn < jmax; nr: radial nodes of the Galerkin blocks
        (None: radial_nodes(jmax)); layer: absorbing fraction of jmax at its top (0: none); gamma_max: amplitude damping
        rate at jmax [1/m]; blocks: build the Galerkin blocks of cf4; wall: exact-roots or effective-radius."""
        self.a, self.g, self.wall = float(a), (float(g[0]), float(g[1])), wall
        self.layer, self.gamma_max = float(layer), float(gamma_max)
        self.k, self.delta, self.beta = float(k), float(delta), float(beta)
        self.ell = wall_offset(k, delta, beta)
        self.A0 = self.a + self.ell.real
        self.disc = d = Disc(aw=self.A0, jmax=jmax, nr=8)                # zeros, norms and index maps; blocks on their own quadrature
        self.R, self.S, self.T = _galerkin(d, nr or radial_nodes(jmax)) if blocks else (None, None, None)
        self.cnt = np.array([z.size for z in d.zeros])
        self.off = np.concatenate([[0], np.cumsum(self.cnt)])
        self.mr = np.repeat(np.arange(d.M + 1), self.cnt)
        self.nr_ = np.concatenate([np.arange(n) for n in self.cnt])
        zc = d.zp[self.mr, self.nr_]
        js = (1.0 - self.layer) * d.jmax
        x = np.clip((zc - js) / (d.jmax - js), 0.0, 1.0) if self.layer > 0.0 else np.zeros(zc.size)
        self.damp = self.gamma_max * x * x                     # amplitude damping rate of the top layer, per row
        self.zc = zc
        self._h, self.fallback, self._tab = {}, {}, {}
        self._warm = None                                      # (G, kappa per row) of the last roots
        self._grid = None

    def G(self, z: float) -> float:
        """Profile G = 1 + g0 z + g1 z^2. z: distance from the entrance [m]."""
        return 1.0 + self.g[0] * z + self.g[1] * z * z

    def dG(self, z: float) -> float:
        """dG/dz [1/m]. z: distance from the entrance [m]."""
        return self.g[0] + 2.0 * self.g[1] * z

    def scale(self, z: float) -> tuple[float, float]:
        """Scale S = (a G + Re ell) / A0 of the bore frame and its z-derivative. z: distance from the entrance [m]."""
        return (self.a * self.G(z) + self.ell.real) / self.A0, self.a * self.dG(z) / self.A0

    def h(self, z: float) -> np.ndarray:
        """kappa^2 / 2k of the bore of radius a G(z) per row. effective-radius: kappa = j / (a G + ell). exact-roots:
        damped Newton, warm-started from the roots of the previous G scaled by (a G' + ell) / (a G + ell), with the
        effective radius where Newton fails. z: distance from the entrance [m]."""
        G = self.G(z)
        if self.wall == "effective-radius":
            return (self.zc / (self.a * G + self.ell)) ** 2 / (2.0 * self.k)
        key = round(G, 12)
        if key not in self._h:
            ag = self.a * G
            dir_guess = self.zc / (ag + self.ell)
            guess = dir_guess.astype(complex)
            if self._warm is not None:
                Gw, kw = self._warm
                guess = np.where(kw != 0.0, kw * (self.a * Gw + self.ell) / (ag + self.ell), guess)
            k2 = self.k ** 2 * (1.0 - (1.0 - self.delta + 1j * self.beta) ** 2)
            mr = self.mr.astype(float)

            def run(ix):
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    return newton_kappa(mr[ix], guess[ix], ag, k2)
            nt = os.cpu_count() or 1
            with ThreadPoolExecutor(nt) as ex:                 # the Bessel ufuncs release the GIL
                parts = list(ex.map(run, np.array_split(np.arange(mr.size), 4 * nt)))
            kap, ok = np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])
            ok &= np.isfinite(kap) & (kap.imag <= 0.0) & (np.abs(kap - dir_guess) * ag < 1.0)
            full = np.where(ok, kap, dir_guess)
            self._warm = (G, np.where(ok, kap, 0.0))
            self._h[key] = full * full / (2.0 * self.k)
            self.fallback[key] = int(np.sum(~ok))
        return self._h[key]

    def pair(self, c: np.ndarray) -> np.ndarray:
        """Signed (2M+1, nmax, B) -> paired (N, 2, B). c: coefficients, rows m = -M..M, columns n, B source nodes."""
        M = self.disc.M
        return np.ascontiguousarray(np.stack([c[M + self.mr, self.nr_], c[M - self.mr, self.nr_]], axis=1))

    def unpair(self, z: np.ndarray) -> np.ndarray:
        """Paired (N, 2, B) -> signed (2M+1, nmax, B). z: coefficients, rows (|m|, n), slots +|m| and -|m|."""
        d = self.disc
        c = np.zeros((2 * d.M + 1, d.nmax, z.shape[2]), complex)
        c[d.M - self.mr, self.nr_] = z[:, 1]
        c[d.M + self.mr, self.nr_] = z[:, 0]
        return c

    def grid(self, T: float) -> SimpleNamespace:
        """(r, phi) grid of the split step: Gauss-Legendre radii with the weights w r, J_|m|(j r / A0) per |m| on them,
        the cosine of the azimuthal nodes, the mode norms. Shared by the bores; rebuilt for a larger T only.
        T: largest gradient of the lens phase of a step [rad per disc radius]."""
        if self._grid is None or self._grid.T < T:
            d = self.disc
            t, w = leggauss(radial_nodes(d.jmax + T))
            r = 0.5 * d.aw * (t + 1.0)
            nphi = sfft.next_fast_len(2 * d.M + int(T + 10.0 * T ** (1.0 / 3.0)) + 11)
            with ThreadPoolExecutor(os.cpu_count() or 1) as ex:  # the Bessel ufuncs release the GIL
                B = list(ex.map(lambda mm: jv(mm, np.outer(r, d.zeros[mm] / d.aw)), range(d.M + 1)))
            self._grid = SimpleNamespace(T=T, r=r, wr=0.5 * d.aw * w * r, nphi=nphi, B=B,
                                         cos=np.cos(2.0 * math.pi * np.arange(nphi) / nphi),
                                         norm=[0.5 * d.aw ** 2 * jv(mm + 1, z) ** 2 for mm, z in enumerate(d.zeros)])
        return self._grid

    def radial_tables(self, m: int, dr: float, cache_bytes: float = 3.0 * 2 ** 30) -> tuple[np.ndarray, np.ndarray]:
        """J_|m|(j r / A0) and its r-derivative on r = 0, dr, ..., A0 (+dr): (nf, n_m) each (float32, cached while the
        tables of all |m| fit in cache_bytes). m: azimuthal order, taken as |m|; dr: radial step [m]; cache_bytes:
        cap on the tables of all |m| [bytes]."""
        d = self.disc
        key = (abs(m), dr)
        if key in self._tab:
            return self._tab[key]
        z = d.zeros[abs(m)]
        rf = np.arange(0.0, d.aw + 2.0 * dr, dr)
        x = np.outer(rf, z / d.aw)
        tab = (jv(abs(m), x).astype(np.float32), (jvp(abs(m), x) * (z / d.aw)[None, :]).astype(np.float32))
        if 8 * rf.size * d.nmodes <= cache_bytes:
            self._tab[key] = tab
        return tab

    def prefetch(self, dr: float, cache_bytes: float = 3.0 * 2 ** 30) -> None:
        """All radial tables of step dr in threads (the Bessel ufuncs release the GIL), when they fit the cache.
        dr: radial step [m]; cache_bytes: cap on the tables of all |m| [bytes]."""
        d = self.disc
        if 8 * (int((d.aw + 2.0 * dr) / dr) + 2) * d.nmodes > cache_bytes:
            return
        todo = [mm for mm in range(d.M + 1) if (mm, dr) not in self._tab]
        with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
            list(ex.map(lambda mm: self.radial_tables(mm, dr, cache_bytes), todo))

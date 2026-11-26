"""Stage 16 provider fb for conformal funnel bores (kind funnel, f = g): every bore of the scene scales about the device
axis with G(z) = 1 + g0 z + g1 z^2 (z from the entrance): the axis of a bore is G(z) c0, its radius a G(z). The field
of a bore lives in the frame of the bore, rho = (r - G c0) / S(z), scaled with its effective Dirichlet radius,
S = (a G + Re ell) / A0, so the wall is the fixed disc A0 = a + Re ell of the Fourier-Bessel basis:
  E(r, z) = S^-1 exp(i k [S' |x|^2 / 2S + G' c0.x + G G' |c0|^2 / 2]) w(rho, z),   x = r - G c0,
  i dw/dz = -Lap_rho w / (2 k S^2) + [V + (k/2) S S'' |rho|^2 + k S G'' c0.rho + (k/2) G G'' |c0|^2] w.
The kinetic part with the wall has the eigenvalues of the physical bore of radius a G(z): fb_wall exact-roots takes the
roots of the scalar characteristic equation, effective-radius the zero field at the radius a G + ell, ell the complex
penetration depth, kappa = j / (a G + ell).
fb_integrator cf4: the lens and the axis term enter as Galerkin blocks (r^2 within m, r e^{+-i phi} between m and
m +- 1); a step is the commutator-free exponential integrator of order 4 (two exponentials of combinations of H at
the Gauss points), each exponential a Chebyshev series.
fb_integrator split-step: a step is exp(-i dz/2 D) exp(-i dz nu K) exp(-i dz/2 D), order 2: D the diagonal of the modes,
nu K = k g1 S [(a / A0) |rho|^2 + 2 c0.rho] the lens and the axis term as one multiplication on the (r, phi) grid; no
Galerkin block and no matrix exponential. The projection back on the modes drops what the product puts above jmax; that
power (projection_loss, summed over the steps) falls as 1 / steps only, so the order is 2 while it stays below the
wanted error: the top of the basis has to be empty. The basis keeps every zero j_mn < jmax; its outer fraction
`layer` absorbs (amplitude rate gamma_max ((j - j_s) / (jmax - j_s))^2). The Kirchhoff edge tail of the entrance is
fixed in j (the frame keeps j while the physical angle falls as 1 / G), so jmax has to cover the guided modes of the
entrance, k theta_c a.
Kirchhoff entrance: the paraxial point source inside the geometric aperture a. Exit plane: the field inside the
geometric exit core a G(L) at the screen pixels (observable point).
The bores are propagated one after another with all source nodes as one block; W and I accumulate bore by bore (the exit
cores do not overlap), and the fields at the bore centres give the matrix of mu between the centres. With dump_field
the exit coefficients of every bore (all nodes) are written to funnel-exit/ for post-processing without propagation."""

import json
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import fft as sfft
from scipy.special import jv, jvp

from ...screen import ScreenGrid
from .fb import CHEB_TOL, Disc, newton_kappa, radial_nodes, wall_offset
from .output import _report_lines
from .screen_sampler import ScreenSampler, _finish
from .source import _rule
from .step.cf4_step import CF4Step
from .step.split_step import SplitStep


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


# ----------------------------------------------------------------- scene driver

def funnel_scene(sim, wave, lines, log, out_dir=None):
    """Stage 16 on a scene of conformal funnel bores: exit-plane screens (point observable), the matrix of mu between
    the bore centres. Returns [(label, res), ...] like fb_capillary_scene."""
    cfg = sim.cfg
    cap = cfg.capillary
    if wave["observable"] != "point":
        raise ValueError("wave_estimator: funnel bores support observable point only")
    if wave["entrance_spot"]:
        raise ValueError("wave_estimator: funnel bores take the source of the scene, not entrance_spot")
    bores = cap.bores
    a = float(bores[0]["radius"])
    g = tuple(float(v) for v in bores[0]["g"])
    for b in bores:
        if tuple(float(v) for v in b["g"]) != g or tuple(float(v) for v in b["f"]) != g or float(b["radius"]) != a:
            raise ValueError("wave_estimator: funnel bores need one radius and one profile g = f for all bores")
    L = float(cap.z1) - float(cap.z0)
    D = float(cap.z0) - float(cap.source.position[2])
    screens = cap.all_screens
    for s in screens:
        if abs(float(s.z) - float(cap.z1)) > 1e-12:
            raise ValueError("wave_estimator: funnel bores support screens on the exit plane only")
    nodes, weights, rule = _rule(wave, "capillary", cap.source, None)
    centres = [np.array([float(b["center"][0]), float(b["center"][1])]) for b in bores]
    G_L = 1.0 + g[0] * L + g[1] * L * L
    if not wave["fb_jmax"]:
        raise ValueError("wave_estimator: funnel bores need fb_jmax (every zero below it; the guided modes reach k theta_c a)")
    jmax, layer = int(wave["fb_jmax"]), 0.1
    dz = float(wave["fb_dz"])
    tol = float(wave["fb_chebyshev_tol"])
    wall, integrator = wave["fb_wall"], wave["fb_integrator"]
    split = integrator == "split-step"
    t0 = time.time()
    samplers = [ScreenSampler(ScreenGrid(s), s.reference or (ScreenGrid(s).cxf, ScreenGrid(s).cyf), "point", 1) for s in screens]
    I = [np.zeros((sm.nx, sm.ny)) for sm in samplers]
    W = [np.zeros((sm.nx, sm.ny), complex) for sm in samplers]
    Iref = [0.0] * len(samplers)
    nb, nn = len(bores), len(nodes)
    centre_amp = np.zeros((len(lines), nb, nn), complex)
    info = []
    dump_dir = None
    if out_dir is not None and wave["dump_field"]:
        dump_dir = os.path.join(out_dir, "funnel-exit")
        os.makedirs(dump_dir, exist_ok=True)
    basis = dr = None
    for li, (k, wl, delta, beta) in enumerate(lines):
        basis = FunnelBasis(a=a, g=g, k=k, delta=delta, beta=beta, jmax=jmax, layer=layer, blocks=not split, wall=wall)
        dr = min(float(wave["fb_dr"]), basis.disc.aw / 1500.0)
        chans = [FunnelChannel(basis=basis, c0=c, integrator=integrator) for c in centres]
        exits = {}

        def exit_coeffs(bi):
            if bi not in exits:
                t1 = time.time()
                cin = chans[bi].entrance_modes(nodes, D)
                p_in = (np.abs(cin) ** 2 * basis.disc.mnorm[:, :, None]).sum((0, 1))
                cL, n_steps, deg = chans[bi].propagate(cin, L, dz, tol)
                st = chans[bi].stepper
                p_out = (np.abs(cL) ** 2 * basis.disc.mnorm[:, :, None]).sum((0, 1))
                exits[bi] = cL
                if dump_dir is not None:
                    np.save(os.path.join(dump_dir, f"line{li}-bore{bi}.npy"), cL)
                info.append({"line": li, "bore": bi, "steps": n_steps, "chebyshev_degree": deg, "seconds": time.time() - t1,
                             "phase_gradient_rad": st.phase_gradient_rad if split else None,
                             "projection_loss": st.projection_loss.tolist() if split else None,
                             "transmission": (p_out / p_in).tolist(), "layer_power": chans[bi].layer_power.tolist()})
                how = (f"phase gradient {st.phase_gradient_rad:.2f} rad per step, projection loss {st.projection_loss.max():.1e}"
                       if split else f"Chebyshev degree {deg}")
                log(f"  16 [funnel]: line {li} bore {bi}: {n_steps} steps, {how}, {time.time() - t1:.0f} s")
            return exits[bi]

        # reference fields of every node at every screen reference
        refs = []
        for sm in samplers:
            rx, ry = sm.ref_xy
            hit = [bi for bi, c in enumerate(centres) if math.hypot(rx - G_L * c[0], ry - G_L * c[1]) < a * G_L]
            if not hit:
                raise ValueError(f"wave_estimator: reference ({rx * 1e6:g}, {ry * 1e6:g}) um outside every exit core")
            e, _ = chans[hit[0]].points(exit_coeffs(hit[0]), L, np.array([rx]), np.array([ry]), dr)
            refs.append(e[0] if e.shape[0] else np.zeros(nn, complex))
        wn = wl * weights
        for bi in range(nb):
            cL = exit_coeffs(bi)
            centre_amp[li, bi] = chans[bi].centre_values(cL, L)
            cx, cy = G_L * centres[bi]
            parts = []
            for si, sm in enumerate(samplers):                  # the pixels of every screen inside this exit core, one call
                ix = np.nonzero(np.abs(sm.px - cx) < a * G_L)[0]
                iy = np.nonzero(np.abs(sm.py - cy) < a * G_L)[0]
                if ix.size and iy.size:
                    X, Y = np.meshgrid(sm.px[ix], sm.py[iy], indexing="ij")
                    parts.append((si, ix, iy, X.ravel(), Y.ravel()))
            if parts:
                e, inside = chans[bi].points(cL, L, np.concatenate([p[3] for p in parts]), np.concatenate([p[4] for p in parts]), dr)
                full = np.zeros((inside.size, nn), complex)
                full[inside] = e
                o = 0
                for si, ix, iy, X, Y in parts:
                    blk, ins = full[o:o + X.size], inside[o:o + X.size]
                    o += X.size
                    shape = (ix.size, iy.size)
                    Ib = (np.abs(blk) ** 2 * wn[None, :]).sum(axis=1).reshape(shape)
                    Wb = (blk * np.conj(refs[si])[None, :] * wn[None, :]).sum(axis=1).reshape(shape)
                    I[si][np.ix_(ix, iy)] += np.where(ins.reshape(shape), Ib, 0.0)
                    W[si][np.ix_(ix, iy)] += np.where(ins.reshape(shape), Wb, 0.0)
            exits.pop(bi, None)                                 # free the block after its last use
        for si in range(len(samplers)):
            Iref[si] += float(np.sum(wl * weights * np.abs(refs[si]) ** 2))
    seconds = time.time() - t0
    # centre matrix: W_bc = sum_lines,nodes w E_b conj E_c
    wn = np.array([wl for _, wl, _, _ in lines])[:, None, None] * weights[None, None, :]
    Wc = np.einsum("lbn,lcn->bc", wn * centre_amp, np.conj(centre_amp))
    Ic = np.real(np.diag(Wc))
    mu_c = Wc / np.sqrt(np.outer(Ic, Ic))
    files, out = [], []
    meta_base = {"provider": "fb", "source_rule": rule, "n_nodes": int(nn),
                 "model_class": {"exact-roots": "exact-jk", "effective-radius": "effective-radius"}[wall] + ",funnel-xi",
                 "funnel": {"g": list(g), "G_exit": G_L, "radius_m": a, "jmax": jmax, "layer": layer,
                            "wall": wall, "integrator": integrator, "n_modes": int(basis.disc.nmodes),
                            "A0_m": basis.A0, "dz_m": dz, "chebyshev_tol": tol, "dr_m": dr,
                            "exact_fallback_modes": sum(basis.fallback.values())},
                 "propagation": info, "distance_m": 0.0}
    if out_dir is not None:
        json.dump({"centres_in_m": [c.tolist() for c in centres], "centres_out_m": [(G_L * c).tolist() for c in centres],
                   "I": Ic.tolist(), "mu_re": mu_c.real.tolist(), "mu_im": mu_c.imag.tolist(),
                   "amplitudes_re": centre_amp.real.tolist(), "amplitudes_im": centre_amp.imag.tolist(),
                   "weights": weights.tolist(), "nodes_m": np.asarray(nodes).tolist(),
                   "basis": {"a_m": a, "g": list(g), "jmax": jmax, "layer": layer, "A0_m": basis.A0,
                             "lines": [[k, wl, delta, beta] for k, wl, delta, beta in lines], "D_m": D, "L_m": L,
                             "dump": None if dump_dir is None else "funnel-exit/line{l}-bore{b}.npy, (2M+1, nmax, nodes)"}},
                  open(os.path.join(out_dir, "funnel-centres.json"), "w", encoding="utf-8"))
        files.append("funnel-centres.json")
        if dump_dir is not None:
            files += [f"funnel-exit/{f}" for f in sorted(os.listdir(dump_dir))]
    labels = ["capillary"] + [f"capillary-s{i}" for i in range(1, len(screens))]
    for si, (label, sm) in enumerate(zip(labels, samplers)):
        acc = SimpleNamespace(I=I[si], W=W[si], I_ref=Iref[si], I_pixel=None, I_bore=None, G12=None, ref_index=sm.ref_index)
        meta = dict(meta_base, scene=label)
        res = _finish(acc, sm, meta, wave, seconds)
        res["report"] = _report_lines(label, res, wave, [f"- funnel: G(L) = {G_L:.6f}, jmax {jmax}, {basis.disc.nmodes} modes, "
                                                          f"wall {wall}, integrator {integrator}, "
                                                          f"sub-step {dz * 1e3:g} mm, centres matrix in funnel-centres.json"])
        res["fb_files"] = files
        out.append((label, res))
    return out

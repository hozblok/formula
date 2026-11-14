"""Stage 16 provider fb for conformal funnel bores (kind funnel, f = g): every bore of the scene scales about the device
axis with G(z) = 1 + g0 z + g1 z^2 (z from the entrance). In xi = r / G(z) the bores are fixed discs and
  E(r, z) = G^-1 exp(i k G' |r|^2 / 2G) u(xi, z),   i du/dz = -Lap_xi u / (2 k G^2) + [V(G xi) + (k/2) G G'' |xi|^2] u.
In a bore of centre c0 (q = xi - c0) the kinetic part with the wall has the eigenvalues of the physical bore of radius
a G(z) (exact roots of the scalar characteristic equation). Its effective Dirichlet radius a G + Re ell is, in xi,
A(z) = a + Re ell / G, so the Fourier-Bessel basis of the disc A0 = a + Re ell lives in rho = q / s(z), s = A / A0:
  u(q) = s^-1 exp(i M s' |q|^2 / 2s) w(q / s),  M = k G^2,
  i dw/dz = -Lap_rho w / (2 M s^2) + [V + (k/2) G G'' (s^2 |rho|^2 + 2 s c0.rho + |c0|^2) + (M' s' + M s'') s |rho|^2 / 2] w.
The lens and the axis term enter as Galerkin blocks (r^2 within m, r e^{+-i phi} between m and m +- 1). A step is the
commutator-free exponential integrator of order 4 (two exponentials of combinations of H at the Gauss points), each
exponential a Chebyshev series. The basis keeps every zero j_mn < jmax; its outer fraction `layer` absorbs (amplitude
rate gamma_max ((j - j_s) / (jmax - j_s))^2). The Kirchhoff edge tail of the entrance is fixed in j (the conformal map
keeps j while the physical angle falls as 1 / G), so jmax has to cover the guided modes of the entrance, k theta_c a.
Kirchhoff entrance: the paraxial point source inside the geometric aperture a. Exit plane: the field inside the
geometric exit core a G(L) at the screen pixels (observable point).
The bores are propagated one after another with all source nodes as one block; W and I accumulate bore by bore (the exit
cores do not overlap), and the fields at the bore centres give the matrix of mu between the centres. With dump_field
the exit coefficients of every bore (all nodes) are written to funnel-exit/ for post-processing without propagation."""

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg.blas import dgemm as _gemm, zaxpy as _axpy
from scipy.special import jv, jvp

from .wave_fb import CHEB_TOL, Disc, newton_kappa, wall_offset


_C1, _C2 = 0.5 - math.sqrt(3.0) / 6.0, 0.5 + math.sqrt(3.0) / 6.0      # Gauss points of a step
_WP, _WM = 0.25 + math.sqrt(3.0) / 6.0, 0.25 - math.sqrt(3.0) / 6.0    # CF4 weights


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

    def __init__(self, a, g, k, delta, beta, jmax, nr=None, layer=0.0, gamma_max=5.0e2, blocks=True):
        self.a, self.g = float(a), (float(g[0]), float(g[1]))
        self.layer, self.gamma_max = float(layer), float(gamma_max)
        self.k, self.delta, self.beta = float(k), float(delta), float(beta)
        self.ell = wall_offset(k, delta, beta)
        self.A0 = self.a + self.ell.real
        self.disc = d = Disc(aw=self.A0, jmax=jmax, nr=8)                # zeros, norms and index maps; blocks on their own quadrature
        # blocks=False: only the evaluation of stored coefficients (points, centre values)
        self.R, self.S, self.T = _galerkin(d, nr or int(1.2 * jmax) + 40) if blocks else (None, None, None)
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

    def G(self, z):
        return 1.0 + self.g[0] * z + self.g[1] * z * z

    def dG(self, z):
        return self.g[0] + 2.0 * self.g[1] * z

    def s(self, z):
        """Dilation s = A / A0 of the basis about the bore centre and its first two derivatives."""
        G, dG, d2G, lr = self.G(z), self.dG(z), 2.0 * self.g[1], self.ell.real
        return ((self.a + lr / G) / self.A0, -lr * dG / G ** 2 / self.A0,
                -lr * (d2G / G ** 2 - 2.0 * dG ** 2 / G ** 3) / self.A0)

    def h(self, z):
        """kappa^2 / 2k of the bore of radius a G(z) per row: exact roots (damped Newton, warm-started from the roots of
        the previous G scaled by (a G' + ell) / (a G + ell)), DIR-ell where Newton fails."""
        G = self.G(z)
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

    def pair(self, c):
        """Signed (2M+1, nmax, B) -> paired (N, 2, B)."""
        M = self.disc.M
        return np.ascontiguousarray(np.stack([c[M + self.mr, self.nr_], c[M - self.mr, self.nr_]], axis=1))

    def unpair(self, z):
        d = self.disc
        c = np.zeros((2 * d.M + 1, d.nmax, z.shape[2]), complex)
        c[d.M - self.mr, self.nr_] = z[:, 1]
        c[d.M + self.mr, self.nr_] = z[:, 0]
        return c

    def radial_tables(self, m, dr, cache_bytes=3.0 * 2 ** 30):
        """J_|m|(j r / A0) and its r-derivative on r = 0, dr, ..., A0 (+dr): (nf, n_m) each (float32, cached while the
        tables of all |m| fit in cache_bytes)."""
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

    def prefetch(self, dr, cache_bytes=3.0 * 2 ** 30):
        """All radial tables of step dr in threads (the Bessel ufuncs release the GIL), when they fit the cache."""
        d = self.disc
        if 8 * (int((d.aw + 2.0 * dr) / dr) + 2) * d.nmodes > cache_bytes:
            return
        todo = [mm for mm in range(d.M + 1) if (mm, dr) not in self._tab]
        with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
            list(ex.map(lambda mm: self.radial_tables(mm, dr, cache_bytes), todo))


class FunnelChannel:
    """One funnel bore of entrance centre c0 on a shared FunnelBasis. The propagation frame is turned by the polar angle
    of c0, so the axis term is real: c_m -> c_m exp(i m alpha) on entry and back on exit."""

    def __init__(self, basis, c0):
        self.b, self.c0 = basis, np.asarray(c0, float)
        self.rho0 = float(np.hypot(*self.c0))
        self.alpha = float(np.arctan2(self.c0[1], self.c0[0]))

    def coeffs(self, z):
        b = self.b
        G, dG = b.G(z), b.dG(z)
        s, ds, d2s = b.s(z)
        lam = 0.5 * b.k * G * 2.0 * b.g[1]
        M, dM = b.k * G * G, 2.0 * b.k * G * dG
        return lam * s * s + 0.5 * (dM * ds + M * d2s) * s, lam * s, lam * self.rho0 ** 2

    def operator(self, za, zb, wa, wb, scale=1.0):
        """A = scale (wa H(za) + wb H(zb)) on the paired layout: (apply(z, alpha, shift) = alpha (A - shift) z, spectral
        bounds of the Hermitian part of A, max damping)."""
        b, d = self.b, self.b.disc
        qa, qb = self.coeffs(za), self.coeffs(zb)
        q2, q1, q0 = (scale * (wa * x + wb * y) for x, y in zip(qa, qb))
        f = q1 * self.rho0                                     # 2 c0.rho = rho0 r (e^{i phi} + e^{-i phi}) in the turned frame
        hd = scale * (wa * b.h(za) + wb * b.h(zb) - 1j * (wa + wb) * b.damp) + q0
        R, S, T, off, cnt, M = b.R, b.S, b.T, b.off, b.cnt, d.M

        def apply(z, alpha=1.0, shift=0.0):
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
    def _chebyshev(apply, bounds, c, L, tol):
        """exp(-i A L) c for apply(z, alpha, shift) = alpha (A - shift) z with the spectrum of A inside bounds
        (Tal-Ezer & Kosloff): T_{i+1} = 2 X T_i - T_{i-1}, X = (A - ec) / dE."""
        emin, emax = bounds
        dE, ec = 0.5 * (emax - emin) * 1.02, 0.5 * (emax + emin)
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

    def step(self, z, z0, dz, tol):
        """One CF4 step: exp(-i dz (w+ H1 + w- H2)) then exp(-i dz (w- H1 + w+ H2)), H1, H2 at the Gauss points,
        w+- = 1/4 +- sqrt(3)/6; each as exp(-i (dz/2) (2 A)) in sub-steps that keep the damping small on the scaled
        spectrum (2 g_max (dz/2) / n_sub <= 1.6)."""
        z1, z2 = z0 + _C1 * dz, z0 + _C2 * dz
        deg = 0
        for wa, wb in ((_WP, _WM), (_WM, _WP)):
            apply, bounds, g_max = self.operator(z1, z2, wa, wb, scale=2.0)
            n_sub = max(1, int(math.ceil(0.5 * g_max * dz / 1.6)))
            for _ in range(n_sub):
                z, n = self._chebyshev(apply, bounds, z, 0.5 * dz / n_sub, tol)
                deg += n
        return z, deg

    def propagate(self, c, L, dz_max, tol=CHEB_TOL):
        """Exit coefficients (signed layout) of the entrance coefficients c, steps of at most dz_max; with the power
        fraction left in the absorbing layer at the exit, per node."""
        n_steps = max(1, int(math.ceil(L / dz_max - 1e-9)))
        dz, deg = L / n_steps, 0
        rot = np.exp(1j * np.array(self.b.disc.ms) * self.alpha)[:, None, None]
        z = self.b.pair(c * rot)
        for i in range(n_steps):
            z, n = self.step(z, i * dz, dz, tol)
            deg += n
        cL = self.b.unpair(z) / rot
        d = self.b.disc
        p = np.abs(cL) ** 2 * d.mnorm[:, :, None]
        top = (d.zp[d.am] > (1.0 - self.b.layer) * d.jmax)[:, :, None]
        self.layer_power = np.where(top, p, 0.0).sum((0, 1)) / p.sum((0, 1))
        return cL, n_steps, deg

    def entrance_modes(self, nodes, D, nr=None):
        """w(rho, 0) of paraxial point sources at (node, -D) through the geometric aperture a: (2M+1, nmax, B).
        c_mn = int_0^a int u J_|m|(j r / A0) e^{-i m phi} r dr dphi / (pi A0^2 J_{|m|+1}(j)^2), Gauss-Legendre on [0, a]
        (the aperture edge is the end of the interval), FFT over phi."""
        from scipy import fft as sfft
        b, d = self.b, self.b.disc
        n_r = nr or int(1.2 * d.jmax) + 40
        tq, wq = leggauss(n_r)
        r = 0.5 * b.a * (tq + 1.0)
        wr = 0.5 * b.a * wq
        nphi = d.nphi
        phi = 2.0 * math.pi * np.arange(nphi) / nphi
        X = self.c0[0] + r[:, None] * np.cos(phi)[None, :]
        Y = self.c0[1] + r[:, None] * np.sin(phi)[None, :]
        G0, dG0 = b.G(0.0), b.dG(0.0)
        s, ds, _ = b.s(0.0)
        lam = 2.0 * math.pi / b.k
        gauge = np.exp(-1j * b.k * dG0 * (X * X + Y * Y) / (2.0 * G0) - 1j * b.k * G0 * G0 * ds * r[:, None] ** 2 / (2.0 * s))
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

    def _frame(self, z):
        b = self.b
        G, dG = b.G(z), b.dG(z)
        s, ds, _ = b.s(z)
        return G, dG, s, ds, b.k * G * G

    def centre_values(self, c, z):
        """E at the bore centre G(z) c0 for every node: only m = 0 is non-zero at rho = 0. (B,)"""
        G, dG, s, ds, M = self._frame(z)
        w0 = c[self.b.disc.M].sum(axis=0)                       # J_0(0) = 1
        r2 = float((G * self.c0) @ (G * self.c0))
        return w0 / s * np.exp(1j * self.b.k * dG * r2 / (2.0 * G)) / G

    def points(self, c, z, X, Y, dr):
        """E at lab points (X, Y) inside the geometric core a G(z) for every node: (n_points, B) and the inside mask.
        Radial functions by cubic Hermite interpolation on r = 0, dr, ... (relative error ~ (2 pi dr jmax / A0)^4 / 384)."""
        b, d = self.b, self.b.disc
        G, dG, s, ds, M = self._frame(z)
        qx, qy = X / G - self.c0[0], Y / G - self.c0[1]
        q = np.hypot(qx, qy)
        inside = q * G < b.a * G
        q, phi = q[inside], np.arctan2(qy[inside], qx[inside])
        rho = q / s
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
        u = E * (np.exp(1j * M * ds * q ** 2 / (2.0 * s)) / s)[:, None]
        r2 = X[inside] ** 2 + Y[inside] ** 2
        return u * (np.exp(1j * b.k * dG * r2 / (2.0 * G)) / G)[:, None], inside


# ----------------------------------------------------------------- scene driver

def funnel_scene(sim, wave, lines, log, out_dir=None):
    """Stage 16 on a scene of conformal funnel bores: exit-plane screens (point observable), the matrix of mu between
    the bore centres. Returns [(label, res), ...] like fb_capillary_scene."""
    from .wave import ScreenSampler, _finish, _report_lines, _rule
    from ..screen import ScreenGrid
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
    for li, (k, wl, delta, beta) in enumerate(lines):
        basis = FunnelBasis(a, g, k, delta, beta, jmax, layer=layer)
        dr = min(float(wave["fb_dr"]), basis.disc.aw / 1500.0)
        chans = [FunnelChannel(basis, c) for c in centres]
        exits = {}

        def exit_coeffs(bi):
            if bi not in exits:
                t1 = time.time()
                cin = chans[bi].entrance_modes(nodes, D)
                p_in = (np.abs(cin) ** 2 * basis.disc.mnorm[:, :, None]).sum((0, 1))
                cL, n_steps, deg = chans[bi].propagate(cin, L, dz, tol)
                p_out = (np.abs(cL) ** 2 * basis.disc.mnorm[:, :, None]).sum((0, 1))
                exits[bi] = cL
                if dump_dir is not None:
                    np.save(os.path.join(dump_dir, f"line{li}-bore{bi}.npy"), cL)
                info.append({"line": li, "bore": bi, "steps": n_steps, "chebyshev_degree": deg, "seconds": time.time() - t1,
                             "transmission": (p_out / p_in).tolist(), "layer_power": chans[bi].layer_power.tolist()})
                log(f"  16 [funnel]: line {li} bore {bi}: {n_steps} steps, Chebyshev degree {deg}, {time.time() - t1:.0f} s")
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
    meta_base = {"provider": "fb", "model_class": "exact-jk,funnel-xi", "source_rule": rule, "n_nodes": int(nn),
                 "funnel": {"g": list(g), "G_exit": G_L, "radius_m": a, "jmax": jmax, "layer": layer,
                            "n_modes": int(basis.disc.nmodes),
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
                                                          f"sub-step {dz * 1e3:g} mm, centres matrix in funnel-centres.json"])
        res["fb_files"] = files
        out.append((label, res))
    return out

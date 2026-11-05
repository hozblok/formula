"""Stage 16 provider `fb`: Fourier-Bessel modes of circular bores (cylinder, torus arc).

Paraxial scalar wave in the co-moving frame of the bent axis c(z) = c0 + t z^2 / (2R)
(t = unit vector toward the bend centre):  i d/dz psi = -(1/2k) Lap psi + (k/R) xi psi,
xi along t; Dirichlet wall at r = a + Re(ell), ell = i / (k q0), q0 = sqrt(-2 delta + 2 i beta)
(DIR-ell); Im(ell) enters as the first-order per-mode damping kappa^2 Im(ell) / (a k).
Longitudinal step: a Chebyshev series of exp(-iHL) (no z error; z-independent H).  Exit field in the lab frame on a Cartesian lattice with the phase
exp(i k c'(L).x' + i g), g = k L^3 / (6 R^2); screens by angular-spectrum drift on the
lattice and exact cell integrals of the band-limited field (Fourier box), reference cell as
Stage 14.  Conventions as the uisk provider: W = <E(P) E*(P_ref)>, e^{-i omega t}, carrier
e^{+ikz} dropped; entrance field exp(ik|x - s|^2 / 2 z_in) / (i lambda z_in).
Options: wall `exact` (per-mode roots of the J_m / K_m characteristic equation of the hollow
cylinder instead of DIR-ell), a wall figure dr(phi, z) to first order (Hadamard) with the q = 1
part as an exact co-moving axis path, the far field from the exit modes (Lommel integrals) and
a dump of the entrance / exit mode coefficients.
"""

import math
import os
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import fft as sfft
from scipy.special import jn_zeros, jv, kve

from ...physical_constants import HC_KEV_ANGSTROM
from ..screen import ScreenGrid
from ..shared.units import m_to_um

MODEL_CLASS_FB = "dir-ell"


# ----------------------------------------------------------------- wall

def wall_offset(k, delta, beta):
    """Complex DIR-ell offset ell = i/(k q0), q0 = sqrt(-2 delta + 2 i beta), Im q0 >= 0."""
    q0 = np.sqrt(complex(-2.0 * delta, 2.0 * beta))
    if q0.imag < 0.0:
        q0 = -q0
    return 1j / (k * q0)


def mode_damping(disc, k, delta, beta):
    """Amplitude damping per mode, kappa^2 Im ell / (k a_w), times the exact Fresnel loss at the
    mode's wall grazing angle theta_w relative to its theta -> 0 form: -> (1 - theta_w^2/theta_c^2)^(-1/2)."""
    ell = wall_offset(k, delta, beta)
    thw = np.sqrt(np.maximum(disc.kap2 - (disc.am[:, None] / disc.aw) ** 2, 0.0)) / k
    q = np.sqrt(thw * thw - 2.0 * delta + 2j * beta)
    q = np.where(q.imag < 0.0, -q, q)
    loss = 4.0 * thw * q.real / np.abs(thw + q) ** 2
    loss0 = 4.0 * thw * beta / (2.0 * delta) ** 1.5
    factor = np.divide(loss, loss0, out=np.ones_like(loss), where=loss0 > 0.0)
    return disc.kap2 * ell.imag / (disc.aw * k) * factor


def exact_kappa(disc, a, k, delta, beta, iters=40, rtol=1e-13):
    """Complex wavenumbers kappa_mn of the hollow circular bore of radius a in the wall n = 1 - delta + i beta:
    p = s with p = kappa J_m'(kappa a) / J_m(kappa a), s = Q K_m'(Q a) / K_m(Q a), Q^2 = k^2 (1 - n^2) - kappa^2,
    Re Q > 0 (scalar continuity of psi and psi'); damped Newton (|step| <= 0.2 / a) from the DIR-ell guess
    j_mn / (a + ell) with the analytic derivative p' - s' (Bessel equations). Returns (kappa, converged) on the
    padded |m| x n grid; invalid entries are zero; a root is accepted only if it converged, decays
    (Im kappa <= 0) and stays within 1 / a of the guess."""
    k2 = k * k * (1.0 - (1.0 - delta + 1j * beta) ** 2)
    ell = wall_offset(k, delta, beta)
    rows, cols = np.nonzero(disc.validp)
    mm = rows.astype(float)
    kap = disc.zp[rows, cols] / (a + ell)

    def g_and_slope(kap):
        x = kap * a
        Q = np.sqrt(k2 - kap * kap)
        Q = np.where(Q.real < 0.0, -Q, Q)
        z = Q * a
        p = kap * (jv(mm - 1.0, x) / jv(mm, x) - mm / x)
        s = Q * (-kve(mm - 1.0, z) / kve(mm, z) - mm / z)
        dp = -(x - mm * mm / x) - a * p * p / kap
        ds = -kap / Q * (z + mm * mm / z) + a * kap * s * s / (Q * Q)
        return p - s, dp - ds

    guess = kap.copy()
    ok = np.zeros(kap.shape, bool)
    cap = 0.2 / a
    for _ in range(iters):
        g, s = g_and_slope(kap)
        step = g / s
        big = np.abs(step) > cap
        step = np.where(big, step * cap / np.where(big, np.abs(step), 1.0), step)
        kap = kap - step
        ok = (np.abs(step) <= rtol * np.abs(kap)) & ~big
        if ok.all():
            break
    out = np.zeros(disc.validp.shape, complex)
    conv = np.zeros(disc.validp.shape, bool)
    out[rows, cols] = kap
    conv[rows, cols] = ok & np.isfinite(kap) & (kap.imag <= 0.0) & (np.abs(kap - guess) * a < 1.0)
    return out, conv


# ----------------------------------------------------------------- disc basis

class Disc:
    """Dirichlet disc basis J_|m|(j_mn r / aw) e^{i m phi}, all zeros j_mn < jmax, with wall_cut J only
    the guided ones (j_mn^2 - m^2 < J^2); Gauss-Legendre radial quadrature, FFT in phi; padded arrays over |m|."""

    def __init__(self, aw, jmax, nr=None, nphi=None, wall_cut=None):
        self.aw, self.jmax, self.wall_cut = float(aw), int(jmax), wall_cut
        zeros, m = [], 0
        while True:
            z = jn_zeros(m, max(8, int(jmax / math.pi) + 4))
            z = z[z < jmax]
            if wall_cut is not None:
                z = z[z * z - m * m < wall_cut * wall_cut]
            if z.size == 0:
                break
            zeros.append(z)
            m += 1
        self.M = len(zeros) - 1
        self.zeros = zeros
        self.nmodes = sum(2 * z.size for z in zeros) - zeros[0].size
        self.nr = nr or int(1.2 * jmax) + 40
        self.nphi = nphi or (2 * self.M + 3 + 7) // 8 * 8
        t, w = leggauss(self.nr)
        self.r = 0.5 * self.aw * (t + 1.0)
        self.wr = 0.5 * self.aw * w
        self.phi = 2.0 * math.pi * np.arange(self.nphi) / self.nphi
        self.nmax = max(z.size for z in zeros)
        self.ms = list(range(-self.M, self.M + 1))
        self.am = np.abs(np.array(self.ms))
        self.col = np.array(self.ms) % self.nphi
        nm = self.M + 1
        self.Bp = np.zeros((nm, self.nr, self.nmax))
        self.Pp = np.zeros((nm, self.nmax, self.nr))
        self.kap2p = np.zeros((nm, self.nmax))
        self.normp = np.zeros((nm, self.nmax))
        self.validp = np.zeros((nm, self.nmax), bool)
        self.zp = np.zeros((nm, self.nmax))
        self.jp1p = np.zeros((nm, self.nmax))
        for mm, z in enumerate(zeros):
            kap, n = z / self.aw, z.size
            Bm = jv(mm, np.outer(self.r, kap))
            jp1 = jv(mm + 1, z)
            norm = 0.5 * self.aw * self.aw * jp1 ** 2
            self.Bp[mm, :, :n] = Bm
            self.Pp[mm, :n, :] = (Bm * (self.wr * self.r)[:, None]).T / norm[:, None]
            self.kap2p[mm, :n] = kap * kap
            self.normp[mm, :n] = 2.0 * math.pi * norm
            self.validp[mm, :n] = True
            self.zp[mm, :n] = z
            self.jp1p[mm, :n] = jp1
        self.B = self.Bp[self.am]
        self.P = self.Pp[self.am]
        self.kap2 = self.kap2p[self.am]
        self.mnorm = self.normp[self.am]
        self.valid = self.validp[self.am]
        self.kap2max = float(self.kap2.max())
        # wall traces for boundary (Hadamard) perturbations: d_n psi_mn = -(j/aw) J_{|m|+1}(j) e^{i m phi}
        self.wall_u = (self.zp * self.jp1p)[self.am]
        self.wall_v = np.divide(self.zp, self.jp1p, out=np.zeros_like(self.zp), where=self.validp)[self.am]
        self.wall_s2 = float(np.max(np.sum(self.zp ** 2, axis=1)))
        self._fine = None
        self._blocks = None

    def shift_blocks(self):
        """Galerkin blocks of r e^{+i phi} (m -> m+1) and r e^{-i phi} (m -> m-1) over signed m:
        A_up[m] = P_{|m+1|} diag(r) B_{|m|}, A_dn[m] = P_{|m-1|} diag(r) B_{|m|}; zero rows beyond M."""
        if self._blocks is None:
            nm, M = 2 * self.M + 1, self.M
            up = np.zeros((nm, self.nmax, self.nmax)); dn = np.zeros((nm, self.nmax, self.nmax))
            rB = self.Bp * self.r[None, :, None]
            for i, m in enumerate(self.ms):
                if m + 1 <= M:
                    up[i] = self.Pp[abs(m + 1)] @ rB[abs(m)]
                if m - 1 >= -M:
                    dn[i] = self.Pp[abs(m - 1)] @ rB[abs(m)]
            self._blocks = (up, dn)
        return self._blocks

    def empty(self):
        return np.zeros((2 * self.M + 1, self.nmax), complex)

    def to_grid(self, c):
        F = np.zeros((self.nr, self.nphi), complex)
        F[:, self.col] = np.einsum("mrn,mn->rm", self.B, c)
        return self.nphi * sfft.ifft(F, axis=1)

    def to_modes(self, G):
        F = sfft.fft(G, axis=1) / self.nphi
        return np.einsum("mnr,rm->mn", self.P, F[:, self.col])

    def norm(self, c):
        return float(np.sum(np.abs(c) ** 2 * self.mnorm))

    def fine_table(self, dr, dtype=np.float32):
        if self._fine is None:
            rf = np.arange(0.0, self.aw + dr, dr)
            J = np.zeros((self.M + 1, rf.size, self.nmax), dtype)
            for mm, z in enumerate(self.zeros):
                J[mm, :, :z.size] = jv(mm, np.outer(rf, z / self.aw))
            self._fine = (rf, J)
        return self._fine

    def radial_table(self, c, dr):
        """R_m(r) on the fine radial grid for each signed m: (2M+1, nf)."""
        rf, J = self.fine_table(dr)
        M = self.M
        pos = np.einsum("mrn,mn->mr", J, c[M:])
        neg = np.einsum("mrn,mn->mr", J[1:], c[M - 1::-1])
        return rf, np.concatenate([neg[::-1], pos], axis=0)


# ----------------------------------------------------------------- wall figure

class WallFigure:
    """Radial wall displacement dr(phi, z) [m]: harmonic terms amp mod(z) cos(q (phi - angle - rate z)),
    mod = cos(2 pi z / period + phase) or 1, plus an optional random field sum_q a_q(z) cos q phi + b_q(z) sin q phi
    (gaussian z-correlation of length lc, equal power per listed q, total rms sigma). coeffs(z) -> dr_hat[p],
    p = -Q..Q, with dr = sum_p dr_hat[p] e^{i p phi}."""

    def __init__(self, terms=(), noise=None, length=0.0, axis=True):
        self.axis = bool(axis)                # q = 1 (a lateral shift of the bore) goes to AxisPath, not to Hadamard
        self.length = float(length)
        self.terms = [dict(t) for t in terms]
        qs = [int(t["q"]) for t in self.terms]
        self.noise = None
        if noise:
            rng = np.random.default_rng(int(noise["seed"]))
            lc, nq = float(noise["lc"]), len(noise["q"])
            dz = lc / 8.0
            n = int(math.ceil(length / dz)) + 2
            pad = int(math.ceil(4.0 * lc / dz))
            kern = np.exp(-0.5 * (np.arange(-pad, pad + 1) * dz / lc) ** 2)
            series = []
            for q in noise["q"]:
                ab = []
                for _ in range(1 if int(q) == 0 else 2):
                    s = np.convolve(rng.standard_normal(n + 2 * pad), kern, mode="valid")
                    ab.append(s * (float(noise["sigma"]) / math.sqrt(nq) / s.std()))
                series.append((int(q), ab))
            self.noise = (np.arange(n) * dz, series)
            self.noise_lc = lc
            qs += [int(q) for q in noise["q"]]
        self.Q = max(qs) if qs else 0
        self.z_dependent = bool(self.noise) or any(t.get("rate") or t.get("period") for t in self.terms)
        self.has_wall = any(q != 1 or not self.axis for q in qs)
        self.has_axis = self.axis and 1 in qs

    def coeffs(self, z):
        d = np.zeros(2 * self.Q + 1, complex)
        for t in self.terms:
            q, amp = int(t["q"]), float(t["amp"])
            if q == 1 and self.axis:
                continue
            if t.get("period"):
                amp *= math.cos(2.0 * math.pi * z / float(t["period"]) + float(t.get("phase", 0.0)))
            psi = q * (float(t.get("angle", 0.0)) + float(t.get("rate", 0.0)) * z)
            if q == 0:
                d[self.Q] += amp
            else:
                d[self.Q + q] += 0.5 * amp * np.exp(-1j * psi)
                d[self.Q - q] += 0.5 * amp * np.exp(1j * psi)
        if self.noise:
            zs, series = self.noise
            for q, ab in series:
                if q == 1 and self.axis:
                    continue
                a = np.interp(z, zs, ab[0])
                if q == 0:
                    d[self.Q] += a
                else:
                    b = np.interp(z, zs, ab[1])
                    d[self.Q + q] += 0.5 * (a - 1j * b)
                    d[self.Q - q] += 0.5 * (a + 1j * b)
        return d

    def min_scale(self):
        """Shortest z scale of the figure (period, pattern turn 2 pi / q rate, noise correlation length) or None."""
        s = []
        for t in self.terms:
            if t.get("period"):
                s.append(float(t["period"]))
            if t.get("rate"):
                s.append(2.0 * math.pi / (max(int(t["q"]), 1) * abs(float(t["rate"]))))
        if self.noise:
            s.append(self.noise_lc)
        return min(s) if s else None


class AxisPath:
    """Lateral displacement c(z) of the bore axis (the q = 1 figure): harmonic terms amp mod(z) (cos psi, sin psi),
    psi = angle + rate z, plus the q = 1 noise pair (a_1, b_1); c, c', c'' at any z (analytic for the terms,
    finite differences on the smooth noise grid)."""

    def __init__(self, figure):
        self.terms = [t for t in figure.terms if int(t["q"]) == 1]
        self.series = None
        if figure.noise:
            zs, series = figure.noise
            for q, ab in series:
                if q == 1:
                    c = np.vstack(ab)
                    d1 = np.gradient(c, zs, axis=1)
                    self.series = (zs, c, d1, np.gradient(d1, zs, axis=1))

    def derivs(self, z):
        """(c, c', c'') at z, each a 2-vector in the bore frame."""
        out = np.zeros((3, 2))
        for t in self.terms:
            amp, rate = float(t["amp"]), float(t.get("rate", 0.0))
            psi = float(t.get("angle", 0.0)) + rate * z
            u, up = np.array([math.cos(psi), math.sin(psi)]), np.array([-math.sin(psi), math.cos(psi)])
            if t.get("period"):
                K, ph = 2.0 * math.pi / float(t["period"]), float(t.get("phase", 0.0))
                f, f1, f2 = amp * math.cos(K * z + ph), -amp * K * math.sin(K * z + ph), -amp * K * K * math.cos(K * z + ph)
            else:
                f, f1, f2 = amp, 0.0, 0.0
            out[0] += f * u
            out[1] += f1 * u + f * rate * up
            out[2] += (f2 - f * rate * rate) * u + 2.0 * f1 * rate * up
        if self.series is not None:
            zs, c, d1, d2 = self.series
            for i, arr in enumerate((c, d1, d2)):
                out[i] += [np.interp(z, zs, arr[0]), np.interp(z, zs, arr[1])]
        return out


# ----------------------------------------------------------------- one bore

class FBChannel:
    """One circular bore: axis c(z) = c0 + t z^2/(2R) (t toward the bend centre; R None =
    straight); frame coordinates xi along t, eta along t_perp; potential k xi / R."""

    def __init__(self, c0, t, rbend, disc, k, gamma=None, h0=None, figure=None, dz=None):
        self.c0 = np.asarray(c0, float)
        self.t = None if t is None else np.asarray(t, float) / np.linalg.norm(t)
        self.rb = None if rbend is None else float(rbend)
        self.disc, self.k = disc, float(k)
        xi = disc.r[:, None] * np.cos(disc.phi)[None, :]
        cpp = 0.0 if self.rb is None else 1.0 / self.rb
        self.pot = self.k * cpp * xi                                # V on the polar grid
        self.gamma = np.zeros_like(disc.kap2) if gamma is None else gamma
        self.h0 = disc.kap2 / (2.0 * self.k) - 1j * self.gamma if h0 is None else np.asarray(h0, complex)
        self.vscale = 0.0 if self.rb is None else self.k / (2.0 * self.rb)   # V = (k/R) r cos phi = vscale r (e^{i phi} + e^{-i phi})
        self.figure = figure if figure is not None and figure.has_wall else None
        self.path = AxisPath(figure) if figure is not None and figure.has_axis else None
        self.dz = dz                                                 # z step of a z-dependent figure or axis
        self.fig_scale = -1.0 / (self.k * disc.aw ** 3)
        self._dr = None                                              # figure coefficients of the current step
        self._cpp = np.zeros(2)                                      # axis curvature c'' of the current step (frame)
        self._g = None
        if self.path is not None:                                    # lab phase g(z) = (k/2) int |c'|^2 on a fine grid
            L = figure.length
            n = max(2001, int(math.ceil(64.0 * L / (figure.min_scale() or L))) + 1)
            zg = np.linspace(0.0, L, n)
            s2 = np.array([float(np.sum(self.slope_frame(z) ** 2)) for z in zg])
            self._g = (zg, np.concatenate([[0.0], np.cumsum(0.5 * (s2[1:] + s2[:-1]) * np.diff(zg))]) * self.k / 2.0)

    def slope_frame(self, z):
        """Axis slope c'(z) in the frame (xi, eta): the bend t z / R along xi plus the axis path."""
        s = np.array([0.0 if self.rb is None else z / self.rb, 0.0])
        return s + (self.path.derivs(z)[1] if self.path is not None else 0.0)

    def to_lab(self, v):
        t, tp = self.frame_axes()
        return v[0] * t + v[1] * tp

    def frame_axes(self):
        t = self.t if self.t is not None else np.array([1.0, 0.0])
        return t, np.array([-t[1], t[0]])

    def axis(self, z):
        c = self.c0 if self.rb is None else self.c0 + self.t * z * z / (2.0 * self.rb)
        return c if self.path is None else c + self.to_lab(self.path.derivs(z)[0])

    def slope(self, z):
        if self.path is None:
            return np.zeros(2) if self.rb is None else self.t * z / self.rb
        return self.to_lab(self.slope_frame(z))

    def lab_phase_g(self, z):
        if self._g is not None:
            return float(np.interp(z, *self._g))
        return 0.0 if self.rb is None else self.k / 2.0 * (z ** 3 / (3.0 * self.rb ** 2))

    def _entrance_grid(self):
        """Lab coordinates of the polar quadrature points at z = 0 and the frame gauge exp(-i k c'(0).x')."""
        t, tp = self.frame_axes()
        d = self.disc
        xi = d.r[:, None] * np.cos(d.phi)[None, :]
        eta = d.r[:, None] * np.sin(d.phi)[None, :]
        c = self.axis(0.0)
        X = c[0] + xi * t[0] + eta * tp[0]
        Y = c[1] + xi * t[1] + eta * tp[1]
        if self.path is None:
            return X, Y, 1.0
        s = self.slope_frame(0.0)
        return X, Y, np.exp(-1j * self.k * (s[0] * xi + s[1] * eta))

    def entrance_field(self, src_xy, z_in):
        """Paraxial point source at (src_xy, -z_in) on the entrance disc (polar grid),
        amplitude 1/(i lambda z_in) like the stage's free-space Fresnel step."""
        X, Y, gauge = self._entrance_grid()
        lam = 2.0 * math.pi / self.k
        return (gauge * np.exp(1j * self.k * ((X - src_xy[0]) ** 2 + (Y - src_xy[1]) ** 2) / (2.0 * z_in))
                / (1j * lam * z_in))

    def entrance_modes(self, src_xy, z_in):
        return self.disc.to_modes(self.entrance_field(src_xy, z_in))

    def spot_field(self, spot):
        """Coherent gaussian on the entrance disc: FWHM in intensity at the waist, unit power, plane-wave tilt
        (tx, ty); waist at z = focus (0: on the entrance, < 0: before it), q = -focus + i z_R per axis."""
        from .wave import spot_profile
        X, Y, gauge = self._entrance_grid()
        return gauge * spot_profile(spot, self.k, X, Y)

    def spot_modes(self, spot):
        return self.disc.to_modes(self.spot_field(spot))

    def apply_v(self, c):
        """Galerkin V c: the bend through the block-tridiagonal m -> m +- 1 matrices, the wall figure
        through its Hadamard form -(1/k aw^3) v_i sum_j dr_hat[m_i - m_j] u_j c_j (a convolution over m)."""
        out = np.zeros_like(c)
        if self.path is None and self.vscale != 0.0:     # bend alone: V = (k/R) xi
            up, dn = self.disc.shift_blocks()
            out[1:] += np.einsum("mij,mj->mi", up[:-1], c[:-1])
            out[:-1] += np.einsum("mij,mj->mi", dn[1:], c[1:])
            out *= self.vscale
        elif self.path is not None:          # k c''.x = (k/2) r [(c''x - i c''y) e^{i phi} + (c''x + i c''y) e^{-i phi}]
            cx = self._cpp[0] + (0.0 if self.rb is None else 1.0 / self.rb)
            cy = self._cpp[1]
            up, dn = self.disc.shift_blocks()
            out[1:] += 0.5 * self.k * (cx - 1j * cy) * np.einsum("mij,mj->mi", up[:-1], c[:-1])
            out[:-1] += 0.5 * self.k * (cx + 1j * cy) * np.einsum("mij,mj->mi", dn[1:], c[1:])
        if self._dr is not None:
            d = self.disc
            b = np.sum(d.wall_u * c, axis=1)
            q = (self._dr.size - 1) // 2
            out += self.fig_scale * d.wall_v * np.convolve(b, self._dr)[q:q + b.size, None]
        return out

    def apply_h(self, c):
        """H c = (kappa^2/2k - i gamma) c + V c (Galerkin blocks)."""
        return self.h0 * c + self.apply_v(c)

    def apply_h_grid(self, c):
        """Same operator with V applied pointwise on the polar grid and projected (check)."""
        d = self.disc
        return self.h0 * c + d.to_modes(d.to_grid(c) * self.pot)

    def chebyshev_degree(self, L, tol):
        emin, emax = self._spectral_bounds()
        dE = 0.5 * (emax - emin) * (1.0 + 0.02)
        t = dE * L
        n = int(t) + 8
        while abs(jv(n, t)) > tol:
            n += 8
        return n, t, 0.5 * (emax + emin), dE

    def _spectral_bounds(self):
        vmax = float(np.max(np.abs(self.pot))) if self.rb is not None else 0.0
        vmax += self.k * float(np.hypot(*self._cpp)) * self.disc.aw
        if self._dr is not None:            # Hadamard term: |V| <= |scale| max_m sum_n j^2 sum_p |dr_hat|
            vmax += abs(self.fig_scale) * self.disc.wall_s2 * float(np.sum(np.abs(self._dr)))
        hr = self.h0.real[self.disc.valid]
        return min(0.0, float(hr.min())) - vmax, float(hr.max()) + vmax

    def propagate(self, c, L, tol=1e-12, z0=0.0):
        """exp(-iHL) c: diagonal for a straight bore without figure; otherwise the Chebyshev series in
        sub-steps short enough that the damping stays small on the scaled spectrum (gamma_max dL <= 1.6)
        and, for a z-dependent figure, no longer than dz (figure taken at each sub-step midpoint)."""
        if self.rb is None and self.figure is None and self.path is None:
            return c * np.exp(-1j * self.h0 * L), 0
        steps = max(1, int(math.ceil(float(np.max(-self.h0.imag)) * L / 1.6)))
        if (self.figure is not None and self.figure.z_dependent) or self.path is not None:
            steps = max(steps, int(math.ceil(L / self.dz - 1e-9)))
        dl, total = L / steps, 0
        try:
            for s in range(steps):
                zm = z0 + (s + 0.5) * dl
                if self.figure is not None:
                    self._dr = self.figure.coeffs(zm)
                if self.path is not None:
                    self._cpp = self.path.derivs(zm)[2]
                c, deg = self.propagate_chebyshev(c, dl, tol)
                total += deg
        finally:
            self._dr = None
            self._cpp = np.zeros(2)
        return c, total

    def propagate_chebyshev(self, c, L, tol=1e-12):
        """exp(-iHL) c as a Chebyshev series (Tal-Ezer & Kosloff): exp(-i t x) =
        sum_n (2 - delta_n0) (-i)^n J_n(t) T_n(x) on the spectrum scaled to [-1, 1]."""
        n_max, t, ec, dE = self.chebyshev_degree(L, tol)

        def hs(v):
            return (self.apply_h(v) - ec * v) / dE

        phi0 = c
        phi1 = hs(c)
        res = jv(0, t) * phi0 + 2.0 * (-1j) * jv(1, t) * phi1
        for n in range(2, n_max + 1):
            phi2 = 2.0 * hs(phi1) - phi0
            res = res + 2.0 * (-1j) ** n * jv(n, t) * phi2
            phi0, phi1 = phi1, phi2
        return res * np.exp(-1j * ec * L), n_max

    def lab_field(self, c, z, xg, yg, dr):
        """Modes at z -> lab field on the Cartesian mesh (xg, yg) inside the disc with the
        lab-frame phase exp(i k c'(z).x' + i g)."""
        d = self.disc
        t, tp = self.frame_axes()
        cz, sl, g = self.axis(z), self.slope(z), self.lab_phase_g(z)
        rf, tab = d.radial_table(c, dr)
        X, Y = np.meshgrid(xg, yg)
        dx, dy = X - cz[0], Y - cz[1]
        xi = dx * t[0] + dy * t[1]
        eta = dx * tp[0] + dy * tp[1]
        r = np.hypot(xi, eta)
        inside = r < d.aw
        ri, phi = r[inside], np.arctan2(eta[inside], xi[inside])
        E = np.zeros(ri.size, complex)
        for i, m in enumerate(d.ms):
            Rm = np.interp(ri, rf, tab[i].real) + 1j * np.interp(ri, rf, tab[i].imag)
            E += Rm * np.exp(1j * m * phi)
        E *= np.exp(1j * self.k * (sl[0] * dx[inside] + sl[1] * dy[inside]) + 1j * g)
        out = np.zeros(X.shape, complex)
        out[inside] = E
        return out


# ----------------------------------------------------------------- far field from the modes

def far_field_amplitude(disc, c, k, thetas, nphi):
    """Fraunhofer amplitude A(theta, phi) = int E exp(-i k theta.x) d^2x of the field sum_mn c_mn J_|m|(j_mn r/aw)
    e^{i m phi} (zero outside aw): A = sum_m e^{i m phi} (-i)^|m| sum_n c_mn G_mn(k theta), with the Lommel form
    G_mn(q) = 2 pi aw kappa J_{|m|+1}(j) J_|m|(q aw) / (kappa^2 - q^2) (J_|m|(j) = 0; pi aw^2 J_{|m|+1}(j)^2 at
    q = kappa); azimuths phi_j = 2 pi j / nphi (nphi >= 2M + 1) by an FFT over m. Shape (len(thetas), nphi)."""
    aw, M = disc.aw, disc.M
    if nphi < 2 * M + 1:
        raise ValueError(f"far field: {nphi} azimuths alias the basis |m| <= {M}")
    q = k * np.asarray(thetas, float)
    a = np.zeros((q.size, nphi), complex)
    for mm in range(M + 1):
        n = disc.zeros[mm].size
        kap, jp1 = disc.zp[mm, :n] / aw, disc.jp1p[mm, :n]
        den = kap[None, :] ** 2 - q[:, None] ** 2
        near = np.abs(den) < 1e-9 * kap[None, :] ** 2
        g = 2.0 * math.pi * aw * (kap * jp1)[None, :] * jv(mm, q * aw)[:, None] / np.where(near, 1.0, den)
        g = np.where(near, math.pi * aw * aw * jp1[None, :] ** 2, g)
        for m in sorted({mm, -mm}):
            a[:, m % nphi] += (-1j) ** mm * (g @ c[m + M, :n])
    return nphi * sfft.ifft(a, axis=1)


def far_field_profiles(P, thetas, k, band):
    """Azimuthal power density dP/dphi = (k / 2 pi)^2 int_band |A|^2 theta dtheta (trapezoid) and the
    azimuthal mean of |A|^2 per theta; with Parseval, sum(ring) dphi is the power inside the band."""
    t = np.asarray(thetas, float)
    w = np.zeros_like(t)
    dt = np.diff(t)
    w[:-1] += 0.5 * dt
    w[1:] += 0.5 * dt
    w *= t * (k / (2.0 * math.pi)) ** 2 * ((t >= band[0]) & (t <= band[1]))
    return w @ P, P.mean(axis=1)


def ring_summary(ring):
    """Ring statistics: mean density, values at phi = 0 and pi over the mean, azimuthal harmonics |c_q| / c_0."""
    n = ring.size
    mean = float(ring.mean())
    if mean <= 0.0:
        return {"mean": mean, "at_0": None, "at_pi": None, "harmonics": []}
    f = np.abs(sfft.rfft(ring)) / abs(sfft.rfft(ring)[0])
    return {"mean": mean, "at_0": float(ring[0] / mean), "at_pi": float(ring[n // 2] / mean) if n % 2 == 0 else None,
            "harmonics": [float(v) for v in f[1:33]]}


# ----------------------------------------------------------------- lattice and screens

class Lattice:
    """Square lab lattice of step h covering [-half, half] with cell edges on edge0 + j h
    (edge0 = a screen window edge, so pixels of size b h align); angular-spectrum drift;
    exact Fourier-box cell integrals onto a commensurate pixel grid."""

    def __init__(self, half, h, dtype=np.complex128, workers=-1, edge0=0.0, cache_bytes=None):
        h = float(h)
        i0 = int(math.floor((-half - edge0) / h))             # first cell edge at or below -half
        i1 = int(math.ceil((half - edge0) / h))               # last cell edge at or above +half
        n = i1 - i0
        self.n, self.h, self.dtype, self.workers = n, h, dtype, workers
        self.x0 = edge0 + i0 * h                               # left edge of the first cell
        self.x = self.x0 + (np.arange(n) + 0.5) * h
        self.half = max(-self.x0, self.x0 + n * h)
        self.kx = 2.0 * np.pi * sfft.fftfreq(n, h)
        self._prop = {}
        self._filt = {}
        self._cache_bytes = None if cache_bytes is None else int(cache_bytes)    # kernel cache budget; None = unlimited
        self._cached = 0

    def _cache(self, store, key, arr):
        """Keep a kernel if the budget allows; large lattices recompute it per use instead."""
        if self._cache_bytes is None or self._cached + arr.nbytes <= self._cache_bytes:
            store[key] = arr
            self._cached += arr.nbytes
        return arr

    def propagator(self, k, d):
        """Angular-spectrum transfer function exp(-i (kx^2 + ky^2) d / 2k), cached per (k, d)."""
        key = (float(k), float(d))
        if key in self._prop:
            return self._prop[key]
        k2 = self.kx[None, :] ** 2 + self.kx[:, None] ** 2
        return self._cache(self._prop, key, np.exp(-1j * k2 / (2.0 * k) * d).astype(self.dtype))

    def drift(self, E, k, d):
        if d == 0.0:
            return E
        F = sfft.fft2(E, workers=self.workers)
        F *= self.propagator(k, d)
        return sfft.ifft2(F, workers=self.workers, overwrite_x=True)

    def cell_plan(self, grid):
        """Pixel grid -> (b, ix0, iy0): pixels b x b lattice cells, window edges on cell edges."""
        px, py = grid.exf / grid.nx, grid.eyf / grid.ny
        b = int(round(px / self.h))
        if b < 1 or abs(b * self.h - px) > 1e-9 * px or abs(py - px) > 1e-9 * px:
            raise ValueError(f"wave_estimator (fb): pixel {m_to_um(px):.6f} x {m_to_um(py):.6f} um is not "
                             f"a square multiple of the lattice step {m_to_um(self.h):.6f} um")
        fx = (grid.x0f - self.x0) / self.h
        fy = (grid.y0f - self.x0) / self.h
        ix0, iy0 = int(round(fx)), int(round(fy))
        if abs(fx - ix0) > 1e-6 or abs(fy - iy0) > 1e-6:
            raise ValueError("wave_estimator (fb): screen window edges must lie on lattice cell edges")
        if ix0 < 0 or iy0 < 0 or ix0 + b * grid.nx > self.n or iy0 + b * grid.ny > self.n:
            raise ValueError("wave_estimator (fb): screen window exceeds the lattice; raise pad")
        return b, ix0, iy0

    def cell_filter(self, grid):
        """Transfer function of the exact box average over b x b lattice cells with the
        half-block shift that puts the samples on the pixel centres; cached per block size."""
        b, _, _ = self.cell_plan(grid)
        if b in self._filt:
            return self._filt[b]
        cell = b * self.h
        shift = (b - 1) * self.h / 2.0
        H = np.sinc(self.kx * cell / (2.0 * np.pi))
        return self._cache(self._filt, b, ((H[None, :] * H[:, None]) * np.exp(1j * (self.kx[None, :] + self.kx[:, None]) * shift)).astype(self.dtype))

    def cells(self, E, grid, filt):
        """Exact cell integrals / cell area of the band-limited lattice field: [ix, iy]."""
        b, ix0, iy0 = self.cell_plan(grid)
        F = sfft.fft2(E, workers=self.workers)
        F *= filt
        Eb = sfft.ifft2(F, workers=self.workers, overwrite_x=True)
        sub = Eb[iy0:iy0 + b * grid.ny:b, ix0:ix0 + b * grid.nx:b]
        return np.ascontiguousarray(sub.T)

    def subsamples(self, E, grid):
        """Lattice samples inside each pixel as the sub-pixel field [ix*b+s, iy*b+t]."""
        b, ix0, iy0 = self.cell_plan(grid)
        blk = E[iy0:iy0 + b * grid.ny, ix0:ix0 + b * grid.nx]          # [y, x]
        return np.ascontiguousarray(blk.T)                              # [x, y] with x fastest in blocks


# ----------------------------------------------------------------- scene spec (picklable)

def fb_jmax_rule(k, aw, theta_cut):
    return int(math.ceil(k * aw * theta_cut))


def build_spec(sim, wave, lines):
    """Plain floats only: bores, source distance, length, screens, lattice and numerics."""
    cfg = sim.cfg
    cap = cfg.capillary
    z_in = float(cap.z0) - float(cap.source.position[2])
    length = float(cap.z1) - float(cap.z0)
    bores = []
    for b in cap.bores:
        if b.get("kind") not in ("cylinder", "torus"):
            raise ValueError("wave_estimator: provider fb supports circular bores (radius, optional bend); "
                             f"got kind {b.get('kind')!r}")
        bend = b.get("bend")
        bores.append({"center": (float(b["center"][0]), float(b["center"][1])), "radius": float(b["radius"]),
                      "rbend": None if bend is None else float(bend["radius"]),
                      "toward": None if bend is None else (float(bend["toward"][0]), float(bend["toward"][1]))})
    screens = cap.all_screens
    scr = [{"z": float(s.z), "nx": int(s.nx), "ny": int(s.ny), "center": (float(s.center[0]), float(s.center[1])),
            "edge_x": float(s.edge_x), "edge_y": float(s.edge_y),
            "reference": None if s.reference is None else (float(s.reference[0]), float(s.reference[1]))}
           for s in screens]
    k_max = max(k for k, *_ in lines)
    lam_min = 2.0 * math.pi / k_max
    a_max = max(b["radius"] for b in bores)
    ells = [wall_offset(k, delta, beta) for k, _, delta, beta in lines]
    aw = a_max + max(e.real for e in ells)
    theta_cut = float(wave["fb_theta_cut"])
    jmax = int(wave["fb_jmax"]) if wave["fb_jmax"] else fb_jmax_rule(k_max, aw, theta_cut)
    theta_modal = jmax / (k_max * aw)
    wall_cut = max(k * aw * math.sqrt(2.0 * delta) for k, _, delta, _ in lines) if wave["fb_wall_cut"] else None
    tilt = max((length / b["rbend"]) if b["rbend"] else 0.0 for b in bores)
    h_auto = lam_min / (2.0 * (theta_modal + tilt) * float(wave["fb_angle_margin"]))
    px = scr[0]["edge_x"] / scr[0]["nx"]
    if wave["grid_dx"]:
        h = float(wave["grid_dx"])
    else:
        h = px / max(1, int(math.ceil(px / h_auto)))
    # lattice half-width: windows, spread of the modal band over the drift, Fresnel tails, bore exits
    half = 0.0
    for s in scr:
        d = s["z"] - float(cap.z1)
        corner = max(abs(s["center"][0]) + s["edge_x"] / 2, abs(s["center"][1]) + s["edge_y"] / 2)
        half = max(half, corner + float(wave["pad"]) * ((theta_modal + tilt) * d + 4.0 * math.sqrt(lam_min * d)))
    for b in bores:
        c0 = np.array(b["center"]); t = np.array(b["toward"]) if b["toward"] else np.zeros(2)
        cl = c0 + (t * length ** 2 / (2.0 * b["rbend"]) if b["rbend"] else 0.0)
        half = max(half, float(np.max(np.abs(cl))) + aw + 2.0 * h)
    if wave["fb_lattice_half"]:
        half = float(wave["fb_lattice_half"])
    figure = {"terms": wave["fb_figure"] or [], "noise": wave["fb_figure_noise"]}
    scale = WallFigure(figure["terms"], figure["noise"], length).min_scale()
    dz = min(float(wave["fb_dz"]), scale / 32.0) if scale else float(wave["fb_dz"])
    ff = wave["far_field"]
    far = None
    if ff is not None:
        n_theta = ff.get("n_theta") or int(math.ceil(ff["theta_max"] / (lam_min / (8.0 * aw)))) + 1
        far = {"theta_max": ff["theta_max"], "n_theta": n_theta, "n_phi": ff.get("n_phi"),
               "ring": ff.get("ring") or [0.0, ff["theta_max"]]}
    return {"z_in": z_in, "length": length, "z_exit": float(cap.z1), "bores": bores, "screens": scr, "aw": aw, "jmax": jmax, "wall_cut": wall_cut,
            "theta_modal": theta_modal, "tilt": tilt, "h": h, "h_auto": h_auto, "half": half,
            "lines": [(float(k), float(wl), float(dl), float(bt)) for k, wl, dl, bt in lines],
            "ells": [(e.real, e.imag) for e in ells], "cheb_tol": float(wave["fb_chebyshev_tol"]),
            "dr": float(wave["fb_dr"]), "wall": wave["fb_wall"], "dtype": wave["fb_grid_dtype"],
            "observable": wave["observable"], "per_bore": wave["fb_per_bore_maps"], "spot": wave["entrance_spot"],
            "workers": wave["workers"] or -1, "cache_gb": float(wave["cache_gb"]),
            "figure": figure, "dz": dz, "far": far, "dump": bool(wave["dump_field"])}


class _PixelCfg:
    def __init__(self, s):
        self.z, self.nx, self.ny = s["z"], s["nx"], s["ny"]
        self.center, self.edge_x, self.edge_y = s["center"], s["edge_x"], s["edge_y"]
        self.reference = s["reference"]


# ----------------------------------------------------------------- worker

_W = {}


def _init_worker(spec):
    """Per-process state: basis, channels per line, lattice, samplers and accumulators."""
    from .wave import ScreenSampler, Accumulator
    t0 = time.time()
    dtype = np.complex64 if spec["dtype"] == "complex64" else np.complex128
    disc = Disc(spec["aw"], spec["jmax"], wall_cut=spec["wall_cut"])
    disc.fine_table(spec["dr"], dtype=np.float32)
    figure = WallFigure(spec["figure"]["terms"], spec["figure"]["noise"], spec["length"])
    chans, fallback = {}, {}
    for k, wl, delta, beta in spec["lines"]:
        if spec["wall"] == "exact":
            row = []
            for b in spec["bores"]:
                kap, ok = exact_kappa(disc, b["radius"], k, delta, beta)
                dirl = disc.kap2 / (2.0 * k) - 1j * mode_damping(disc, k, delta, beta)
                h0 = np.where(disc.valid, np.where(ok[disc.am], (kap * kap)[disc.am] / (2.0 * k), dirl), 0.0)
                fallback[k] = fallback.get(k, 0) + int(np.sum(disc.validp & ~ok))
                row.append(FBChannel(b["center"], b["toward"], b["rbend"], disc, k, h0=h0, figure=figure, dz=spec["dz"]))
            chans[k] = row
        else:
            gamma = mode_damping(disc, k, delta, beta) if spec["wall"] == "dir-ell" else None
            chans[k] = [FBChannel(b["center"], b["toward"], b["rbend"], disc, k, gamma, figure=figure, dz=spec["dz"])
                        for b in spec["bores"]]
    lat = Lattice(spec["half"], spec["h"], dtype=dtype, workers=spec["workers"], edge0=spec["screens"][0]["center"][0] - spec["screens"][0]["edge_x"] / 2,
                  cache_bytes=spec["cache_gb"] * 2 ** 30)
    grids = [ScreenGrid(_PixelCfg(s)) for s in spec["screens"]]
    b = [lat.cell_plan(g)[0] for g in grids]
    samplers = [ScreenSampler(g, s["reference"] or (g.cxf, g.cyf), spec["observable"], bb)
                for g, s, bb in zip(grids, spec["screens"], b)]
    nb = len(spec["bores"])
    per_bore = spec["per_bore"] if spec["per_bore"] is not None else nb <= 2
    far = None
    if spec["far"] is not None:
        f = spec["far"]
        nphi = f["n_phi"] or 1 << int(math.ceil(math.log2(max(4 * disc.M + 4, 256))))
        far = {"thetas": np.linspace(0.0, f["theta_max"], f["n_theta"]), "n_phi": int(nphi), "ring": f["ring"]}
    _W.update(spec=spec, disc=disc, chans=chans, lat=lat, grids=grids, samplers=samplers,
              nb=nb, per_bore=per_bore, dtype=dtype, t_init=time.time() - t0, degrees=[],
              fallback=fallback, far=far)


def _node_fields(spec, k, chans, lat, xi, per_bore=True):
    """Exit-plane lattice fields for one source node and one line: one array per bore, or
    their sum in a single array (one lattice-sized buffer instead of one per bore); with the
    entrance and exit mode coefficients of every bore."""
    out, coeffs = [], []
    for ch in chans:
        c0 = ch.spot_modes(spec["spot"]) if spec["spot"] else ch.entrance_modes(xi, spec["z_in"])
        cL, deg = ch.propagate(c0, spec["length"], spec["cheb_tol"])
        coeffs.append((c0, cL))
        _W["degrees"].append(deg)
        cz = ch.axis(spec["length"])
        lo = np.searchsorted(lat.x, cz - ch.disc.aw - 2 * lat.h)
        hi = np.searchsorted(lat.x, cz + ch.disc.aw + 2 * lat.h)
        xg, yg = lat.x[lo[0]:hi[0]], lat.x[lo[1]:hi[1]]
        block = ch.lab_field(cL, spec["length"], xg, yg, spec["dr"]).astype(lat.dtype)
        if per_bore or not out:
            out.append(np.zeros((lat.n, lat.n), lat.dtype))
        out[-1][lo[1]:hi[1], lo[0]:hi[0]] += block
    return out, coeffs


def _run_nodes(args):
    """Accumulate W/I over a chunk of (node, weight) pairs; returns the partial sums."""
    from .wave import Accumulator
    nodes, weights, idx = args
    spec, lat, per_bore = _W["spec"], _W["lat"], _W["per_bore"]
    # fresh sums per task: a pool worker may serve several chunks
    accs = [Accumulator(smp, _W["nb"] if per_bore else 1) for smp in _W["samplers"]]
    _W["degrees"] = []
    z1 = spec["z_exit"]                                     # exit plane (lab z), not z_in + L
    disc, far = _W["disc"], _W["far"]
    nl = len(spec["lines"])
    extra = {"p_in": np.zeros(nl), "p_out": np.zeros(nl), "dump": {}, "fallback": _W["fallback"], "far_meta": far,
             "basis": {"aw": disc.aw, "zeros": disc.zp, "valid": disc.validp, "m": np.array(disc.ms)}}
    if far is not None:
        nt = far["thetas"].size
        extra.update(ring=np.zeros((nl, far["n_phi"])), radial=np.zeros((nl, nt)), map=np.zeros((nt, far["n_phi"])))
    for j, xi, w in zip(idx, nodes, weights):
        for li, (k, wl, delta, beta) in enumerate(spec["lines"]):
            chans = _W["chans"][k]
            fields, coeffs = _node_fields(spec, k, chans, lat, xi, per_bore)
            extra["p_in"][li] += w * sum(disc.norm(c0) for c0, _ in coeffs)
            extra["p_out"][li] += w * sum(disc.norm(cL) for _, cL in coeffs)
            if spec["dump"]:
                for bi, (c0, cL) in enumerate(coeffs):
                    extra["dump"][f"c_in_line{li}_node{j}_bore{bi}"] = c0
                    extra["dump"][f"c_out_line{li}_node{j}_bore{bi}"] = cL
            if far is not None:
                P = np.abs(far_field_amplitude(disc, coeffs[0][1], k, far["thetas"], far["n_phi"])) ** 2
                ring, radial = far_field_profiles(P, far["thetas"], k, far["ring"])
                extra["ring"][li] += w * ring
                extra["radial"][li] += w * radial
                extra["map"] += w * wl * P
            for s, g, smp, acc in zip(spec["screens"], _W["grids"], _W["samplers"], accs):
                d = s["z"] - z1
                cells, subs = [], []
                for E in fields:
                    Ed = lat.drift(E, k, d)
                    cells.append(lat.cells(Ed, g, lat.cell_filter(g)))
                    subs.append(lat.subsamples(Ed, g) if spec["observable"] == "coherent_cell" else None)
                acc.add(w * wl, cells, [None] * len(cells), subs)
    return [(a.I, a.W, a.I_ref, a.I_pixel, a.I_bore, a.G12) for a in accs], len(nodes), _W["t_init"], \
        (int(np.median(_W["degrees"])) if _W["degrees"] else None), int(_W["disc"].nmodes), extra


# ----------------------------------------------------------------- scene driver

def fb_capillary_scene(sim, wave, lines, rays_paths, log, out_dir=None):
    from .wave import _rule, ScreenSampler, Accumulator, _finish, _report_lines
    cfg = sim.cfg
    cap = cfg.capillary
    if wave["entrance_spot"]:
        sp = wave["entrance_spot"]
        nodes, weights = np.array([sp["center"]]), np.array([1.0])
        rule = (f"entrance spot: gaussian FWHM {sp['fwhm'] * 1e9:.0f} nm at ({sp['center'][0] * 1e6:.3f}, "
                f"{sp['center'][1] * 1e6:.3f}) um, tilt ({sp['tilt'][0] * 1e3:.3f}, {sp['tilt'][1] * 1e3:.3f}) mrad")
    else:
        nodes, weights, rule = _rule(wave, "capillary", cap.source, rays_paths)
    spec = build_spec(sim, wave, lines)
    labels = ["capillary"] + [f"capillary-s{i}" for i in range(1, len(cap.screens) + 1)]
    lam0 = float(sim.lam)
    nyq = spec["h"] <= spec["h_auto"] * (1.0 + 1e-12)
    log(f"  16 [capillary/fb]: {rule}; j_max {spec['jmax']} (theta_modal {spec['theta_modal']*1e3:.3f} mrad, "
        f"tilt {spec['tilt']*1e3:.3f} mrad), lattice h = {m_to_um(spec['h']):.4f} um (auto {m_to_um(spec['h_auto']):.4f}), "
        f"half {m_to_um(spec['half']):.1f} um, chebyshev"
        + (f", guided cut j^2 - m^2 < {spec['wall_cut']:.1f}^2" if spec["wall_cut"] else "")
        + f", wall {spec['wall']} ell = {spec['ells'][0][0]*1e9:.3f} + {spec['ells'][0][1]*1e9:.3f}i nm"
        + ("" if nyq else " — note: lattice coarser than the auto step"))
    t0 = time.time()
    jobs = int(wave["fb_jobs"] or 1)
    order = np.arange(len(nodes))
    chunks = [(nodes[i::jobs], weights[i::jobs], order[i::jobs]) for i in range(jobs)]
    chunks = [c for c in chunks if len(c[0])]
    if len(chunks) == 1:
        _init_worker(spec)
        parts = [_run_nodes(chunks[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(chunks), initializer=_init_worker, initargs=(spec,)) as pool:
            parts = pool.map(_run_nodes, chunks)
    grids = [ScreenGrid(_PixelCfg(s)) for s in spec["screens"]]
    lat = Lattice(spec["half"], spec["h"], edge0=spec["screens"][0]["center"][0] - spec["screens"][0]["edge_x"] / 2)
    bs = [lat.cell_plan(g)[0] for g in grids]
    samplers = [ScreenSampler(g, s["reference"] or (g.cxf, g.cyf), wave["observable"], bb)
                for g, s, bb in zip(grids, spec["screens"], bs)]
    nb = len(spec["bores"])
    per_bore = spec["per_bore"] if spec["per_bore"] is not None else nb <= 2
    accs = [Accumulator(smp, nb if per_bore else 1) for smp in samplers]
    if sum(p[1] for p in parts) != len(nodes):
        raise RuntimeError(f"wave_estimator (fb): {sum(p[1] for p in parts)} nodes accumulated, {len(nodes)} expected")
    for part, n_nodes, t_init, deg, _, _ in parts:
        for acc, (I, W, I_ref, I_pix, I_bore, G12) in zip(accs, part):
            acc.I += I; acc.W += W; acc.I_ref += I_ref
            if acc.I_pixel is not None and I_pix is not None:
                acc.I_pixel += I_pix
            if acc.I_bore is not None and I_bore is not None:
                acc.I_bore += I_bore
            if acc.G12 is not None and G12 is not None:
                acc.G12 += G12
    degrees = [p[3] for p in parts if p[3] is not None]
    extra = parts[0][5]
    for p in parts[1:]:
        for key, v in p[5].items():
            if key == "dump":
                extra["dump"].update(v)
            elif key not in ("fallback", "far_meta", "basis"):
                extra[key] = extra[key] + v
    fig = WallFigure(spec["figure"]["terms"], spec["figure"]["noise"], spec["length"])
    fallback = ({f"{float(HC_KEV_ANGSTROM) * k / (2.0e10 * math.pi):.6f}": n for k, n in sorted(extra["fallback"].items())}
                if spec["wall"] == "exact" else None)
    files = []
    if out_dir is not None and spec["far"] is not None:
        files += _emit_far_field(out_dir, spec, lines, extra, wave)
    if out_dir is not None and spec["dump"]:
        files += _emit_dump(out_dir, spec, lines, extra, nodes, weights)
    transmission = [float(po / pi) if pi > 0.0 else None for pi, po in zip(extra["p_in"], extra["p_out"])]
    wall_label = {"dir-ell": "dir-ell", "dir-ell-real": "dir-ell,no-absorption", "exact": "exact-jk"}[spec["wall"]]
    out = []
    a = spec["bores"][0]["radius"]
    for label, s, smp, acc in zip(labels, spec["screens"], samplers, accs):
        d = s["z"] - float(cap.z1)
        nf = {"entrance": a * a / (lam0 * spec["length"]),
              "source": None if spec["spot"] else a * a / (lam0 * spec["z_in"]),
              "exit": (a * a / (lam0 * d)) if d > 0 else None}
        bent = any(b["rbend"] for b in spec["bores"])
        meta = {"scene": label, "provider": "fb",
                "model_class": wall_label + (",parabolic-axis" if bent else "") + (",hadamard-figure" if fig.has_wall else "")
                + (",axis-path" if fig.has_axis else ""),
                "figure": spec["figure"], "figure_dz_m": spec["dz"] if (fig.z_dependent or fig.has_axis) else None,
                "exact_fallback_modes": fallback, "transmission": transmission, "far_field": spec["far"],
                "source_rule": rule, "n_nodes": int(len(nodes)),
                "lattice_h_m": spec["h"], "lattice_h_auto_m": spec["h_auto"], "lattice_half_m": spec["half"],
                "lattice_n": lat.n, "theta_modal_rad": spec["theta_modal"], "tilt_rad": spec["tilt"],
                "sampling": {"nyquist_ratio": 2.0 * spec["h"] * (spec["theta_modal"] + spec["tilt"]) / (2.0 * math.pi / max(k for k, *_ in lines))},
                "jmax": spec["jmax"], "wall_cut": spec["wall_cut"], "n_modes": parts[0][4],
                "propagator": "chebyshev" if bent else "diagonal",
                "chebyshev_degree_median": (int(np.median(degrees)) if degrees and bent else None),
                "wall": spec["wall"], "wall_offset_nm": [[e[0] * 1e9, e[1] * 1e9] for e in spec["ells"]],
                "aw_m": spec["aw"], "fresnel_numbers": nf, "distance_m": d, "bores": spec["bores"],
                "per_bore_maps": per_bore, "jobs": jobs, "worker_init_s": max(p[2] for p in parts)}
        res = _finish(acc, smp, meta, wave, time.time() - t0)
        wall_text = {"dir-ell": f"wall DIR-ℓ, ℓ = {spec['ells'][0][0]*1e9:.3f} + {spec['ells'][0][1]*1e9:.3f}i nm, damping × Fresnel loss at θ_w",
                     "dir-ell-real": f"wall DIR-ℓ, ℓ = {spec['ells'][0][0]*1e9:.3f} nm (absorption off)",
                     "exact": "wall exact: J_m / K_m characteristic equation per mode and line"
                              + (f" (DIR-ℓ fallback for {sum(fallback.values())} mode(s))" if fallback and sum(fallback.values()) else "")}
        lines_r = [
            f"- provider fb: j_max = {spec['jmax']} (θ_modal = {spec['theta_modal']*1e3:.3f} mrad, tilt {spec['tilt']*1e3:.3f} mrad), "
            + (f"chebyshev degree ≈ {int(np.median(degrees)) if degrees else '?'}" if (bent or fig.has_wall or fig.has_axis)
               else "diagonal propagator (straight bore)")
            + (f", guided-mode cut j² − m² < {spec['wall_cut']:.1f}²" if spec["wall_cut"] else "")
            + f"; {wall_text[spec['wall']]}",
            f"- lattice h = {m_to_um(spec['h']):.4f} µm (auto {m_to_um(spec['h_auto']):.4f}), half-width {m_to_um(spec['half']):.1f} µm, "
            f"{lat.n}² points; cells = exact Fourier box of the band-limited field; screens by angular-spectrum drift",
            f"- N_F: entrance {nf['entrance']:.2f}, source " + (f"{nf['source']:.2f}" if nf["source"] else "— (entrance spot)") + ", exit "
            + (f"{nf['exit']:.3f}" if nf["exit"] else "— (screen on the exit plane)") + "; ray optics is not used anywhere in this stage"]
        if fig.has_wall or fig.has_axis:
            lines_r.append(f"- figure: {len(spec['figure']['terms'])} term(s)" + (", random field" if spec["figure"]["noise"] else "")
                           + (" (Hadamard wall, first order)" if fig.has_wall else "") + (" (axis path, co-moving frame)" if fig.has_axis else "")
                           + (f", z step {spec['dz'] * 1e6:.1f} µm" if (fig.z_dependent or fig.has_axis) else ""))
        if label == "capillary":
            lines_r.append("- modal transmission P_exit / P_entrance: " + ", ".join(f"{t:.4f}" for t in transmission if t is not None))
        res["report"] = _report_lines(label, res, wave, lines_r)
        out.append((label, res))
    out[0][1]["fb_files"] = files
    return out


def _emit_far_field(out_dir, spec, lines, extra, wave):
    """far-field.json (per line and weighted ring statistics, modal transmission), far-field-fb.npz (ring and
    radial profiles per line, weighted map) and the weighted ring profile as SVG."""
    import json
    from .. import render
    far = extra["far_meta"]
    th, nphi = far["thetas"], far["n_phi"]
    phi = 2.0 * math.pi * np.arange(nphi) / nphi
    rows = []
    wsum = np.zeros(nphi)
    for li, (k, wl, *_) in enumerate(lines):
        r = ring_summary(extra["ring"][li])
        r.update(e_kev=float(HC_KEV_ANGSTROM) * k / (2.0e10 * math.pi), weight=float(wl),
                 transmission=float(extra["p_out"][li] / extra["p_in"][li]) if extra["p_in"][li] > 0.0 else None)
        rows.append(r)
        wsum += wl * extra["ring"][li]
    summary = {"theta_max_rad": far["thetas"][-1], "n_theta": int(th.size), "n_phi": nphi, "ring_rad": far["ring"],
               "lines": rows, "weighted": ring_summary(wsum),
               "convention": "A(theta, phi) = int E exp(-i k theta.x) d^2x from the exit modes (Lommel form); ring = "
                             "dP/dphi = (k/2pi)^2 int_ring |A|^2 theta dtheta; phi measured in the bore frame; "
                             "lines and nodes summed by intensity"}
    with open(os.path.join(out_dir, "far-field.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    np.savez(os.path.join(out_dir, "far-field-fb.npz"), theta=th, phi=phi, ring=extra["ring"], radial=extra["radial"],
             map=extra["map"], k=np.array([k for k, *_ in lines]), weight=np.array([wl for _, wl, *_ in lines]),
             p_in=extra["p_in"], p_out=extra["p_out"])
    mean = float(wsum.mean()) or 1.0
    name = "16-far-field.svg"
    render.save(os.path.join(out_dir, name), render.line_chart(
        [{"xs": [float(v) for v in np.degrees(phi)], "ys": [float(v) / mean for v in wsum], "label": "dP/dφ / mean"}],
        "far-field ring profile (weighted over lines)", "φ, deg", "dP/dφ / mean",
        f"ring {far['ring'][0] * 1e3:.3f}–{far['ring'][1] * 1e3:.3f} mrad, {nphi} azimuths", w=760))
    return ["far-field.json", "far-field-fb.npz", name]


def _emit_dump(out_dir, spec, lines, extra, nodes, weights):
    """field-capillary.npz: entrance and exit mode coefficients per line / node / bore and the basis
    (padded zeros j_mn over |m|, validity, a_w, signed m order); fields follow from sum c J_|m|(j r / a_w) e^{i m phi}."""
    b = extra["basis"]
    np.savez(os.path.join(out_dir, "field-capillary.npz"), k=np.array([k for k, *_ in lines]),
             line_weights=np.array([wl for _, wl, *_ in lines]), nodes=np.asarray(nodes, float),
             node_weights=np.asarray(weights, float), aw=np.array(b["aw"]), zeros=b["zeros"], valid=b["valid"],
             m=b["m"], length=np.array(spec["length"]), z_exit=np.array(spec["z_exit"]), **extra["dump"])
    return ["field-capillary.npz"]

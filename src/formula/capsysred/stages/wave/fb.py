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
Options: wall `exact-roots` (per-mode roots of the J_m / K_m characteristic equation of the hollow
cylinder instead of DIR-ell), a wall figure dr(phi, z) to first order (Hadamard) with the q = 1
part as an exact co-moving axis path, the far field from the exit modes (Lommel integrals) and
a dump of the entrance / exit mode coefficients.
"""

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import fft as sfft
from scipy.special import jn_zeros, jv, kve

from .physics import spot_profile

MODEL_CLASS_FB = "dir-ell"
CHEB_TOL = 1e-12        # default cut |J_n(t)| <= CHEB_TOL of the Chebyshev series of exp(-iHL)
CHEB_MARGIN = 1.02      # widening of the spectrum half-width of H scaled to [-1, 1]
CHEB_DAMP = 1.6         # max damping gamma_max dL of one Chebyshev sub-step
NEWTON_ITERS = 40       # default iterations of the damped Newton on the characteristic equation
NEWTON_RTOL = 1e-13     # default convergence |step| <= NEWTON_RTOL |kappa|
NEWTON_STEP = 0.2       # step cap |step| <= NEWTON_STEP / a


def radial_nodes(jmax: float) -> int:
    """Default number of Gauss-Legendre radial nodes on [0, aw] for the zeros below jmax: int(1.2 jmax) + 40.

    Need. The nodes project a field onto the modes J_|m|(j r / aw), j < jmax = k aw theta_cut. For a field of tilt
    T = k aw theta_in, exp(i T (r / aw) cos phi) with theta_in the incidence angle on the bore, the projection is
    exact to 1e-10 of the field norm with

        N = (T + jmax) / 4 + 3 (T + jmax)^(1/3)

    (measured for jmax 8..640 and T from 0 to 4 jmax: the coefficient of the cube root is 2.9..3.3; tilt-only
    field, eight azimuthal orders per jmax; a chirp of the same edge frequency needs fewer nodes).

    Reason. With r = aw (t + 1) / 2 the integrand field x mode oscillates on t in [-1, 1] as exp(i w t),
    w = (T + jmax) / 2. N nodes are exact through the polynomial degree 2 N - 1, and a polynomial follows
    exp(i w t) once its degree passes w by a transition zone ~ w^(1/3) (the coefficients J_n(w) of its Chebyshev
    series decay beyond it), so N = w / 2 + margin. T = jmax is a mode of the basis projected onto the basis.

    Rule. The factor 1.2 sets how far the field may exceed the cut of the basis: (T + jmax) / 4 = 1.2 jmax gives
    T = 3.8 jmax, an incidence up to 3.8 theta_cut. The term + 40 covers the cube root, which equals 40 at
    jmax ~ 500. The basis does not represent such a field; too few nodes alias it into the kept modes.

    Largest tilt T_max projected to 1e-10, against the alternative max(int(1.2 jmax), 100):

                 int(1.2 jmax) + 40      max(int(1.2 jmax), 100)
        jmax     nodes   T_max / jmax    nodes   T_max / jmax
           8        49       15.8          100       38.8
          20        64        8.4          100       14.9
          40        88        5.9          100        7.0
          80       136        4.7          100        3.0
         160       232        4.1          192        3.2
         320       424        3.9          384        3.4
         640       808        3.8          768        3.5

    Conclusion. int(1.2 jmax) + 40 keeps T_max >= 3.8 jmax up to jmax ~ 500 and loses it slowly beyond (the exact
    margin is 5 jmax^(1/3): 27 at jmax 160, 44 at 640, 66 at 2200). The floor of 100 nodes gains only below
    jmax 50, dips to 3.0 near jmax 80 and stays at 3.2..3.5 in the working range, for a saving of 40 nodes
    (17 % at jmax 160, 5 % at 640). The rule stays."""
    return int(1.2 * jmax) + 40


# ----------------------------------------------------------------- wall

def wall_offset(k: float, delta: float, beta: float) -> complex:
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


def newton_kappa(mm: np.ndarray, kap: np.ndarray, a: float, k2: complex, iters: int = NEWTON_ITERS,
                 rtol: float = NEWTON_RTOL) -> tuple[np.ndarray, np.ndarray]:
    """Damped Newton (|step| <= NEWTON_STEP / a) on p = s of the scalar characteristic equation (exact_kappa) from the
    guesses kap of the orders mm, k2 = k^2 (1 - n^2). Returns (kappa, converged)."""
    cap = NEWTON_STEP / a
    ok = np.zeros(kap.shape, bool)
    for _ in range(iters):
        x = kap * a
        Q = np.sqrt(k2 - kap * kap)
        Q = np.where(Q.real < 0.0, -Q, Q)
        z = Q * a
        p = kap * (jv(mm - 1.0, x) / jv(mm, x) - mm / x)
        s = Q * (-kve(mm - 1.0, z) / kve(mm, z) - mm / z)
        dp = -(x - mm * mm / x) - a * p * p / kap
        ds = -kap / Q * (z + mm * mm / z) + a * kap * s * s / (Q * Q)
        step = (p - s) / (dp - ds)
        big = np.abs(step) > cap
        step = np.where(big, step * cap / np.where(big, np.abs(step), 1.0), step)
        kap = kap - step
        ok = (np.abs(step) <= rtol * np.abs(kap)) & ~big
        if ok.all():
            break
    return kap, ok


def exact_kappa(disc: "Disc", *, a: float, k: float, delta: float, beta: float, iters: int = NEWTON_ITERS,
                rtol: float = NEWTON_RTOL) -> tuple[np.ndarray, np.ndarray]:
    """Complex wavenumbers kappa_mn of the hollow circular bore of radius a in the wall n = 1 - delta + i beta:
    p = s with p = kappa J_m'(kappa a) / J_m(kappa a), s = Q K_m'(Q a) / K_m(Q a), Q^2 = k^2 (1 - n^2) - kappa^2,
    Re Q > 0 (scalar continuity of psi and psi'); damped Newton (|step| <= NEWTON_STEP / a) from the DIR-ell guess
    j_mn / (a + ell) with the analytic derivative p' - s' (Bessel equations). Returns (kappa, converged) on the
    padded |m| x n grid; invalid entries are zero; a root is accepted only if it converged, decays
    (Im kappa <= 0) and stays within 1 / a of the guess."""
    k2 = k * k * (1.0 - (1.0 - delta + 1j * beta) ** 2)
    ell = wall_offset(k, delta, beta)
    rows, cols = np.nonzero(disc.validp)
    guess = disc.zp[rows, cols] / (a + ell)
    kap, ok = newton_kappa(rows.astype(float), guess, a, k2, iters, rtol)
    out = np.zeros(disc.validp.shape, complex)
    conv = np.zeros(disc.validp.shape, bool)
    out[rows, cols] = kap
    conv[rows, cols] = ok & np.isfinite(kap) & (kap.imag <= 0.0) & (np.abs(kap - guess) * a < 1.0)
    return out, conv


# ----------------------------------------------------------------- disc basis

class Disc:
    """Dirichlet disc basis J_|m|(j_mn r / aw) e^{i m phi}, all zeros j_mn < jmax, with wall_cut J only
    the guided ones (j_mn^2 - m^2 < J^2); Gauss-Legendre radial quadrature, FFT in phi; padded arrays over |m|."""

    def __init__(self, *, aw: float, jmax: float, nr: int | None = None, nphi: int | None = None,
                 wall_cut: float | None = None) -> None:
        """aw: disc radius [m]; jmax: zeros cut j_mn < jmax; nr, nphi: radial / azimuthal nodes (None: auto); wall_cut: J or None."""
        self.aw, self.jmax, self.wall_cut = float(aw), float(jmax), wall_cut
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
        self.nr = nr or radial_nodes(jmax)
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

    def shift_blocks(self) -> tuple[np.ndarray, np.ndarray]:
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

    def empty(self) -> np.ndarray:
        return np.zeros((2 * self.M + 1, self.nmax), complex)

    def to_grid(self, c: np.ndarray) -> np.ndarray:
        """c: mode coefficients (2M+1, nmax), row i is signed m = i - M."""
        F = np.zeros((self.nr, self.nphi), complex)
        F[:, self.col] = np.einsum("mrn,mn->rm", self.B, c)
        return self.nphi * sfft.ifft(F, axis=1)

    def to_modes(self, G: np.ndarray) -> np.ndarray:
        """G: field at the (r, phi) nodes, (nr, nphi)."""
        F = sfft.fft(G, axis=1) / self.nphi
        return np.einsum("mnr,rm->mn", self.P, F[:, self.col])

    def norm(self, c: np.ndarray) -> float:
        """c: mode coefficients (2M+1, nmax)."""
        return float(np.sum(np.abs(c) ** 2 * self.mnorm))

    def fine_table(self, dr: float, dtype: type[np.floating] = np.float32) -> tuple[np.ndarray, np.ndarray]:
        """dr: fine radial step [m]; dtype of the J table; both are fixed by the first call."""
        if self._fine is None:
            rf = np.arange(0.0, self.aw + dr, dr)
            J = np.zeros((self.M + 1, rf.size, self.nmax), dtype)
            for mm, z in enumerate(self.zeros):
                J[mm, :, :z.size] = jv(mm, np.outer(rf, z / self.aw))
            self._fine = (rf, J)
        return self._fine

    def radial_table(self, c: np.ndarray, dr: float) -> tuple[np.ndarray, np.ndarray]:
        """R_m(r) on the fine radial grid for each signed m: (2M+1, nf).
        c: mode coefficients (2M+1, nmax); dr: fine radial step [m]."""
        rf, J = self.fine_table(dr)
        M = self.M
        pos = np.einsum("mrn,mn->mr", J, c[M:])
        neg = np.einsum("mrn,mn->mr", J[1:], c[M - 1::-1])
        return rf, np.concatenate([neg[::-1], pos], axis=0)


# ----------------------------------------------------------------- wall figure

class WallFigure:
    """Radial wall displacement dr(phi, z) [m]: harmonic terms amp mod(z) cos(q (phi - angle - rate z)),
    mod = cos(2 pi z / period + phase) or 1, plus an optional random field sum_q a_q(z) cos q phi + b_q(z) sin q phi
    (white noise smoothed by exp(-z^2 / 2 lc^2), correlation exp(-dz^2 / 4 lc^2), equal power per listed q, total rms
    sigma). coeffs(z) -> dr_hat[p],
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

def _blocks(B, c):
    """Real blocks B[m] times complex vectors c[m] as two real batched matmuls."""
    return (np.matmul(B, c.real[:, :, None]) + 1j * np.matmul(B, c.imag[:, :, None]))[:, :, 0]


class FBChannel:
    """One circular bore: axis c(z) = c0 + t z^2/(2R) (t toward the bend centre; R None =
    straight); frame coordinates xi along t, eta along t_perp; potential k xi / R."""

    def __init__(self, *, c0: np.ndarray | tuple[float, float], t: np.ndarray | tuple[float, float] | None,
                 rbend: float | None, disc: Disc, k: float, gamma: np.ndarray | None = None,
                 h0: np.ndarray | None = None, figure: WallFigure | None = None, dz: float | None = None,
                 wall_loss: complex | None = None) -> None:
        """c0: entrance centre (x, y) [m]; t: toward the bend centre; rbend [m], None = straight; k [rad/m]; gamma: mode
        damping (2M+1, nmax) [1/m]; h0: H diagonal instead of kap2/2k - i gamma; dz: z step of a z-dependent figure [m];
        wall_loss: a uniform complex wall displacement [m] (i Im ell for the DIR-ell wall) applied through the boundary
        (Hadamard) form at every step: the non-diagonal first-order loss operator whose diagonal is the gamma of
        mode_damping without its Fresnel factor."""
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
        self.wall_loss = None if wall_loss is None else complex(wall_loss)
        self._dr = None                                              # figure coefficients of the current step
        self._cpp = np.zeros(2)                                      # axis curvature c'' of the current step (frame)
        self._g = None
        if self.path is not None:                                    # lab phase g(z) = (k/2) int |c'|^2 on a fine grid
            L = figure.length
            n = max(2001, int(math.ceil(64.0 * L / (figure.min_scale() or L))) + 1)
            zg = np.linspace(0.0, L, n)
            s2 = np.array([float(np.sum(self.slope_frame(z) ** 2)) for z in zg])
            self._g = (zg, np.concatenate([[0.0], np.cumsum(0.5 * (s2[1:] + s2[:-1]) * np.diff(zg))]) * self.k / 2.0)

    def slope_frame(self, z: float) -> np.ndarray:
        """Axis slope c'(z) in the frame (xi, eta): the bend t z / R along xi plus the axis path.
        z: distance from the entrance [m]."""
        s = np.array([0.0 if self.rb is None else z / self.rb, 0.0])
        return s + (self.path.derivs(z)[1] if self.path is not None else 0.0)

    def to_lab(self, v: np.ndarray) -> np.ndarray:
        """v: 2-vector in the frame (xi, eta)."""
        t, tp = self.frame_axes()
        return v[0] * t + v[1] * tp

    def frame_axes(self) -> tuple[np.ndarray, np.ndarray]:
        t = self.t if self.t is not None else np.array([1.0, 0.0])
        return t, np.array([-t[1], t[0]])

    def axis(self, z: float) -> np.ndarray:
        """z: distance from the entrance [m]."""
        c = self.c0 if self.rb is None else self.c0 + self.t * z * z / (2.0 * self.rb)
        return c if self.path is None else c + self.to_lab(self.path.derivs(z)[0])

    def slope(self, z: float) -> np.ndarray:
        """z: distance from the entrance [m]."""
        if self.path is None:
            return np.zeros(2) if self.rb is None else self.t * z / self.rb
        return self.to_lab(self.slope_frame(z))

    def lab_phase_g(self, z: float) -> float:
        """z: distance from the entrance [m]."""
        if self._g is not None:
            return float(np.interp(z, *self._g))
        return 0.0 if self.rb is None else self.k / 2.0 * (z ** 3 / (3.0 * self.rb ** 2))

    def _entrance_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | float]:
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

    def entrance_field(self, src_xy: np.ndarray | tuple[float, float], z_in: float) -> np.ndarray:
        """Paraxial point source at (src_xy, -z_in) on the entrance disc (polar grid),
        amplitude 1/(i lambda z_in) like the stage's free-space Fresnel step.
        src_xy: source (x, y) [m]; z_in: source distance before the entrance [m]."""
        X, Y, gauge = self._entrance_grid()
        lam = 2.0 * math.pi / self.k
        return (gauge * np.exp(1j * self.k * ((X - src_xy[0]) ** 2 + (Y - src_xy[1]) ** 2) / (2.0 * z_in))
                / (1j * lam * z_in))

    def entrance_modes(self, src_xy: np.ndarray | tuple[float, float], z_in: float) -> np.ndarray:
        """src_xy: source (x, y) [m]; z_in: source distance before the entrance [m]."""
        return self.disc.to_modes(self.entrance_field(src_xy, z_in))

    def spot_field(self, spot: dict) -> np.ndarray:
        """Coherent gaussian on the entrance disc: FWHM in intensity at the waist, unit power, plane-wave tilt
        (tx, ty); waist at z = focus (0: on the entrance, < 0: before it), q = -focus + i z_R per axis.
        spot: the wave spot config (fwhm, center, tilt, focus)."""
        X, Y, gauge = self._entrance_grid()
        return gauge * spot_profile(spot, self.k, X, Y)

    def spot_modes(self, spot: dict) -> np.ndarray:
        """spot: the wave spot config (fwhm, center, tilt, focus)."""
        return self.disc.to_modes(self.spot_field(spot))

    def apply_v(self, c: np.ndarray) -> np.ndarray:
        """Galerkin V c: the bend through the block-tridiagonal m -> m +- 1 matrices, the wall figure
        through its Hadamard form -(1/k aw^3) v_i sum_j dr_hat[m_i - m_j] u_j c_j (a convolution over m).
        c: mode coefficients (2M+1, nmax)."""
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
            out[1:] += 0.5 * self.k * (cx - 1j * cy) * _blocks(up[:-1], c[:-1])
            out[:-1] += 0.5 * self.k * (cx + 1j * cy) * _blocks(dn[1:], c[1:])
        if self._dr is not None:
            d = self.disc
            b = np.sum(d.wall_u * c, axis=1)
            q = (self._dr.size - 1) // 2
            out += self.fig_scale * d.wall_v * np.convolve(b, self._dr)[q:q + b.size, None]
        return out

    def apply_h(self, c: np.ndarray) -> np.ndarray:
        """H c = (kappa^2/2k - i gamma) c + V c (Galerkin blocks).
        c: mode coefficients (2M+1, nmax)."""
        return self.h0 * c + self.apply_v(c)

    def apply_h_grid(self, c: np.ndarray) -> np.ndarray:
        """Same operator with V applied pointwise on the polar grid and projected (check).
        c: mode coefficients (2M+1, nmax)."""
        d = self.disc
        return self.h0 * c + d.to_modes(d.to_grid(c) * self.pot)

    def chebyshev_degree(self, L: float, tol: float) -> tuple[int, float, float, float]:
        """L: step length [m]; tol: the series stops where |J_n(t)| <= tol."""
        emin, emax = self._spectral_bounds()
        dE = 0.5 * (emax - emin) * CHEB_MARGIN
        t = dE * L
        n = int(t) + 8
        while abs(jv(n, t)) > tol:
            n += 8
        return n, t, 0.5 * (emax + emin), dE

    def _spectral_bounds(self) -> tuple[float, float]:
        vmax = float(np.max(np.abs(self.pot))) if self.rb is not None else 0.0
        vmax += self.k * float(np.hypot(*self._cpp)) * self.disc.aw
        if self._dr is not None:            # Hadamard term: |V| <= |scale| max_m sum_n j^2 sum_p |dr_hat|
            vmax += abs(self.fig_scale) * self.disc.wall_s2 * float(np.sum(np.abs(self._dr)))
        hr = self.h0.real[self.disc.valid]
        return min(0.0, float(hr.min())) - vmax, float(hr.max()) + vmax

    def propagate(self, c: np.ndarray, L: float, tol: float = CHEB_TOL, z0: float = 0.0) -> tuple[np.ndarray, int]:
        """exp(-iHL) c: diagonal for a straight bore without figure; otherwise the Chebyshev series in
        sub-steps short enough that the damping stays small on the scaled spectrum (gamma_max dL <= CHEB_DAMP)
        and, for a z-dependent figure, no longer than dz (figure taken at each sub-step midpoint).
        c: mode coefficients (2M+1, nmax); L: length [m]; tol: Chebyshev cut |J_n| <= tol; z0: start z [m]."""
        if self.rb is None and self.figure is None and self.path is None and self.wall_loss is None:
            return c * np.exp(-1j * self.h0 * L), 0
        steps = max(1, int(math.ceil(float(np.max(-self.h0.imag)) * L / CHEB_DAMP)))
        if (self.figure is not None and self.figure.z_dependent) or self.path is not None:
            steps = max(steps, int(math.ceil(L / self.dz - 1e-9)))
        dl, total = L / steps, 0
        try:
            for s in range(steps):
                zm = z0 + (s + 0.5) * dl
                if self.figure is not None or self.wall_loss is not None:
                    dr = self.figure.coeffs(zm) if self.figure is not None else np.zeros(1, complex)
                    if self.wall_loss is not None:
                        dr = dr.astype(complex)
                        dr[(dr.size - 1) // 2] += self.wall_loss
                    self._dr = dr
                if self.path is not None:
                    self._cpp = self.path.derivs(zm)[2]
                c, deg = self.propagate_chebyshev(c, dl, tol)
                total += deg
        finally:
            self._dr = None
            self._cpp = np.zeros(2)
        return c, total

    def propagate_chebyshev(self, c: np.ndarray, L: float, tol: float = CHEB_TOL) -> tuple[np.ndarray, int]:
        """exp(-iHL) c as a Chebyshev series (Tal-Ezer & Kosloff): exp(-i t x) =
        sum_n (2 - delta_n0) (-i)^n J_n(t) T_n(x) on the spectrum scaled to [-1, 1].
        c: mode coefficients (2M+1, nmax); L: length [m]; tol: Chebyshev cut |J_n| <= tol."""
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

    def lab_field(self, *, c: np.ndarray, z: float, xg: np.ndarray, yg: np.ndarray, dr: float) -> np.ndarray:
        """Modes at z -> lab field on the Cartesian mesh (xg, yg) inside the disc with the
        lab-frame phase exp(i k c'(z).x' + i g).
        c: mode coefficients (2M+1, nmax) at z [m]; xg, yg: 1-D lab mesh axes [m]; dr: fine radial step [m]."""
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


def fb_jmax_rule(*, k: float, aw: float, theta_cut: float) -> int:
    return int(math.ceil(k * aw * theta_cut))

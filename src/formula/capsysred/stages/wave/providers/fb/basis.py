"""Disc basis of the provider fb: Disc, the wall offset and the damping of the modes, the roots of the characteristic
equation and the rule of jmax."""

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import fft as sfft
from scipy.special import jn_zeros, jv, kve

MODEL_CLASS_FB = "dir-ell"
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


def fb_jmax_rule(*, k: float, aw: float, theta_cut: float) -> int:
    return int(math.ceil(k * aw * theta_cut))

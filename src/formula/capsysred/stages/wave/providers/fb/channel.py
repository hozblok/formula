"""One bore of the provider fb: FBChannel with the wall figure and the axis path."""

import math

import numpy as np
from scipy.special import jv

from ...physics import spot_profile
from .basis import Disc

CHEB_TOL = 1e-12        # default cut |J_n(t)| <= CHEB_TOL of the Chebyshev series of exp(-iHL)
CHEB_MARGIN = 1.02      # widening of the spectrum half-width of H scaled to [-1, 1]
CHEB_DAMP = 1.6         # max damping gamma_max dL of one Chebyshev sub-step


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

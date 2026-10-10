"""Stage 16 provider `fb`: Fourier-Bessel modes of circular bores (cylinder, torus arc).

Paraxial scalar wave in the co-moving frame of the bent axis c(z) = c0 + t z^2 / (2R)
(t = unit vector toward the bend centre):  i d/dz psi = -(1/2k) Lap psi + (k/R) xi psi,
xi along t; Dirichlet wall at r = a + Re(ell), ell = i / (k q0), q0 = sqrt(-2 delta + 2 i beta)
(DIR-ell); Im(ell) enters as the first-order per-mode damping kappa^2 Im(ell) / (a k).
Longitudinal step: Strang split-step (fb_dz) or a Chebyshev series of exp(-iHL) (no z
error; z-independent H).  Exit field in the lab frame on a Cartesian lattice with the phase
exp(i k c'(L).x' + i g), g = k L^3 / (6 R^2); screens by angular-spectrum drift on the
lattice and exact cell integrals of the band-limited field (Fourier box), reference cell as
Stage 14.  Conventions as the uisk provider: W = <E(P) E*(P_ref)>, e^{-i omega t}, carrier
e^{+ikz} dropped; entrance field exp(ik|x - s|^2 / 2 z_in) / (i lambda z_in).
"""

import math
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import fft as sfft
from scipy.special import jn_zeros, jv

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


# ----------------------------------------------------------------- disc basis

class Disc:
    """Dirichlet disc basis J_|m|(j_mn r / aw) e^{i m phi}, all zeros j_mn < jmax; Gauss-Legendre
    radial quadrature, FFT in phi; padded arrays over |m|."""

    def __init__(self, aw, jmax, nr=None, nphi=None):
        self.aw, self.jmax = float(aw), int(jmax)
        zeros, m = [], 0
        while True:
            z = jn_zeros(m, max(8, int(jmax / math.pi) + 4))
            z = z[z < jmax]
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
        for mm, z in enumerate(zeros):
            kap, n = z / self.aw, z.size
            Bm = jv(mm, np.outer(self.r, kap))
            norm = 0.5 * self.aw * self.aw * jv(mm + 1, z) ** 2
            self.Bp[mm, :, :n] = Bm
            self.Pp[mm, :n, :] = (Bm * (self.wr * self.r)[:, None]).T / norm[:, None]
            self.kap2p[mm, :n] = kap * kap
            self.normp[mm, :n] = 2.0 * math.pi * norm
            self.validp[mm, :n] = True
        self.B = self.Bp[self.am]
        self.P = self.Pp[self.am]
        self.kap2 = self.kap2p[self.am]
        self.mnorm = self.normp[self.am]
        self.valid = self.validp[self.am]
        self.kap2max = float(self.kap2.max())
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


# ----------------------------------------------------------------- one bore

class FBChannel:
    """One circular bore: axis c(z) = c0 + t z^2/(2R) (t toward the bend centre; R None =
    straight); frame coordinates xi along t, eta along t_perp; potential k xi / R."""

    def __init__(self, c0, t, rbend, disc, k, dz, gamma=None):
        self.c0 = np.asarray(c0, float)
        self.t = None if t is None else np.asarray(t, float) / np.linalg.norm(t)
        self.rb = None if rbend is None else float(rbend)
        self.disc, self.k, self.dz = disc, float(k), float(dz)
        xi = disc.r[:, None] * np.cos(disc.phi)[None, :]
        cpp = 0.0 if self.rb is None else 1.0 / self.rb
        self.pot = self.k * cpp * xi                                # V on the polar grid
        self.pot_half = np.exp(-1j * self.pot * dz / 2.0)
        self.gamma = np.zeros_like(disc.kap2) if gamma is None else gamma
        self.phase = np.exp(-1j * disc.kap2 / (2.0 * self.k) * dz - self.gamma * dz)
        self.h0 = disc.kap2 / (2.0 * self.k) - 1j * self.gamma     # diagonal part of H
        self.vscale = 0.0 if self.rb is None else self.k / (2.0 * self.rb)   # V = (k/R) r cos phi = vscale r (e^{i phi} + e^{-i phi})

    def frame_axes(self):
        t = self.t if self.t is not None else np.array([1.0, 0.0])
        return t, np.array([-t[1], t[0]])

    def axis(self, z):
        return self.c0 if self.rb is None else self.c0 + self.t * z * z / (2.0 * self.rb)

    def slope(self, z):
        return np.zeros(2) if self.rb is None else self.t * z / self.rb

    def lab_phase_g(self, z):
        return 0.0 if self.rb is None else self.k / 2.0 * (z ** 3 / (3.0 * self.rb ** 2))

    def entrance_field(self, src_xy, z_in):
        """Paraxial point source at (src_xy, -z_in) on the entrance disc (polar grid),
        amplitude 1/(i lambda z_in) like the stage's free-space Fresnel step."""
        t, tp = self.frame_axes()
        d = self.disc
        xi = d.r[:, None] * np.cos(d.phi)[None, :]
        eta = d.r[:, None] * np.sin(d.phi)[None, :]
        X = self.c0[0] + xi * t[0] + eta * tp[0]
        Y = self.c0[1] + xi * t[1] + eta * tp[1]
        lam = 2.0 * math.pi / self.k
        return (np.exp(1j * self.k * ((X - src_xy[0]) ** 2 + (Y - src_xy[1]) ** 2) / (2.0 * z_in))
                / (1j * lam * z_in))

    def entrance_modes(self, src_xy, z_in):
        return self.disc.to_modes(self.entrance_field(src_xy, z_in))

    def propagate_split(self, c, L):
        """Strang split-step over L (nz = round(L/dz) steps; dz adjusted to divide L)."""
        d = self.disc
        nz = max(1, int(round(L / self.dz)))
        if abs(nz * self.dz - L) > 1e-12 * max(L, 1e-9):
            dz = L / nz
            pot_half = np.exp(-1j * self.pot * dz / 2.0)
            phase = np.exp(-1j * d.kap2 / (2.0 * self.k) * dz - self.gamma * dz)
        else:
            pot_half, phase = self.pot_half, self.phase
        G = d.to_grid(c)
        for _ in range(nz):
            G = G * pot_half
            c = d.to_modes(G) * phase
            G = d.to_grid(c)
            G = G * pot_half
        return d.to_modes(G)

    def apply_v(self, c):
        """Galerkin V c through the block-tridiagonal m -> m +- 1 matrices (no transforms)."""
        if self.vscale == 0.0:
            return np.zeros_like(c)
        up, dn = self.disc.shift_blocks()
        out = np.zeros_like(c)
        out[1:] += np.einsum("mij,mj->mi", up[:-1], c[:-1])
        out[:-1] += np.einsum("mij,mj->mi", dn[1:], c[1:])
        return self.vscale * out

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
        return -vmax, self.disc.kap2max / (2.0 * self.k) + vmax

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
    screens = [cap.screen, *cap.screens]
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
    return {"z_in": z_in, "length": length, "z_exit": float(cap.z1), "bores": bores, "screens": scr, "aw": aw, "jmax": jmax,
            "theta_modal": theta_modal, "tilt": tilt, "h": h, "h_auto": h_auto, "half": half,
            "lines": [(float(k), float(wl), float(dl), float(bt)) for k, wl, dl, bt in lines],
            "ells": [(e.real, e.imag) for e in ells], "dz": float(wave["fb_dz"]),
            "propagator": wave["fb_propagator"], "cheb_tol": float(wave["fb_chebyshev_tol"]),
            "dr": float(wave["fb_dr"]), "wall": wave["fb_wall"], "dtype": wave["fb_grid_dtype"],
            "observable": wave["observable"], "per_bore": wave["fb_per_bore_maps"],
            "workers": wave["workers"] or -1, "cache_gb": float(wave["cache_gb"])}


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
    disc = Disc(spec["aw"], spec["jmax"])
    disc.fine_table(spec["dr"], dtype=np.float32)
    chans = {}
    for (k, wl, delta, beta), (er, ei) in zip(spec["lines"], spec["ells"]):
        gamma = disc.kap2 * (ei if spec["wall"] == "dir-ell" else 0.0) / (spec["aw"] * k)   # per-mode damping
        chans[k] = [FBChannel(b["center"], b["toward"], b["rbend"], disc, k, spec["dz"], gamma) for b in spec["bores"]]
    lat = Lattice(spec["half"], spec["h"], dtype=dtype, workers=spec["workers"], edge0=spec["screens"][0]["center"][0] - spec["screens"][0]["edge_x"] / 2,
                  cache_bytes=spec["cache_gb"] * 2 ** 30)
    grids = [ScreenGrid(_PixelCfg(s)) for s in spec["screens"]]
    b = [lat.cell_plan(g)[0] for g in grids]
    samplers = [ScreenSampler(g, s["reference"] or (g.cxf, g.cyf), spec["observable"], bb)
                for g, s, bb in zip(grids, spec["screens"], b)]
    nb = len(spec["bores"])
    per_bore = spec["per_bore"] if spec["per_bore"] is not None else nb <= 2
    _W.update(spec=spec, disc=disc, chans=chans, lat=lat, grids=grids, samplers=samplers,
              nb=nb, per_bore=per_bore, dtype=dtype, t_init=time.time() - t0, degrees=[])


def _node_fields(spec, k, chans, lat, xi, per_bore=True):
    """Exit-plane lattice fields for one source node and one line: one array per bore, or
    their sum in a single array (one lattice-sized buffer instead of one per bore)."""
    out = []
    for ch in chans:
        c0 = ch.entrance_modes(xi, spec["z_in"])
        if spec["propagator"] == "chebyshev":
            cL, deg = ch.propagate_chebyshev(c0, spec["length"], spec["cheb_tol"])
            _W["degrees"].append(deg)
        else:
            cL = ch.propagate_split(c0, spec["length"])
        cz = ch.axis(spec["length"])
        lo = np.searchsorted(lat.x, cz - ch.disc.aw - 2 * lat.h)
        hi = np.searchsorted(lat.x, cz + ch.disc.aw + 2 * lat.h)
        xg, yg = lat.x[lo[0]:hi[0]], lat.x[lo[1]:hi[1]]
        block = ch.lab_field(cL, spec["length"], xg, yg, spec["dr"]).astype(lat.dtype)
        if per_bore or not out:
            out.append(np.zeros((lat.n, lat.n), lat.dtype))
        out[-1][lo[1]:hi[1], lo[0]:hi[0]] += block
    return out


def _run_nodes(args):
    """Accumulate W/I over a chunk of (node, weight) pairs; returns the partial sums."""
    from .wave import Accumulator
    nodes, weights = args
    spec, lat, per_bore = _W["spec"], _W["lat"], _W["per_bore"]
    # fresh sums per task: a pool worker may serve several chunks
    accs = [Accumulator(smp, _W["nb"] if per_bore else 1) for smp in _W["samplers"]]
    _W["degrees"] = []
    z1 = spec["z_exit"]                                     # exit plane (lab z), not z_in + L
    for xi, w in zip(nodes, weights):
        for k, wl, delta, beta in spec["lines"]:
            chans = _W["chans"][k]
            fields = _node_fields(spec, k, chans, lat, xi, per_bore)
            for s, g, smp, acc in zip(spec["screens"], _W["grids"], _W["samplers"], accs):
                d = s["z"] - z1
                cells, subs = [], []
                for E in fields:
                    Ed = lat.drift(E, k, d)
                    cells.append(lat.cells(Ed, g, lat.cell_filter(g)))
                    subs.append(lat.subsamples(Ed, g) if spec["observable"] == "coherent_cell" else None)
                acc.add(w * wl, cells, [None] * len(cells), subs)
    return [(a.I, a.W, a.I_ref, a.I_pixel, a.I_bore, a.G12) for a in accs], len(nodes), _W["t_init"], \
        (int(np.median(_W["degrees"])) if _W["degrees"] else None), int(_W["disc"].nmodes)


# ----------------------------------------------------------------- scene driver

def fb_capillary_scene(sim, wave, lines, rays_paths, log):
    from .wave import _rule, ScreenSampler, Accumulator, _finish, _report_lines
    cfg = sim.cfg
    cap = cfg.capillary
    nodes, weights, rule = _rule(wave, "capillary", cap.source, rays_paths)
    spec = build_spec(sim, wave, lines)
    labels = ["capillary"] + [f"capillary-s{i}" for i in range(1, len(cap.screens) + 1)]
    lam0 = float(sim.lam)
    nyq = spec["h"] <= spec["h_auto"] * (1.0 + 1e-12)
    log(f"  16 [capillary/fb]: {rule}; j_max {spec['jmax']} (theta_modal {spec['theta_modal']*1e3:.3f} mrad, "
        f"tilt {spec['tilt']*1e3:.3f} mrad), lattice h = {m_to_um(spec['h']):.4f} um (auto {m_to_um(spec['h_auto']):.4f}), "
        f"half {m_to_um(spec['half']):.1f} um, {spec['propagator']}" + (f" dz {spec['dz']*1e3:.2f} mm" if spec['propagator'] == 'split_step' else "")
        + f", wall {spec['wall']} ell = {spec['ells'][0][0]*1e9:.3f} + {spec['ells'][0][1]*1e9:.3f}i nm"
        + ("" if nyq else " — note: lattice coarser than the auto step"))
    t0 = time.time()
    jobs = int(wave["fb_jobs"] or 1)
    chunks = [(nodes[i::jobs], weights[i::jobs]) for i in range(jobs)]
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
    for part, n_nodes, t_init, deg, _ in parts:
        for acc, (I, W, I_ref, I_pix, I_bore, G12) in zip(accs, part):
            acc.I += I; acc.W += W; acc.I_ref += I_ref
            if acc.I_pixel is not None and I_pix is not None:
                acc.I_pixel += I_pix
            if acc.I_bore is not None and I_bore is not None:
                acc.I_bore += I_bore
            if acc.G12 is not None and G12 is not None:
                acc.G12 += G12
    degrees = [p[3] for p in parts if p[3] is not None]
    out = []
    a = spec["bores"][0]["radius"]
    for label, s, smp, acc in zip(labels, spec["screens"], samplers, accs):
        d = s["z"] - float(cap.z1)
        nf = {"entrance": a * a / (lam0 * spec["length"]), "source": a * a / (lam0 * spec["z_in"]),
              "exit": (a * a / (lam0 * d)) if d > 0 else None}
        bent = any(b["rbend"] for b in spec["bores"])
        meta = {"scene": label, "provider": "fb",
                "model_class": MODEL_CLASS_FB + (",parabolic-axis" if bent else "") + ("" if spec["wall"] == "dir-ell" else ",no-absorption"),
                "source_rule": rule, "n_nodes": int(len(nodes)),
                "lattice_h_m": spec["h"], "lattice_h_auto_m": spec["h_auto"], "lattice_half_m": spec["half"],
                "lattice_n": lat.n, "theta_modal_rad": spec["theta_modal"], "tilt_rad": spec["tilt"],
                "sampling": {"nyquist_ratio": 2.0 * spec["h"] * (spec["theta_modal"] + spec["tilt"]) / (2.0 * math.pi / max(k for k, *_ in lines))},
                "jmax": spec["jmax"], "n_modes": parts[0][4],
                "propagator": spec["propagator"], "dz_m": spec["dz"] if spec["propagator"] == "split_step" else None,
                "chebyshev_degree_median": (int(np.median(degrees)) if degrees else None),
                "wall": spec["wall"], "wall_offset_nm": [[e[0] * 1e9, e[1] * 1e9] for e in spec["ells"]],
                "aw_m": spec["aw"], "fresnel_numbers": nf, "distance_m": d, "bores": spec["bores"],
                "per_bore_maps": per_bore, "jobs": jobs, "worker_init_s": max(p[2] for p in parts)}
        res = _finish(acc, smp, meta, wave, time.time() - t0)
        res["report"] = _report_lines(label, res, wave, [
            f"- provider fb: j_max = {spec['jmax']} (θ_modal = {spec['theta_modal']*1e3:.3f} mrad, tilt {spec['tilt']*1e3:.3f} mrad), "
            f"{spec['propagator']}" + (f" dz = {spec['dz']*1e3:.2f} mm" if spec["propagator"] == "split_step" else f" degree ≈ {int(np.median(degrees)) if degrees else '?'}")
            + f"; wall DIR-ℓ, ℓ = {spec['ells'][0][0]*1e9:.3f} + {spec['ells'][0][1]*1e9:.3f}i nm"
            + ("" if spec["wall"] == "dir-ell" else " (absorption off)"),
            f"- lattice h = {m_to_um(spec['h']):.4f} µm (auto {m_to_um(spec['h_auto']):.4f}), half-width {m_to_um(spec['half']):.1f} µm, "
            f"{lat.n}² points; cells = exact Fourier box of the band-limited field; screens by angular-spectrum drift",
            f"- N_F: entrance {nf['entrance']:.2f}, source {nf['source']:.2f}, exit "
            + (f"{nf['exit']:.3f}" if nf["exit"] else "— (screen on the exit plane)") + "; ray optics is not used anywhere in this stage",
        ])
        out.append((label, res))
    return out

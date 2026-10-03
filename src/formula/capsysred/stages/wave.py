"""Stage 16: wave-optics degree of coherence (YAML section `wave_estimator`).

Baseline estimator: one positive source quadrature; complex W(P, P_ref) and
I(P) accumulate from the same fields (PSD by construction). Providers:
`free` (source -> screen Fresnel) and `uisk` (unfolded-image spectral
Kirchhoff: straight regular-polygon bores, plane-wave r(kappa) per bounce,
Kirchhoff entrance/exit masks). Paraxial scalar optics, carrier e^{+ikz}
dropped, phasor e^{-i omega t}. Rows: stage16/[screen-N/|free/]mu-wave.jsonl.
"""

import json
import math
import os
import shutil
import time

import numpy as np
from scipy import fft as sfft
from scipy.special import fresnel

from ... import __version__
from .. import render, rays_v3
from ..screen import ScreenGrid
from ..shared.units import m_to_um

RESULT_DIR = "stage16"
MODEL_CLASS = {"free": "free-space", "uisk": "kirchhoff-rims"}
# reference intensity below this fraction of max I = deep diffraction tail: the lattice
# representation of the aperture edges dominates W
FAR_TAIL_RATIO = 1e-3
UNITS = ("|E|^2 of unit-amplitude point-source Fresnel fields (1/(i lambda z) per step), "
         "source weights normalized to 1, spectral weights normalized to 1; arbitrary but "
         "consistent within one run; I_coherent (area^2 scale) and I_pixel (area scale) "
         "are different detector quantities")


# ----------------------------------------------------------------- physics

def fresnel_r(theta, delta, beta):
    """Plane-wave amplitude (theta - q)/(theta + q), q = sqrt(theta^2 - 2delta + 2i beta), Im q >= 0."""
    theta = np.asarray(theta, dtype=float)
    q = np.sqrt(theta.astype(complex) ** 2 - 2.0 * delta + 2j * beta)
    q = np.where(q.imag < 0.0, -q, q)
    return (theta - q) / (theta + q)


def cell_kernel(edges_lo, edges_hi, xs, k, d):
    """int_lo^hi exp(ik (P - x)^2 / 2d) dP for every (cell, x): Fresnel integrals."""
    s = math.sqrt(k / (math.pi * d))
    s_hi, c_hi = fresnel((edges_hi[:, None] - xs[None, :]) * s)
    s_lo, c_lo = fresnel((edges_lo[:, None] - xs[None, :]) * s)
    return math.sqrt(math.pi * d / k) * ((c_hi - c_lo) + 1j * (s_hi - s_lo))


def point_kernel(pts, xs, k, d):
    """exp(ik (P - x)^2 / 2d) for every (point, x)."""
    return np.exp(1j * k * (pts[:, None] - xs[None, :]) ** 2 / (2.0 * d))


def _f1(u, k, d):
    """int_0^u exp(ik t^2/2d) dt."""
    s = math.sqrt(k / (math.pi * d))
    si, ci = fresnel(u * s)
    return math.sqrt(math.pi * d / k) * (ci + 1j * si)


def _f2(u, k, d):
    """int_0^u F1(t) dt = u F1(u) - (d/ik)(exp(ik u^2/2d) - 1)."""
    return u * _f1(u, k, d) - (d / (1j * k)) * (np.exp(1j * k * u * u / (2.0 * d)) - 1.0)


def point_cell_kernel(pts, xs, h, k, d):
    """Mean over the exit cell [x - h/2, x + h/2] of exp(ik (P - x')^2 / 2d): exact in x',
    so a fast chirp (far reference point) no longer aliases on the exit lattice."""
    lo, hi = xs[None, :] - h / 2.0, xs[None, :] + h / 2.0
    return (_f1(pts[:, None] - lo, k, d) - _f1(pts[:, None] - hi, k, d)) / h


def cell_cell_kernel(edges_lo, edges_hi, xs, h, k, d):
    """(1/h) int_cell_P int_cell_x exp(ik (P - x')^2 / 2d) dx' dP via the second antiderivative."""
    a, b = edges_lo[:, None], edges_hi[:, None]
    c, e = xs[None, :] - h / 2.0, xs[None, :] + h / 2.0
    return (_f2(b - c, k, d) - _f2(b - e, k, d) - _f2(a - c, k, d) + _f2(a - e, k, d)) / h


# ----------------------------------------------------------------- source

def source_rule(src, wave):
    """Positive quadrature of the source: (nodes (N, 2) [m], weights (N,), label)."""
    shape, size = src.shape, float(src.size)
    cx, cy = float(src.position[0]), float(src.position[1])
    n_req = int(wave["source_nodes"])
    if shape == "grid":                     # before the degenerate-size branch: size 0 = equal weights
        n, s = src.grid_n, float(src.grid_step)
        half = (n - 1) / 2.0
        rot = math.radians(getattr(src, "grid_rot_deg", 0.0))
        c, q = math.cos(rot), math.sin(rot)
        sites = np.array([((i - half) * s * c - (j - half) * s * q,
                           (i - half) * s * q + (j - half) * s * c)
                          for i in range(n) for j in range(n)])
        r_max = getattr(src, "grid_r_max", None)
        if r_max is not None:
            sites = sites[np.hypot(sites[:, 0], sites[:, 1]) <= r_max]
        w = (np.exp(-(sites ** 2).sum(1) / (2.0 * size * size)) if size > 0.0
             else np.ones(len(sites)))
        return _checked(sites + np.array([cx, cy]), w, f"grid: {len(sites)} lattice nodes")
    if shape == "point" or size <= 0.0:
        return np.array([[cx, cy]]), np.array([1.0]), "point"
    if shape == "disk":
        n_r = max(1, math.ceil(math.sqrt(n_req / 4.0)))
        n_phi = 4 * n_r
        t, wt = np.polynomial.legendre.leggauss(n_r)      # uniform in r^2
        rad = size * np.sqrt((t + 1.0) / 2.0)
        phi = (np.arange(n_phi) + 0.5) * 2.0 * math.pi / n_phi
        nodes = np.stack([cx + rad[:, None] * np.cos(phi)[None, :],
                          cy + rad[:, None] * np.sin(phi)[None, :]], -1).reshape(-1, 2)
        w = np.repeat(wt / (2.0 * n_phi), n_phi)
        return _checked(nodes, w, f"disk: Gauss-Legendre(r^2) {n_r} x {n_phi} angles")
    if shape == "gaussian":
        n = max(1, math.ceil(math.sqrt(n_req)))
        t, wt = np.polynomial.hermite_e.hermegauss(n)
        gx, gy = np.meshgrid(t, t, indexing="ij")
        nodes = np.stack([cx + size * gx, cy + size * gy], -1).reshape(-1, 2)
        w = (wt[:, None] * wt[None, :]).reshape(-1)
        return _checked(nodes, w, f"gaussian: Gauss-Hermite {n} x {n}")
    raise ValueError(f"wave_estimator: unsupported source shape {shape!r}")


def _checked(nodes, w, label):
    w = np.asarray(w, dtype=float)
    if len(nodes) == 0 or not np.all(np.isfinite(w)) or np.any(w < 0.0) or w.sum() <= 0.0:
        raise ValueError(f"wave_estimator: source rule {label!r} has no nodes or invalid weights")
    return np.asarray(nodes, dtype=float), w / w.sum(), label


def recorded_rule(archives, scene, src):
    """Empirical source of rays v3 archives: every recorded origin of every archive, weight 1/N."""
    nodes = []
    for archive in archives:
        if not rays_v3.is_v3(archive):
            raise ValueError(f"{archive}: recorded_origins needs a v3 rays archive")
        index = rays_v3.load_index(archive)
        origins = rays_v3.origins(archive, index, scene)
        if not origins or any(o is None for o in origins):
            raise ValueError(f"{archive}: scene {scene!r} has no recorded origins")
        zs = {round(float(o[2]), 12) for o in origins}
        if zs != {round(float(src.position[2]), 12)}:
            raise ValueError(f"{archive}: origin z {sorted(zs)} differs from the "
                             f"source z {float(src.position[2])}")
        nodes += [[float(o[0]), float(o[1])] for o in origins]
    nodes = np.array(nodes)
    return nodes, np.full(len(nodes), 1.0 / len(nodes)), \
        f"recorded_origins: {len(nodes)} modes from {len(archives)} archive(s)"


# ----------------------------------------------------------------- polygon bores

class Polygon:
    """Regular-polygon bore: face k has unit normal at rotation + 2 pi k / n, plane (p - c).n = a."""

    def __init__(self, bore):
        self.n = int(bore["sides"])
        self.a = float(bore["radius"])
        self.c = np.array([float(bore["center"][0]), float(bore["center"][1])])
        ang = float(bore["rotation"]) + 2.0 * np.pi * np.arange(self.n) / self.n
        self.normals = np.stack([np.cos(ang), np.sin(ang)], -1)
        self.verts = np.array([
            self.c + np.linalg.solve(np.stack([self.normals[k], self.normals[(k + 1) % self.n]]),
                                     [self.a, self.a])
            for k in range(self.n)])

    def reflect_point(self, k, p):
        n = self.normals[k]
        s = (p - self.c) @ n - self.a
        return p - 2.0 * s[..., None] * n

    def reflect_vector(self, k, v):
        n = self.normals[k]
        return v - 2.0 * (v @ n)[..., None] * n


class Family:
    """One reflection sequence: image polygon, image source map, unfolded face normals."""

    def __init__(self, poly, seq):
        self.seq = tuple(seq)
        self.poly = poly
        self.center = poly.c.copy()
        self.verts = poly.verts.copy()
        self.face_normals = poly.normals.copy()
        for i in reversed(self.seq):                  # U = s_i1 o ... o s_im
            self.center = poly.reflect_point(i, self.center)
            self.verts = poly.reflect_point(i, self.verts)
            self.face_normals = poly.reflect_vector(i, self.face_normals)
        self.unfolded = []
        for t, i in enumerate(self.seq):              # n'_t = s_i1 ... s_i(t-1) n_it
            v = poly.normals[i].copy()
            for j in reversed(self.seq[:t]):
                v = poly.reflect_vector(j, v)
            self.unfolded.append(v)

    def same_tile(self, other, a):
        """Same image tile: centres within 1e-9 a, image normals within 1e-9 (dimensionless)."""
        return (np.allclose(self.center, other.center, rtol=0.0, atol=1e-9 * a)
                and np.allclose(self.face_normals, other.face_normals, rtol=0.0, atol=1e-9))

    def map_point(self, p):
        for i in reversed(self.seq):
            p = self.poly.reflect_point(i, p)
        return p


def enumerate_families(poly, max_bounces):
    """Reflection sequences up to max_bounces; sequences that unfold onto the same
    tile (commuting reflections, e.g. both corner orders of a square) are one family."""
    seqs, frontier = [()], [()]
    for _ in range(max_bounces):
        new = [s + (i,) for s in frontier for i in range(poly.n) if not s or s[-1] != i]
        seqs += new
        frontier = new
    out = []
    for s in seqs:
        fam = Family(poly, s)
        if not any(fam.same_tile(f, poly.a) for f in out):
            out.append(fam)
    return out


def sub_offsets(h, supersample):
    return ((np.arange(supersample) + 0.5) / supersample - 0.5) * h


def sub_lattice_mask(xs, ys, center, normals, a, supersample, strip=256):
    """Indicator of the convex polygon (p - center).n_k <= a on the S x S sub-lattice,
    shape (nx, S, ny, S); built in x-strips so the float temporaries stay small."""
    h = xs[1] - xs[0] if len(xs) > 1 else (ys[1] - ys[0] if len(ys) > 1 else 1.0)
    off = sub_offsets(h, supersample)
    Y = (ys[:, None] + off[None, :]).reshape(-1)
    out = np.empty((len(xs), supersample, len(ys), supersample), dtype=bool)
    for i0 in range(0, len(xs), strip):
        X = (xs[i0:i0 + strip, None] + off[None, :]).reshape(-1)
        inside = np.ones((X.size, Y.size), dtype=bool)
        for n in normals:
            inside &= ((X[:, None] - center[0]) * n[0] + (Y[None, :] - center[1]) * n[1]) <= a
        out[i0:i0 + strip] = inside.reshape(-1, supersample, len(ys), supersample)
    return out


def coverage(xs, ys, center, normals, a, supersample):
    """Fractional cell coverage of the polygon (mean of the sub-lattice indicator)."""
    return sub_lattice_mask(xs, ys, center, normals, a, supersample).mean(axis=(1, 3))


class PackedMask:
    """Sub-lattice indicator stored bit-packed per (s, t) offset: 1 bit per sub-cell."""

    def __init__(self, inside):
        self.shape = (inside.shape[0], inside.shape[2])
        self.s = inside.shape[1]
        self.bits = [[np.packbits(inside[:, s, :, t]) for t in range(self.s)]
                     for s in range(self.s)]
        self.coverage = inside.mean(axis=(1, 3))

    def slab(self, s, t):
        n = self.shape[0] * self.shape[1]
        return np.unpackbits(self.bits[s][t], count=n).reshape(self.shape).astype(bool)


def _fast_len(n):
    return sfft.next_fast_len(max(int(n), 1), real=False)


class BoreSolver:
    """UISK for one polygon bore on the global lattice x = i h, y = j h.

    lam_h: the shortest wavelength of the run (lattice step); lam_reach: the
    longest (Fresnel tails in the box margin)."""

    def __init__(self, bore, z_in, length, theta_max, lam_h, wave, lam_reach=None):
        lam_reach = lam_h if lam_reach is None else lam_reach
        self.poly = Polygon(bore)
        self.z_in, self.length = z_in, length
        self.h = wave["grid_dx"] or lam_h / (2.0 * theta_max * wave["angle_margin"])
        self.supersample = int(wave["mask_supersample"])
        # box margin: geometric spread + Fresnel tails, scaled by `pad`; also the
        # angular band limit of the propagation (waves that would wrap around)
        self.reach = max(wave["pad"] * (theta_max * length + 4.0 * math.sqrt(lam_reach * length)),
                         4.0 * self.h)
        self.theta_max, self.theta_cut = theta_max, self.reach / length
        lo, hi = self.poly.verts.min(0), self.poly.verts.max(0)
        self.win = self._window(lo - 2 * self.h, hi + 2 * self.h)      # real exit region
        self.xs, self.ys = self._coords(self.win)
        self.exit_mask = coverage(self.xs, self.ys, self.poly.c, self.poly.normals,
                                  self.poly.a, self.supersample)
        self.families, self.pruned = [], 0
        for fam in enumerate_families(self.poly, int(wave["max_bounces"])):
            flo, fhi = fam.verts.min(0), fam.verts.max(0)
            gap = np.maximum(0.0, np.maximum(flo - hi, lo - fhi))
            if math.hypot(*gap) > self.reach:            # cannot reach the exit region
                self.pruned += 1
                continue
            g = self._window(np.minimum(flo, lo) - self.reach, np.maximum(fhi, hi) + self.reach)
            fam.grid = g
            fam.xs, fam.ys = self._coords(g)
            fam.mask = PackedMask(sub_lattice_mask(fam.xs, fam.ys, fam.center, fam.face_normals,
                                                  self.poly.a, self.supersample))
            fam.slice = (slice(self.win[0] - g[0], self.win[1] - g[0]),
                         slice(self.win[2] - g[2], self.win[3] - g[2]))
            self.families.append(fam)
        self.cache = {}

    def _window(self, lo, hi):
        i0, j0 = math.floor(lo[0] / self.h), math.floor(lo[1] / self.h)
        nx = _fast_len(math.ceil(hi[0] / self.h) - i0 + 1)
        ny = _fast_len(math.ceil(hi[1] / self.h) - j0 + 1)
        return (i0, i0 + nx, j0, j0 + ny)

    def _coords(self, g):
        return (np.arange(g[0], g[1]) * self.h, np.arange(g[2], g[3]) * self.h)

    def transfer(self, fam, k, delta, beta):
        """exp(-i kappa^2 L / 2k) prod_j r(|kappa . n'_j| / k) on the family grid."""
        key = (id(fam), k)
        if key in self.cache:
            return self.cache[key]
        nx, ny = fam.grid[1] - fam.grid[0], fam.grid[3] - fam.grid[2]
        kx = 2.0 * np.pi * sfft.fftfreq(nx, self.h)
        ky = 2.0 * np.pi * sfft.fftfreq(ny, self.h)
        H = np.exp(-1j * (kx[:, None] ** 2 + ky[None, :] ** 2) * self.length / (2.0 * k))
        for n in fam.unfolded:
            H = H * fresnel_r(np.abs(kx[:, None] * n[0] + ky[None, :] * n[1]) / k, delta, beta)
        # band limit: waves steeper than reach/L would leave the box and wrap around;
        # raised cosine per axis over the upper half of the margin band above theta_max
        # (a separable window keeps the square problem exactly separable)
        t_hi = self.theta_cut
        t_lo = t_hi - 0.5 * (t_hi - self.theta_max)
        if t_lo > self.theta_max:
            def window(kk):
                theta = np.abs(kk) / k
                return np.where(theta <= t_lo, 1.0,
                                np.where(theta >= t_hi, 0.0,
                                         0.5 * (1.0 + np.cos(np.pi * (theta - t_lo) / (t_hi - t_lo)))))
            H = H * (window(kx)[:, None] * window(ky)[None, :])
        return H

    def cache_transfer(self, k, delta, beta, budget_bytes):
        need = sum(16 * (f.grid[1] - f.grid[0]) * (f.grid[3] - f.grid[2]) for f in self.families)
        if need <= budget_bytes:
            for fam in self.families:
                self.cache[(id(fam), k)] = self.transfer(fam, k, delta, beta)

    def exit_field(self, xi, k, delta, beta, workers):
        """Field on the real exit window for a point source at xi in the source plane."""
        lam = 2.0 * np.pi / k
        out = np.zeros((self.win[1] - self.win[0], self.win[3] - self.win[2]), dtype=complex)
        off = sub_offsets(self.h, self.supersample)
        for fam in self.families:
            src = fam.map_point(np.asarray(xi, dtype=float))
            # cell average of chirp x indicator on the sub-lattice (anti-aliased edges)
            cx = np.exp(1j * k * (fam.xs[:, None] + off[None, :] - src[0]) ** 2 / (2.0 * self.z_in))
            cy = np.exp(1j * k * (fam.ys[:, None] + off[None, :] - src[1]) ** 2 / (2.0 * self.z_in))
            u = np.zeros(fam.mask.shape, dtype=complex)
            tmp = np.empty(fam.mask.shape, dtype=complex)
            for s in range(self.supersample):
                for t in range(self.supersample):
                    np.multiply(cx[:, s, None], cy[None, :, t], out=tmp)
                    tmp *= fam.mask.slab(s, t)
                    u += tmp
            u *= 1.0 / (self.supersample ** 2 * 1j * lam * self.z_in)
            u = sfft.ifft2(sfft.fft2(u, workers=workers) * self.transfer(fam, k, delta, beta),
                           workers=workers)
            out += u[fam.slice]
        return out * self.exit_mask


# ----------------------------------------------------------------- screens

class ScreenSampler:
    """Fresnel step exit lattice -> screen pixels: point values or cell integrals."""

    def __init__(self, grid: ScreenGrid, ref_xy, observable, subsamples):
        self.grid, self.observable, self.sub = grid, observable, int(subsamples)
        self.nx, self.ny = grid.nx, grid.ny
        self.hx, self.hy = grid.exf / grid.nx, grid.eyf / grid.ny
        self.px = np.array(grid.xs())
        self.py = np.array(grid.ys())
        self.ref_xy = (float(ref_xy[0]), float(ref_xy[1]))
        self.ref_inside = (grid.x0f <= self.ref_xy[0] < grid.x0f + grid.exf
                           and grid.y0f <= self.ref_xy[1] < grid.y0f + grid.eyf)
        if observable == "coherent_cell" and not self.ref_inside:
            raise ValueError(
                f"wave_estimator: reference ({m_to_um(self.ref_xy[0]):g}, "
                f"{m_to_um(self.ref_xy[1]):g}) um lies outside the screen window; "
                "observable coherent_cell needs the reference cell on the screen")
        # reference cell (coherent_cell) or the flagged pixel of an inside point reference
        self.ref_index = None
        if self.ref_inside:
            iy, ix = divmod(grid.ref_pixel(self.ref_xy), self.nx)
            self.ref_index = (ix, iy)
        off = (np.arange(self.sub) + 0.5) / self.sub - 0.5
        self.sub_x = (self.px[:, None] + off[None, :] * self.hx).reshape(-1)
        self.sub_y = (self.py[:, None] + off[None, :] * self.hy).reshape(-1)

    def reference_cell_um(self):
        if self.observable != "coherent_cell":
            return None
        ix, iy = self.ref_index
        return [m_to_um(self.px[ix]), m_to_um(self.py[iy])]

    def kernels(self, xs, ys, k, d, h):
        """(Kx, Ky, Kx_ref, Ky_ref, Sx, Sy): pixel, reference-point and sub-pixel kernels;
        every kernel is integrated exactly over the exit lattice cell of width h."""
        if self.observable == "coherent_cell":
            kx = cell_cell_kernel(self.px - self.hx / 2, self.px + self.hx / 2, xs, h, k, d)
            ky = cell_cell_kernel(self.py - self.hy / 2, self.py + self.hy / 2, ys, h, k, d)
            kxr = kyr = None
            sx, sy = point_cell_kernel(self.sub_x, xs, h, k, d), point_cell_kernel(self.sub_y, ys, h, k, d)
        else:
            kx, ky = point_cell_kernel(self.px, xs, h, k, d), point_cell_kernel(self.py, ys, h, k, d)
            kxr = point_cell_kernel(np.array([self.ref_xy[0]]), xs, h, k, d)
            kyr = point_cell_kernel(np.array([self.ref_xy[1]]), ys, h, k, d)
            sx = sy = None
        return kx, ky, kxr, kyr, sx, sy

    def fields(self, u, kern, k, d, h):
        """Screen field [ix, iy], reference-point field (point mode) and the complex
        sub-pixel field [ix*S+s, iy*S+t] (cell mode; None otherwise)."""
        kx, ky, kxr, kyr, sx, sy = kern
        lam = 2.0 * np.pi / k
        c = h * h / (1j * lam * d)
        e = c * (kx @ u @ ky.T)
        ref = None if kxr is None else complex(c * (kxr @ u @ kyr.T)[0, 0])
        es = None if sx is None else c * (sx @ u @ sy.T)
        return e, ref, es

    def free_fields(self, xi, k, d):
        """Direct source -> screen fields of a point source at xi (free scene)."""
        lam = 2.0 * np.pi / k
        c = 1.0 / (1j * lam * d)
        x0, y0 = np.array([xi[0]]), np.array([xi[1]])
        if self.observable == "coherent_cell":
            fx = cell_kernel(self.px - self.hx / 2, self.px + self.hx / 2, x0, k, d)[:, 0]
            fy = cell_kernel(self.py - self.hy / 2, self.py + self.hy / 2, y0, k, d)[:, 0]
            gx = point_kernel(self.sub_x, x0, k, d)[:, 0]
            gy = point_kernel(self.sub_y, y0, k, d)[:, 0]
            return c * fx[:, None] * fy[None, :], None, c * gx[:, None] * gy[None, :]
        fx = point_kernel(self.px, x0, k, d)[:, 0]
        fy = point_kernel(self.py, y0, k, d)[:, 0]
        ref = (c * point_kernel(np.array([self.ref_xy[0]]), x0, k, d)[0, 0]
               * point_kernel(np.array([self.ref_xy[1]]), y0, k, d)[0, 0])
        return c * fx[:, None] * fy[None, :], complex(ref), None

    def pixel_intensity(self, es):
        """integral_cell |E|^2 dA from the sub-pixel field of the TOTAL wave."""
        return (np.abs(es) ** 2).reshape(self.nx, self.sub, self.ny, self.sub).mean(axis=(1, 3)) \
            * (self.hx * self.hy)


class Accumulator:
    """Joint W/I sums of one screen from the same fields (weights >= 0)."""

    def __init__(self, sampler, n_bores):
        self.s = sampler
        nx, ny = sampler.nx, sampler.ny
        self.I = np.zeros((nx, ny))
        self.W = np.zeros((nx, ny), dtype=complex)
        self.I_ref = 0.0
        self.I_pixel = np.zeros((nx, ny)) if sampler.observable == "coherent_cell" else None
        self.I_bore = np.zeros((n_bores, nx, ny)) if n_bores > 1 else None
        self.G12 = np.zeros((nx, ny), dtype=complex) if n_bores == 2 else None
        self.ref_index = sampler.ref_index

    def add(self, w, fields, refs, subfields):
        """fields/refs/subfields: per-bore screen fields, reference-point fields (point
        mode) and sub-pixel fields (cell mode); bores are summed coherently first."""
        e = fields[0] if len(fields) == 1 else sum(fields)
        e_ref = sum(refs) if refs[0] is not None else e[self.ref_index]
        self.I += w * np.abs(e) ** 2
        self.W += w * e * np.conj(e_ref)
        self.I_ref += w * abs(e_ref) ** 2
        if self.I_pixel is not None and subfields[0] is not None:
            es = subfields[0] if len(subfields) == 1 else sum(subfields)
            self.I_pixel += w * self.s.pixel_intensity(es)
        if self.I_bore is not None:
            for b, eb in enumerate(fields):
                self.I_bore[b] += w * np.abs(eb) ** 2
        if self.G12 is not None:
            self.G12 += w * fields[0] * np.conj(fields[1])


# ----------------------------------------------------------------- stage driver

def preflight_wave_output(out_dir):
    """Fail before any work when the result directory already exists (no clobber)."""
    final = os.path.join(out_dir, RESULT_DIR)
    if os.path.lexists(final):
        raise ValueError(f"{final}: publication conflict; remove it manually")


def _raw_spectral_weights(cfg):
    """Weights as configured, before spectral_lines() normalizes them."""
    spec = cfg.spectrum
    mode = spec.get("mode", "monochromatic")
    if mode == "lines":
        return [float(l.get("weight", 1.0)) for l in spec["lines"]]
    if mode == "table":
        from ..spectrum import _table_pairs
        return [w for _, w in _table_pairs(spec["file"], cfg.precision)]
    return [float(ln.weight) for ln in spectral_lines_of(cfg)]


def spectral_lines_of(cfg):
    from ..spectrum import spectral_lines
    return spectral_lines(cfg.spectrum, cfg.energy_kev)


def preflight_wave_inputs(sim, wave):
    """Stage-16 input contract, cheap and before any heavy work: raw spectral weights,
    screens past the exit, coherent-cell references inside their windows."""
    weights = _raw_spectral_weights(sim.cfg)
    energies = [float(ln.e_kev) for ln in sim.lines]
    if (not all(math.isfinite(w) and w >= 0.0 for w in weights) or sum(weights) <= 0.0
            or not all(math.isfinite(e) and e > 0.0 for e in energies)):
        raise ValueError("wave_estimator: spectral lines need finite non-negative weights "
                         "with a positive sum and finite positive energies")
    cfg = sim.cfg
    scenes = []
    if cfg.free_source is not None and wave["provider"] in ("auto", "free"):
        scenes.append([cfg.free_screen])
    cap = cfg.capillary
    if wave["provider"] == "free" and cfg.free_source is None:
        raise ValueError("wave_estimator: provider free needs a configured free.source")
    if wave["provider"] == "uisk" and cap is None:
        raise ValueError("wave_estimator: provider uisk needs a configured capillary scene")
    if cap is not None and wave["provider"] in ("auto", "uisk"):
        for bore in cap.bores:
            if bore.get("kind") != "polygon":
                raise ValueError("wave_estimator: provider uisk supports regular-polygon bores "
                                 f"only (sides: n); got kind {bore.get('kind', 'cylinder')!r}")
        for i, scr in enumerate([cap.screen, *cap.screens]):
            if float(scr.z) <= float(cap.z1):
                raise ValueError(f"wave_estimator: capillary screen {i} at z = {float(scr.z)} "
                                 f"is not past the exit z1 = {float(cap.z1)}; a screen on the "
                                 "exit plane is outside the Stage-16 MVP")
        scenes.append([cap.screen, *cap.screens])
    for screens in scenes:
        for scr in screens:
            g = ScreenGrid(scr)
            ScreenSampler(g, scr.reference or (g.cxf, g.cyf), wave["observable"], 1)


def _polygon_circumradius(bore):
    return float(bore["radius"]) / math.cos(math.pi / int(bore["sides"]))


def _theta_max(cap, src_nodes, grids, refs=()):
    """Largest paraxial angle the lattice must resolve: entrance cone, screen chirps
    and the chirps of point references (which may lie outside the windows)."""
    z_in = float(cap.z0) - float(cap.source.position[2])
    theta = 0.0
    for bore in cap.bores:
        c = np.array([float(bore["center"][0]), float(bore["center"][1])])
        rc = _polygon_circumradius(bore)
        far = max(float(np.hypot(*(n - c))) for n in src_nodes) + rc
        theta = max(theta, far / z_in)
        for i, g in enumerate(grids):
            d = float(g.z) - float(cap.z1)
            corner = max(abs(g.x0f - c[0]), abs(g.x0f + g.exf - c[0]),
                         abs(g.y0f - c[1]), abs(g.y0f + g.eyf - c[1]))
            if i < len(refs) and refs[i] is not None:
                corner = max(corner, abs(refs[i][0] - c[0]), abs(refs[i][1] - c[1]))
            theta = max(theta, (corner + rc) / d)
    return theta


def run_wave_stage(sim, out_dir, wave, rays_paths=None, log=print):
    """Stage 16 on every configured scene; publishes out_dir/stage16 atomically."""
    preflight_wave_output(out_dir)
    preflight_wave_inputs(sim, wave)
    partial = os.path.join(out_dir, RESULT_DIR + ".partial")
    if os.path.lexists(partial):
        raise ValueError(f"{partial}: stale partial output; remove it manually")
    os.makedirs(partial)
    try:
        return _run(sim, out_dir, partial, wave, rays_paths, log)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)     # never leave a stale partial tree
        raise


def _run(sim, out_dir, partial, wave, rays_paths, log):
    cfg = sim.cfg
    t0 = time.time()
    files, report, results = [], [], {}
    workers = wave["workers"] or -1
    p = cfg.precision
    lines = [(float(ln.k), float(ln.weight),
              float(cfg.material.delta(ln.e_kev, precision=p)) if sim.per_line else sim.delta_f,
              float(cfg.material.beta(ln.e_kev, precision=p)) if sim.per_line else sim.beta_f)
             for ln in sim.lines if float(ln.weight) > 0.0]
    if cfg.free_source is not None and wave["provider"] in ("auto", "free"):
        res = _free_scene(sim, wave, lines, rays_paths)
        results["free"] = res
        files += _emit(partial, "free", res, wave, log)
        report += res["report"]
    cap = cfg.capillary
    if cap is not None and wave["provider"] in ("auto", "uisk"):
        for bore in cap.bores:
            if bore.get("kind") != "polygon":
                raise ValueError("wave_estimator: provider uisk supports regular-polygon bores "
                                 f"only (sides: n); got kind {bore.get('kind', 'cylinder')!r}")
        for label, res in _capillary_scene(sim, wave, lines, rays_paths, workers, log):
            results[label] = res
            sub = "" if label == "capillary" else "screen-" + label.rsplit("-s", 1)[1]
            files += _emit(partial, sub, res, wave, log)
            report += res["report"]
    if not results:
        raise ValueError(f"wave_estimator: no scene matches provider {wave['provider']!r} "
                         "(free scene: auto|free; capillary polygon bores: auto|uisk)")
    os.rename(partial, os.path.join(out_dir, RESULT_DIR))
    return {"files": [RESULT_DIR + "/" + f for f in files], "report": report,
            "results": results, "seconds": time.time() - t0}


def _rule(wave, scene, src, rays_paths):
    if wave["source_mode"] == "recorded_origins":
        if not rays_paths:
            raise ValueError("wave_estimator: source_mode recorded_origins needs a rays archive "
                             "(out/rays-modes or --replay)")
        return recorded_rule(list(rays_paths), scene, src)
    return source_rule(src, wave)


def _free_scene(sim, wave, lines, rays_paths):
    cfg = sim.cfg
    src, scr = cfg.free_source, cfg.free_screen
    nodes, weights, rule = _rule(wave, "free", src, rays_paths)
    grid = ScreenGrid(scr)
    sampler = ScreenSampler(grid, scr.reference or (grid.cxf, grid.cyf),
                            wave["observable"], wave["pixel_subsamples"])
    acc = Accumulator(sampler, 1)
    d = float(scr.z) - float(src.position[2])
    t0 = time.time()
    for k, wl, _, _ in lines:
        for xi, w in zip(nodes, weights):
            e, ref, es = sampler.free_fields(xi, k, d)
            acc.add(w * wl, [e], [ref], [es])
    lam = float(sim.lam)
    meta = {"scene": "free", "provider": "free", "model_class": MODEL_CLASS["free"],
            "source_rule": rule, "n_nodes": int(len(nodes)), "distance_m": d,
            "fresnel_numbers": {"source": float(src.size) ** 2 / (lam * d)}}
    res = _finish(acc, sampler, meta, wave, time.time() - t0)
    res["report"] = _report_lines("free", res, wave, [])
    return res


def _capillary_scene(sim, wave, lines, rays_paths, workers, log):
    cfg = sim.cfg
    cap = cfg.capillary
    src = cap.source
    nodes, weights, rule = _rule(wave, "capillary", src, rays_paths)
    screens = [cap.screen, *cap.screens]
    labels = ["capillary"] + [f"capillary-s{i}" for i in range(1, len(cap.screens) + 1)]
    z_in = float(cap.z0) - float(src.position[2])
    length = float(cap.z1) - float(cap.z0)
    grids = [ScreenGrid(s) for s in screens]
    # samplers first: the cheap reference checks precede the heavy solver grids
    samplers = [ScreenSampler(g, s.reference or (g.cxf, g.cyf), wave["observable"],
                              wave["pixel_subsamples"]) for s, g in zip(screens, grids)]
    refs = [smp.ref_xy for smp in samplers] if wave["observable"] == "point" else ()
    theta_max = _theta_max(cap, nodes, grids, refs)
    lam_h = 2.0 * np.pi / max(k for k, *_ in lines)        # shortest wavelength sets h
    lam_reach = 2.0 * np.pi / min(k for k, *_ in lines)    # longest sets the Fresnel margin
    solvers = [BoreSolver(b, z_in, length, theta_max, lam_h, wave, lam_reach) for b in cap.bores]
    accs = [Accumulator(smp, len(solvers)) for smp in samplers]
    # sampling contract: Nyquist angle lam_h/(2h) against theta_max (ratio > 1 = aliasing)
    h_auto = lam_h / (2.0 * theta_max * wave["angle_margin"])
    sampling = {"h_auto_m": h_auto, "sampling_ratio": 2.0 * solvers[0].h * theta_max / lam_h,
                "angle_margin_effective": lam_h / (2.0 * solvers[0].h * theta_max)}
    sampling = {k: float(v) for k, v in sampling.items()}
    sampling["violation"] = bool(sampling["sampling_ratio"] > 1.0)
    if sampling["violation"]:
        log(f"  16 [capillary]: WARNING explicit grid_dx {m_to_um(solvers[0].h):.4f} um under-samples "
            f"theta_max {theta_max * 1e3:.3f} mrad at lambda_min {lam_h * 1e10:.4f} A "
            f"(Nyquist ratio {sampling['sampling_ratio']:.2f} > 1; auto h = {m_to_um(h_auto):.4f} um)")
    elif solvers[0].h > h_auto * (1.0 + 1e-12):
        log(f"  16 [capillary]: note explicit grid_dx {m_to_um(solvers[0].h):.4f} um above auto "
            f"{m_to_um(h_auto):.4f} um: angle margin {sampling['angle_margin_effective']:.2f} "
            f"instead of {wave['angle_margin']:g}")
    n_fam = sum(len(b.families) for b in solvers)
    n_cells = max((f.grid[1] - f.grid[0]) * (f.grid[3] - f.grid[2])
                  for b in solvers for f in b.families)
    log(f"  16 [capillary]: {rule}; lattice h = {m_to_um(solvers[0].h):.4f} um, "
        f"theta_max = {theta_max * 1e3:.3f} mrad, {n_fam} families "
        f"({sum(b.pruned for b in solvers)} pruned), largest grid {n_cells:,} cells")
    t0 = time.time()
    budget = int(wave["cache_gb"] * 1024 ** 3 / max(len(lines), 1))
    dists = [float(s.z) - float(cap.z1) for s in screens]
    for k, wl, delta, beta in lines:
        for b in solvers:
            b.cache_transfer(k, delta, beta, budget)
        kerns = [[g.kernels(b.xs, b.ys, k, d, b.h) for b in solvers] for g, d in zip(samplers, dists)]
        every = max(1, len(nodes) // 10)
        for j, (xi, w) in enumerate(zip(nodes, weights)):
            exits = [b.exit_field(xi, k, delta, beta, workers) for b in solvers]
            for g, acc, kern_b, d in zip(samplers, accs, kerns, dists):
                fields, refs, subs = [], [], []
                for b, u, kern in zip(solvers, exits, kern_b):
                    e, ref, es = g.fields(u, kern, k, d, b.h)
                    fields.append(e)
                    refs.append(ref)
                    subs.append(es)
                acc.add(w * wl, fields, refs, subs)
            if (j + 1) % every == 0 or j + 1 == len(nodes):
                log(f"    nodes {j + 1}/{len(nodes)}, {time.time() - t0:.0f} s")
    out = []
    a = float(cap.bores[0]["radius"])
    lam0 = float(sim.lam)
    model = MODEL_CLASS["uisk"] + ("" if all(int(b["sides"]) == 4 for b in cap.bores)
                                   else ",first-order-polygon")
    for label, g, acc, d in zip(labels, samplers, accs, dists):
        nf = {"entrance": a * a / (lam0 * length), "source": a * a / (lam0 * z_in),
              "exit": a * a / (lam0 * d)}
        meta = {"scene": label, "provider": "uisk", "model_class": model,
                "source_rule": rule, "n_nodes": int(len(nodes)),
                "lattice_h_m": solvers[0].h, "lattice_wavelength_m": lam_h,
                "theta_max_rad": theta_max,
                "box_margin_m": solvers[0].reach, "theta_cut_rad": solvers[0].theta_cut,
                "sampling": sampling,
                "families": [{"bore": i, "count": len(b.families), "pruned": b.pruned,
                              "sequences": [list(f.seq) for f in b.families],
                              "grids": [[int(v) for v in f.grid] for f in b.families]}
                             for i, b in enumerate(solvers)],
                "max_bounces": int(wave["max_bounces"]), "fresnel_numbers": nf,
                "distance_m": d, "apothem_m": a}
        res = _finish(acc, g, meta, wave, time.time() - t0)
        res["report"] = _report_lines(label, res, wave, [
            f"- lattice h = {m_to_um(solvers[0].h):.4f} µm ({wave['mask_supersample']}× mask "
            f"supersampling), θ_max = {theta_max * 1e3:.3f} mrad, families "
            f"{[len(b.families) for b in solvers]} (pruned {[b.pruned for b in solvers]}), "
            f"max_bounces = {wave['max_bounces']}",
            f"- N_F: entrance {nf['entrance']:.2f}, source {nf['source']:.2f}, exit "
            f"{nf['exit']:.3f}; ray optics is not used anywhere in this stage",
            f"- sampling: h/h_auto = {solvers[0].h / h_auto:.3f}, Nyquist ratio 2hθ_max/λ_min = "
            f"{sampling['sampling_ratio']:.3f}"
            + (" — ALIASING: the lattice under-samples the scene" if sampling["violation"] else ""),
        ])
        out.append((label, res))
    return out


def _finish(acc, sampler, meta, wave, seconds):
    nx, ny = sampler.nx, sampler.ny
    I, W, I_ref = acc.I, acc.W, acc.I_ref
    floor = wave["intensity_floor"] * (float(I.max()) if I.size else 0.0)
    denom = np.sqrt(I * I_ref)
    good = denom > 0.0
    mu = np.full((nx, ny), np.nan + 0j)
    mu[good] = W[good] / denom[good]
    trusted = (I >= floor) & (I_ref >= floor) & good
    imax = float(I.max()) if I.size else 0.0
    ref_rel = float(I_ref / imax) if imax > 0.0 else None
    meta = dict(meta, I_ref_over_max_I=ref_rel,
                reference_in_far_tail=bool(ref_rel is not None and ref_rel < FAR_TAIL_RATIO))
    return {"nx": nx, "ny": ny, "I": I, "W": W, "I_ref": float(I_ref), "mu": mu,
            "I_pixel": acc.I_pixel, "I_bore": acc.I_bore, "G12": acc.G12,
            "trusted": trusted, "ref_index": acc.ref_index, "ref_xy": sampler.ref_xy,
            "sampler": sampler, "meta": meta, "seconds": seconds,
            "n_trusted": int(trusted.sum()),
            "over_unity": int((np.abs(mu[trusted]) > 1.0 + 1e-6).sum())}


def _report_lines(label, res, wave, extra):
    m = res["meta"]
    return [f"## Stage 16 — wave estimator [{label}]",
            f"- provider {m['provider']} ({m['model_class']}); observable {wave['observable']}; "
            f"source {m['source_rule']}; {m['n_nodes']} nodes",
            f"- trusted pixels {res['n_trusted']} of {res['nx'] * res['ny']} "
            f"(I ≥ {wave['intensity_floor']:g} × max I); |μ| > 1 on trusted: {res['over_unity']}",
            f"- status: unverified (declared target_error {wave['target_error']:g}; "
            "convergence ladders are not run inside this stage)",
            *([f"- WARNING: reference in a far diffraction tail (I_ref / max I = "
               f"{m['I_ref_over_max_I']:.1e} < {FAR_TAIL_RATIO:g}): W and μ there are dominated by the "
               "lattice representation of the aperture edges; accuracy not controlled"]
              if m.get("reference_in_far_tail") else []),
            *extra,
            f"- time: {res['seconds']:.1f} s"]


def _emit(partial, sub, res, wave, log):
    """Rows, meta and maps of one screen under partial/[sub/]."""
    out = os.path.join(partial, sub) if sub else partial
    os.makedirs(out, exist_ok=True)
    s, nx, ny = res["sampler"], res["nx"], res["ny"]
    xs_um, ys_um = [m_to_um(x) for x in s.px], [m_to_um(y) for y in s.py]
    ref_index = res["ref_index"]
    I, W, mu, tr = res["I"], res["W"], res["mu"], res["trusted"]
    with open(os.path.join(out, "mu-wave.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for iy in range(ny):
            for ix in range(nx):
                m = mu[ix, iy]
                ok = bool(np.isfinite(m.real))
                row = {"stage": res["meta"]["scene"], "pixel": iy * nx + ix,
                       "x_um": xs_um[ix], "y_um": ys_um[iy],
                       "W_re": float(W[ix, iy].real), "W_im": float(W[ix, iy].imag),
                       "I": float(I[ix, iy]), "I_ref": res["I_ref"],
                       "I_pixel": (None if res["I_pixel"] is None
                                   else float(res["I_pixel"][ix, iy])),
                       "mu_re": float(m.real) if ok else None,
                       "mu_im": float(m.imag) if ok else None,
                       "mu_abs": float(abs(m)) if ok else None,
                       "mu_null_reason": (None if ok else
                                          ("I_ref=0" if res["I_ref"] <= 0.0 else "I=0")),
                       "observable": wave["observable"], "trusted": bool(tr[ix, iy]),
                       "is_reference": ref_index is not None and (ix, iy) == tuple(ref_index)}
                if res["I_bore"] is not None:
                    row["I_bores"] = [float(v[ix, iy]) for v in res["I_bore"]]
                    row["I_int"] = float(I[ix, iy] - sum(v[ix, iy] for v in res["I_bore"]))
                if res["G12"] is not None:
                    row["G12_re"] = float(res["G12"][ix, iy].real)
                    row["G12_im"] = float(res["G12"][ix, iy].imag)
                fh.write(json.dumps(row) + "\n")
    meta = dict(res["meta"])
    meta.update({
        "wave_estimator": dict(wave),
        "screen": {"nx": nx, "ny": ny, "z_m": float(s.grid.z),
                   "cell_um": [m_to_um(s.hx), m_to_um(s.hy)],
                   "reference_requested_um": [m_to_um(res["ref_xy"][0]), m_to_um(res["ref_xy"][1])],
                   "reference_inside_window": bool(s.ref_inside),
                   "reference_cell_um": s.reference_cell_um(),
                   "reference_pixel": (None if ref_index is None
                                       else ref_index[1] * nx + ref_index[0])},
        "conventions": {"W": "<E(P) conj E(P_ref)>", "phasor": "exp(-i omega t), exp(+ikz)",
                        "I_coherent": "|integral_cell E dA|^2 (observable coherent_cell)",
                        "I_pixel": "integral_cell |sum_bores E|^2 dA = cell area x mean sub-pixel "
                                   "intensity of the total wave",
                        "units": UNITS},
        "status": "unverified", "I_ref": res["I_ref"], "n_trusted": res["n_trusted"],
        "seconds": res["seconds"], "capsysred_version": __version__})
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    names = ["mu-wave.jsonl", "meta.json"] + _maps(out, res, wave)
    prefix = (sub + "/") if sub else ""
    for n in names:
        log(f"  → {RESULT_DIR}/{prefix}{n}")
    return [prefix + n for n in names]


def _json_default(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    raise TypeError(f"not serializable: {type(v)}")


def _grid(arr, nx, ny, mask=None):
    """[iy][ix] rows for render; masked cells become None (checkerboard)."""
    return [[(float(arr[ix, iy]) if mask is None or mask[ix, iy] else None)
             for ix in range(nx)] for iy in range(ny)]


PHASE_MU_MIN = 1e-3     # phase of mu is drawn only where |mu| exceeds this (and is not exactly 0)


def _phase_mask(mu, trusted):
    """Trusted, finite, |mu| >= PHASE_MU_MIN and not an exact zero (W = 0 has no phase)."""
    a = np.abs(mu)
    return trusted & np.isfinite(a) & (a >= PHASE_MU_MIN) & (a > 0.0)


def _log_rows(values, imax):
    """log10(I / max): None for exact zeros (gap), true value for weak positives."""
    return [None if v <= 0.0 else math.log10(v / imax) for v in values]


def _maps(out, res, wave):
    """SVG maps of the contract: |mu|, I linear/log (zeros left blank), phase of mu with
    its own mask, I_pixel, per-bore intensities, Gamma_12 (Re/Im, modulus/phase) and the
    signed interference term. Empty curves are dropped, never faked."""
    s, nx, ny = res["sampler"], res["nx"], res["ny"]
    names = []
    ext = (m_to_um(s.grid.x0f), m_to_um(s.grid.x0f + s.grid.exf),
           m_to_um(s.grid.y0f), m_to_um(s.grid.y0f + s.grid.eyf))
    mark = (m_to_um(res["ref_xy"][0]), m_to_um(res["ref_xy"][1])) if s.ref_inside else None
    tr = res["trusted"]
    mu_abs, mu_arg = np.abs(res["mu"]), np.angle(res["mu"])
    pmask = _phase_mask(res["mu"], tr)
    sub = f"{res['meta']['provider']}, {res['meta']['n_nodes']} nodes, {wave['observable']}"
    two = res["I_bore"] is not None and len(res["I_bore"]) == 2
    zero = res["I"] <= 0.0

    def save(name, fig):
        render.save(os.path.join(out, name), fig)
        names.append(name)

    if ny > 1:
        row1 = [render.heatmap(_grid(mu_abs, nx, ny, tr), ext, "|μ(P, P_ref)| (wave)", "x, µm",
                               "y, µm", sub + f"; trusted {int(tr.sum())}/{nx * ny}", "|μ|",
                               mark=mark, vmax=1.0, w=430, equal=True),
                render.heatmap(_grid(res["I"], nx, ny), ext, "intensity", "x, µm", "y, µm",
                               "linear; " + wave["observable"], "I", w=430, equal=True),
                render.heatmap(_grid(res["I"], nx, ny, ~zero), ext, "intensity (log)", "x, µm",
                               "y, µm", "3 decades below max; exact zeros blank", "I",
                               w=430, equal=True, log=True)]
        save("16-wave-mu-intensity.svg", render.hstack(row1))
        row2 = [render.heatmap(_grid(mu_arg, nx, ny, pmask), ext, "arg μ(P, P_ref)", "x, µm", "y, µm",
                               f"trusted and |μ| ≥ {PHASE_MU_MIN:g}; masked elsewhere", "rad",
                               mark=mark, vmax=math.pi, w=430, equal=True, diverging=True)]
        if res["I_pixel"] is not None:
            row2.append(render.heatmap(_grid(res["I_pixel"], nx, ny), ext, "I_pixel (ordinary detector)",
                                       "x, µm", "y, µm", "∫cell |E|² dA of the total wave", "I_pixel",
                                       w=430, equal=True))
        save("16-wave-phase-pixel.svg", render.hstack(row2))
        if res["I_bore"] is not None:
            row3 = [render.heatmap(_grid(res["I_bore"][b], nx, ny), ext, f"I_{b + 1} (bore {b + 1} alone)",
                                   "x, µm", "y, µm", "same source, weights and observable", "I",
                                   w=430, equal=True) for b in range(len(res["I_bore"]))]
            if two:
                g12 = res["G12"]
                ga = np.abs(g12)
                gmask = (ga > 0.0) & (ga >= wave["intensity_floor"] * float(ga.max() or 1.0))
                i_int = res["I"] - res["I_bore"].sum(0)
                row3 += [render.heatmap(_grid(ga, nx, ny), ext, "|Γ₁₂(P)|", "x, µm", "y, µm",
                                        "cross term of the two bores at one point", "|Γ₁₂|",
                                        w=430, equal=True),
                         render.heatmap(_grid(np.angle(g12), nx, ny, gmask), ext, "arg Γ₁₂(P)", "x, µm",
                                        "y, µm", "masked where |Γ₁₂| = 0 or < floor × max", "rad",
                                        vmax=math.pi, w=430, equal=True, diverging=True),
                         render.heatmap(_grid(i_int, nx, ny), ext, "interference I_both − I₁ − I₂",
                                        "x, µm", "y, µm", "signed, symmetric scale", "I_int",
                                        w=430, equal=True, diverging=True)]
            save("16-wave-bores.svg", render.vstack(
                [render.hstack(row3[:3])] + ([render.hstack(row3[3:])] if len(row3) > 3 else [])))
    else:
        xs_um = [m_to_um(x) for x in s.px]
        vl = [(mark[0], "ref")] if mark else []
        series = []
        if tr.any():
            series.append({"xs": xs_um, "ys": _grid(mu_abs, nx, ny, tr)[0], "label": "wave |μ|"})
        if pmask.any():
            series.append({"xs": xs_um, "ys": [v / math.pi if v is not None else None
                                               for v in _grid(mu_arg, nx, ny, pmask)[0]],
                           "label": f"arg μ / π (|μ| ≥ {PHASE_MU_MIN:g})", "dash": "6,4"})
        if series:
            save("16-wave-mu.svg", render.line_chart(
                series, "|μ(x, x_ref)| and phase (wave)", "x, µm", "|μ|, arg μ/π",
                sub + f"; trusted {int(tr.sum())}/{nx}", vlines=vl, w=760))
        i_row = [float(v) for v in res["I"][:, 0]]
        imax = max(i_row)
        # coherent-observable scale S_coh = max(I, I_b): I, I_b, I_int and Gamma_12 share the
        # observable's units; I_pixel (ordinary detector, area scale) gets its own S_pixel
        s_coh = max([imax] + ([float(res["I_bore"].max())] if res["I_bore"] is not None else []))
        scale = s_coh if s_coh > 0.0 else 1.0           # technical divisor only; S_coh stays the true max
        pmax = float(res["I_pixel"].max()) if res["I_pixel"] is not None else None
        if s_coh > 0.0:
            note = "" if imax > 0.0 else "total I is zero everywhere; bores on the shared coherent scale"
        elif pmax:
            note = "coherent intensities are zero (shown without normalization); I_pixel is not"
        else:
            note = "all intensities are exactly zero"
        series = [{"xs": xs_um, "ys": [v / scale for v in i_row], "label": "I / S_coh"}]
        if res["I_bore"] is not None:
            for b in range(len(res["I_bore"])):
                series.append({"xs": xs_um, "label": f"I_{b + 1} / S_coh", "dash": "6,4",
                               "ys": [float(res["I_bore"][b][ix, 0]) / scale for ix in range(nx)]})
        if res["I_pixel"] is not None:
            series.append({"xs": xs_um, "ys": [float(v) / (pmax or 1.0) for v in res["I_pixel"][:, 0]],
                           "label": "I_pixel / S_pixel (ordinary detector, own scale)", "dash": "2,3"})
            note = (note + "; " if note else "") + f"S_pixel = {pmax:.3e} (area scale, not comparable)"
        save("16-wave-intensity.svg", render.line_chart(
            series, "intensity (profiles; two observables, two scales)", "x, µm", "I / S",
            (note + "; " if note else "") + f"S_coh = max(I, I_b) = {s_coh:.3e}", w=760))
        if imax > 0.0:
            save("16-wave-intensity-log.svg", render.line_chart(
                [{"xs": xs_um, "ys": _log_rows(i_row, imax), "label": "log10(I / max)"}],
                "intensity (log)", "x, µm", "log10(I / max)",
                f"exact zeros blank: {int(sum(v <= 0.0 for v in i_row))} of {nx}", y_zero=False, w=760))
        if two:
            i_int = [float(v) for v in (res["I"] - res["I_bore"].sum(0))[:, 0]]
            g12 = res["G12"][:, 0]
            save("16-wave-interference.svg", render.line_chart(
                [{"xs": xs_um, "ys": [v / scale for v in i_int], "label": "I_int / S_coh"},
                 {"xs": xs_um, "ys": [2.0 * float(v.real) / scale for v in g12],
                  "label": "2 Re Γ₁₂ / S_coh", "dash": "6,4"},
                 {"xs": xs_um, "ys": [2.0 * float(v.imag) / scale for v in g12],
                  "label": "2 Im Γ₁₂ / S_coh", "dash": "2,3"}],
                "interference term and complex Γ₁₂", "x, µm", "signed",
                f"S_coh = max(I, I_b) = {s_coh:.3e}", y_zero=False, w=760))
            ga = np.abs(g12)
            gmax = float(ga.max())
            gmask = (ga > 0.0) & (ga >= wave["intensity_floor"] * (gmax or 1.0))
            gseries = [{"xs": xs_um, "ys": [float(v) / (gmax or 1.0) for v in ga], "label": "|Γ₁₂| / max"}]
            if gmask.any():
                gseries.append({"xs": xs_um, "ys": [float(np.angle(v)) / math.pi if m else None
                                                    for v, m in zip(g12, gmask)],
                                "label": "arg Γ₁₂ / π (masked)", "dash": "6,4"})
            save("16-wave-gamma12.svg", render.line_chart(
                gseries, "Γ₁₂(P): modulus and phase", "x, µm", "|Γ₁₂|/max, arg/π",
                f"max |Γ₁₂| = {gmax:.3e}", y_zero=False, w=760))
    return names

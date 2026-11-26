"""Provider uisk of Stage 16: unfolded-image spectral Kirchhoff in straight polygon bores."""

import math

import numpy as np
from scipy import fft as sfft

from ..physics import fresnel_r
from ..polygon import Polygon, _fast_len, coverage, sub_lattice_mask, sub_offsets


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
    tile (commuting reflections, e.g. both corner orders of a square) are one family.
    Four faces (rectangle): the tiles are the lattice (kx, ky), |kx| + |ky| <= max_bounces."""
    if poly.n == 4:
        out = []
        for kx in range(-max_bounces, max_bounces + 1):
            for ky in range(-(max_bounces - abs(kx)), max_bounces - abs(kx) + 1):
                sx, sy = ((0, 2) if kx > 0 else (2, 0)), ((1, 3) if ky > 0 else (3, 1))
                out.append(Family(poly, [sx[i % 2] for i in range(abs(kx))] + [sy[i % 2] for i in range(abs(ky))]))
        return out
    seqs, frontier = [()], [()]
    for _ in range(max_bounces):
        new = [s + (i,) for s in frontier for i in range(poly.n) if not s or s[-1] != i]
        seqs += new
        frontier = new
    out = []
    for s in seqs:
        fam = Family(poly, s)
        if not any(fam.same_tile(f, poly.scale) for f in out):
            out.append(fam)
    return out


class PackedMask:
    """Sub-lattice indicator stored bit-packed per (s, t) offset: 1 bit per sub-cell."""

    def __init__(self, inside):
        self.shape = (inside.shape[0], inside.shape[2])
        self.s = inside.shape[1]
        self.bits = [[np.packbits(inside[:, s, :, t]) for t in range(self.s)]
                     for s in range(self.s)]

    def slab(self, s, t):
        n = self.shape[0] * self.shape[1]
        return np.unpackbits(self.bits[s][t], count=n).reshape(self.shape).astype(bool)


class BoreSolver:
    """UISK for one polygon bore on the global lattice x = i h, y = j h.

    lam_h: the shortest wavelength of the run (lattice step); lam_reach: the
    longest (Fresnel tails in the box margin)."""

    def __init__(self, bore, z_in, length, theta_max, lam_h, wave, lam_reach=None, cache_masks=True):
        lam_reach = lam_h if lam_reach is None else lam_reach
        self.poly = Polygon(bore)
        self.cache_masks = cache_masks                      # False: masks rebuilt per node (one node, one line)
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
                                                  self.poly.a, self.supersample)) if cache_masks else None
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

    @staticmethod
    def _axis_counts(fam, tol=1e-12):
        """(reflections off x faces, off y faces) when every unfolded normal is a lattice axis, else None."""
        a = b = 0
        for n in fam.unfolded:
            if abs(abs(n[0]) - 1.0) < tol and abs(n[1]) < tol:
                a += 1
            elif abs(abs(n[1]) - 1.0) < tol and abs(n[0]) < tol:
                b += 1
            else:
                return None
        return a, b

    def transfer(self, fam, k, delta, beta, rows=256, general=False):
        """exp(-i kappa^2 L / 2k) prod_j r(|kappa . n'_j| / k) on the family grid: an outer product of
        two 1D factors for axis-aligned faces, else filled in row chunks so that only the result is grid-sized."""
        key = (id(fam), k)
        if key in self.cache:
            return self.cache[key]
        nx, ny = fam.grid[1] - fam.grid[0], fam.grid[3] - fam.grid[2]
        kx = 2.0 * np.pi * sfft.fftfreq(nx, self.h)
        ky = 2.0 * np.pi * sfft.fftfreq(ny, self.h)
        # band limit: waves steeper than reach/L would leave the box and wrap around;
        # raised cosine per axis over the upper half of the margin band above theta_max
        # (a separable window keeps the square problem exactly separable)
        t_hi = self.theta_cut
        t_lo = t_hi - 0.5 * (t_hi - self.theta_max)

        def window(kk):
            theta = np.abs(kk) / k
            return np.where(theta <= t_lo, 1.0,
                            np.where(theta >= t_hi, 0.0,
                                     0.5 * (1.0 + np.cos(np.pi * (theta - t_lo) / (t_hi - t_lo)))))
        wx, wy = (window(kx), window(ky)) if t_lo > self.theta_max else (None, None)
        axes = None if general else self._axis_counts(fam)
        if axes is not None:
            ex = np.exp(-1j * kx ** 2 * self.length / (2.0 * k)) * fresnel_r(np.abs(kx) / k, delta, beta) ** axes[0]
            ey = np.exp(-1j * ky ** 2 * self.length / (2.0 * k)) * fresnel_r(np.abs(ky) / k, delta, beta) ** axes[1]
            if wx is not None:
                ex, ey = ex * wx, ey * wy
            return np.outer(ex, ey)
        H = np.empty((nx, ny), dtype=complex)
        for i0 in range(0, nx, rows):
            kxc = kx[i0:i0 + rows, None]
            h = np.exp(-1j * (kxc ** 2 + ky[None, :] ** 2) * self.length / (2.0 * k))
            for n in fam.unfolded:
                h *= fresnel_r(np.abs(kxc * n[0] + ky[None, :] * n[1]) / k, delta, beta)
            if wx is not None:
                h *= wx[i0:i0 + rows, None] * wy[None, :]
            H[i0:i0 + rows] = h
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
            if fam.mask is None:
                full = sub_lattice_mask(fam.xs, fam.ys, fam.center, fam.face_normals, self.poly.a, self.supersample)
                slab = lambda s, t: full[:, s, :, t]
            else:
                full, slab = None, fam.mask.slab
            shape = (fam.xs.size, fam.ys.size)
            u = np.zeros(shape, dtype=complex)
            tmp = np.empty(shape, dtype=complex)
            for s in range(self.supersample):
                for t in range(self.supersample):
                    np.multiply(cx[:, s, None], cy[None, :, t], out=tmp)
                    tmp *= slab(s, t)
                    u += tmp
            del tmp, full
            u *= 1.0 / (self.supersample ** 2 * 1j * lam * self.z_in)
            # two grid-sized arrays alive at most (out-of-place FFTs keep the bit pattern)
            F = sfft.fft2(u, workers=workers)
            del u
            F *= self.transfer(fam, k, delta, beta)
            u = sfft.ifft2(F, workers=workers)
            del F
            out += u[fam.slice]
            del u
        return out * self.exit_mask

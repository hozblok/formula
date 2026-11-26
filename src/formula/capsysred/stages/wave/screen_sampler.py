"""Screens of Stage 16: the Fresnel step from the exit lattice to the pixels and the joint W, I sums."""

import numpy as np

from ...screen import ScreenGrid
from ...shared.units import m_to_um
from .physics import (cell_cell_kernel, cell_kernel, cell_overlap_kernel, point_cell_kernel, point_kernel,
                      point_pick_kernel)

# reference intensity below this fraction of max I = deep diffraction tail: the lattice
# representation of the aperture edges dominates W
FAR_TAIL_RATIO = 1e-3


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
        every kernel is integrated exactly over the exit lattice cell of width h.
        d = 0 (screen on the exit plane): no propagation, pixels pick / average the lattice."""
        if d == 0.0:
            point = lambda pts, zs: point_pick_kernel(pts, zs, h)
            cell = lambda lo, hi, zs: cell_overlap_kernel(lo, hi, zs, h)
        else:
            point = lambda pts, zs: point_cell_kernel(pts, zs, h, k, d)
            cell = lambda lo, hi, zs: cell_cell_kernel(lo, hi, zs, h, k, d)
        if self.observable == "coherent_cell":
            kx = cell(self.px - self.hx / 2, self.px + self.hx / 2, xs)
            ky = cell(self.py - self.hy / 2, self.py + self.hy / 2, ys)
            kxr = kyr = None
            sx, sy = point(self.sub_x, xs), point(self.sub_y, ys)
        else:
            kx, ky = point(self.px, xs), point(self.py, ys)
            kxr = point(np.array([self.ref_xy[0]]), xs)
            kyr = point(np.array([self.ref_xy[1]]), ys)
            sx = sy = None
        return kx, ky, kxr, kyr, sx, sy

    def fields(self, u, kern, k, d, h):
        """Screen field [ix, iy], reference-point field (point mode) and the complex
        sub-pixel field [ix*S+s, iy*S+t] (cell mode; None otherwise)."""
        kx, ky, kxr, kyr, sx, sy = kern
        lam = 2.0 * np.pi / k
        c = h * h if d == 0.0 else h * h / (1j * lam * d)   # d = 0: overlap kernels, cell area
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

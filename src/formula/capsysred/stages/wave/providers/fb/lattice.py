"""Lab lattice of the provider fb: the angular-spectrum drift to the screens and the cell integrals onto the pixels."""

import math

import numpy as np
from scipy import fft as sfft

from .....shared.units import m_to_um


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

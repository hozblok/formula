"""Polygon bores of Stage 16 and their indicators on the lattice."""

import numpy as np
from scipy import fft as sfft


class Polygon:
    """Polygon bore: face k has unit normal at rotation + 2 pi k / n, plane (p - c).n = a_k."""

    def __init__(self, bore):
        self.n = int(bore["sides"])
        self.a = np.array([float(v) for v in bore.get("apothems") or [bore["radius"]] * self.n])
        self.scale = float(self.a.max())
        self.c = np.array([float(bore["center"][0]), float(bore["center"][1])])
        ang = float(bore["rotation"]) + 2.0 * np.pi * np.arange(self.n) / self.n
        self.normals = np.stack([np.cos(ang), np.sin(ang)], -1)
        self.verts = np.array([
            self.c + np.linalg.solve(np.stack([self.normals[k], self.normals[(k + 1) % self.n]]),
                                     [self.a[k], self.a[(k + 1) % self.n]])
            for k in range(self.n)])

    def reflect_point(self, k, p):
        n = self.normals[k]
        s = (p - self.c) @ n - self.a[k]
        return p - 2.0 * s[..., None] * n

    def reflect_vector(self, k, v):
        n = self.normals[k]
        return v - 2.0 * (v @ n)[..., None] * n


def sub_offsets(h, supersample):
    return ((np.arange(supersample) + 0.5) / supersample - 0.5) * h


def sub_lattice_mask(xs, ys, center, normals, a, supersample, strip=256):
    """Indicator of the convex polygon (p - center).n_k <= a_k on the S x S sub-lattice,
    shape (nx, S, ny, S); built in x-strips so the float temporaries stay small."""
    h = xs[1] - xs[0] if len(xs) > 1 else (ys[1] - ys[0] if len(ys) > 1 else 1.0)
    off = sub_offsets(h, supersample)
    Y = (ys[:, None] + off[None, :]).reshape(-1)
    out = np.empty((len(xs), supersample, len(ys), supersample), dtype=bool)
    for i0 in range(0, len(xs), strip):
        X = (xs[i0:i0 + strip, None] + off[None, :]).reshape(-1)
        inside = np.ones((X.size, Y.size), dtype=bool)
        for n, ak in zip(normals, np.broadcast_to(a, (len(normals),))):
            inside &= ((X[:, None] - center[0]) * n[0] + (Y[None, :] - center[1]) * n[1]) <= ak
        out[i0:i0 + strip] = inside.reshape(-1, supersample, len(ys), supersample)
    return out


def coverage(xs, ys, center, normals, a, supersample):
    """Fractional cell coverage of the polygon (mean of the sub-lattice indicator)."""
    return sub_lattice_mask(xs, ys, center, normals, a, supersample).mean(axis=(1, 3))


def _fast_len(n):
    return sfft.next_fast_len(max(int(n), 1), real=False)

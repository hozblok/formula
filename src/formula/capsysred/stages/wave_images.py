"""Stage 16 provider `images`: a rectangular bore with faces along the lattice axes is the
product of two one-dimensional unfolded-image sums. Every image up to the guided horizon
(tiles whose bounce angle stays below ~1.5 theta_c) is summed with the plane-wave r(kappa)
per bounce; Kirchhoff entrance / exit masks; no band window (the line lattice is wide enough
for nothing to wrap). Same lattice, screens and accumulators as `uisk`; optional coherent
`entrance_spot`; optional far-field observable of the exit field."""

import math

import numpy as np
from scipy import fft as sfft


def axis_rectangle(bore):
    """(cx, ax, cy, ay) of a polygon bore whose four faces lie along the lattice axes, else None."""
    from .wave import Polygon
    poly = Polygon(bore)
    if poly.n != 4:
        return None
    face = {}
    for n, a in zip(poly.normals, poly.a):
        if abs(abs(n[0]) - 1.0) < 1e-12 and abs(n[1]) < 1e-12:
            face[("x", n[0] > 0.0)] = float(a)
        elif abs(abs(n[1]) - 1.0) < 1e-12 and abs(n[0]) < 1e-12:
            face[("y", n[1] > 0.0)] = float(a)
        else:
            return None
    cx = poly.c[0] + (face[("x", True)] - face[("x", False)]) / 2.0
    cy = poly.c[1] + (face[("y", True)] - face[("y", False)]) / 2.0
    return cx, (face[("x", True)] + face[("x", False)]) / 2.0, cy, (face[("y", True)] + face[("y", False)]) / 2.0


class AxisSolver:
    """One axis: tiles n of the unfolded entrance (even n shifted by nW, odd n mirrored) on a
    line lattice x = i h, each tile's field FFT-propagated with exp(-i kappa^2 L / 2k) r^|n|."""

    def __init__(self, c, a, z_in, length, h, supersample, n_images, spread):
        from .wave import _fast_len, sub_offsets
        self.c, self.a, self.z_in, self.L, self.h = float(c), float(a), float(z_in), float(length), float(h)
        self.n_img = int(n_images)
        half = (self.n_img + 0.5) * 2.0 * self.a + spread
        i0 = math.floor((self.c - half) / self.h)
        n = _fast_len(math.ceil((self.c + half) / self.h) - i0 + 1)
        self.xs = np.arange(i0, i0 + n) * self.h
        self.kap = 2.0 * np.pi * sfft.fftfreq(n, self.h)
        e0 = math.floor((self.c - self.a - 2.0 * self.h) / self.h)
        ne = _fast_len(math.ceil((self.c + self.a + 2.0 * self.h) / self.h) - e0 + 1)
        self.win = (e0, e0 + ne)
        self.exit_xs = np.arange(e0, e0 + ne) * self.h
        self.slice = slice(e0 - i0, e0 - i0 + ne)
        self.off = sub_offsets(self.h, supersample)

    def preimage(self, n):
        """Entrance-frame coordinate of every sub-lattice point for tile n."""
        X = self.xs[:, None] + self.off[None, :]
        return X - n * 2.0 * self.a if n % 2 == 0 else 2.0 * self.c + n * 2.0 * self.a - X

    def exit_line(self, entrance, k, delta, beta):
        """Exit samples on the window for a 1D entrance factor `entrance(x)` given in the aperture frame."""
        from .wave import fresnel_r
        r = fresnel_r(np.abs(self.kap) / k, delta, beta)
        prop = np.exp(-1j * self.kap ** 2 * self.L / (2.0 * k))
        S = np.zeros(self.xs.size, dtype=complex)
        for n in range(-self.n_img, self.n_img + 1):
            X = self.preimage(n)
            u = (entrance(X) * (np.abs(X - self.c) <= self.a)).mean(axis=1)
            S += r ** abs(n) * sfft.fft(u)
        return sfft.ifft(S * prop)[self.slice]


class ImageSolver:
    """Separable image sum for one axis-aligned rectangular bore; the interface of BoreSolver."""

    def __init__(self, bore, z_in, length, theta_max, lam_h, wave, lam_reach=None, theta_c_max=None, spot=None):
        from .wave import Polygon, coverage
        lam_reach = lam_h if lam_reach is None else lam_reach
        self.poly = Polygon(bore)
        rect = axis_rectangle(bore)
        if rect is None:
            raise ValueError("wave_estimator: provider images needs a rectangle with faces along the lattice axes")
        cx, ax, cy, ay = rect
        self.z_in, self.length, self.spot = float(z_in), float(length), spot
        self.h = wave["grid_dx"] or lam_h / (2.0 * theta_max * wave["angle_margin"])
        self.theta_max = theta_max
        thc = float(theta_c_max) if theta_c_max else theta_max
        self.theta_horizon = 1.5 * thc
        spread = self.theta_horizon * self.length + 4.0 * math.sqrt(lam_reach * self.length)
        self.reach, self.theta_cut = spread, self.theta_horizon
        S = int(wave["mask_supersample"])
        self.axes = []
        for c, a in ((cx, ax), (cy, ay)):
            n_img = wave["images"] if wave["images"] is not None else int(math.ceil(self.theta_horizon * self.length / (2.0 * a))) + 4
            self.axes.append(AxisSolver(c, a, z_in, length, self.h, S, n_img, spread))
        self.n_images = [ax_.n_img for ax_ in self.axes]
        self.line_points = [ax_.xs.size for ax_ in self.axes]
        self.win = (*self.axes[0].win, *self.axes[1].win)
        self.xs, self.ys = self.axes[0].exit_xs, self.axes[1].exit_xs
        self.exit_mask = coverage(self.xs, self.ys, self.poly.c, self.poly.normals, self.poly.a, S)
        self.families, self.pruned = [], 0

    def cache_transfer(self, k, delta, beta, budget_bytes):
        return None

    def entrance_factors(self, xi, k):
        """1D factors of the entrance field and the 2D prefactor: point-source chirps or the coherent spot."""
        if self.spot is None:
            z = self.z_in
            return ([lambda x, s=float(xi[0]): np.exp(1j * k * (x - s) ** 2 / (2.0 * z)),
                     lambda y, s=float(xi[1]): np.exp(1j * k * (y - s) ** 2 / (2.0 * z))],
                    1.0 / (1j * (2.0 * np.pi / k) * z))
        s = self.spot["fwhm"] / math.sqrt(8.0 * math.log(2.0))
        (cx, cy), (tx, ty) = self.spot["center"], self.spot["tilt"]
        norm = (2.0 * math.pi * s * s) ** -0.25
        return ([lambda x: norm * np.exp(-(x - cx) ** 2 / (4.0 * s * s)) * np.exp(1j * k * tx * x),
                 lambda y: norm * np.exp(-(y - cy) ** 2 / (4.0 * s * s)) * np.exp(1j * k * ty * y)], 1.0)

    def exit_field(self, xi, k, delta, beta, workers):
        """Field on the real exit window: product of the two axis sums times the exit mask."""
        factors, pref = self.entrance_factors(xi, k)
        ux = self.axes[0].exit_line(factors[0], k, delta, beta)
        uy = self.axes[1].exit_line(factors[1], k, delta, beta)
        return pref * np.outer(ux, uy) * self.exit_mask


def far_field(u, h, k, theta_max, pad):
    """Angular power |FFT u|^2 on theta = lambda f (zero-padded): window marginals, centroids, P(+)/P(-)."""
    F = sfft.fftshift(sfft.fft2(u, s=(pad, pad)))
    th = sfft.fftshift(sfft.fftfreq(pad, h)) * (2.0 * np.pi / k)
    P = np.abs(F) ** 2
    m = np.abs(th) < theta_max
    Pw, t = P[np.ix_(m, m)], th[m]
    return t, Pw.sum(axis=1), Pw.sum(axis=0), float(P.sum()), float(Pw.sum())


def far_field_summary(t, px, py, total, inside):
    def lobes(p):
        tot = float(p.sum())
        if tot <= 0.0:
            return {"centroid_rad": None, "P_plus_over_minus": None}
        return {"centroid_rad": float((p * t).sum() / tot),
                "P_plus_over_minus": float(p[t > 0.0].sum() / max(float(p[t < 0.0].sum()), 1e-300))}
    return {"theta_rad": t.tolist(), "P_x": px.tolist(), "P_y": py.tolist(),
            "x": lobes(px), "y": lobes(py), "power_in_window": inside / total if total > 0.0 else None}

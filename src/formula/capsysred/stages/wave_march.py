"""Stage 16 provider `march`: scalar paraxial split-step propagation through the bore with the
wall as a medium n = 1 - delta + i beta on the lattice (fractional cell filling), so the cross
section may change along z: wall roughness, width steps, tapers, sidewall slope, rounded corners,
elliptic or drifting circular bores. Kirchhoff entrance mask; exit field on the same window as
`images` / `uisk`. Section `march` of wave_estimator: dx, dz, box, sponge, defects."""

import math

import numpy as np
from scipy import fft as sfft

DEFECT_KEYS = {
    "roughness": {"sigma", "lc", "seed", "walls"},
    "step": {"z", "dw"},
    "taper": {"dw"},
    "slope": {"deg"},
    "rounded": {"rc"},
    "ellipse": {"ratio"},
    "drift": {"dxy"},
}
WALLS = ("x-", "x+", "y-", "y+")


def rough_series(n, dz, sigma, lc, rng):
    """Gaussian series, rms sigma, Lorentzian spectrum 1/(1 + (k_z l_c)^2) (exp(-|z|/l_c) correlation)."""
    white = rng.standard_normal(n)
    kz = 2.0 * np.pi * sfft.fftfreq(n, dz)
    h = np.real(sfft.ifft(sfft.fft(white) / (1.0 + (kz * lc) ** 2)))
    return h * (sigma / h.std()) if h.std() > 0.0 else h


def coverage_1d(x, h, lo, hi):
    """Fraction of each cell [x - h/2, x + h/2] inside [lo, hi]."""
    return np.clip((np.minimum(x + h / 2.0, hi) - np.maximum(x - h / 2.0, lo)) / h, 0.0, 1.0)


class MarchSolver:
    """Split-step march for one bore; the interface of BoreSolver / ImageSolver."""

    def __init__(self, bore, z_in, length, theta_max, lam_h, wave, lam_reach=None, spot=None):
        from .wave import Polygon, _fast_len, sub_offsets
        m = wave["march"]
        self.z_in, self.length, self.spot = float(z_in), float(length), spot
        self.theta_max = theta_max
        self.h = m["dx"] or wave["grid_dx"] or lam_h / (2.0 * theta_max * wave["angle_margin"])
        self.n_steps = max(1, int(math.ceil(self.length / m["dz"] - 1e-9)))
        self.dz = self.length / self.n_steps
        self.S = int(wave["mask_supersample"])
        self.circular = bore.get("sides") is None
        if self.circular:
            self.c = np.array([float(bore["center"][0]), float(bore["center"][1])])
            self.half = np.array([float(bore["radius"])] * 2)
            self.poly = None
        else:
            self.poly = Polygon(bore)
            lo, hi = self.poly.verts.min(0), self.poly.verts.max(0)
            self.c, self.half = (lo + hi) / 2.0, (hi - lo) / 2.0
        self.axis_rect = (not self.circular and self.poly.n == 4
                          and all(min(abs(abs(n[0]) - 1.0), abs(abs(n[1]) - 1.0)) < 1e-12 for n in self.poly.normals))
        margin = float(m["box"])
        self.reach, self.theta_cut = margin, lam_h / (2.0 * self.h)
        self.grid, self.xs, self.ys = [], [], []
        for c, a in zip(self.c, self.half):
            i0 = math.floor((c - a - margin) / self.h)
            n = _fast_len(math.ceil((c + a + margin) / self.h) - i0 + 1)
            self.grid.append((i0, i0 + n))
        self.lx = np.arange(self.grid[0][0], self.grid[0][1]) * self.h       # full lattice
        self.ly = np.arange(self.grid[1][0], self.grid[1][1]) * self.h
        self.win, self.slices = [], []
        for c, a, (g0, g1) in zip(self.c, self.half, self.grid):
            e0 = math.floor((c - a - 2.0 * self.h) / self.h)
            ne = _fast_len(math.ceil((c + a + 2.0 * self.h) / self.h) - e0 + 1)
            self.win += [e0, e0 + ne]
            self.slices.append(slice(e0 - g0, e0 - g0 + ne))
        self.win = tuple(self.win)
        self.xs, self.ys = self.lx[self.slices[0]], self.ly[self.slices[1]]    # exit window
        self.off = sub_offsets(self.h, self.S)
        kinds = {d["type"] for d in m["defects"]}
        if self.circular and kinds - {"ellipse", "taper", "drift"}:
            raise ValueError("wave_estimator.march: a circular bore takes defects ellipse, taper, drift only")
        if not self.circular and (not self.axis_rect and kinds or "ellipse" in kinds):
            raise ValueError("wave_estimator.march: wall defects need a rectangle with faces along the lattice axes")
        self._defects(m["defects"], m.get("seed"))
        self.kx = 2.0 * np.pi * sfft.fftfreq(self.lx.size, self.h)
        self.ky = 2.0 * np.pi * sfft.fftfreq(self.ly.size, self.h)
        sw = float(m["sponge"])
        ramp = lambda v, lo, hi: np.clip(np.maximum(lo + sw - v, v - (hi - sw)) / sw, 0.0, 1.0) ** 2 if sw > 0.0 else 0.0 * v
        self.sponge = (ramp(self.lx, self.lx[0], self.lx[-1])[:, None] + ramp(self.ly, self.ly[0], self.ly[-1])[None, :])
        self.families, self.pruned = [], 0
        self.exit_mask_full = self.fill(self.n_steps)
        self.exit_mask = self.exit_mask_full[self.slices[0], self.slices[1]]
        self.n_images, self.line_points = [], []

    # ------------------------------------------------------------- geometry along z

    def _defects(self, defects, seed):
        n = self.n_steps
        self.offsets = np.zeros((4, n + 1))              # wall offsets (x-, x+, y-, y+) at every step plane
        self.center_shift = np.zeros((2, n + 1))
        self.slope = self.rc = 0.0
        self.ratio = 1.0
        z = np.arange(n + 1) * self.dz
        for d in defects:
            t = d["type"]
            if t == "roughness":
                rng = np.random.default_rng(int(d["seed"]))
                for wall in d.get("walls") or WALLS:
                    s = rough_series(n + 1, self.dz, float(d["sigma"]), float(d["lc"]), rng)
                    self.offsets[WALLS.index(wall)] += s
            elif t == "step":
                after = z >= float(d["z"])
                for ax, dw in enumerate(d["dw"]):
                    self.offsets[2 * ax] -= 0.5 * float(dw) * after
                    self.offsets[2 * ax + 1] += 0.5 * float(dw) * after
            elif t == "taper":
                for ax, dw in enumerate(d["dw"]):
                    self.offsets[2 * ax] -= 0.5 * float(dw) * z / self.length
                    self.offsets[2 * ax + 1] += 0.5 * float(dw) * z / self.length
            elif t == "slope":
                self.slope = math.tan(math.radians(float(d["deg"])))
            elif t == "rounded":
                self.rc = float(d["rc"])
            elif t == "ellipse":
                self.ratio = float(d["ratio"])
            elif t == "drift":
                for ax in range(2):
                    self.center_shift[ax] += float(d["dxy"][ax]) * z / self.length
        self.z_dependent = bool(np.any(self.offsets != 0.0) or np.any(self.center_shift != 0.0))
        self.separable = self.axis_rect and self.slope == 0.0 and self.rc == 0.0
        self._static = None if (self.z_dependent or self.separable) else self._mask2d(0)

    def _walls(self, i):
        """(xl, xr, yl, yr) of the rectangle or ellipse box at step plane i."""
        cx, cy = self.c + self.center_shift[:, i]
        o = self.offsets[:, i]
        return cx - self.half[0] + o[0], cx + self.half[0] + o[1], cy - self.half[1] + o[2], cy + self.half[1] + o[3]

    def _mask2d(self, i):
        """S x S supersampled inside fraction for polygons with slope / rounding and for ellipses."""
        xl, xr, yl, yr = self._walls(i)
        X = (self.lx[:, None] + self.off[None, :]).reshape(-1)
        Y = (self.ly[:, None] + self.off[None, :]).reshape(-1)
        cx, cy, ax, ay = (xl + xr) / 2.0, (yl + yr) / 2.0, (xr - xl) / 2.0, (yr - yl) / 2.0
        if self.circular:
            inside = ((X[:, None] - cx) / ax) ** 2 + ((Y[None, :] - cy) / (ay * self.ratio)) ** 2 < 1.0
        elif self.axis_rect:
            hx = ax + (Y[None, :] - cy) * self.slope                 # sidewall slope: wider at +y
            inside = (np.abs(X[:, None] - cx) < hx) & (np.abs(Y[None, :] - cy) < ay)
            if self.rc > 0.0:
                ex = np.maximum(np.abs(X[:, None] - cx) - (hx - self.rc), 0.0)
                ey = np.maximum(np.abs(Y[None, :] - cy) - (ay - self.rc), 0.0)
                inside &= ex ** 2 + ey ** 2 < self.rc ** 2
        else:
            inside = np.ones((X.size, Y.size), dtype=bool)
            for nrm, a in zip(self.poly.normals, self.poly.a):
                inside &= ((X[:, None] - self.poly.c[0]) * nrm[0] + (Y[None, :] - self.poly.c[1]) * nrm[1]) <= a
        return inside.reshape(self.lx.size, self.S, self.ly.size, self.S).mean(axis=(1, 3))

    def fill(self, i):
        """Core fraction of every lattice cell at step plane i."""
        if self._static is not None:
            return self._static
        if self.separable:
            xl, xr, yl, yr = self._walls(i)
            return np.outer(coverage_1d(self.lx, self.h, xl, xr), coverage_1d(self.ly, self.h, yl, yr))
        if self.circular:                                    # row-wise chord of the ellipse
            xl, xr, yl, yr = self._walls(i)
            cx, cy, ax, ay = (xl + xr) / 2.0, (yl + yr) / 2.0, (xr - xl) / 2.0, (yr - yl) / 2.0 * self.ratio
            t = np.clip(1.0 - ((self.ly - cy) / ay) ** 2, 0.0, None)
            w = ax * np.sqrt(t)
            lo, hi = cx - w[None, :], cx + w[None, :]
            x = self.lx[:, None]
            return np.clip((np.minimum(x + self.h / 2.0, hi) - np.maximum(x - self.h / 2.0, lo)) / self.h, 0.0, 1.0)
        return self._mask2d(i)

    # ------------------------------------------------------------- propagation

    def cache_transfer(self, k, delta, beta, budget_bytes):
        return None

    def entrance(self, xi, k):
        X, Y = self.lx[:, None], self.ly[None, :]
        if self.spot is None:
            lam = 2.0 * np.pi / k
            return np.exp(1j * k * ((X - xi[0]) ** 2 + (Y - xi[1]) ** 2) / (2.0 * self.z_in)) / (1j * lam * self.z_in)
        s = self.spot["fwhm"] / math.sqrt(8.0 * math.log(2.0))
        (cx, cy), (tx, ty) = self.spot["center"], self.spot["tilt"]
        return (np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (4.0 * s * s)) * np.exp(1j * k * (tx * X + ty * Y))
                / math.sqrt(2.0 * math.pi * s * s))

    def exit_field(self, xi, k, delta, beta, workers):
        """Kirchhoff entrance, n_steps of free propagation + medium factor, exit window times the exit mask."""
        prop = np.exp(-1j * (self.kx[:, None] ** 2 + self.ky[None, :] ** 2) * self.dz / (2.0 * k))
        wall = np.exp(-1j * k * (delta - 1j * beta) * self.dz)
        absorb = np.exp(-k * 5.0e-5 * self.dz * self.sponge)
        u = self.entrance(xi, k) * self.fill(0)
        f = None
        for i in range(1, self.n_steps + 1):
            u = sfft.ifft2(sfft.fft2(u, workers=workers) * prop, workers=workers)
            if f is None or self.z_dependent:
                f = self.fill(i)
            u *= (f + (1.0 - f) * wall) * absorb
        return u[self.slices[0], self.slices[1]] * self.exit_mask

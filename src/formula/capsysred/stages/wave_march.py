"""Stage 16 provider `march`: scalar paraxial split-step propagation through the bore with the
wall as a medium n = 1 - delta + i beta on the lattice (fractional cell filling), so the cross
section may change along z: wall roughness, width steps, tapers, sidewall slope, rounded corners,
elliptic, oval (rotating along z) or drifting circular bores, and conformal funnels (centre and radius times
G(z) = 1 + g0 z + g1 z^2). Kirchhoff entrance mask; exit field on the same window as `images` / `uisk`.
Section `march` of wave_estimator: dx, dz, box, sponge, defects."""

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
    "ovality": {"amp", "rate", "angle"},
    "drift": {"dxy"},
}
WALLS = ("x-", "x+", "y-", "y+")


def rough_series(n, dz, sigma, lc, rng):
    """Gaussian series, rms sigma: white noise filtered by 1/(1 + (k_z l_c)^2), correlation (1 + |z|/l_c) exp(-|z|/l_c)."""
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
        self.funnel = None
        if bore.get("kind") == "funnel":
            g, f = tuple(float(v) for v in bore["g"]), tuple(float(v) for v in bore["f"])
            if g != f:
                raise ValueError("wave_estimator.march: a funnel needs f = g (centre and radius on one parabola G)")
            if m["defects"]:
                raise ValueError("wave_estimator.march: a funnel takes no defects")
            self.funnel = g
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
        gs = (1.0, 1.0) if self.funnel is None else self._g_range()
        for c, a in zip(self.c, self.half):
            lo, hi = min(g * (c - a) for g in gs), max(g * (c + a) for g in gs)      # a funnel sweeps G(z) (c -+ a)
            i0 = math.floor((lo - margin) / self.h)
            n = _fast_len(math.ceil((hi + margin) / self.h) - i0 + 1)
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
        if self.circular and kinds - {"ellipse", "taper", "drift", "ovality"}:
            raise ValueError("wave_estimator.march: a circular bore takes defects ellipse, ovality, taper, drift only")
        if not self.circular and (not self.axis_rect and kinds or kinds & {"ellipse", "ovality"}):
            raise ValueError("wave_estimator.march: wall defects need a rectangle with faces along the lattice axes")
        self._defects(m["defects"], m.get("seed"))
        self.kx = 2.0 * np.pi * sfft.fftfreq(self.lx.size, self.h)
        self.ky = 2.0 * np.pi * sfft.fftfreq(self.ly.size, self.h)
        sw = float(m["sponge"])
        ramp = lambda v, lo, hi: np.clip(np.maximum(lo + sw - v, v - (hi - sw)) / sw, 0.0, 1.0) ** 2 if sw > 0.0 else 0.0 * v
        self.sponge = (ramp(self.lx, self.lx[0], self.lx[-1])[:, None] + ramp(self.ly, self.ly[0], self.ly[-1])[None, :])
        self.families, self.pruned = [], 0
        self.exit_mask_full = self.fill(self.n_steps)
        self._cover_exit()
        self.n_images, self.line_points = [], []

    def _cover_exit(self):
        """Grow the exit window to every core cell of the exit section (slope, wall offsets) with 2 cells of margin."""
        from .wave import _fast_len
        win = list(self.win)
        for ax in range(2):
            cells = np.nonzero(self.exit_mask_full.sum(axis=1 - ax) > 0.0)[0]
            sl, g0 = self.slices[ax], self.grid[ax][0]
            lo, hi = min(sl.start, int(cells[0]) - 2), max(sl.stop, int(cells[-1]) + 3)
            if (lo, hi) != (sl.start, sl.stop):
                lo = max(lo, 0)
                n = min(_fast_len(hi - lo), self.exit_mask_full.shape[ax] - lo)
                self.slices[ax] = slice(lo, lo + n)
                win[2 * ax], win[2 * ax + 1] = g0 + lo, g0 + lo + n
        self.win = tuple(win)
        self.xs, self.ys = self.lx[self.slices[0]], self.ly[self.slices[1]]
        self.exit_mask = self.exit_mask_full[self.slices[0], self.slices[1]]

    # ------------------------------------------------------------- geometry along z

    def _g(self, z):
        """Funnel scale G(z) = 1 + g0 z + g1 z^2 of the centre and the radius (1 without a funnel)."""
        return 1.0 if self.funnel is None else 1.0 + self.funnel[0] * z + self.funnel[1] * z * z

    def _g_range(self):
        """Smallest and largest G on [0, L] (ends and the vertex of the parabola)."""
        g0, g1 = self.funnel
        vals = [self._g(0.0), self._g(self.length)]
        if g1 != 0.0 and 0.0 < -g0 / (2.0 * g1) < self.length:
            vals.append(self._g(-g0 / (2.0 * g1)))
        return min(vals), max(vals)

    def _defects(self, defects, seed):
        n = self.n_steps
        self.offsets = np.zeros((4, n + 1))              # wall offsets (x-, x+, y-, y+) at every step plane
        self.center_shift = np.zeros((2, n + 1))
        self.slope = self.rc = 0.0
        self.ratio = 1.0
        self.oval = None                                 # (amp, rate, angle): dr = amp cos 2 (phi - angle - rate z)
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
            elif t == "ovality":
                self.oval = (float(d["amp"]), float(d.get("rate", 0.0)), float(d.get("angle", 0.0)))
            elif t == "drift":
                for ax in range(2):
                    self.center_shift[ax] += float(d["dxy"][ax]) * z / self.length
        if self.oval is not None and self.ratio != 1.0:
            raise ValueError("wave_estimator.march: defects ellipse and ovality exclude each other")
        self.z_dependent = bool(np.any(self.offsets != 0.0) or np.any(self.center_shift != 0.0)
                                or (self.oval is not None and self.oval[1] != 0.0) or self.funnel is not None)
        self.separable = self.axis_rect and self.slope == 0.0 and self.rc == 0.0
        self._static = None if (self.z_dependent or self.separable) else self._mask2d(0)

    def _walls(self, i):
        """(xl, xr, yl, yr) of the rectangle or ellipse box at step plane i (a funnel scales centre and half-widths by G)."""
        g = self._g(i * self.dz)
        cx, cy = g * (self.c + self.center_shift[:, i])
        hx, hy = g * self.half
        o = self.offsets[:, i]
        return cx - hx + o[0], cx + hx + o[1], cy - hy + o[2], cy + hy + o[3]

    def _mask2d(self, i):
        """S x S supersampled inside fraction for polygons with slope / rounding and for ellipses."""
        xl, xr, yl, yr = self._walls(i)
        X = (self.lx[:, None] + self.off[None, :]).reshape(-1)
        Y = (self.ly[:, None] + self.off[None, :]).reshape(-1)
        cx, cy, ax, ay = (xl + xr) / 2.0, (yl + yr) / 2.0, (xr - xl) / 2.0, (yr - yl) / 2.0
        if self.circular and self.oval is not None:
            amp, rate, angle = self.oval
            psi = angle + rate * i * self.dz
            u = (X[:, None] - cx) * math.cos(psi) + (Y[None, :] - cy) * math.sin(psi)
            v = -(X[:, None] - cx) * math.sin(psi) + (Y[None, :] - cy) * math.cos(psi)
            inside = (u / (ax + amp)) ** 2 + (v / (ax - amp)) ** 2 < 1.0
        elif self.circular:
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
            if self.oval is not None:
                lo, hi = self._oval_chord(i, cx, cy, ax)
            else:
                t = np.clip(1.0 - ((self.ly - cy) / ay) ** 2, 0.0, None)
                w = ax * np.sqrt(t)
                lo, hi = cx - w[None, :], cx + w[None, :]
            x = self.lx[:, None]
            return np.clip((np.minimum(x + self.h / 2.0, hi) - np.maximum(x - self.h / 2.0, lo)) / self.h, 0.0, 1.0)
        return self._mask2d(i)

    def _oval_chord(self, i, cx, cy, a):
        """x interval of every lattice row inside the ellipse of semi-axes a + amp (along psi) and a - amp at plane i."""
        amp, rate, angle = self.oval
        psi = angle + rate * i * self.dz
        c, s = math.cos(psi), math.sin(psi)
        y = self.ly - cy
        qa = c * c / (a + amp) ** 2 + s * s / (a - amp) ** 2
        qb = 2.0 * y * c * s * (1.0 / (a + amp) ** 2 - 1.0 / (a - amp) ** 2)
        qc = y * y * (s * s / (a + amp) ** 2 + c * c / (a - amp) ** 2) - 1.0
        disc = qb * qb - 4.0 * qa * qc
        r = np.sqrt(np.clip(disc, 0.0, None))
        lo = np.where(disc > 0.0, cx + (-qb - r) / (2.0 * qa), cx)
        hi = np.where(disc > 0.0, cx + (-qb + r) / (2.0 * qa), cx)
        return lo[None, :], hi[None, :]

    # ------------------------------------------------------------- propagation

    def cache_transfer(self, k, delta, beta, budget_bytes):
        return None

    def entrance(self, xi, k):
        X, Y = self.lx[:, None], self.ly[None, :]
        if self.spot is None:
            lam = 2.0 * np.pi / k
            return np.exp(1j * k * ((X - xi[0]) ** 2 + (Y - xi[1]) ** 2) / (2.0 * self.z_in)) / (1j * lam * self.z_in)
        from .wave import spot_profile
        return spot_profile(self.spot, k, X, Y)

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

"""Stage 11: beamlet (Gaussian-beam summation) estimator, float64.

Each ray becomes a Gaussian beamlet: the central ray is
the engine trace; the complex 2x2 beam tensor Q = Gamma^-1 rides the
segments and bounces by tensor ABCD (gamma.py, Arnaud-Kogelnik general
astigmatism — a bounce is the wall's projected curvature tensor plus the
mirror flip; skew bounces couple the planes). On the screen the beamlet
deposits an elliptic phase spot over the pixels inside the window_sigmas
ellipse's bounding box instead of one bin. Implicit bores carry no
closed-form curvature or normal: their bounces are no-ops (the scalar-q
model of stage 11a, exact for an isotropic launch on flat walls).

The estimator is honest — no ray self-pair subtraction: the mode field is a
true coherent sum of beamlet fields, mu = |W| / sqrt(I*I_ref) with
I = sum_m w_m |g_m|^2 as is. beamlet.self_pairs: subtract switches to the
U-statistic: the ray diagonal sum_i |e_i|^2 (and e_i conj(e_i,ref)) comes
off each mode and n/(n-1) restores the scale. The rng stream matches the
other estimators, so the rays are the stage-2/14 rays.

One path for any CAPSYSRED_STAGE11_JOBS: the unit of work is a mode, each
mode's float64 W/I contribution enters an exact fixed-point accumulator (so
the totals do not depend on the order of completion), the float32 delete-one
rows go to a disk store, and the jackknife runs natively over pixel tiles.
jobs = 1 runs the same functions in-process; mu-beamlet.jsonl and the SVGs
are byte-identical for every jobs value.
"""

import cmath
import math
from array import array

from .._contracts import _physics_contract
from .gamma import EXACT_KINDS, bounce_lenses, inv2, propagate
from ...native import make_beamlet_grid
from ...screen import ScreenGrid
from ...shared.utils import zeros


def _flat_walls(optic) -> bool:
    walls = getattr(optic, "walls", None)
    return optic is not None and (
        walls is None or any(w.kind not in EXACT_KINDS for w in walls))


class BeamletField:
    """Per-mode complex beamlet fields per spectral line; honest mu totals."""

    def __init__(self, lines, screen: ScreenGrid, ref_pixel: int,
                 w0: float, n_sigmas: float, optic=None, use_native=True,
                 w0_t=None, waist_dz=0.0, self_pairs="keep"):
        self.kms = [float(l.k) for l in lines]
        # a beamlet fan sums to lambda * (point-source field): (k/k0)^2
        # restores the configured line weights in W and I
        self.wfs = [l.weight * (k / self.kms[0]) ** 2
                    for l, k in zip(lines, self.kms)]
        self.nl = len(self.kms)
        self.zrs = [0.5 * w0 * w0 * k for k in self.kms]   # z_R = w0^2*k/2
        self.w0, self.ns = w0, n_sigmas
        # anisotropic launch: w0_t along x, w0 along y, the same for every ray
        self.w0_t = w0 if w0_t is None else float(w0_t)
        self.zrt = [0.5 * self.w0_t * self.w0_t * k for k in self.kms]
        # waist plane waist_dz past the source: a virtual drift -waist_dz
        # before the first segment; its on-axis factor is divided out
        self.waist_dz = float(waist_dz)
        self.vfix = [(cmath.sqrt(1 + 1j * self.waist_dz / zt)
                      * cmath.sqrt(1 + 1j * self.waist_dz / zs)).conjugate()
                     for zt, zs in zip(self.zrt, self.zrs)]
        self.optic = optic
        walls = getattr(optic, "walls", None)
        self.flat_walls = _flat_walls(optic)
        self.curved = any(w.kind in EXACT_KINDS and w.kind != "polygon"
                          for w in walls or ())
        self.ref = ref_pixel
        self.grid = screen
        # delete-one cut: float32 rows leave ~nl*2^-24 relative residue
        # where exact 0 is due; x16 headroom, ~1e-6 at nl = 1
        self.loo_eps = (16 + self.nl) * 2.0 ** -24
        self.nx, self.ny = screen.nx, screen.ny
        self.zf = float(screen.z)
        self.xs = screen.xs()
        self.ys = screen.ys()
        self.dx = screen.exf / screen.nx
        self.dy = screen.eyf / screen.ny
        self.x0f, self.y0f = screen.x0f, screen.y0f
        # mu comes from the float64 totals; the delete-one-mode rows are
        # dense float32 arrays (sigma is statistical, 7 digits are plenty):
        # 12 B/pixel/mode
        self.W = {}         # pixel -> complex float64 total
        self.I = {}         # pixel -> float float64 total
        self.Ws = []        # [mode] array('f') interleaved re, im
        self.Is = []        # [mode] array('f')
        self.i_refs = []    # [mode] float32-consistent intensity at ref
        # arrival-point bins (uint32 per pixel); the shared path exports and
        # zeroes it per mode, the fold/finalize oracle keeps it running
        self.density = array("I", bytes(4 * self.nx * self.ny))
        self.w_sum, self.w_n = 0.0, 0   # mean spot width at screen (line 0)
        self.gamma_bad = 0  # deposits skipped: Im(G) lost negative-definiteness
        self.native = (make_beamlet_grid(screen.nx, screen.ny,
                                         screen.x0f, screen.y0f,
                                         screen.exf, screen.eyf,
                                         self.kms, self.zrs, self.zrt,
                                         n_sigmas)
                       if use_native else None)
        self._g = None
        # subtract: the ray self-pair diagonal per line, sum_i |e_i(P)|^2
        # and sum_i e_i(P) conj(e_i(ref)), tracked beside the field
        self.subtract = self_pairs == "subtract"
        self._si = self._sw = None
        if self.native is not None:
            self.native.set_reference(self.ref)
            if self.subtract:
                self.native.set_self_pairs(True)

    def new_mode(self):
        if self.native is not None:
            self.native.clear()
        else:
            self._g = [{} for _ in range(self.nl)]
            if self.subtract:
                self._si = [{} for _ in range(self.nl)]
                self._sw = [{} for _ in range(self.nl)]

    def prep(self, rec, zf=None):
        """The shared per-record work: floats, segments to the plane zf (the
        recorded arrival plane; default this one), wall lenses. Extra planes
        shift the result arithmetically (straight final flight) instead of
        re-doing it."""
        zf = self.zf if zf is None else zf
        opl = float(rec.opl)
        x, y = float(rec.point[0]), float(rec.point[1])
        dxf, dyf = float(rec.direction[0]), float(rec.direction[1])
        dzf = float(rec.direction[2])
        pts = [tuple(float(c) for c in p) for p in rec.refl]
        if pts:
            segs = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
            segs.append(math.dist(pts[-1], (x, y, zf)))
            segs.insert(0, max(opl - sum(segs), 0.0))
            # outgoing direction per bounce (curved walls only): toward the
            # next hit, the last one the recorded final direction
            outs = [None] * len(pts)
            if self.curved:
                outs = [tuple(q - p for p, q in zip(a, b))
                        for a, b in zip(pts, pts[1:])]
                outs.append((dxf, dyf, dzf))
            lenses = bounce_lenses(self.optic, pts, outs)
        else:
            segs, lenses = [opl], []
        # one launch frame for every ray: GBS needs a ray-independent Q0
        return x, y, dxf, dyf, dzf, opl, 0.0, segs, lenses

    def add_ray(self, rec, amps):
        """Deposit one beamlet; rec.pixel None = tail only (the center lies
        outside the window), still deposited. The spot is elliptic from
        G = Q^-1 at the screen: phase = tilt + (k/2)·δᵀRe(G)δ, envelope
        exp((k/2)·δᵀIm(G)δ) — without the tilt term k*(d_x*δx + d_y*δy) the
        beamlets of one point source disagree at a pixel and the spherical
        front never reconstructs."""
        x, y, dxf, dyf, _, opl, psi, segs, lenses = self.prep(rec)
        self.deposit(x, y, dxf, dyf, opl, psi, segs, lenses, amps, rec.pixel)

    def deposit(self, x, y, dxf, dyf, opl, psi, segs, lenses, amps, pixel):
        if pixel is not None:
            self.density[pixel] += 1
        if self.waist_dz:
            segs = [-self.waist_dz] + segs
            lenses = [(math.nan, 0.0, 0.0, 0.0)] + lenses
            amps = [a * f for a, f in zip(amps, self.vfix)]
        if self.native is not None:
            w_spot, bad = self.native.add_ray(
                x, y, dxf, dyf, opl, psi, segs,
                [v for lens in lenses for v in lens], list(amps))
            self.gamma_bad += bad
            if w_spot >= 0.0:
                self.w_sum += w_spot
                self.w_n += 1
            return
        for m in range(self.nl):
            km = self.kms[m]
            q, a_geo = propagate((self.zrt[m], self.zrs[m], psi), segs, lenses)
            gm = inv2(q)
            gi = (gm[0].imag, gm[1].imag, gm[2].imag)
            mean = 0.5 * (gi[0] + gi[2])
            dev = math.hypot(0.5 * (gi[0] - gi[2]), gi[1])
            if mean + dev >= 0.0:      # beam blew up: no Gaussian to deposit
                self.gamma_bad += 1
                continue
            w_hi = math.sqrt(-2.0 / (km * (mean + dev)))   # widest spot axis
            if m == 0:
                w_lo = math.sqrt(-2.0 / (km * (mean - dev)))
                self.w_sum += math.sqrt(w_hi * w_lo)
                self.w_n += 1
            pref = amps[m] * a_geo.conjugate() * cmath.exp(1j * km * opl)
            tx, ty = km * dxf, km * dyf
            hxx, hxy, hyy = (0.5 * km * v for v in gm)   # (k/2)·G, complex
            # per-axis bounding box of the ns-sigma ellipse (det > 0 here)
            det_gi = gi[0] * gi[2] - gi[1] * gi[1]
            rx = self.ns * math.sqrt(-2.0 * gi[2] / (km * det_gi))
            ry = self.ns * math.sqrt(-2.0 * gi[0] / (km * det_gi))
            ix_lo = max(0, int(math.floor((x - rx - self.x0f) / self.dx)))
            ix_hi = min(self.nx - 1, int(math.floor((x + rx - self.x0f) / self.dx)))
            iy_lo = max(0, int(math.floor((y - ry - self.y0f) / self.dy)))
            iy_hi = min(self.ny - 1, int(math.floor((y + ry - self.y0f) / self.dy)))
            if ix_lo > ix_hi or iy_lo > iy_hi:
                continue
            g = self._g[m]
            si = sw = None
            ref_c = None
            if self.subtract:
                si, sw = self._si[m], self._sw[m]
                iy_r, ix_r = divmod(self.ref, self.nx)
                if iy_lo <= iy_r <= iy_hi and ix_lo <= ix_r <= ix_hi:
                    dx_r, dy_r = self.xs[ix_r] - x, self.ys[iy_r] - y
                    quad = (hxx * (dx_r * dx_r) + 2.0 * hxy * (dx_r * dy_r)
                            + hyy * (dy_r * dy_r))
                    ref_c = (pref * cmath.exp(complex(
                        quad.imag, quad.real + tx * dx_r + ty * dy_r))).conjugate()
            for iy in range(iy_lo, iy_hi + 1):
                dy_off = self.ys[iy] - y
                cy = hyy * (dy_off * dy_off)
                phase_y = ty * dy_off
                row = iy * self.nx
                for ix in range(ix_lo, ix_hi + 1):
                    dx_off = self.xs[ix] - x
                    quad = (hxx * (dx_off * dx_off)
                            + 2.0 * hxy * (dx_off * dy_off) + cy)
                    val = pref * cmath.exp(complex(
                        quad.imag, quad.real + tx * dx_off + phase_y))
                    pix = row + ix
                    prev = g.get(pix)
                    g[pix] = val if prev is None else prev + val
                    if si is not None:
                        si[pix] = si.get(pix, 0.0) + abs(val) ** 2
                        if ref_c is not None:
                            sw[pix] = sw.get(pix, 0j) + val * ref_c

    def _check_pairs(self, n_rays):
        if self.subtract and int(n_rays) < 2:
            raise ValueError("beamlet self_pairs: subtract pairs distinct rays — needs at least "
                             f"two rays per mode, got {int(n_rays)}")

    def fold_mode(self, n_rays=0):
        """Oracle path (tests): fold into the retained native rows/totals,
        or into the Python dicts without the native grid. n_rays: the mode's
        quadrature rays, the U-statistic scale n/(n-1) under subtract."""
        self._check_pairs(n_rays)
        if self.native is not None:
            # scan, totals AND the delete-one rows all stay in C++
            self.native.fold(self.wfs, self.ref, self.subtract, int(n_rays))
            return
        npix = self.nx * self.ny
        w_row = array("f", bytes(8 * npix))
        i_row = array("f", bytes(4 * npix))
        f = n_rays / (n_rays - 1) if self.subtract and n_rays > 1 else 1.0
        for m in range(self.nl):
            wf = self.wfs[m] * f
            g = self._g[m]
            g_ref = g.get(self.ref)
            ref_c = g_ref.conjugate() if g_ref is not None else None
            si = self._si[m] if self.subtract else {}
            sw = self._sw[m] if self.subtract else {}
            for pixel, value in g.items():
                a2 = abs(value) ** 2 - si.get(pixel, 0.0)
                self.I[pixel] = self.I.get(pixel, 0.0) + wf * a2
                i_row[pixel] += wf * a2
                if ref_c is not None:
                    cross = value * ref_c - sw.get(pixel, 0j)
                    self.W[pixel] = self.W.get(pixel, 0j) + wf * cross
                    w_row[2 * pixel] += wf * cross.real
                    w_row[2 * pixel + 1] += wf * cross.imag
        self.Ws.append(w_row)
        self.Is.append(i_row)
        self.i_refs.append(float(i_row[self.ref]))
        self._g = self._si = self._sw = None

    def fold_export(self, n_rays=0):
        """Shared path: one mode's (w_row f32, i_row f32, dW f64, dI f64,
        i_ref) exactly as fold builds the rows; nothing is retained."""
        if self.native is None:
            raise RuntimeError("fold_export needs the native BeamletGrid")
        self._check_pairs(n_rays)
        return self.native.fold_export(self.wfs, self.ref, self.subtract,
                                       int(n_rays))

    def take_density(self) -> bytes:
        """The mode's arrival counts (uint32 per pixel), then zeroed."""
        out = self.density.tobytes()
        self.density = array("I", bytes(4 * self.nx * self.ny))
        return out

    def _density_rows(self, nx: int, ny: int):
        density = [[0.0] * nx for _ in range(ny)]
        for pixel, count in enumerate(self.density):
            if count:
                iy, ix = divmod(pixel, nx)
                density[iy][ix] = float(count)
        return density

    def finalize(self, nx: int, ny: int):
        """Row-major [iy][ix] maps: mu, mu_err (delete-one-mode jackknife),
        dubious, intensity, density. I itself is the mu denominator (under
        subtract the U-statistic I can go negative at sparse pixels: such
        pixels keep mu = 0 and are flagged dubious); the trust flags are
        sigma > 1, |mu| pinned at 1 or 0 with sigma below loo_eps, fewer
        than 2 usable leave-one-out modes, or a reference with I <= 0."""
        if self.native is not None:
            return self._finalize_native(nx, ny)
        n_modes = len(self.Ws)
        W, I = self.W, self.I
        i_ref = I.get(self.ref, 0.0)
        mu, err, dubious = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
        intensity = zeros(nx, ny)
        for pixel, value in I.items():
            iy, ix = divmod(pixel, nx)
            intensity[iy][ix] = value
        density = self._density_rows(nx, ny)
        Ws, Is, i_refs, eps = self.Ws, self.Is, self.i_refs, self.loo_eps
        if not i_ref > 0.0:   # unlit or U-negative reference: no lit pixel is estimable
            for pixel, i_pix in I.items():
                if i_pix != 0.0:
                    iy, ix = divmod(pixel, nx)
                    dubious[iy][ix] = 1.0
        else:
            for pixel, i_pix in I.items():
                if i_pix <= 0.0:
                    if i_pix < 0.0:          # U-negative: lit, not estimable
                        iy, ix = divmod(pixel, nx)
                        dubious[iy][ix] = 1.0
                    continue
                w = W.get(pixel, 0j)
                iy, ix = divmod(pixel, nx)
                mu[iy][ix] = min(abs(w) / math.sqrt(i_pix * i_ref), 1.0)
                wr, wi = w.real, w.imag
                # sole-mode pixels are skipped; the cut is loo_eps-relative
                eps_i, eps_r = self.loo_eps * i_pix, self.loo_eps * i_ref
                loo = []
                for s in range(n_modes):
                    i_s = i_pix - Is[s][pixel]
                    iref_s = i_ref - i_refs[s]
                    if i_s > eps_i and iref_s > eps_r:
                        row_s = Ws[s]
                        dr = wr - row_s[2 * pixel]
                        di = wi - row_s[2 * pixel + 1]
                        loo.append(min(math.hypot(dr, di)
                                       / math.sqrt(i_s * iref_s), 1.0))
                if len(loo) > 1:
                    mean = sum(loo) / len(loo)
                    err[iy][ix] = math.sqrt(
                        sum((v - mean) ** 2 for v in loo)
                        * (len(loo) - 1) / len(loo))
                if (err[iy][ix] > 1.0 or len(loo) < 2
                        or ((mu[iy][ix] >= 1.0 - eps or mu[iy][ix] == 0.0)
                            and err[iy][ix] <= eps)):
                    dubious[iy][ix] = 1.0
        w_mean = self.w_sum / self.w_n if self.w_n else 0.0
        return {"mu": mu, "mu_err": err, "dubious": dubious,
                "intensity": intensity, "density": density,
                "ref_pixel": self.ref, "i_ref": i_ref, "w_mean": w_mean,
                "gamma_bad": self.gamma_bad, "flat_walls": self.flat_walls}

    def _finalize_native(self, nx: int, ny: int):
        """Oracle path: totals and the delete-one scan live in C++; here
        the dense results are only unpacked into row lists."""
        i_tot = array("d")
        i_tot.frombytes(self.native.totals()[1])
        mu_b, err_b, dub_b = self.native.jackknife(self.ref, self.loo_eps)
        mu_t = array("d")
        mu_t.frombytes(mu_b)
        err_t = array("d")
        err_t.frombytes(err_b)
        rows = lambda a: [list(a[iy * nx:(iy + 1) * nx]) for iy in range(ny)]
        w_mean = self.w_sum / self.w_n if self.w_n else 0.0
        return {"mu": rows(mu_t), "mu_err": rows(err_t),
                "dubious": rows(array("d", (float(b) for b in dub_b))),
                "intensity": rows(i_tot), "density": self._density_rows(nx, ny),
                "ref_pixel": self.ref, "i_ref": i_tot[self.ref],
                "w_mean": w_mean, "gamma_bad": self.gamma_bad,
                "flat_walls": self.flat_walls}


def _reference_pixel(grid, reference):
    xy = reference if reference else (grid.cxf, grid.cyf)
    pix = grid.pixel(xy)
    if pix is None:
        raise ValueError(f"screen reference {tuple(xy)} lies outside the window")
    return pix


def _scene_source(sim, scene):
    cfg = sim.cfg
    return cfg.free_source if scene == "free" else cfg.capillary.source


def _scene_optic(sim, scene):
    if scene != "capillary":
        return None
    from ...surfaces import CapillaryBundle
    cap = sim.cfg.capillary
    return CapillaryBundle(cap.bores, cap.z0, cap.z1)


def _launch(sim, scene, plane_z: float):
    """(w0_t, waist_dz) of the scene's beamlet launch, from the typed config."""
    cfg = sim.cfg
    z_src = float(_scene_source(sim, scene).position[2])
    w0_t = cfg.beamlet_w0_t
    if w0_t == "auto":   # Fresnel scale of the scene's source->screen flight
        w0_t = math.sqrt(float(sim.lam) * (plane_z - z_src) / math.pi)
    waist_dz = 0.0 if cfg.beamlet_waist_z is None else cfg.beamlet_waist_z - z_src
    return (None if w0_t is None else float(w0_t)), float(waist_dz)


def _stage11_contract(sim, scene, planes, n_modes) -> dict:
    """Everything the deposit depends on, from the typed config: the worker
    rebuilds it from cfg.raw and must obtain the same dict, else the run
    refuses (a programmatic drift between raw and typed config)."""
    cfg = sim.cfg
    w0_t, waist_dz = _launch(sim, scene, planes[0]["z"])
    return {"scene": scene, "n_modes": int(n_modes), "planes": planes,
            "physics": _physics_contract(sim),
            "beamlet": {"w0": float(cfg.beamlet_w0), "w0_t": w0_t,
                        "ns": float(cfg.beamlet_ns), "waist_dz": waist_dz,
                        "self_pairs": cfg.beamlet_self_pairs,
                        "loo_eps": (16 + len(sim.lines)) * 2.0 ** -24},
            "amplitude_min": float(cfg.amplitude_min)}

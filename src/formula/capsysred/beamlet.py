"""Stage 11: beamlet (Gaussian-beam summation) estimator, float64.

Each ray becomes a Gaussian beamlet (doc/beamlets.ru.md): the central ray is
the engine trace; the complex 2x2 beam tensor Q = Gamma^-1 rides the
segments and bounces by tensor ABCD (gamma.py, Arnaud-Kogelnik general
astigmatism — grazing bounces focus sagittally f_s = R/(2 sin) and, on
curved walls, meridionally f_t = R sin/2; skew rays rotate the azimuth and
couple the planes). On the screen the beamlet deposits an elliptic phase
spot over the pixels within window_sigmas of the widest axis instead of one
bin. Implicit bores carry no closed-form curvature: their bounces are flat
(the scalar-q model of stage 11a).

The estimator is honest — no ray self-pair subtraction: the mode field is a
true coherent sum of beamlet fields, mu = |W| / sqrt(I*I_ref) with
I = sum_m w_m |g_m|^2 as is. The rng stream matches _mc_stage, so the rays
are the stage-2/6 rays.
"""

import cmath
import math
import time

from .altcoh import FloatLineAmplitudes
from .gamma import EXACT_KINDS, bounce_lenses, inv2, propagate
from .native import make_beamlet_grid
from .progress import Progress
from .rays import scene_stream
from .screen import ScreenGrid


class BeamletField:
    """Per-mode complex beamlet fields per spectral line; honest mu totals."""

    def __init__(self, lines, screen: ScreenGrid, ref_pixel: int,
                 w0: float, n_sigmas: float, optic=None, use_native=True,
                 w0_t=None):
        self.kms = [float(l.k) for l in lines]
        self.wfs = [l.weight for l in lines]
        self.nl = len(self.kms)
        self.zrs = [0.5 * w0 * w0 * k for k in self.kms]   # z_R = w0^2*k/2
        self.w0, self.ns = w0, n_sigmas
        # anisotropic launch: tangential waist may differ from the sagittal
        # one (the channel wants the matched mode, the free flight the
        # Fresnel scale); the ellipse is oriented per ray, see add_ray
        self.w0_t = w0 if w0_t is None else float(w0_t)
        self.zrt = [0.5 * self.w0_t * self.w0_t * k for k in self.kms]
        self.optic = optic
        self.flat_walls = any(w.kind not in EXACT_KINDS
                              for w in getattr(optic, "walls", ()))
        self.ref = ref_pixel
        self.grid = screen
        self.nx, self.ny = screen.nx, screen.ny
        self.zf = float(screen.z)
        self.xs = screen.xs()
        self.ys = screen.ys()
        self.dx = screen.exf / screen.nx
        self.dy = screen.eyf / screen.ny
        self.x0f, self.y0f = screen.x0f, screen.y0f
        self.W = {}         # pixel -> complex sum_s w_m g g*_ref
        self.I = {}         # pixel -> float sum_s w_m |g|^2
        self.density = {}   # pixel -> ray count (arrival-point bin)
        self.w_sum, self.w_n = 0.0, 0   # mean spot width at screen (line 0)
        self.gamma_bad = 0  # deposits skipped: Im(G) lost negative-definiteness
        self.native = (make_beamlet_grid(screen.nx, screen.ny,
                                         screen.x0f, screen.y0f,
                                         screen.exf, screen.eyf,
                                         self.kms, self.zrs, self.zrt,
                                         n_sigmas)
                       if use_native else None)
        self._g = None

    def new_mode(self):
        if self.native is not None:
            self.native.clear()
        else:
            self._g = [{} for _ in range(self.nl)]

    def prep(self, rec):
        """The shared per-record work: floats, segments to THIS plane, wall
        lenses, ellipse azimuth. Extra planes shift the result arithmetically
        (straight final flight) instead of re-doing it."""
        opl = float(rec.opl)
        x, y = float(rec.point[0]), float(rec.point[1])
        dxf, dyf = float(rec.direction[0]), float(rec.direction[1])
        dzf = float(rec.direction[2])
        pts = [tuple(float(c) for c in p) for p in rec.refl]
        if pts:
            segs = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
            segs.append(math.dist(pts[-1], (x, y, self.zf)))
            segs.insert(0, max(opl - sum(segs), 0.0))
            lenses = bounce_lenses(self.optic, pts,
                                   [float(s) for s in rec.sins])
        else:
            segs, lenses = [opl], []
        # ellipse orientation: the channel frame of the first bounce, or the
        # ray's own transverse azimuth when it never touches a wall
        psi = lenses[0][0] if lenses else math.atan2(dyf, dxf)
        return x, y, dxf, dyf, dzf, opl, psi, segs, lenses

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
            self.density[pixel] = self.density.get(pixel, 0) + 1
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
            r = self.ns * w_hi
            ix_lo = max(0, int(math.floor((x - r - self.x0f) / self.dx)))
            ix_hi = min(self.nx - 1, int(math.floor((x + r - self.x0f) / self.dx)))
            iy_lo = max(0, int(math.floor((y - r - self.y0f) / self.dy)))
            iy_hi = min(self.ny - 1, int(math.floor((y + r - self.y0f) / self.dy)))
            if ix_lo > ix_hi or iy_lo > iy_hi:
                continue
            g = self._g[m]
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

    def fold_mode(self):
        for m in range(self.nl):
            wf = self.wfs[m]
            if self.native is not None:
                items = self.native.items(m)
                g_ref = self.native.at(m, self.ref)
                ref_c = g_ref.conjugate() if g_ref != 0 else None
            else:
                g = self._g[m]
                items = g.items()
                g_ref = g.get(self.ref)
                ref_c = g_ref.conjugate() if g_ref is not None else None
            for pixel, value in items:
                a2 = value.real * value.real + value.imag * value.imag
                self.I[pixel] = self.I.get(pixel, 0.0) + wf * a2
                if ref_c is not None:
                    self.W[pixel] = self.W.get(pixel, 0j) + wf * (value * ref_c)
        self._g = None

    def finalize(self, nx: int, ny: int):
        """Row-major [iy][ix] maps: mu, intensity, density."""
        zeros = lambda: [[0.0] * nx for _ in range(ny)]
        mu, intensity, density = zeros(), zeros(), zeros()
        i_ref = self.I.get(self.ref, 0.0)
        for pixel, value in self.I.items():
            iy, ix = divmod(pixel, nx)
            intensity[iy][ix] = value
        for pixel, count in self.density.items():
            iy, ix = divmod(pixel, nx)
            density[iy][ix] = float(count)
        if i_ref > 0.0:
            for pixel, w in self.W.items():
                i_pix = self.I.get(pixel, 0.0)
                if i_pix <= 0.0:
                    continue
                iy, ix = divmod(pixel, nx)
                mu[iy][ix] = min(abs(w) / math.sqrt(i_pix * i_ref), 1.0)
        w_mean = self.w_sum / self.w_n if self.w_n else 0.0
        return {"mu": mu, "intensity": intensity, "density": density,
                "ref_pixel": self.ref, "i_ref": i_ref, "w_mean": w_mean,
                "gamma_bad": self.gamma_bad, "flat_walls": self.flat_walls}


def run_beamlet_stage(sim, label, scene, src_cfg, scr_cfg, optic, aim_factory,
                      seed_offset: int, quick: int, extra_screens=()):
    """The beamlet deposit over the scene's ray records — from the shared
    rays file when it matches, else traced (the stage-2/6 rng stream).

    ONE pass serves every screen: the per-record prep (parse, segments, wall
    lenses, azimuth) is shared, and each extra plane only shifts the straight
    final flight — arrival x + dx*s, opl + s, last segment + s with
    s = (z_i - z_main)/dz. The launch waist is resolved once from the main
    flight (one beamlet, one launch). Returns the main-plane result with the
    extra planes under "extras"."""
    cfg = sim.cfg
    screen = ScreenGrid(scr_cfg)
    n_modes, n_rays = src_cfg.budget(quick)
    amps_of = FloatLineAmplitudes(cfg.material, sim.lines, cfg.precision)
    w0_t = cfg.beamlet_w0_t
    if w0_t == "auto":   # Fresnel scale of the scene's source->screen flight
        flight = float(scr_cfg.z) - float(src_cfg.position[2])
        w0_t = math.sqrt(float(sim.lam) * flight / math.pi)

    def make(scr):
        grid = ScreenGrid(scr)
        return BeamletField(sim.lines, grid, grid.ref_pixel(scr.reference),
                            cfg.beamlet_w0, cfg.beamlet_ns, optic, w0_t=w0_t)

    def blank():
        return {"emitted": 0, "screen": 0, "absorbed": 0, "lost": 0,
                "off_window": 0}

    planes = [(make(scr_cfg), 0.0, blank())]
    for scr in extra_screens:
        field = make(scr)
        planes.append((field, float(field.zf) - float(screen.z), blank()))
    records, rays_from = scene_stream(sim, scene, src_cfg, scr_cfg, optic,
                                      aim_factory, seed_offset, quick)
    progress = Progress(label, n_modes * n_rays)
    t0 = time.time()
    mode_cur = None
    for rec in records:
        if rec.mode != mode_cur:
            if mode_cur is not None:
                for field, _, _ in planes:
                    field.fold_mode()
            for field, _, _ in planes:
                field.new_mode()
            mode_cur = rec.mode
        fate, amps = rec.fate, None
        if fate == "screen":
            amps = amps_of([float(s) for s in rec.sins])
            if (cfg.amplitude_min > 0.0
                    and max(abs(a) for a in amps) < cfg.amplitude_min):
                fate = "absorbed"
        if fate == "screen":
            # tail beamlets (center outside a window) still deposit there
            x, y, dxf, dyf, dzf, opl, psi, segs, lenses = planes[0][0].prep(rec)
            for field, dz, st in planes:
                st["emitted"] += 1
                if dz == 0.0:
                    xi, yi, opl_i, segs_i, pix = x, y, opl, segs, rec.pixel
                else:
                    step = dz / dzf
                    xi, yi = x + dxf * step, y + dyf * step
                    opl_i = opl + step
                    segs_i = ([opl_i] if not lenses and len(segs) == 1
                              else segs[:-1] + [segs[-1] + step])
                    pix = field.grid.pixel((xi, yi))
                field.deposit(xi, yi, dxf, dyf, opl_i, psi, segs_i, lenses,
                              amps, pix)
                st["screen" if pix is not None else "off_window"] += 1
        else:
            for _, _, st in planes:
                st["emitted"] += 1
                st[fate] += 1
        progress.step()
    if mode_cur is not None:
        for field, _, _ in planes:
            field.fold_mode()
    progress.finish(f"on screen {planes[0][2]['screen']:,}")
    seconds = time.time() - t0

    def pack(field, st):
        return {"maps": field.finalize(field.nx, field.ny), "screen": field.grid,
                "stats": st, "rays_from": rays_from, "w0_t": field.w0_t,
                "n_modes": n_modes, "n_rays": n_rays, "seconds": seconds}

    out = pack(planes[0][0], planes[0][2])
    out["extras"] = [pack(field, st) for field, _, st in planes[1:]]
    return out

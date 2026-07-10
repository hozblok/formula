"""Stage 11: beamlet (Gaussian-beam summation) estimator, float64.

Each ray becomes a Gaussian beamlet (doc/beamlets.ru.md): the central ray is
the engine trace; the scalar complex beam parameter only drifts, q = i*pi*
w0^2/lambda + L — straight walls reflect the meridional plane like a flat
mirror (curved-wall focusing is the stage-11b Gamma tensor). On the screen
the beamlet deposits a soft phase spot over the pixels within window_sigmas*w
of the arrival point instead of one bin.

The estimator is honest — no ray self-pair subtraction: the mode field is a
true coherent sum of beamlet fields, mu = |W| / sqrt(I*I_ref) with
I = sum_m w_m |g_m|^2 as is. The rng stream matches _mc_stage, so the rays
are the stage-2/6 rays.
"""

import cmath
import math
import random
import time

from .altcoh import FloatLineAmplitudes
from .native import make_tracer
from .progress import Progress
from .screen import ScreenGrid
from .source import Source


class BeamletField:
    """Per-mode complex beamlet fields per spectral line; honest mu totals."""

    def __init__(self, lines, screen: ScreenGrid, ref_pixel: int,
                 w0: float, n_sigmas: float):
        self.kms = [float(l.k) for l in lines]
        self.wfs = [l.weight for l in lines]
        self.nl = len(self.kms)
        self.q0s = [complex(0.0, 0.5 * w0 * w0 * k)
                    for k in self.kms]   # i*z_R, z_R = pi*w0^2/lambda = w0^2*k/2
        self.w0, self.ns = w0, n_sigmas
        self.ref = ref_pixel
        self.nx, self.ny = screen.nx, screen.ny
        self.xs = screen.xs()
        self.ys = screen.ys()
        self.dx = screen.exf / screen.nx
        self.dy = screen.eyf / screen.ny
        self.x0f, self.y0f = screen.x0f, screen.y0f
        self.W = {}         # pixel -> complex sum_s w_m g g*_ref
        self.I = {}         # pixel -> float sum_s w_m |g|^2
        self.density = {}   # pixel -> ray count (arrival-point bin)
        self.w_sum, self.w_n = 0.0, 0   # mean beam width at screen (line 0)
        self._g = None

    def new_mode(self):
        self._g = [{} for _ in range(self.nl)]

    def add_ray(self, point, direction, amps, opl: float, pixel):
        """Deposit one beamlet; `pixel` is the arrival-point bin (None = tail
        only, the center lies outside the window). The spot phase is
        tilt + curvature: k*(d_x*δx + d_y*δy) + k*ρ²*Re(1/q)/2 — without the
        tilt term the beamlets of one point source disagree at a pixel and
        the spherical front never reconstructs."""
        x, y = float(point[0]), float(point[1])
        dxf, dyf = float(direction[0]), float(direction[1])
        if pixel is not None:
            self.density[pixel] = self.density.get(pixel, 0) + 1
        for m in range(self.nl):
            q = self.q0s[m] + opl
            invq = 1.0 / q
            lam = 2.0 * math.pi / self.kms[m]
            w = math.sqrt(-lam / (math.pi * invq.imag))
            if m == 0:
                self.w_sum += w
                self.w_n += 1
            gouy = math.atan2(q.real, q.imag)
            km = self.kms[m]
            pref = amps[m] * (self.w0 / w) * cmath.exp(1j * (km * opl - gouy))
            curv = 0.5 * km * invq.real
            tx, ty = km * dxf, km * dyf
            inv_w2 = 1.0 / (w * w)
            r = self.ns * w
            ix_lo = max(0, int(math.floor((x - r - self.x0f) / self.dx)))
            ix_hi = min(self.nx - 1, int(math.floor((x + r - self.x0f) / self.dx)))
            iy_lo = max(0, int(math.floor((y - r - self.y0f) / self.dy)))
            iy_hi = min(self.ny - 1, int(math.floor((y + r - self.y0f) / self.dy)))
            if ix_lo > ix_hi or iy_lo > iy_hi:
                continue
            g = self._g[m]
            for iy in range(iy_lo, iy_hi + 1):
                dy_off = self.ys[iy] - y
                dy2 = dy_off * dy_off
                phase_y = ty * dy_off
                row = iy * self.nx
                for ix in range(ix_lo, ix_hi + 1):
                    dx_off = self.xs[ix] - x
                    rho2 = dx_off * dx_off + dy2
                    val = pref * cmath.exp(complex(
                        -rho2 * inv_w2, curv * rho2 + tx * dx_off + phase_y))
                    pix = row + ix
                    prev = g.get(pix)
                    g[pix] = val if prev is None else prev + val

    def fold_mode(self):
        for m in range(self.nl):
            g, wf = self._g[m], self.wfs[m]
            g_ref = g.get(self.ref)
            ref_c = g_ref.conjugate() if g_ref is not None else None
            for pixel, value in g.items():
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
                "ref_pixel": self.ref, "i_ref": i_ref, "w_mean": w_mean}


def run_beamlet_stage(sim, label, src_cfg, scr_cfg, optic, aim_factory,
                      seed_offset: int, quick: int):
    """The stage-6 MC loop with the beamlet deposit; the rng stream matches
    _mc_stage exactly, so the rays are the stage-2/6 rays."""
    cfg = sim.cfg
    rng = random.Random(cfg.seed * 1000003 + seed_offset)
    source = Source(src_cfg, rng)
    screen = ScreenGrid(scr_cfg)
    n_modes = max(2, src_cfg.n_modes // quick)
    n_rays = max(20, src_cfg.n_rays // quick)
    amps_of = FloatLineAmplitudes(cfg.material, sim.lines, cfg.precision)
    field = BeamletField(sim.lines, screen, screen.ref_pixel(scr_cfg.reference),
                         cfg.beamlet_w0, cfg.beamlet_ns)
    aim = aim_factory(source, screen, rng)
    tracer = make_tracer(optic)
    stats = {"emitted": 0, "screen": 0, "absorbed": 0, "lost": 0,
             "off_window": 0}
    progress = Progress(label, n_modes * n_rays)
    t0 = time.time()
    for _ in range(n_modes):
        origin = source.mode_origin()
        field.new_mode()
        for _ in range(n_rays):
            direction = aim(origin)
            tr = tracer(origin, direction, optic, screen.z, cfg.max_bounces)
            stats["emitted"] += 1
            fate, amps = tr.fate, None
            if fate == "screen":
                amps = amps_of([float(s) for _, s in tr.reflections])
                if (cfg.amplitude_min > 0.0
                        and max(abs(a) for a in amps) < cfg.amplitude_min):
                    fate = "absorbed"
            if fate == "screen":
                # tail beamlets (center outside the window) still deposit
                pixel = screen.pixel(tr.point)
                field.add_ray(tr.point, tr.direction, amps, float(tr.opl), pixel)
                stats["screen" if pixel is not None else "off_window"] += 1
            else:
                stats[fate] += 1
            progress.step()
        field.fold_mode()
    progress.finish(f"on screen {stats['screen']:,}")
    maps = field.finalize(screen.nx, screen.ny)
    return {"maps": maps, "screen": screen, "stats": stats,
            "n_modes": n_modes, "n_rays": n_rays, "seconds": time.time() - t0}

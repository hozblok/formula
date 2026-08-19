"""Stage 10: the stage-6 estimator with per-mode rows and jackknife errors.

Float64 port of doc/mu28_legacy_fixed3_renamed.py brought back to running
sums: per mode s the sparse per-line fields g, sq fold into per-mode rows
W_s = sum_m w_m (g g*_ref - sq|_ref), Ic_s = sum_m w_m (|g|^2 - sq) instead
of summing in place. The totals reproduce the stage-6 |mu| (same rays,
float64 vs Number); the stored rows add what running sums cannot emit — a
delete-one-mode jackknife standard error for every pixel. 1D and 2D screens
(sparse pixel dicts throughout); memory O(n_modes * lit pixels).

Pixels that never see two rays of one mode have Ic = 0 by construction — a
float residual decides the sign, and mu degenerates to a coin flip between 0
and the 1.0 clamp (the isolated bright pixels of the stage-6 map). Stage 10
masks them: mu = err = 0, the "solid" map marks the estimable pixels. On top
of that the "dubious" map marks solid pixels whose estimate cannot be
trusted: sigma > 1 (nearly single-mode), every leave-one-out value pinned at
the 1.0 clamp (sigma = 0 is a lie), no usable jackknife, or Ic <= 0.
"""

import cmath
import math
import time

from .altcoh import FloatLineAmplitudes
from ..shared.progress import Progress
from ..rays import rescreen, scene_stream
from ..screen import ScatterRaster, ScreenGrid
from ..shared.utils import zeros


class JackknifeCoherence:
    """Per-mode W/Ic rows; mu from the totals, sigma from delete-one-mode."""

    def __init__(self, lines, ref_pixel: int):
        self.kms = [float(l.k) for l in lines]
        self.wfs = [l.weight for l in lines]
        self.nl = len(self.kms)
        self.ref = ref_pixel
        self.n_folded = 0
        self.Ws = []       # [mode] {pixel: complex W_s}
        self.Ics = []      # [mode] {pixel: float Ic_s}
        self.ic_refs = []  # [mode] float Ic_s at ref
        self.I = {}        # pixel -> full intensity (self-pairs kept)
        self.density = {}  # pixel -> ray count
        self.pairs = set()  # pixels with >= 2 same-mode rays (Ic estimable)
        self._g = self._sq = self._n = None

    def new_mode(self):
        self._g = [{} for _ in range(self.nl)]
        self._sq = [{} for _ in range(self.nl)]
        self._n = {}

    def add_ray(self, rec, amps):
        pixel, opl = rec.pixel, float(rec.opl)
        for m, (km, amp) in enumerate(zip(self.kms, amps)):
            term = amp * cmath.exp(1j * km * opl)
            g = self._g[m]
            g[pixel] = g.get(pixel, 0j) + term
            sq = self._sq[m]
            sq[pixel] = (sq.get(pixel, 0.0)
                         + amp.real * amp.real + amp.imag * amp.imag)
        self.density[pixel] = self.density.get(pixel, 0) + 1
        self._n[pixel] = self._n.get(pixel, 0) + 1

    def fold_mode(self):
        self.pairs.update(p for p, n in self._n.items() if n >= 2)
        w_row, ic_row = {}, {}
        for m in range(self.nl):
            g, sq, wf = self._g[m], self._sq[m], self.wfs[m]
            g_ref = g.get(self.ref)
            ref_c = g_ref.conjugate() if g_ref is not None else None
            for pixel, value in g.items():
                a2 = value.real * value.real + value.imag * value.imag
                self.I[pixel] = self.I.get(pixel, 0.0) + wf * a2
                ic_row[pixel] = ic_row.get(pixel, 0.0) + wf * (a2 - sq[pixel])
                if ref_c is not None:
                    cross = value * ref_c
                    if pixel == self.ref:        # drop ray self-pairs
                        cross -= sq[pixel]
                    w_row[pixel] = w_row.get(pixel, 0j) + wf * cross
        self.n_folded += 1
        self.Ws.append(w_row)
        self.Ics.append(ic_row)
        self.ic_refs.append(ic_row.get(self.ref, 0.0))
        self._g = self._sq = self._n = None

    def finalize(self, nx: int, ny: int):
        """Row-major [iy][ix] maps: mu, mu_err (jackknife), intensity, density."""
        n_modes = self.n_folded
        W, Ic = {}, {}
        for w_row in self.Ws:
            for pixel, w in w_row.items():
                W[pixel] = W.get(pixel, 0j) + w
        for ic_row in self.Ics:
            for pixel, v in ic_row.items():
                Ic[pixel] = Ic.get(pixel, 0.0) + v
        ic_ref = sum(self.ic_refs)
        mu, err, intensity, density = (zeros(nx, ny), zeros(nx, ny),
                                        zeros(nx, ny), zeros(nx, ny))
        solid, dubious = zeros(nx, ny), zeros(nx, ny)
        for pixel in self.pairs:
            iy, ix = divmod(pixel, nx)
            solid[iy][ix] = 1.0
        for pixel, value in self.I.items():
            iy, ix = divmod(pixel, nx)
            intensity[iy][ix] = value
        for pixel, count in self.density.items():
            iy, ix = divmod(pixel, nx)
            density[iy][ix] = float(count)
        ref_solid = self.ref in self.pairs
        if not ref_solid:
            # no reference pairs: nothing is estimable — flag every solid
            # pixel instead of leaving a confident-looking zero map
            for pixel in self.pairs:
                iy, ix = divmod(pixel, nx)
                dubious[iy][ix] = 1.0
        for pixel, ic in Ic.items():
            if pixel not in self.pairs or not ref_solid:   # Ic is a float residual
                continue
            iy, ix = divmod(pixel, nx)
            if ic <= 0.0 or ic_ref <= 0.0:   # shot-dominated pixel: mu stays 0
                dubious[iy][ix] = 1.0
                continue
            w = W.get(pixel)
            if w is None:   # no mode ever co-lit P and ref: zero cross data
                dubious[iy][ix] = 1.0
                continue
            mu[iy][ix] = min(abs(w) / math.sqrt(ic * ic_ref), 1.0)
            loo = []   # leave-one-mode-out mu; pixels pinned at the clamp give err 0
            for s in range(n_modes):
                w_s = w - self.Ws[s].get(pixel, 0j)
                ic_s = ic - self.Ics[s].get(pixel, 0.0)
                ic_ref_s = ic_ref - self.ic_refs[s]
                if ic_s > 0.0 and ic_ref_s > 0.0:
                    loo.append(min(abs(w_s) / math.sqrt(ic_s * ic_ref_s), 1.0))
            if len(loo) > 1:   # jackknife needs >= 2 usable modes
                mean = sum(loo) / len(loo)
                err[iy][ix] = math.sqrt(
                    sum((v - mean) ** 2 for v in loo) * (len(loo) - 1) / len(loo))
            # sigma beyond the |mu| range, all loo at the clamp, or no jackknife
            if (err[iy][ix] > 1.0 or len(loo) < 2
                    or (mu[iy][ix] >= 1.0 and err[iy][ix] == 0.0)):
                dubious[iy][ix] = 1.0
        return {"mu": mu, "mu_err": err, "intensity": intensity,
                "density": density, "solid": solid, "dubious": dubious,
                "ref_pixel": self.ref, "i_ref": max(ic_ref, 0.0),
                "n_modes": n_modes}


def run_jack_stage(sim, label, scene, src_cfg, scr_cfg, optic, aim_factory,
                   seed_offset: int, screen_cfg=None):
    """The stage-6 estimator over the scene's ray records — from the shared
    rays file when it matches, else traced (the stage-2/6 rng stream).
    screen_cfg re-bins the scr_cfg-plane records onto another screen."""
    cfg = sim.cfg
    target = screen_cfg or scr_cfg
    screen = ScreenGrid(target)
    scat = ScatterRaster(target)
    n_modes, n_rays = src_cfg.budget()
    amps_of = FloatLineAmplitudes(cfg.material, sim.lines, cfg.precision)
    jack = JackknifeCoherence(sim.lines, screen.ref_pixel(target.reference))
    records, rays_from = scene_stream(sim, scene, src_cfg, scr_cfg, optic,
                                      aim_factory, seed_offset)
    if screen_cfg is not None or getattr(sim.rays, "readonly", False):
        # --replay records may carry pixel ids of a different grid — re-bin
        records = rescreen(records, float(scr_cfg.z), screen)
    stats = {"emitted": 0, "screen": 0, "absorbed": 0, "lost": 0,
             "off_window": 0, "reflected_rays": 0, "reflections": 0,
             "bounce_hist": {}}
    progress = Progress(label, n_modes * n_rays)
    t0 = time.time()
    mode_cur = None
    for rec in records:
        if rec.mode != mode_cur:
            if mode_cur is not None:
                jack.fold_mode()
            jack.new_mode()
            mode_cur = rec.mode
        stats["emitted"] += 1
        nb = len(rec.sins)
        if nb:
            stats["reflected_rays"] += 1
            stats["reflections"] += nb
            stats["bounce_hist"][nb] = stats["bounce_hist"].get(nb, 0) + 1
        fate, amps = rec.fate, None
        if fate == "screen":
            amps = amps_of([float(s) for s in rec.sins])
            if (cfg.amplitude_min > 0.0
                    and max(abs(a) for a in amps) < cfg.amplitude_min):
                fate = "absorbed"
        if fate == "screen":
            scat.add(rec.point)
            if rec.pixel is None:
                stats["off_window"] += 1
            else:
                jack.add_ray(rec, amps)
                stats["screen"] += 1
        else:
            stats[fate] += 1
        progress.step()
    if mode_cur is not None:
        jack.fold_mode()
    progress.finish(f"on screen {stats['screen']:,}")
    maps = jack.finalize(screen.nx, screen.ny)
    return {"maps": maps, "screen": screen, "stats": stats, "scatter": scat,
            "rays_from": rays_from, "n_modes": n_modes, "n_rays": n_rays,
            "seconds": time.time() - t0, "src_cfg": src_cfg}

"""Stage 8: streaming low-rank sketch of W -> numerical CMD of the field.

W = Σ_s Σ_m w_m·g·g† is PSD of
low effective rank, so a fixed random Ω (M×r) and the running sum
Y += Σ_m w_m·g·(g†Ω) capture it in O(M·r) memory — the M×M matrix is never
materialized, which is what makes 2D screens (ny > 1) affordable. Nystrom
reconstruction gives any μ(P, P_ref) column in O(M·r) and the coherent-mode
spectrum λ_n (N_eff, n99) of the transmitted field from the r×r core.

A pairwise reference column (same algebra as coherence.CoherenceAccumulator,
self-pairs dropped) accumulates alongside on the same rays: the sketch column
must match it within the Nystrom tail error.

float64 like stage 7 (the maps are statistical estimators, noise >> 1e-15);
numpy is imported lazily — stage 8 is opt-in and the core stages stay
dependency-free.
"""

import math
import time

from .altcoh import FloatLineAmplitudes
from ..shared.progress import Progress
from ..rays import scene_stream
from ..screen import ScreenGrid


class SketchCoherence:
    """Pairwise reference column + streaming sketch over one mode stream."""

    def __init__(self, lines, npix: int, ref_pixel: int, rank: int, seed: int):
        import numpy as np
        self.np = np
        self.kms = np.array([float(l.k) for l in lines])
        self.wfs = [l.weight for l in lines]
        self.nl = len(self.wfs)
        self.M = npix
        self.ref = ref_pixel
        self.r = min(rank, npix)
        rng = np.random.default_rng(seed)
        self.Om = (rng.standard_normal((npix, self.r))
                   + 1j * rng.standard_normal((npix, self.r))) / math.sqrt(2.0)
        self.Y = np.zeros((npix, self.r), dtype=complex)
        self.Wp = np.zeros(npix, dtype=complex)      # pairwise vs ref
        self.I = np.zeros(npix)
        self.Ic = np.zeros(npix)                     # self-pair-free
        self.density = np.zeros(npix, dtype=int)
        self._g = self._sq = None

    def new_mode(self):
        self._g = [{} for _ in range(self.nl)]
        self._sq = [{} for _ in range(self.nl)]

    def add_ray(self, rec, amps):
        pixel, opl = rec.pixel, float(rec.opl)
        phases = self.np.exp(1j * self.kms * opl)
        for m in range(self.nl):
            amp = amps[m]
            g = self._g[m]
            g[pixel] = g.get(pixel, 0j) + amp * phases[m]
            sq = self._sq[m]
            sq[pixel] = sq.get(pixel, 0.0) + amp.real ** 2 + amp.imag ** 2
        self.density[pixel] += 1

    def fold_mode(self):
        np = self.np
        for m in range(self.nl):
            g, wf = self._g[m], self.wfs[m]
            if not g:
                continue
            idx = np.fromiter(g.keys(), dtype=int, count=len(g))
            vals = np.fromiter(g.values(), dtype=complex, count=len(g))
            sqv = np.fromiter(self._sq[m].values(), dtype=float, count=len(g))
            a2 = vals.real ** 2 + vals.imag ** 2
            self.I[idx] += wf * a2
            self.Ic[idx] += wf * (a2 - sqv)
            g_ref = g.get(self.ref)
            if g_ref is not None:
                self.Wp[idx] += wf * (vals * g_ref.conjugate())
                self.Wp[self.ref] -= wf * self._sq[m][self.ref]   # self-pairs
            y = vals.conj() @ self.Om[idx]                        # (r,) = g†Ω
            self.Y[idx] += wf * vals[:, None] * y[None, :]
        self._g = self._sq = None

    def _mu(self, col):
        np = self.np
        # float |e^{iθ}|²-1 ~ 1e-16 leaves a fake positive Ic on 1-ray pixels
        # where the Number engine gets an exact 0: gate Ic against I.
        mu = np.zeros(self.M)
        if self.Ic[self.ref] <= 1e-12 * self.I[self.ref]:
            return mu
        live = self.Ic > 1e-12 * self.I
        den = self.Ic * self.Ic[self.ref]
        mu[live] = np.minimum(np.abs(col[live]) / np.sqrt(den[live]), 1.0)
        return mu

    def finalize(self, nx: int, ny: int):
        np = self.np
        B = self.Om.conj().T @ self.Y
        B = (B + B.conj().T) / 2.0
        col = self.Y @ (np.linalg.pinv(B, rcond=1e-12) @ self.Y[self.ref].conj())
        col[self.ref] -= self.I[self.ref] - self.Ic[self.ref]     # diag pedestal
        mu_pair = self._mu(self.Wp)
        mu_sk = self._mu(col)
        # spectrum of Ŵ = Y·B⁺·Y† from the r×r core: eig(B^{+1/2}·Y†Y·B^{+1/2})
        s, U = np.linalg.eigh(B)
        keep = s > max(s.max(), 0.0) * 1e-12
        Bph = U[:, keep] / np.sqrt(s[keep])
        H = Bph.conj().T @ (self.Y.conj().T @ self.Y) @ Bph
        lam = np.linalg.eigvalsh((H + H.conj().T) / 2.0)[::-1]
        lam = np.maximum(lam, 0.0)
        neff = lam.sum() ** 2 / (lam ** 2).sum() if lam.any() else 0.0
        n99 = (int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.99) + 1)
               if lam.any() else 0)
        solid = self.Ic > 1e-6 * max(self.Ic.max(), 0.0)
        rms = float(np.sqrt(np.mean((mu_pair[solid] - mu_sk[solid]) ** 2)))
        grid = lambda v: [[float(v[iy * nx + ix]) for ix in range(nx)]
                          for iy in range(ny)]
        return {"mu_pair": grid(mu_pair), "mu_sketch": grid(mu_sk),
                "mu_diff": grid(abs(mu_pair - mu_sk)),
                "solid": grid(solid.astype(float)),
                "intensity": grid(self.I), "density": grid(self.density),
                "lam": [float(v) for v in lam], "neff": float(neff), "n99": n99,
                "rank": self.r, "rms_pair_sketch": rms,
                "solid_px": int(solid.sum()), "dark_px": int((self.density == 0).sum()),
                "ref_pixel": self.ref}


def run_sketch_stage(sim, label, scene, src_cfg, scr_cfg, optic, aim_factory,
                     seed_offset: int):
    """The sketch estimator over the scene's ray records — from the shared
    rays file when it matches, else traced (the stage-2/6 rng stream)."""
    cfg = sim.cfg
    screen = ScreenGrid(scr_cfg)
    n_modes, n_rays = src_cfg.budget()
    amps_of = FloatLineAmplitudes(cfg.material, sim.lines, cfg.precision)
    acc = SketchCoherence(sim.lines, screen.nx * screen.ny,
                          screen.ref_pixel(scr_cfg.reference),
                          cfg.sketch_rank, cfg.seed)
    records, rays_from = scene_stream(sim, scene, src_cfg, scr_cfg, optic,
                                      aim_factory, seed_offset)
    stats = {"emitted": 0, "screen": 0, "absorbed": 0, "lost": 0,
             "off_window": 0}
    progress = Progress(label, n_modes * n_rays)
    t0 = time.time()
    mode_cur = None
    for rec in records:
        if rec.mode != mode_cur:
            if mode_cur is not None:
                acc.fold_mode()
            acc.new_mode()
            mode_cur = rec.mode
        stats["emitted"] += 1
        fate, amps = rec.fate, None
        if fate == "screen":
            amps = amps_of([float(s) for s in rec.sins])
            if (cfg.amplitude_min > 0.0
                    and max(abs(a) for a in amps) < cfg.amplitude_min):
                fate = "absorbed"
        if fate == "screen":
            if rec.pixel is None:
                stats["off_window"] += 1
            else:
                acc.add_ray(rec, amps)
                stats["screen"] += 1
        else:
            stats[fate] += 1
        progress.step()
    if mode_cur is not None:
        acc.fold_mode()
    progress.finish(f"on screen {stats['screen']:,}")
    return {"maps": acc.finalize(screen.nx, screen.ny), "screen": screen,
            "stats": stats, "rays_from": rays_from, "n_modes": n_modes,
            "n_rays": n_rays, "seconds": time.time() - t0}

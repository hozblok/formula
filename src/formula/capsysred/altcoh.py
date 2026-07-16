"""Stage 7: alternative coherence estimators on one shared ray stream.

Axis change per doc/coherence-methods-analysis.ru.md §3.3/§3.4: three
estimators are fed IDENTICAL rays (same rng stream as stages 2/6), so map
differences are estimator effects, not statistics:

  pairwise  — the reference W(P, P_ref) estimator, float mirror of
              coherence.CoherenceAccumulator (baseline);
  fullw     — axis C: the complete Hermitian W(x1, x2) matrix from per-mode
              fields; whole mu map, no reference-pixel choice;
  wigner    — axis D: phase-space histogram B(x, u) of ray hits and exit
              directions, W(x1, x2) = sum_u B((x1+x2)/2, u) e^{i k u (x1-x2)};
              carries no intra-mode ray phases (the radiometric limit).

All three run in float64: the maps are statistical estimators with noise
floors far above 1e-15 (doc/mu28_capsysred.py: the float mirror of the Number
estimator lands on the same maps). 1D screens only (ny = 1).
"""

import cmath
import math
import time

from ..formula import Number
from .fresnel import FresnelAmplitude
from .progress import Progress
from .rays import scene_stream
from .screen import ScreenGrid


class FloatLineAmplitudes:
    """prod_j r(sin θ_j; E_m) per line in float64, constants from the exact
    material model; cross-checked against the Number Fresnel via check()."""

    def __init__(self, material, lines, precision):
        self.db = [(2.0 * float(material.delta(l.e_kev, precision=precision)),
                    2.0 * float(material.beta(l.e_kev, precision=precision)))
                   for l in lines]

    def __call__(self, sins):
        out = []
        for d2, b2 in self.db:
            a = 1.0 + 0.0j
            for s in sins:
                root = cmath.sqrt(s * s - d2 - 1j * b2)
                a *= (s - root) / (s + root)
            out.append(a)
        return out



class AltCoherence:
    """The three estimators over one mode stream (mode = coherent group)."""

    def __init__(self, lines, screen: ScreenGrid, ref_pixel: int):
        self.kms = [float(l.k) for l in lines]
        self.wfs = [l.weight for l in lines]
        self.nl = len(self.kms)
        self.nx = screen.nx
        self.xs = screen.xs()
        self.ref = ref_pixel
        nx = self.nx
        self.W = [0j] * nx          # pairwise vs ref
        self.I = [0.0] * nx         # full intensity
        self.Ic = [0.0] * nx        # self-pair-free intensity
        self.density = [0] * nx
        self.W2 = [[0j] * nx for _ in range(nx)]   # upper triangle j >= i
        # Wigner u-bin: phase error k_max*(du/2)*|Δx|_max stays below ~0.2 rad
        dx_max = max(self.xs[-1] - self.xs[self.ref],
                     self.xs[self.ref] - self.xs[0]) or screen.exf
        self.du = 0.4 / (max(self.kms) * dx_max)
        self.B = {}                 # (pixel, iu) -> per-line |amp|^2 sums
        self._g = self._sq = None

    def new_mode(self):
        self._g = [{} for _ in range(self.nl)]
        self._sq = [{} for _ in range(self.nl)]

    def add_ray(self, rec, amps):
        pixel, opl = rec.pixel, float(rec.opl)
        u = float(rec.direction[0]) / float(rec.direction[2])
        iu = math.floor(u / self.du + 0.5)
        cell = self.B.get((pixel, iu))
        if cell is None:
            cell = self.B[(pixel, iu)] = [0.0] * self.nl
        for m, (km, amp) in enumerate(zip(self.kms, amps)):
            term = amp * cmath.exp(1j * km * opl)
            g = self._g[m]
            g[pixel] = g.get(pixel, 0j) + term
            a2 = amp.real * amp.real + amp.imag * amp.imag
            sq = self._sq[m]
            sq[pixel] = sq.get(pixel, 0.0) + a2
            cell[m] += a2
        self.density[pixel] += 1

    def fold_mode(self):
        for m in range(self.nl):
            g, sq, wf = self._g[m], self._sq[m], self.wfs[m]
            g_ref = g.get(self.ref)
            ref_c = g_ref.conjugate() if g_ref is not None else None
            items = sorted(g.items())
            for pixel, value in items:
                a2 = value.real * value.real + value.imag * value.imag
                self.I[pixel] += wf * a2
                self.Ic[pixel] += wf * (a2 - sq[pixel])
                if ref_c is not None:
                    cross = value * ref_c
                    if pixel == self.ref:        # drop ray self-pairs
                        cross -= sq[pixel]
                    self.W[pixel] += wf * cross
            n = len(items)
            for a in range(n):
                i, gi = items[a]
                wgi = wf * gi
                row = self.W2[i]
                for b in range(a, n):
                    j, gj = items[b]
                    row[j] += wgi * gj.conjugate()
        self._g = self._sq = None

    # ------------------------------------------------------------ finalize

    def _mu(self, w, ic1, ic2) -> float:
        den = ic1 * ic2
        return min(abs(w) / math.sqrt(den), 1.0) if den > 0.0 else 0.0

    def finalize(self):
        nx, ref = self.nx, self.ref
        ic_ref = max(self.Ic[ref], 0.0)
        mu_pair = [self._mu(self.W[i], self.Ic[i], ic_ref) for i in range(nx)]
        selfsq = [self.I[i] - self.Ic[i] for i in range(nx)]
        mu_full = [[0.0] * nx for _ in range(nx)]
        for i in range(nx):
            ici = self.Ic[i]
            row = self.W2[i]
            for j in range(i, nx):
                w = row[j] - selfsq[i] if j == i else row[j]
                mu_full[i][j] = mu_full[j][i] = self._mu(w, ici, self.Ic[j])
        mu_wig, i_wig = self._wigner_mu()
        return {"mu_pair": mu_pair, "mu_full": mu_full,
                "mu_full_col": [mu_full[i][ref] for i in range(nx)],
                "mu_wigner": mu_wig, "i_wigner": i_wig,
                "intensity": list(self.I), "density": list(self.density),
                "ref_pixel": ref}

    def _wigner_mu(self):
        """mu(x, x_ref) and I(x) from the phase-space histogram alone."""
        nx, ref, du = self.nx, self.ref, self.du
        rows = [dict() for _ in range(nx)]
        i_wig = [0.0] * nx
        for (pixel, iu), cell in self.B.items():
            rows[pixel][iu] = cell
            i_wig[pixel] += sum(w * c for w, c in zip(self.wfs, cell))
        mu = [0.0] * nx
        for ix in range(nx):
            if i_wig[ix] <= 0.0 or i_wig[ref] <= 0.0:
                continue
            s = ix + ref
            mids = (s // 2,) if s % 2 == 0 else (s // 2, s // 2 + 1)
            dx = self.xs[ix] - self.xs[ref]
            tot = 0j
            for mid in mids:
                for iu, cell in rows[mid].items():
                    ph = iu * du * dx
                    for m in range(self.nl):
                        tot += (self.wfs[m] * cell[m]
                                * cmath.exp(1j * self.kms[m] * ph))
            mu[ix] = min(abs(tot) / len(mids)
                         / math.sqrt(i_wig[ix] * i_wig[ref]), 1.0)
        return mu, i_wig

    def wigner_grid(self, max_rows: int = 200):
        """Dense line-summed B(x, u) for display: (rows, u_lo, u_hi)."""
        if not self.B:
            return [[0.0] * self.nx], 0.0, 0.0
        ius = [iu for _, iu in self.B]
        lo, hi = min(ius), max(ius)
        step = max(1, (hi - lo + 1 + max_rows - 1) // max_rows)
        nrow = (hi - lo) // step + 1
        grid = [[0.0] * self.nx for _ in range(nrow)]
        for (pixel, iu), cell in self.B.items():
            grid[(iu - lo) // step][pixel] += sum(
                w * c for w, c in zip(self.wfs, cell))
        return grid, lo * self.du, (hi + 1) * self.du


def run_alt_stage(sim, label, scene, src_cfg, scr_cfg, optic, aim_factory,
                  seed_offset: int, quick: int):
    """The alternative estimators over the scene's ray records — from the
    shared rays file when it matches, else traced (the stage-2/6 rng stream)."""
    cfg = sim.cfg
    screen = ScreenGrid(scr_cfg)
    if screen.ny != 1:
        raise ValueError("stage 7 supports 1D screens (ny = 1) only")
    n_modes, n_rays = src_cfg.budget(quick)
    amps_of = FloatLineAmplitudes(cfg.material, sim.lines, cfg.precision)
    alt = AltCoherence(sim.lines, screen, screen.ref_pixel(scr_cfg.reference))
    records, rays_from = scene_stream(sim, scene, src_cfg, scr_cfg, optic,
                                      aim_factory, seed_offset, quick)
    stats = {"emitted": 0, "screen": 0, "absorbed": 0, "lost": 0,
             "off_window": 0}
    progress = Progress(label, n_modes * n_rays)
    t0 = time.time()
    mode_cur = None
    for rec in records:
        if rec.mode != mode_cur:
            if mode_cur is not None:
                alt.fold_mode()
            alt.new_mode()
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
                alt.add_ray(rec, amps)
                stats["screen"] += 1
        else:
            stats[fate] += 1
        progress.step()
    if mode_cur is not None:
        alt.fold_mode()
    progress.finish(f"on screen {stats['screen']:,}")
    mid = len(sim.lines) // 2
    r_num = FresnelAmplitude(cfg.material, sim.lines[mid].e_kev)
    s_probe = math.sin(1.0e-4)
    check = abs(amps_of([s_probe])[mid]
                - complex(r_num(Number(repr(s_probe), cfg.precision))))
    return {"maps": alt.finalize(), "alt": alt, "screen": screen,
            "stats": stats, "rays_from": rays_from, "n_modes": n_modes,
            "n_rays": n_rays, "seconds": time.time() - t0,
            "fresnel_check": check}

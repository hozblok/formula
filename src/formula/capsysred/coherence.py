"""Cross-spectral density vs a reference pixel and the degree of coherence.

Textbook estimator (legacy CAPSYS bugs C1..C6 fixed, plan-cap.ru.md §0): per
mode s and line m the screen field is g_{s,m}[pix] = sum_rays r_prod_m *
exp(i*k_m*L); W[pix] = sum_s sum_m w_m g g*(ref) — complex, averaged BEFORE the
modulus — and mu = |W| / sqrt(Ic*Ic_ref) with no seed in the denominator.

The per-ray amplitude may differ per line (energy-dependent Fresnel): add_ray
takes one Number for all lines or a list of len(lines).

Ray self-pairs are removed from W's diagonal and from the normalizing Ic
(|g|^2 - sum|term|^2): with finite rays per pixel the shot noise otherwise
biases |mu| down as 1/(1+1/n), and removing it also cancels the intra-pixel
phase-spread factor exactly. The displayed intensity keeps the full |g|^2.
Fields, W and the self-pair sums accumulate in Number; the intensity maps and
final mu are float (statistical estimators, noise floor >> 1e-15).
"""

import math

from .shared.nums import conj, exp_i, lift

class CoherenceAccumulator:
    def __init__(self, lines, ref_pixel: int, precision: int):
        # lines: [SpectralLine(e_kev, k: Number, weight: float)]
        self.kms = [line.k for line in lines]
        self.wfs = [line.weight for line in lines]
        self.wns = [lift(line.weight, precision) for line in lines]
        self.mono = len(lines) == 1
        self.ref = ref_pixel
        self.W = {}        # pixel -> Number (complex cross-spectral density vs ref)
        self.I = {}        # pixel -> float  (full intensity, weighted |g|^2)
        self.Ic = {}       # pixel -> float  (self-pair-free intensity for mu)
        self.density = {}  # pixel -> int    (ray counts)
        self._g = self._sq = None

    def new_mode(self):
        """Per-line sparse complex fields + per-line sum|term|^2 of one mode."""
        self._g = [{} for _ in self.kms]
        self._sq = [{} for _ in self.kms]

    def add_ray(self, rec, amplitudes):
        """amplitudes: one Number for every line, or [Number] per line."""
        pixel, opl = rec.pixel, rec.opl
        amps = (list(amplitudes) if isinstance(amplitudes, (list, tuple))
                else [amplitudes] * len(self.kms))
        if len(amps) != len(self.kms):
            raise ValueError(
                f"expected {len(self.kms)} per-line amplitudes, got {len(amps)}")
        for km, amp, g, sq in zip(self.kms, amps, self._g, self._sq):
            term = amp * exp_i(km * opl)
            prev = g.get(pixel)
            g[pixel] = term if prev is None else prev + term
            a2 = abs(amp)
            a2 = a2 * a2
            prev = sq.get(pixel)
            sq[pixel] = a2 if prev is None else prev + a2
        self.density[pixel] = self.density.get(pixel, 0) + 1

    def fold_mode(self):
        """Incoherent mode sum: W += w*g*conj(g_ref), I += w*|g|^2."""
        for m, (g, sq) in enumerate(zip(self._g, self._sq)):
            wf = self.wfs[m]
            for pixel, value in g.items():
                a2 = float(abs(value)) ** 2
                self.I[pixel] = self.I.get(pixel, 0.0) + wf * a2
                self.Ic[pixel] = (self.Ic.get(pixel, 0.0)
                                  + wf * (a2 - float(sq[pixel])))
            g_ref = g.get(self.ref)
            if g_ref is None:
                continue
            ref_c = conj(g_ref)
            for pixel, value in g.items():
                cross = value * ref_c
                if pixel == self.ref:            # drop ray self-pairs
                    cross = cross - sq[pixel]
                if not self.mono:
                    cross = cross * self.wns[m]
                prev = self.W.get(pixel)
                self.W[pixel] = cross if prev is None else prev + cross
        self._g = self._sq = None

    def finalize(self, nx: int, ny: int):
        """Maps as row-major [iy][ix] float lists: mu in [0,1], intensity, density."""
        ic_ref = max(self.Ic.get(self.ref, 0.0), 0.0)
        mu = [[0.0] * nx for _ in range(ny)]
        intensity = [[0.0] * nx for _ in range(ny)]
        density = [[0.0] * nx for _ in range(ny)]
        for pixel, value in self.I.items():
            iy, ix = divmod(pixel, nx)
            intensity[iy][ix] = value
            ic = self.Ic.get(pixel, 0.0)
            w = self.W.get(pixel)
            if w is not None and ic > 0.0 and ic_ref > 0.0:
                mu[iy][ix] = min(float(abs(w)) / math.sqrt(ic * ic_ref), 1.0)
        for pixel, count in self.density.items():
            iy, ix = divmod(pixel, nx)
            density[iy][ix] = float(count)
        return {"mu": mu, "intensity": intensity, "density": density,
                "ref_pixel": self.ref, "i_ref": ic_ref}

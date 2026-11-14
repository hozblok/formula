"""Spectral lines (E_m, k_m, weight) straight from config — no wave train, no FFT.

Temporal coherence is carried by the per-line phase sum over exp(i*k_m*L); the
line energy E_m feeds the per-line Fresnel amplitude (symbolic.LineAmplitudes).
"""

import math
from collections import namedtuple

from ..formula import NumberContext
from ..physical_constants import HC_KEV_ANGSTROM

_FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

SpectralLine = namedtuple("SpectralLine", ["e_kev", "k", "weight"])


def wavevector(energy_kev):
    """k = 2*pi/lambda [rad/m] for a photon energy in keV."""
    N = NumberContext(energy_kev.precision)
    return N.eval(f"2*pi*E/(({HC_KEV_ANGSTROM})*1e-10)", E=energy_kev)


def wavelength_m(energy_kev):
    N = NumberContext(energy_kev.precision)
    return N.eval(f"(({HC_KEV_ANGSTROM})*1e-10)/E", E=energy_kev)


def _table_pairs(path, N: NumberContext):
    """Two columns E_keV weight; '#' comments, blank lines and commas allowed."""
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            row = raw.split("#")[0].replace(",", " ").split()
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"spectrum table row needs 'E weight': {raw!r}")
            pairs.append((N(row[0], label=f"spectrum table {path}"), float(row[1])))
    if not pairs:
        raise ValueError(f"spectrum table {path!r} is empty")
    return pairs


def spectral_lines(cfg: dict, energy_kev):
    """Config -> [SpectralLine(e_kev, k, weight)], weights normalized to 1."""
    N = NumberContext(energy_kev.precision)
    mode = cfg.get("mode", "monochromatic")
    if mode == "monochromatic":
        return [SpectralLine(energy_kev, wavevector(energy_kev), 1.0)]
    if mode == "lines":
        pairs = [(N(l["energy_kev"], label="spectrum lines energy_kev"), float(l.get("weight", 1.0)))
                 for l in cfg["lines"]]
    elif mode == "table":
        pairs = _table_pairs(cfg["file"], N)
    elif mode == "gaussian":
        e0 = float(energy_kev)
        sigma = e0 * float(cfg["rel_fwhm"]) / _FWHM_TO_SIGMA
        n = int(cfg.get("n_lines", 7))
        span = float(cfg.get("n_sigma", 3.0)) * sigma
        pairs = []
        for j in range(n):
            e = e0 - span + 2.0 * span * j / (n - 1) if n > 1 else e0
            w = math.exp(-0.5 * ((e - e0) / sigma) ** 2)
            pairs.append((N(repr(e)), w))
    else:
        raise ValueError(f"unknown spectrum mode: {mode!r}")
    total = sum(w for _, w in pairs)
    return [SpectralLine(e, wavevector(e), w / total) for e, w in pairs]

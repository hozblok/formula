"""Optical constants and Fresnel reflection amplitudes."""

import cmath
from functools import lru_cache
from typing import Sequence

from ..formula import Number
from ..xray import GlassMaterial
from .spectrum import SpectralLine


@lru_cache(maxsize=256)
def optical_constants(material, energy, precision: int) -> tuple[float, float]:
    """Cached float64 (delta, beta) for a material and energy in keV."""
    return (
        float(material.delta(energy, precision=precision)),
        float(material.beta(energy, precision=precision)),
    )


class FresnelAmplitude:
    """Complex r(sin theta), s-pol, with 2*delta and 2*beta fixed at the energy.

    Same formula as xray.reflect_amplitude (cross-checked per run); constants are
    precomputed once so the per-bounce cost is pure Number arithmetic.
    """

    def __init__(self, material, energy_kev: Number):
        p = energy_kev.precision
        two = Number("2", p)
        self.d2 = material.delta(energy_kev, precision=p) * two
        self.b2 = material.beta(energy_kev, precision=p) * two
        self.b2i = self.b2 * Number("i", p)

    def __call__(self, sin_theta: Number) -> Number:
        root = (sin_theta * sin_theta - self.d2 + self.b2i).sqrt()
        return (sin_theta - root) / (sin_theta + root)

    def product(self, sins: list[Number]) -> Number:
        amp = Number("1", self.d2.precision)
        for s in sins:
            amp = amp * self(s)
        return amp


class FloatLineAmplitudes:
    """prod_j r(sin θ_j; E_m) per line in float64, constants from the exact
    material model; cross-checked against the Number Fresnel via check()."""

    def __init__(self, material: GlassMaterial, lines: Sequence[SpectralLine],
                 precision: int):
        # (2*delta, 2*beta) per line: the root is sqrt(sin^2 - 2d + 2ib)
        self.two_delta_beta: list[tuple[float, float]] = []
        for line in lines:
            delta, beta = optical_constants(material, line.e_kev, precision)
            self.two_delta_beta.append((2.0 * delta, 2.0 * beta))

    def __call__(self, sins: Sequence[float]) -> list[complex]:
        out = []
        for d2, b2 in self.two_delta_beta:
            a = 1.0 + 0.0j
            for s in sins:
                root = cmath.sqrt(s * s - d2 + 1j * b2)
                a *= (s - root) / (s + root)
            out.append(a)
        return out

"""Optical constants and Fresnel reflection amplitudes."""

import cmath
from typing import Sequence

from ..formula import NumberContext
from ..xray import GlassMaterial
from .spectrum import SpectralLine


class FresnelAmplitude:
    """Complex r(sin theta), s-pol, with 2*delta and 2*beta fixed at the energy.

    Same formula as xray.reflect_amplitude (cross-checked per run); constants are
    precomputed once so the per-bounce cost is pure mp arithmetic.
    """

    def __init__(self, material, energy_kev, numbers: NumberContext):
        self.numbers = N = numbers
        delta, beta = material.optical(energy_kev)
        self.d2, self.b2 = 2.0 * delta, 2.0 * beta
        self.b2i = self.b2 * N.i

    def __call__(self, sin_theta):
        root = (sin_theta * sin_theta - self.d2 + self.b2i).sqrt()
        return (sin_theta - root) / (sin_theta + root)

    def product(self, sins):
        amp = self.numbers(1)
        for s in sins:
            amp = amp * self(s)
        return amp


class FloatLineAmplitudes:
    """prod_j r(sin θ_j; E_m) per line in float64, constants from the exact
    material model; cross-checked against the mp Fresnel via check()."""

    def __init__(self, material: GlassMaterial, lines: Sequence[SpectralLine],
                 numbers: NumberContext):
        self.numbers = numbers
        # (2*delta, 2*beta) per line: the root is sqrt(sin^2 - 2d + 2ib)
        self.two_delta_beta: list[tuple[float, float]] = []
        for line in lines:
            delta, beta = material.optical(line.e_kev)
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

"""Fresnel reflection amplitude with energy-fixed optical constants."""

from ..formula import Number
from .nums import const


class FresnelAmplitude:
    """Complex r(sin theta), s-pol, with 2*delta and 2*beta fixed at the energy.

    Same formula as xray.reflect_amplitude (cross-checked per run); constants are
    precomputed once so the per-bounce cost is pure Number arithmetic.
    """

    def __init__(self, material, energy_kev: Number):
        p = energy_kev.precision
        two = const("2", p)
        self.d2 = material.delta(energy_kev, p) * two
        self.b2i = material.beta(energy_kev, p) * two * const("i", p)
        self._half = const("0.5", p)
        self._one = const("1", p)

    def __call__(self, sin_theta: Number) -> Number:
        root = (sin_theta * sin_theta - self.d2 - self.b2i) ** self._half
        return (sin_theta - root) / (sin_theta + root)

    def product(self, sins: list[Number]) -> Number:
        amp = self._one
        for s in sins:
            amp = amp * self(s)
        return amp

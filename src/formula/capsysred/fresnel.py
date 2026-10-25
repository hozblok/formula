"""Optical constants and Fresnel reflection amplitudes."""

from functools import lru_cache

from ..formula import Number


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

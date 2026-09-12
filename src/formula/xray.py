"""Grazing-incidence X-ray reflection from glass.

For X-rays the refractive index is n = 1 - delta + i*beta (convention e^{-i*omega*t},
propagation e^{+i*k*L}) with delta, beta << 1,
so a ray hitting a surface below the critical angle theta_c = sqrt(2*delta)
undergoes near-total external reflection. Reflectivity is the Fresnel equation
evaluated with the multiprecision complex engine; delta and beta come from the
material's complex effective electron density (tabulated Henke f1/f2 for
silica). This is the building block for tracing X-rays down a glass capillary
(see XRAY.md).
"""

from collections import namedtuple
from typing import Callable, Union

from .formula import Number
from .intersect import RaySurface
from .henke import silica_electron_density
from .physical_constants import ANGSTROM, HC_KEV_ANGSTROM, R_E


def wavelength_angstrom(energy_kev, *, precision: int) -> Number:
    """Photon wavelength (angstrom) for an energy in keV."""
    return Number(f"({HC_KEV_ANGSTROM})/({_s(energy_kev)})", precision)


class GlassMaterial:
    """Optical constants of a glass for the X-ray refractive index n=1-delta+i*beta.

    Exactly one input channel:
    - electron_density(E_keV) -> complex effective electron density (m^-3);
      delta + i*beta = r_e * lambda^2 / (2*pi) * n_eff(E) (tabulated Henke
      f1/f2 for silica);
    - delta_beta_ref = complex(delta, beta) at energy_ref_kev — an empirical
      anchor scaled as delta ~ 1/E^2, beta ~ 1/E^4; for a published epsilon
      use delta = (1 - Re eps)/2, beta = Im eps/2.
    """

    def __init__(
        self,
        name: str = "glass",
        *,
        electron_density: Callable[[float], complex] | None = None,
        delta_beta_ref: complex | None = None,
        energy_ref_kev: str | None = None,
    ):
        if (electron_density is None) == (delta_beta_ref is None):
            raise ValueError("exactly one of electron_density / delta_beta_ref")
        if delta_beta_ref is not None and energy_ref_kev is None:
            raise ValueError("delta_beta_ref requires energy_ref_kev")
        self.electron_density = electron_density  # E [keV] -> complex, m^-3
        self.delta_beta_ref = delta_beta_ref
        self.energy_ref_kev = energy_ref_kev
        self.name = name

    def _optical(self, n_eff_part: float, energy_kev_value, precision: int) -> Number:
        lam_m = f"({HC_KEV_ANGSTROM})/({_s(energy_kev_value)})*({ANGSTROM})"
        expr = f"({R_E})*({lam_m})^2*({repr(n_eff_part)})/(2*pi)"
        return Number(expr, precision)

    def delta(self, energy_kev_value, *, precision: int) -> Number:
        """delta(E): prefactor * Re n_eff(E), or the empirical anchor * 1/E^2."""
        if self.electron_density is None:
            ratio = float(self.energy_ref_kev) / float(str(energy_kev_value))
            return Number(repr(self.delta_beta_ref.real * ratio ** 2), precision)
        else:
            n_eff = self.electron_density(float(str(energy_kev_value)))
            return self._optical(n_eff.real, energy_kev_value, precision)

    def beta(self, energy_kev_value, *, precision: int) -> Number:
        """beta(E): prefactor * Im n_eff(E), or the empirical anchor * 1/E^4."""
        if self.electron_density is None:
            ratio = float(self.energy_ref_kev) / float(str(energy_kev_value))
            return Number(repr(self.delta_beta_ref.imag * ratio ** 4), precision)
        else:
            n_eff = self.electron_density(float(str(energy_kev_value)))
            return self._optical(n_eff.imag, energy_kev_value, precision)

    def critical_angle(self, energy_kev_value, *, precision: int) -> Number:
        """Critical grazing angle theta_c = sqrt(2*delta) (radians)."""
        d = self.delta(energy_kev_value, precision=precision)
        return Number(f"sqrt(2*({d}))", precision)


# Fused silica (SiO2, ~2.20 g/cm^3): tabulated Henke n_eff(E).
FUSED_SILICA = GlassMaterial(
    name="fused silica",
    electron_density=silica_electron_density,
)


# Polycapillary glass of Opt. Express 20, 3975 (2012), composition unknown:
# eps = 1 - 9.115e-6 + i*1.145e-7 at 8 keV -> delta = 4.5575e-6, beta = 5.725e-8.
OE2012_GLASS = GlassMaterial(
    name="OE 20:3975 (2012) glass",
    delta_beta_ref=complex(4.5575e-6, 5.725e-8),
    energy_ref_kev="8.0",
)


ReflectionEvent = namedtuple(
    "ReflectionEvent", ["point", "direction", "grazing_angle", "reflectivity", "t"]
)


def reflect_amplitude(
    grazing_angle,
    energy_kev_value,
    material: GlassMaterial = FUSED_SILICA,
    *,
    precision: int,
) -> Number:
    """Complex amplitude reflection coefficient r (s-polarization).

    r = (s - sqrt(s^2 - 2*delta + 2i*beta)) / (s + sqrt(...)), s = sin(theta).
    Carries magnitude and phase; coherent tracing must multiply r, not sqrt(R).
    """
    s = Number(f"sin({_s(grazing_angle)})", precision)
    d2 = material.delta(energy_kev_value, precision=precision) * 2
    b2 = material.beta(energy_kev_value, precision=precision) * 2
    root = (s * s - d2 + b2 * Number("i", precision)) ** Number("0.5", precision)
    return (s - root) / (s + root)


def reflectivity(
    grazing_angle,
    energy_kev_value,
    material: GlassMaterial = FUSED_SILICA,
    *,
    precision: int,
) -> Number:
    """Fresnel reflectivity R = |r|^2 at a grazing angle for a photon energy.

    At grazing incidence s and p polarization are indistinguishable.
    """
    r = reflect_amplitude(grazing_angle, energy_kev_value, material, precision=precision)
    return abs(r) ** Number("2", precision)


def reflect_ray(
    surface_expr: str,
    origin,
    direction,
    energy_kev_value,
    t_max,
    material: GlassMaterial = FUSED_SILICA,
    *,
    precision: int,
    t_min=0,
    method: str = "auto",
):
    """First grazing reflection of a ray off a glass surface F(x,y,z)=0.

    Finds the nearest hit, then returns a ReflectionEvent with the hit point,
    reflected direction, grazing angle and reflectivity. Returns None if the ray
    misses. The reflected direction feeds the next bounce for capillary tracing.
    """
    rs = RaySurface(surface_expr, precision)
    func = rs.function(origin, direction)
    ts = rs.intersect(origin, direction, t_max, method=method, t_min=t_min)
    if not ts:
        return None
    t = ts[0]
    point, refl_dir, grazing = func.reflect_at(t)
    reflec = reflectivity(grazing, energy_kev_value, material, precision=precision)
    return ReflectionEvent(point, refl_dir, grazing, reflec, t)


def _s(value: Union[Number, str, int, float]) -> str:
    """Render a scalar for embedding in a formula expression."""
    return str(value)

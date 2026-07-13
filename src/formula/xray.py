"""Grazing-incidence X-ray reflection from glass.

For X-rays the refractive index is n = 1 - delta - i*beta with delta, beta << 1,
so a ray hitting a surface below the critical angle theta_c = sqrt(2*delta)
undergoes near-total external reflection. Reflectivity is the Fresnel equation
evaluated with the arbitrary-precision complex engine; delta and beta are taken
from a simple material model that scales with photon energy. This is the
building block for tracing X-rays down a glass capillary (see XRAY.md).
"""

from collections import namedtuple
from typing import Union

from .formula import Number
from .intersect import RaySurface

# Classical electron radius (m) and photon energy*wavelength product (keV*angstrom).
R_E = "2.8179403262e-15"
HC_KEV_ANGSTROM = "12.398419843320026"
_ANGSTROM = "1e-10"


def wavelength_angstrom(energy_kev, precision: int = 30) -> Number:
    """Photon wavelength (angstrom) for an energy in keV."""
    return Number(f"({HC_KEV_ANGSTROM})/({_s(energy_kev)})", precision)


def energy_kev(wavelength_angstrom_value, precision: int = 30) -> Number:
    """Photon energy (keV) for a wavelength in angstrom."""
    return Number(f"({HC_KEV_ANGSTROM})/({_s(wavelength_angstrom_value)})", precision)


class GlassMaterial:
    """Optical constants of a glass for the X-ray refractive index n=1-delta-i*beta.

    delta comes from the electron density via the classical formula
    delta = r_e * lambda^2 * rho_e / (2*pi); beta is a reference value scaled as
    1/E^4 (photoelectric absorption far from edges). Both are crude away from
    absorption edges -- replace with tabulated f1/f2 for accurate work.
    """

    def __init__(
        self,
        name: str = "glass",
        *,
        electron_density: str,
        beta_ref: str,
        energy_ref_kev: str,
    ):
        self.electron_density = electron_density  # electrons per m^3
        self.beta_ref = beta_ref
        self.energy_ref_kev = energy_ref_kev
        self.name = name

    def delta(self, energy_kev_value, precision: int = 30) -> Number:
        """Refractive-index decrement delta(E)."""
        lam_m = f"({HC_KEV_ANGSTROM})/({_s(energy_kev_value)})*({_ANGSTROM})"
        expr = f"({R_E})*({lam_m})^2*({self.electron_density})/(2*pi)"
        return Number(expr, precision)

    def beta(self, energy_kev_value, precision: int = 30) -> Number:
        """Absorption index beta(E), scaled as 1/E^4 from the reference value."""
        ratio = f"({self.energy_ref_kev})/({_s(energy_kev_value)})"
        return Number(f"({self.beta_ref})*({ratio})^4", precision)

    def critical_angle(self, energy_kev_value, precision: int = 30) -> Number:
        """Critical grazing angle theta_c = sqrt(2*delta) (radians)."""
        d = self.delta(energy_kev_value, precision)
        return Number(f"sqrt(2*({d}))", precision)


# Fused silica (SiO2, ~2.20 g/cm^3): electron density ~6.6e29 /m^3, beta ~3.9e-8 at 10 keV.
FUSED_SILICA = GlassMaterial(
    name="fused silica",
    electron_density="6.6e29",
    beta_ref="3.89e-8",
    energy_ref_kev="10.0",
)


ReflectionEvent = namedtuple(
    "ReflectionEvent", ["point", "direction", "grazing_angle", "reflectivity", "t"]
)


def reflect_amplitude(
    grazing_angle,
    energy_kev_value,
    material: GlassMaterial = FUSED_SILICA,
    precision: int = 30,
) -> Number:
    """Complex amplitude reflection coefficient r (s-polarization).

    r = (s - sqrt(s^2 - 2*delta - 2i*beta)) / (s + sqrt(...)), s = sin(theta).
    Carries magnitude and phase; coherent tracing must multiply r, not sqrt(R).
    """
    s = Number(f"sin({_s(grazing_angle)})", precision)
    d2 = material.delta(energy_kev_value, precision) * 2
    b2 = material.beta(energy_kev_value, precision) * 2
    root = (s * s - d2 - b2 * Number("i", precision)) ** Number("0.5", precision)
    return (s - root) / (s + root)


def reflectivity(
    grazing_angle,
    energy_kev_value,
    material: GlassMaterial = FUSED_SILICA,
    precision: int = 30,
) -> Number:
    """Fresnel reflectivity R = |r|^2 at a grazing angle for a photon energy.

    At grazing incidence s and p polarization are indistinguishable.
    """
    r = reflect_amplitude(grazing_angle, energy_kev_value, material, precision)
    return abs(r) ** Number("2", precision)


def reflect_ray(
    surface_expr: str,
    origin,
    direction,
    energy_kev_value,
    t_max,
    material: GlassMaterial = FUSED_SILICA,
    precision: int = 30,
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
    reflec = reflectivity(grazing, energy_kev_value, material, precision)
    return ReflectionEvent(point, refl_dir, grazing, reflec, t)


def _s(value: Union[Number, str, int, float]) -> str:
    """Render a scalar for embedding in a formula expression."""
    return str(value)

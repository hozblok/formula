"""Henke atomic scattering factors and derived fused-silica optical inputs.

si.nff / o.nff are verbatim CXRO tables (E [eV], f1, f2 per atom); rows with
the f1 = -9999 sentinel (below the low-energy validity limit) are skipped.
"""

from bisect import bisect_left
from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

from ..physical_constants import AVOGADRO_CONSTANT

Element = Literal["si", "o"]

_DIR = Path(__file__).parent

# CXRO marker for "f1 not tabulated" (a plain negative f1 is physical).
_SENTINEL = -9999.0

# Nominal fused-silica mass density (kg/m^3), equivalent to 2.200 g/cm^3.
# Density is grade-dependent, so material-specific models may override this
# nominal value. Published fused-silica values include:
# - Corning HPFS, 2.201 g/cm^3 at 25 deg C:
#   https://www.corning.com/media/worldwide/csm/documents/1e624b3034474193b5b00ea6f558dd3d2.pdf
# - Heraeus Covantics, 2.201 g/cm^3:
#   https://www.heraeus-covantics.com/knowlegde-base/properties
# - AGC AQ synthetic fused silica, 2.2 g/cm^3:
#   https://www.agc.com/en/products/electoric/pdf/agc_aq.pdf
# - NIST NSRDS-NBS 8, 2.2002 g/cm^3 for high-purity fused silica:
#   https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds8.pdf
# - LBNL/PDG reference tables, 2.200 g/cm^3 for fused quartz:
#   https://pdg.lbl.gov/2025/AtomicNuclearProperties/adndt.pdf
FUSED_SILICA_DENSITY = "2200"

# SiO2 molar mass (kg/mol). Published values include:
# - NIST Chemistry WebBook, 60.0843 g/mol:
#   https://webbook.nist.gov/cgi/cbook.cgi?ID=C14808607&Units=SI
# - NIH PubChem, 60.084 g/mol:
#   https://pubchem.ncbi.nlm.nih.gov/compound/Silicon-dioxide
# - US EPA Substance Registry, 60.08 g/mol:
#   https://cdxapps.epa.gov/oms-substance-registry-services/substance-details/151977
# - ECHA registration dossier, 60.084 g/mol:
#   https://echa.europa.eu/registration-dossier/-/registered-dossier/17236/11/?documentUUID=c8ac29eb-1393-4c3a-9d3b-d73a69fd3226
# - ILO/WHO International Chemical Safety Card, 60.1 g/mol:
#   https://chemicalsafety.ilo.org/dyn/icsc/showcard.display?p_card_id=0808&p_lang=en&p_version=2
SIO2_MOLAR_MASS = "0.0600843"

# SiO2 formula units per m^3: N_FU = (rho / M_SiO2) * N_A.
FUSED_SILICA_N_FU = (
    float(FUSED_SILICA_DENSITY) / float(SIO2_MOLAR_MASS) * float(AVOGADRO_CONSTANT)
)

# Fused silica (SiO2, ~2.20 g/cm^3): X-ray optical-model inputs.
#
# The number density of SiO2 formula units is
#
#   N_FU = (rho / M_SiO2) * N_A
#        = (2200 / 0.0600843) * 6.02214076e23
#        = 2.20502e28 m^-3.
#
# Approximation:
# One SiO2 formula unit contains 14 + 2*8 = 30 electrons. Therefore,
#
#   n_e = (14 + 2*8) * N_FU
#       = 30 * N_FU
#       = 6.61506e29 electrons / m^3.
#
# The nominal value is rounded to two significant figures.
# SILICA_ELECTRON_DENSITY = "6.6e29"   # electrons / m^3


@lru_cache(maxsize=len(get_args(Element)))
def _table(element: Element) -> tuple[list[float], list[float], list[float]]:
    """(evs, f1_vals, f2_vals); rows below the f1 validity limit are dropped."""
    evs, f1_vals, f2_vals = [], [], []
    for line in (_DIR / f"{element}.nff").read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and float(parts[1]) != _SENTINEL:
            evs.append(float(parts[0]))
            f1_vals.append(float(parts[1]))
            f2_vals.append(float(parts[2]))
    return evs, f1_vals, f2_vals


def _interp(element: Element, col: Literal["f1", "f2"], energy_kev: float) -> float:
    evs, f1_vals, f2_vals = _table(element)
    vals = f1_vals if col == "f1" else f2_vals
    energy_ev = energy_kev * 1000.0
    if not evs[0] <= energy_ev <= evs[-1]:
        raise ValueError(f"{element}: {energy_kev} keV outside the Henke table")
    j = max(bisect_left(evs, energy_ev), 1)
    e_prev, e_next = evs[j - 1], evs[j]
    v_prev, v_next = vals[j - 1], vals[j]
    slope = (v_next - v_prev) / (e_next - e_prev)
    return v_prev + slope * (energy_ev - e_prev)


@lru_cache(maxsize=100)
def f1(element: Element, energy_kev: float) -> float:
    """Henke f1 (electrons/atom), linear interpolation in E."""
    return _interp(element, "f1", energy_kev)


@lru_cache(maxsize=100)
def f2(element: Element, energy_kev: float) -> float:
    """Henke f2 (electrons/atom), linear interpolation in E."""
    return _interp(element, "f2", energy_kev)


def silica_electron_density(energy_kev: float) -> complex:
    """Complex effective electron density of fused silica, electrons / m^3.

    n_eff(E) = N_FU * sum over SiO2 of (f1 + i*f2), N_FU = (rho / M_SiO2) * N_A;
    delta + i*beta = r_e*lambda^2/(2*pi) * n_eff in the optics layer, so Re drives
    refraction (far from edges -> 30 * N_FU = 6.615e29) and Im > 0 absorption,
    per the project convention n = 1 - delta + i*beta.

    Tables downloaded 2026-08-31 from the CXRO database,
    https://henke.lbl.gov/optical_constants/asf.html
    (files sf/si.nff, sf/o.nff): B. L. Henke, E. M. Gullikson, J. C. Davis,
    At. Data Nucl. Data Tables 54, 181-342 (1993).
    """
    return FUSED_SILICA_N_FU * complex(
        f1("si", energy_kev) + 2.0 * f1("o", energy_kev),
        f2("si", energy_kev) + 2.0 * f2("o", energy_kev),
    )

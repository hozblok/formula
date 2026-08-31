"""Tabular physical constants (strings for the multiprecision engine)."""

# Metres per angstrom.
ANGSTROM = "1e-10"

# Classical electron radius (m).
R_E = "2.8179403262e-15"

# Avogadro constant (mol^-1).
AVOGADRO_CONSTANT = "6.02214076e23"

# Photon energy * vacuum wavelength product (h*c) (keV * angstrom).
HC_KEV_ANGSTROM = "12.398419843320026"

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

# Fused silica (SiO2, ~2.20 g/cm^3): X-ray optical-model inputs.
SILICA_ELECTRON_DENSITY = "6.6e29"   # electrons / m^3
SILICA_BETA_REF = "3.89e-8"          # absorption index at the reference energy
SILICA_ENERGY_REF_KEV = "10.0"       # reference photon energy (keV)

# Polycapillary glass (Opt. Express 20, 3975): eps = 1 - 9.115e-6 + i*1.145e-7 at
# 8 keV; electron density back-derived from delta = 4.5575e-6.
OE2012_ELECTRON_DENSITY = "4.2308e29"  # electrons / m^3
OE2012_BETA_REF = "5.725e-8"           # absorption index at the reference energy
OE2012_ENERGY_REF_KEV = "8.0"          # reference photon energy (keV)

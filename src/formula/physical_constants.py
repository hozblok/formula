"""Tabular physical constants (strings for the arbitrary-precision engine)."""

# Classical electron radius (m).
R_E = "2.8179403262e-15"

# Photon energy * wavelength product (keV * angstrom).
HC_KEV_ANGSTROM = "12.398419843320026"

# Metres per angstrom.
ANGSTROM = "1e-10"

# Fused silica (SiO2, ~2.20 g/cm^3): X-ray optical-model inputs.
SILICA_ELECTRON_DENSITY = "6.6e29"   # electrons / m^3
SILICA_BETA_REF = "3.89e-8"          # absorption index at the reference energy
SILICA_ENERGY_REF_KEV = "10.0"       # reference photon energy (keV)

# Polycapillary glass (Opt. Express 20, 3975): eps = 1 - 9.115e-6 + i*1.145e-7 at
# 8 keV; electron density back-derived from delta = 4.5575e-6.
OE2012_ELECTRON_DENSITY = "4.2308e29"  # electrons / m^3
OE2012_BETA_REF = "5.725e-8"           # absorption index at the reference energy
OE2012_ENERGY_REF_KEV = "8.0"          # reference photon energy (keV)

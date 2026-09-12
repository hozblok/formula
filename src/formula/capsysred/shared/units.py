"""Unit conversions (all return float).

Display uses these directly. The geometry path lifts m_to_um(1) to a Number at
precision p: the RaySurface cross-check equation is written in micrometre
coordinates because in metres the coefficients are ~a^2 ~ 1e-12 and the
subdivision root finder is badly conditioned; in micrometres they are ~1. 1e6
is exact in double, so lifting it is lossless.
"""


def m_to_um(x):
    """Metres -> micrometres."""
    return float(x) * 1e6


def m_to_mm(x):
    """Metres -> millimetres."""
    return float(x) * 1e3


def m_to_nm(x):
    """Metres -> nanometres."""
    return float(x) * 1e9


def m_to_angstrom(x):
    """Metres -> angstroms."""
    return float(x) * 1e10


def rad_to_mrad(x):
    """Radians -> milliradians."""
    return float(x) * 1e3


def rad_to_urad(x):
    """Radians -> microradians."""
    return float(x) * 1e6

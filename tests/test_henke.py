"""Henke f1 tables and the derived silica effective electron density."""

import math

import pytest

from formula.henke import f1, f2, silica_electron_density


def test_f1_approaches_z_far_from_edges():
    assert math.isclose(f1("si", 30.0), 14.0269, rel_tol=1e-4)
    assert math.isclose(f1("o", 30.0), 8.00194, rel_tol=1e-4)


def test_f2_endpoints():
    assert math.isclose(f2("si", 30.0), 0.0228461, rel_tol=1e-4)
    assert math.isclose(f2("o", 30.0), 0.00180754, rel_tol=1e-4)


def test_outside_table_raises():
    with pytest.raises(ValueError):
        f1("si", 0.010)  # below the common grid (f1 sentinel rows dropped)
    with pytest.raises(ValueError):
        f2("si", 31.0)


def test_silica_rho_real_matches_nominal_constant():
    # 30 * N_FU = 6.615e29; tabulated f1 adds ~0.5 % dispersion at 10 keV.
    assert math.isclose(silica_electron_density(10.0).real, 6.615e29, rel_tol=0.02)


def test_silica_rho_imag_reproduces_legacy_beta():
    # Im(rho) through the r_e*lam^2/(2*pi) prefactor gives retired SILICA_BETA_REF.
    lam = 12.398419843320026 / 10.0 * 1e-10
    beta = 2.8179403262e-15 * lam * lam / (2 * math.pi) * silica_electron_density(10.0).imag
    assert math.isclose(beta, 3.89e-8, rel_tol=0.005)

"""Grazing-incidence X-ray reflection (reflectivity + reflected-ray geometry)."""

import math

from formula import (
    FUSED_SILICA,
    RaySurface,
    energy_kev,
    reflect_ray,
    reflectivity,
    wavelength_angstrom,
)


def _f(number):
    return float(str(number))


def test_wavelength_energy_roundtrip():
    lam = wavelength_angstrom(10.0, precision=32)
    assert math.isclose(_f(lam), 1.23984, rel_tol=1e-4)
    assert math.isclose(_f(energy_kev(lam, precision=32)), 10.0, rel_tol=1e-9)


def test_critical_angle_fused_silica_10kev():
    theta_c = FUSED_SILICA.critical_angle(10.0, precision=32)
    # ~3 mrad for fused silica at 10 keV.
    assert 2.9e-3 < _f(theta_c) < 3.1e-3


def test_delta_scales_as_inverse_energy_squared():
    d5 = _f(FUSED_SILICA.delta(5.0, precision=32))
    d10 = _f(FUSED_SILICA.delta(10.0, precision=32))
    assert math.isclose(d5 / d10, 4.0, rel_tol=1e-6)


def test_reflectivity_high_below_critical_angle():
    # Hard X-ray, well below theta_c -> near-total external reflection.
    r = _f(reflectivity(1e-3, 10.0, precision=32))
    assert r > 0.9


def test_reflectivity_low_above_critical_angle():
    # Same energy, far above theta_c -> reflectivity collapses.
    r = _f(reflectivity(1e-2, 10.0, precision=32))
    assert r < 0.01


def test_soft_xray_has_larger_critical_angle():
    soft = _f(FUSED_SILICA.critical_angle(0.5, precision=32))
    hard = _f(FUSED_SILICA.critical_angle(10.0, precision=32))
    assert soft > hard


def test_reflect_at_specular_off_plane():
    # Flat mirror y=0; a ray skimming down reflects with y-direction flipped.
    rs = RaySurface("y", precision=40)
    func = rs.function((0, 1, 0), (1, -0.1, 0))
    ts = rs.intersect((0, 1, 0), (1, -0.1, 0), t_max=100)
    point, direction, grazing = func.reflect_at(ts[0])
    dx, dy, _ = (_f(c) for c in direction)
    assert abs(_f(point[1])) < 1e-30  # hit lies on the mirror plane y=0
    assert dx > 0 and dy > 0  # x preserved, y mirrored upward
    assert math.isclose(_f(grazing), math.asin(0.1 / math.hypot(1, 0.1)), rel_tol=1e-9)


def test_reflect_ray_off_capillary_wall():
    # Ray inside a unit cylinder hits the glass wall and reflects.
    event = reflect_ray("x*x + y*y - 1", (0, 0, 0), (1, 0.02, 0.2), 10.0, t_max=10, precision=32)
    assert event is not None
    assert math.isclose(_f(event.point[0] ** 2 + event.point[1] ** 2), 1.0, rel_tol=1e-9)
    assert 0.0 <= _f(event.reflectivity) <= 1.0
    assert _f(event.grazing_angle) > 0

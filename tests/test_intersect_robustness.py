"""Robustness, regression and applicability-limit tests for RaySurface.

Companion to test_intersect.py. Three groups:
  * geometry / API correctness (normalization, point recovery, validation);
  * endpoint-root, routing and degree-cap behavior;
  * documented limits of applicability (what each backend can and cannot do).
"""

import math

import pytest

from formula import Number, RaySurface
from formula._roots import is_polynomial


def _pi(prec):
    return Number("4*atan(1)", prec)


def _close(a, b, eps="1e-18"):
    prec = a._precision
    return abs(a - Number(b, prec)) < Number(eps, prec)


def _match(roots, expected, eps="1e-18"):
    return len(roots) == len(expected) and all(
        _close(r, e if isinstance(e, Number) else str(e), eps)
        for r, e in zip(roots, expected)
    )


ALL = ["sturm", "sampling", "subdivision", "auto"]
GENERAL = ["chebyshev", "subdivision", "auto"]


# --------------------------------------------------------------------------- #
# Geometry / API correctness
# --------------------------------------------------------------------------- #

def test_point_at_recovers_axes_the_surface_ignores():
    # Surface depends on x only; the ray carries y=5, z=7 — point_at must keep them.
    rs = RaySurface("x*x - 1", precision=24)
    pts = rs.points((-2, 5, 7), (1, 0, 0), t_max=10, method="sturm")
    assert len(pts) == 2
    for x, y, z in pts:
        assert str(y) == "5" and str(z) == "7"
        assert abs(x * x - Number(1, 24)) < Number("1e-20", 24)


def test_t_is_arc_length_under_nonunit_direction():
    # direction is normalized internally, so a scaled direction gives the same t.
    rs = RaySurface("x*x + y*y - 1", precision=48)
    unit = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    scaled = rs.intersect((-2, 0, 0), (7, 0, 0), t_max=10, method="sturm")
    assert _match(unit, ["1", "3"]) and _match(scaled, ["1", "3"])


def test_oblique_ray_points_lie_on_surface():
    rs = RaySurface("x*x + y*y + z*z - 4", precision=48)
    pts = rs.points((-3, -3, -3), (1, 1, 1), t_max=20, method="sturm")
    assert len(pts) == 2
    for x, y, z in pts:
        assert abs(x * x + y * y + z * z - Number(4, 48)) < Number("1e-18", 48)


def test_ray_along_ignored_axis_returns_empty_not_nan():
    # Cylinder ignores z; a purely-z ray never changes x,y -> no crossing, no nan.
    rs = RaySurface("x*x + y*y - 1", precision=24)
    assert rs.intersect((0, 0, 0), (0, 0, 1), t_max=10, method="sturm") == []


def test_zero_direction_rejected():
    rs = RaySurface("x*x - 1", precision=24)
    with pytest.raises(ValueError, match="non-zero"):
        rs.intersect((0, 0, 0), (0, 0, 0), t_max=5)


def test_tmin_above_tmax_rejected():
    rs = RaySurface("x*x - 1", precision=24)
    with pytest.raises(ValueError, match="t_min"):
        rs.intersect((-2, 0, 0), (1, 0, 0), t_max=1, t_min=5)


def test_unknown_surface_variable_rejected():
    rs = RaySurface("w*w - 1", precision=24)
    with pytest.raises(ValueError, match="surface variables"):
        rs.intersect((0, 0, 0), (1, 0, 0), t_max=5)


# --------------------------------------------------------------------------- #
# Endpoint roots — regression for the critical phantom-root bug
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ALL)
def test_root_exactly_at_tmin(method):
    # Ray origin sits on the surface: t=0 is a root, plus t=2.
    rs = RaySurface("x*x + y*y - 1", precision=48)
    roots = rs.intersect((-1, 0, 0), (1, 0, 0), t_max=10, t_min=0, method=method)
    assert _match(roots, ["0", "2"], eps="1e-15")


@pytest.mark.parametrize("method", ALL)
def test_root_exactly_at_tmax(method):
    # Upper boundary is a root (t=3 with t_max=3); must be kept, not dropped.
    rs = RaySurface("x*x + y*y - 1", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=3, method=method)
    assert _match(roots, ["1", "3"], eps="1e-15")


@pytest.mark.parametrize("method", ALL)
def test_both_endpoints_are_roots(method):
    rs = RaySurface("x*x + y*y - 1", precision=48)
    roots = rs.intersect((-1, 0, 0), (1, 0, 0), t_max=2, t_min=0, method=method)
    assert _match(roots, ["0", "2"], eps="1e-15")


def test_no_phantom_root_with_origin_on_sphere():
    # Origin on the sphere: roots are t=0 and t=2.
    rs = RaySurface("x*x + y*y + z*z - 1", precision=48)
    roots = rs.intersect((-1, 0, 0), (1, 0, 0), t_max=5, t_min=0, method="sturm")
    assert _match(roots, ["0", "2"], eps="1e-15")


@pytest.mark.parametrize("method", GENERAL + ["sampling"])
def test_transcendental_root_at_tmin(method):
    # sin(x) has a root exactly at t_min=0; every backend must report it.
    rs = RaySurface("sin(x)", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=7, t_min=0, method=method)
    pi = _pi(32)
    assert _match(roots, ["0", pi, pi * Number(2, 32)], eps="1e-12")


def test_endpoint_net_adds_no_false_positive():
    # g != 0 at the boundaries -> nothing extra appended.
    rs = RaySurface("x*x + y*y - 1", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert _match(roots, ["1", "3"])


# --------------------------------------------------------------------------- #
# Polynomial / Sturm
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ALL)
def test_quartic_four_roots(method):
    rs = RaySurface("(x*x - 1) * (x*x - 4)", precision=48)
    roots = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method=method)
    assert _match(roots, ["1", "2", "4", "5"], eps="1e-14")


@pytest.mark.parametrize("degree", [15, 16])
def test_polynomial_at_and_below_degree_cap(degree):
    # Degree 15 and 16 are both at/under the cap and must resolve.
    expr = "*".join(f"(x - {k})" for k in range(1, degree + 1))
    rs = RaySurface(expr, precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=degree + 1, method="sturm")
    assert len(roots) == degree


def test_degree_above_cap_raises_for_sturm():
    expr = "*".join(f"(x - {k})" for k in range(1, 18))  # degree 17 > default 16
    rs = RaySurface(expr, precision=32)
    with pytest.raises(ValueError, match="exceeds max_degree"):
        rs.intersect((0, 0, 0), (1, 0, 0), t_max=18, method="sturm")


def test_degree_above_default_cap_with_raised_max_degree():
    expr = "*".join(f"(x - {k})" for k in range(1, 19))  # degree 18
    rs = RaySurface(expr, precision=48)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=19, method="sturm", max_degree=20)
    assert len(roots) == 18


def test_max_degree_below_two_rejected():
    rs = RaySurface("x*x - 1", precision=24)
    with pytest.raises(ValueError, match="max_degree"):
        rs.intersect((-2, 0, 0), (1, 0, 0), t_max=5, method="sturm", max_degree=1)


def test_tangent_double_root_sturm_finds_sampling_misses():
    rs = RaySurface("x*x + y*y + z*z - 1", precision=48)
    found = rs.intersect((-2, 1, 0), (1, 0, 0), t_max=10, method="sturm")
    assert _match(found, ["2"], eps="1e-13")
    missed = rs.intersect((-2, 1, 0), (1, 0, 0), t_max=10, method="sampling")
    assert missed == []  # even multiplicity, no sign change


@pytest.mark.parametrize("method", ["sturm", "sampling"])
def test_odd_multiplicity_three_root(method):
    # g(t)=t^3 has a single (triple) root at 0; the sign change keeps it visible.
    rs = RaySurface("x^3", precision=48)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=2, t_min=-2, method=method)
    assert _match(roots, ["0"], eps="1e-12")


def test_clustered_roots_resolved_by_sturm():
    rs = RaySurface("(x - 1) * (x - 1.001)", precision=64)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="sturm")
    assert _match(roots, ["1", "1.001"], eps="1e-15")


def test_complex_surface_real_intersections():
    rs = RaySurface("(x*x - 1) * (1 + i)", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert _match(roots, ["1", "3"])


def test_only_roots_in_range_are_returned():
    rs = RaySurface("x*x + y*y - 1", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=2, method="sturm")
    assert _match(roots, ["1"])


# --------------------------------------------------------------------------- #
# Chebyshev / subdivision on analytic surfaces
# --------------------------------------------------------------------------- #

def test_chebyshev_sine_roots():
    rs = RaySurface("sin(x)", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=10, t_min="0.5", method="chebyshev")
    pi = _pi(32)
    assert _match(roots, [pi, pi * Number(2, 32), pi * Number(3, 32)], eps="1e-24")


def test_chebyshev_exp_root():
    rs = RaySurface("exp(x) - 2", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="chebyshev")
    assert _match(roots, [Number("log(2)", 32)], eps="1e-24")


@pytest.mark.parametrize("method", ["subdivision", "auto"])
def test_corrugated_wall_twelve_roots(method):
    # Capillary wall radius 1+0.3 sin(4z); a radial ray crosses at z=k*pi/4.
    rs = RaySurface("x*x + y*y - (1 + 0.3*sin(4*z))^2", precision=32)
    roots = rs.intersect((1, 0, 0), (0, 0, 1), t_max=10, t_min="0.1", method=method)
    pi = _pi(32)
    expected = [pi * Number(k, 32) / Number(4, 32) for k in range(1, 13)]
    assert _match(roots, expected, eps="1e-22")


def test_subdivision_handles_moderate_oscillation():
    # sin(10 x) over (0.1, 6): 19 simple roots, all recovered.
    rs = RaySurface("sin(10*x)", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=6, t_min="0.1", method="subdivision")
    assert len(roots) == len([k for k in range(1, 40) if 1 < k * math.pi < 60])


def test_subdivision_tangency_and_rejects_turning_point():
    # cos(t)+1 touches zero at pi,3pi; at 2pi g'=0 but g=2 (no root).
    rs = RaySurface("cos(x) + 1", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=10, t_min="0.1", method="subdivision")
    pi = _pi(32)
    assert _match(roots, [pi, pi * Number(3, 32)], eps="1e-12")


@pytest.mark.parametrize("method", ["chebyshev", "subdivision"])
def test_general_backends_reject_complex(method):
    rs = RaySurface("(x*x - 1) * (1 + i)", precision=32)
    with pytest.raises(NotImplementedError):
        rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method=method)


# --------------------------------------------------------------------------- #
# auto routing
# --------------------------------------------------------------------------- #

def test_is_polynomial_classification():
    def poly(expr):
        return is_polynomial(RaySurface(expr, 24).surface)
    assert poly("x*x + y*y - 1")
    assert poly("x*x/2 + y*y/2 - 1")      # variable-free denominators are fine
    assert poly("x^2 - 2")                # natural power
    assert not poly("1/(x - 1) - 0.5")    # variable in a denominator
    assert not poly("x^-2 - 0.25")        # negative power
    assert not poly("x^0.5 - 2")          # fractional power
    assert not poly("sin(x)")             # transcendental


def test_auto_routes_polynomial_to_exact_sturm():
    rs = RaySurface("(x*x - 1) * (x*x - 4)", precision=48)
    a = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method="auto")
    s = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method="sturm")
    assert _match(a, [str(r) for r in s])


def test_auto_scaled_quadric_stays_exact():
    # x*x/2 + ... must route to Sturm (exact), not the approximate general path.
    rs = RaySurface("x*x/2 + y*y/2 - 2", precision=48)
    auto = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=10, method="auto")
    sturm = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert _match(auto, [str(r) for r in sturm])


def test_auto_nonnatural_power_routed_to_general():
    # x^-2 - 0.25 = 0 -> t = 2; misrouting to Sturm would interpolate garbage.
    rs = RaySurface("x^-2 - 0.25", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=5, t_min="0.5", method="auto")
    assert _match(roots, ["2"], eps="1e-12")


def test_auto_falls_back_when_sturm_raises():
    # Force Sturm to fail (cap below the true degree); auto must fall back to the
    # general backends and still find all four roots instead of propagating.
    rs = RaySurface("(x*x - 1) * (x*x - 4)", precision=48)
    roots = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method="auto", max_degree=3)
    assert _match(roots, ["1", "2", "4", "5"], eps="1e-14")


# --------------------------------------------------------------------------- #
# Torus (R=2, r=1): (x^2+y^2+z^2+3)^2 - 16*(x^2+y^2) = 0; up to four hits per ray.
# --------------------------------------------------------------------------- #

_TORUS = "(x*x+y*y+z*z+3)^2 - 16*(x*x+y*y)"


@pytest.mark.parametrize("method", ALL)
def test_torus_four_intersections(method):
    # Equatorial ray along x hits both walls of both sides: x=-3,-1,1,3 -> t=1,3,5,7.
    rs = RaySurface(_TORUS, precision=48)
    roots = rs.intersect((-4, 0, 0), (1, 0, 0), t_max=10, method=method)
    assert _match(roots, ["1", "3", "5", "7"], eps="1e-10")


def test_torus_is_routed_to_exact_sturm():
    rs = RaySurface(_TORUS, precision=48)
    assert is_polynomial(rs.surface)
    auto = rs.intersect((-4, 0, 0), (1, 0, 0), t_max=10, method="auto")
    sturm = rs.intersect((-4, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert _match(auto, [str(r) for r in sturm])


def test_torus_vertical_ray_through_tube_wall():
    # Ray up the z-axis at x=2 pierces the tube wall at z=-1,1 -> t=2,4.
    rs = RaySurface(_TORUS, precision=48)
    roots = rs.intersect((2, 0, -3), (0, 0, 1), t_max=10, method="sturm")
    assert _match(roots, ["2", "4"], eps="1e-12")


def test_torus_ray_misses_through_the_hole():
    rs = RaySurface(_TORUS, precision=48)
    assert rs.intersect((-4, 0, 5), (1, 0, 0), t_max=10, method="sturm") == []


def test_torus_grazing_outer_rim_is_tangent():
    # Ray grazing the outer equator touches it once (double root) at t=3.
    rs = RaySurface(_TORUS, precision=48)
    sturm = rs.intersect((3, -3, 0), (0, 1, 0), t_max=10, method="sturm")
    assert _match(sturm, ["3"], eps="1e-10")
    missed = rs.intersect((3, -3, 0), (0, 1, 0), t_max=10, method="sampling")
    assert missed == []  # even multiplicity, no sign change


def test_torus_points_lie_on_surface():
    rs = RaySurface(_TORUS, precision=48)
    pts = rs.points((-4, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert len(pts) == 4
    for x, y, z in pts:
        residual = (x * x + y * y + z * z + Number(3, 48)) ** Number(2, 48) - Number(
            16, 48
        ) * (x * x + y * y)
        assert abs(residual) < Number("1e-40", 48)


# --------------------------------------------------------------------------- #
# Documented limits of applicability
# --------------------------------------------------------------------------- #

def test_limit_sampling_misses_subsample_features():
    # Two roots closer than the sample spacing collapse to one (or none).
    rs = RaySurface("(x - 1) * (x - 1.0001)", precision=48)
    coarse = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="sampling", samples=8)
    assert len(coarse) < 2  # the rigorous sturm path gets both; sampling does not


def test_limit_high_multiplicity_use_sturm_not_newton():
    # (x-1)^5: Sturm removes the multiplicity via square-free and nails the root.
    rs = RaySurface("(x - 1)^5", precision=48)
    assert _match(
        rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="sturm"), ["1"], eps="1e-20"
    )
    # A Newton-based backend instead hits the surface's derivative singularity:
    # d/dx of u^5 is formed as u^5 * 5 * u'/u, which is nan at the root u=0.
    with pytest.raises(ValueError):
        rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="sampling")


@pytest.mark.parametrize("method", ["subdivision", "auto"])
def test_limit_oscillation_recommended_path_is_complete(method):
    # The reliable route for dense oscillation is subdivision (and auto, which
    # includes it) — not a direct high-degree Chebyshev fit (see the doc).
    rs = RaySurface("sin(8*x)", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=6, t_min="0.1", method=method)
    assert len(roots) == len([k for k in range(1, 40) if 0.8 < k * math.pi < 48])

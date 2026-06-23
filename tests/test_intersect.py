"""All-intersections ray<->surface finder (RaySurface)."""

import pytest

from formula import Number, RaySurface


def _close(a, b, eps="1e-20"):
    prec = a._precision
    return abs(a - Number(b, prec)) < Number(eps, prec)


def _match(roots, expected, eps="1e-20"):
    assert len(roots) == len(expected)
    return all(_close(r, e, eps) for r, e in zip(roots, expected))


@pytest.mark.parametrize("method", ["sampling", "sturm", "auto"])
def test_cylinder_two_roots(method):
    rs = RaySurface("x*x + y*y - 1", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method=method)
    assert _match(roots, ["1", "3"])


@pytest.mark.parametrize("method", ["sampling", "sturm", "auto"])
def test_sphere_two_roots(method):
    rs = RaySurface("x*x + y*y + z*z - 1", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method=method)
    assert _match(roots, ["1", "3"])


def test_tangent_double_root_sturm_finds_it():
    # Ray grazes the sphere at y=1: (t-2)^2 = 0, a double root sampling misses.
    rs = RaySurface("x*x + y*y + z*z - 1", precision=48)
    roots = rs.intersect((-2, 1, 0), (1, 0, 0), t_max=10, method="sturm")
    assert _match(roots, ["2"], eps="1e-15")

    missed = rs.intersect((-2, 1, 0), (1, 0, 0), t_max=10, method="sampling")
    assert missed == []  # sampling cannot see an even-multiplicity root


def test_quartic_four_roots():
    # (x^2-1)(x^2-4) = x^4 -5x^2 +4, roots at x = -2,-1,1,2 along the ray.
    rs = RaySurface("(x*x - 1) * (x*x - 4)", precision=48)
    roots = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method="sturm")
    assert _match(roots, ["1", "2", "4", "5"])


def test_only_roots_in_range():
    rs = RaySurface("x*x + y*y - 1", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=2, method="sturm")
    assert _match(roots, ["1"])


def test_points_map_back_to_surface():
    rs = RaySurface("x*x + y*y - 1", precision=48)
    pts = rs.points((-2, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert len(pts) == 2
    for x, y, z in pts:
        residual = x * x + y * y - Number(1, 48)
        assert abs(residual) < Number("1e-20", 48)


def test_sampling_matches_sturm():
    rs = RaySurface("(x*x - 1) * (x*x - 4)", precision=48)
    a = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method="sampling")
    b = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method="sturm")
    assert _match(a, [str(r) for r in b], eps="1e-15")


def test_chebyshev_transcendental_roots():
    rs = RaySurface("sin(x)", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=10, t_min="0.5", method="chebyshev")
    pi = Number("4*atan(1)", 32)
    assert _match(roots, [pi, pi * Number(2, 32), pi * Number(3, 32)], eps="1e-28")


def test_chebyshev_exp_root():
    rs = RaySurface("exp(x) - 2", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="chebyshev")
    ln2 = Number("log(2)", 32)
    assert _match(roots, [ln2], eps="1e-28")


def test_auto_routes_transcendental_to_chebyshev():
    rs = RaySurface("exp(x) - 2", precision=32)
    a = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="auto")
    b = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="chebyshev")
    assert _match(a, [str(r) for r in b], eps="1e-28")


def test_complex_surface_real_intersections():
    # Complex-valued surface; real hits are the common roots of Re g and Im g.
    rs = RaySurface("(x*x - 1) * (1 + i)", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert _match(roots, ["1", "3"])


# --- variant 7: subdivision with derivative/range bounds ---

def _kpi_over_4(n, prec=32):
    pi = Number("4*atan(1)", prec)
    return [pi * Number(k, prec) / Number(4, prec) for k in range(1, n + 1)]


@pytest.mark.parametrize("method", ["subdivision", "sturm"])
def test_subdivision_matches_polynomial(method):
    rs = RaySurface("(x*x - 1) * (x*x - 4)", precision=48)
    roots = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method=method)
    assert _match(roots, ["1", "2", "4", "5"], eps="1e-15")


def test_subdivision_finds_corrugated_wall():
    # Capillary wall with radius 1 + 0.3*sin(4z); a radial ray at x=1 crosses it
    # wherever sin(4z)=0, i.e. z = k*pi/4 — twelve closely spaced roots in (0.1, 10].
    rs = RaySurface("x*x + y*y - (1 + 0.3*sin(4*z))^2", precision=32)
    roots = rs.intersect((1, 0, 0), (0, 0, 1), t_max=10, t_min="0.1", method="subdivision")
    assert _match(roots, _kpi_over_4(12), eps="1e-25")


def test_chebyshev_self_validates_on_corrugated_wall():
    # A low starting degree is inadequate for 12 oscillations; the spectral-tail
    # check escalates the degree until every root is captured.
    rs = RaySurface("x*x + y*y - (1 + 0.3*sin(4*z))^2", precision=32)
    cheb = rs.intersect((1, 0, 0), (0, 0, 1), t_max=10, t_min="0.1",
                        method="chebyshev", cheb_degree=8)
    sub = rs.intersect((1, 0, 0), (0, 0, 1), t_max=10, t_min="0.1", method="subdivision")
    assert len(cheb) == 12
    assert _match(cheb, [str(r) for r in sub], eps="1e-20")


def test_subdivision_tangent_and_rejects_turning_point():
    # cos(t)+1 touches zero (double root) at pi, 3pi; at 2pi g'=0 but g=2 (no root).
    rs = RaySurface("cos(x) + 1", precision=32)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=10, t_min="0.1", method="subdivision")
    pi = Number("4*atan(1)", 32)
    assert _match(roots, [pi, pi * Number(3, 32)], eps="1e-12")


def test_auto_unions_chebyshev_and_subdivision():
    # auto on a transcendental surface reconciles both general backends.
    rs = RaySurface("x*x + y*y - (1 + 0.3*sin(4*z))^2", precision=32)
    auto = rs.intersect((1, 0, 0), (0, 0, 1), t_max=10, t_min="0.1", method="auto")
    assert _match(auto, _kpi_over_4(12), eps="1e-25")


def test_subdivision_real_only():
    rs = RaySurface("(x*x - 1) * (1 + i)", precision=32)
    with pytest.raises(NotImplementedError):
        rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method="subdivision")

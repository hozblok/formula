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


def test_complex_surface_real_intersections():
    # Complex-valued surface; real hits are the common roots of Re g and Im g.
    rs = RaySurface("(x*x - 1) * (1 + i)", precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert _match(roots, ["1", "3"])

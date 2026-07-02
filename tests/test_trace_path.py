"""Edge cases of RaySurface.trace_path: miss wall / miss screen / trapped rays,
plus point+direction bookkeeping on wall, cylinder, cone, torus and funnel."""

import math

import pytest

from formula import Number, RaySurface


def _close(a, b, eps="1e-20"):
    prec = a.precision
    return abs(a - Number(b, prec)) < Number(eps, prec)


def _vec_close(v, expected, eps="1e-20"):
    return all(_close(c, e, eps) for c, e in zip(v, expected))


def _unit(v):
    prec = v[0].precision
    norm2 = sum((c * c for c in v), Number(0, prec))
    return _close(norm2, "1")


# --------------------------------------------------------------------------- #
# Vertical wall x + 1 = 0
# --------------------------------------------------------------------------- #

def test_wall_single_reflection_full_bookkeeping():
    rs = RaySurface("x + 1", precision=48)
    path = rs.trace_path((0, 0, 0), (-1, 0, 0), t_max=10, screen_surface="x - 2")
    assert _vec_close(path.source, ["0", "0", "0"])
    assert _vec_close(path.source_direction, ["-1", "0", "0"])
    assert len(path.reflections) == 1
    r = path.reflections[0]
    assert _vec_close(r.point, ["-1", "0", "0"])
    assert _vec_close(r.direction, ["1", "0", "0"])
    assert _close(r.grazing, Number("2*atan(1)", 48))  # normal incidence: pi/2
    assert path.exited
    assert _vec_close(path.screen_point, ["2", "0", "0"])
    assert _vec_close(path.screen_direction, ["1", "0", "0"])
    assert _close(path.opl, "4")  # 1 to the wall + 3 to the screen


def test_wall_missed_ray_goes_straight_to_screen():
    rs = RaySurface("x + 1", precision=48)
    path = rs.trace_path((0, 0, 0), (1, 0, 0), t_max=10, screen_surface="x - 5")
    assert path.reflections == []
    assert path.exited
    assert _vec_close(path.screen_point, ["5", "0", "0"])
    assert _vec_close(path.screen_direction, ["1", "0", "0"])
    assert _close(path.opl, "5")


def test_screen_missed_returns_none_point_and_direction():
    rs = RaySurface("x + 1", precision=48)
    path = rs.trace_path((0, 0, 0), (0, 1, 0), t_max=10, screen_surface="x - 5")
    assert path.reflections == []
    assert path.exited
    assert path.screen_point is None
    assert path.screen_direction is None
    assert _close(path.opl, "0")


def test_no_screen_surface_given():
    rs = RaySurface("x + 1", precision=48)
    path = rs.trace_path((0, 0, 0), (-1, 0, 0), t_max=10)
    assert len(path.reflections) == 1
    assert path.exited
    assert path.screen_point is None
    assert path.screen_direction is None


def test_source_direction_is_normalized():
    rs = RaySurface("x + 1", precision=48)
    path = rs.trace_path((0, 0, 0), (5, 0, 0), t_max=10)
    assert _vec_close(path.source_direction, ["1", "0", "0"])


# --------------------------------------------------------------------------- #
# Cylinder x^2 + y^2 - 1 = 0 (capillary bore)
# --------------------------------------------------------------------------- #

def test_cylinder_missed_ray_flies_down_the_bore():
    rs = RaySurface("x*x + y*y - 1", precision=48)
    path = rs.trace_path((0.2, 0, -1), (0, 0, 1), t_max=10,
                         exit_surface="z - 1", screen_surface="z - 3")
    assert path.reflections == []
    assert path.exited
    assert _vec_close(path.screen_point, ["0.2", "0", "3"])
    assert _vec_close(path.screen_direction, ["0", "0", "1"])
    assert _close(path.opl, "4")


def test_cylinder_two_bounce_zigzag_exact_geometry():
    rs = RaySurface("x*x + y*y - 1", precision=48)
    inv = Number("1/2^0.5", 48)
    path = rs.trace_path((0, 0, 0), (1, 0, 1), t_max=10,
                         exit_surface="z - 4", screen_surface="z - 6")
    assert len(path.reflections) == 2
    r1, r2 = path.reflections
    assert _vec_close(r1.point, ["1", "0", "1"])
    assert _vec_close(r1.direction, [-inv, "0", inv])
    assert _vec_close(r2.point, ["-1", "0", "3"])
    assert _vec_close(r2.direction, [inv, "0", inv])
    for r in (r1, r2):
        assert _unit(r.direction)
        assert _close(r.grazing, Number("atan(1)", 48))  # pi/4
    assert path.exited
    assert _vec_close(path.screen_point, ["2", "0", "6"])
    assert _vec_close(path.screen_direction, [inv, "0", inv])
    assert _close(path.opl, Number("6*2^0.5", 48))


def test_cylinder_trapped_ray_max_bounces():
    # Perpendicular ray bounces between x=1 and x=-1 forever; never reaches z=100.
    rs = RaySurface("x*x + y*y - 1", precision=48)
    path = rs.trace_path((0, 0, 0), (1, 0, 0), t_max=10,
                         exit_surface="z - 100", screen_surface="z - 200",
                         max_bounces=4)
    assert len(path.reflections) == 4
    assert not path.exited
    assert path.screen_point is None
    assert path.screen_direction is None


def test_cylinder_ray_never_reaches_exit_surface():
    # Along the axis away from the exit plane: no wall hit, no exit -> not exited.
    rs = RaySurface("x*x + y*y - 1", precision=48)
    path = rs.trace_path((0, 0, 0), (0, 0, -1), t_max=10,
                         exit_surface="z - 1", screen_surface="z - 3")
    assert path.reflections == []
    assert not path.exited
    assert path.screen_point is None


def test_cylinder_launch_point_on_wall_is_skipped():
    rs = RaySurface("x*x + y*y - 1", precision=48)
    path = rs.trace_path((-1, 0, 0), (1, 0, 0), t_max=10, max_bounces=1)
    assert len(path.reflections) == 1
    assert _vec_close(path.reflections[0].point, ["1", "0", "0"])


def test_cylinder_grazing_no_phantom_reflection_at_launch():
    # Regression: |g(eps)| is tiny at grazing incidence and the endpoint net used
    # to re-add the launch point as a root, reflecting the ray in place.
    phi = 1e-4
    rs = RaySurface("x*x + y*y - 1", precision=32)
    ts = rs.intersect((1, 0, 0), (-math.sin(phi), math.cos(phi), 0),
                      t_max=9e-4, t_min=9e-13)
    assert len(ts) == 1
    assert _close(ts[0], str(2 * math.sin(phi)), eps="1e-10")

    path = rs.trace_path((1, 0, 0), (-math.sin(phi), math.cos(phi), 0),
                         t_max=9e-4, max_bounces=3)
    assert len(path.reflections) == 3
    for k, r in enumerate(path.reflections, start=1):
        x, y, _ = r.point
        assert _close(x * x + y * y, "1")            # on the wall
        assert _close(y, str(2 * math.sin(phi) * k), eps="1e-10")  # marches on
        assert _close(r.grazing, str(phi), eps="1e-10")


# --------------------------------------------------------------------------- #
# Cone, torus, funnel
# --------------------------------------------------------------------------- #

def test_cone_two_hits():
    rs = RaySurface("x*x + y*y - z*z", precision=48)
    pts = rs.points((-2, 0, 1), (1, 0, 0), t_max=10, method="sturm")
    assert len(pts) == 2
    assert _vec_close(pts[0], ["-1", "0", "1"])
    assert _vec_close(pts[1], ["1", "0", "1"])


def test_torus_four_hits():
    # (x^2+y^2+z^2+R^2-r^2)^2 = 4 R^2 (x^2+y^2), R=2, r=1: x = +-1, +-3 on the x-axis.
    rs = RaySurface("(x*x + y*y + z*z + 3)^2 - 16*(x*x + y*y)", precision=48)
    roots = rs.intersect((-4, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert len(roots) == 4
    for r, e in zip(roots, ["1", "3", "5", "7"]):
        assert _close(r, e, eps="1e-15")


def test_torus_missed():
    rs = RaySurface("(x*x + y*y + z*z + 3)^2 - 16*(x*x + y*y)", precision=48)
    assert rs.intersect((-4, 0, 5), (1, 0, 0), t_max=10, method="sturm") == []


def test_funnel_single_hit_and_miss():
    rs = RaySurface("x*x + y*y - z", precision=48)  # paraboloid funnel
    roots = rs.intersect((0.5, 0, -1), (0, 0, 1), t_max=10)
    assert len(roots) == 1
    assert _close(roots[0], "1.25")
    assert rs.intersect((0.5, 0, -1), (0, 0, -1), t_max=10) == []


def test_funnel_reflection_direction_is_unit():
    rs = RaySurface("x*x + y*y - z", precision=48)
    path = rs.trace_path((0.5, 0, -1), (0, 0, 1), t_max=10, max_bounces=1)
    assert len(path.reflections) == 1
    assert _unit(path.reflections[0].direction)

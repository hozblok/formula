"""Finite reflecting patches are checked on archived and polished ray hits."""

import math

import pytest

from formula import Number
from formula.capsysred.shared.implicit_surface import Surface
from formula.capsysred.shared.types import _RAY_POSITION_TOL_M
from formula.capsysred.stages._ray_jets import Hit, transport_ray
from formula.capsysred.surfaces import Mirror
from formula.capsysred.walls.wall_polygon import PolygonWall


def _point(values, number_inputs):
    return tuple(Number(str(value), 64) for value in values) if number_inputs else tuple(values)


@pytest.mark.parametrize("number_inputs", [False, True])
@pytest.mark.parametrize("z,allowed", [
    (-_RAY_POSITION_TOL_M, False), (0.0, False), (_RAY_POSITION_TOL_M / 2, False),
    (_RAY_POSITION_TOL_M, False), (2 * _RAY_POSITION_TOL_M, True), (1.0, True),
    (2 - 2 * _RAY_POSITION_TOL_M, True), (2 - _RAY_POSITION_TOL_M, False),
    (2 - _RAY_POSITION_TOL_M / 2, False), (2.0, False), (2 + _RAY_POSITION_TOL_M, False),
])
def test_mirror_requires_clearance_from_both_span_edges(number_inputs, z, allowed):
    mirror = Mirror(Number("0", 64), Number("2", 64))
    point = _point((0, 0.2, z), number_inputs)
    context = {"origin": _point((1, 0, -1), number_inputs),
               "direction": _point((-1, 0, 1), number_inputs)}
    if allowed:
        mirror.validate(point, **context)
    else:
        with pytest.raises(ValueError, match="reflecting span"):
            mirror.validate(point, **context)


@pytest.mark.parametrize("number_inputs", [False, True])
@pytest.mark.parametrize("origin_x,direction_x", [(-1, -1), (-1, 1), (0, -1), (1, 1), (1, 0)],
                         ids=("behind_moving_away", "backside_approach", "on_plane", "moving_away", "tangent"))
def test_mirror_rejects_nonreflecting_incidence(number_inputs, origin_x, direction_x):
    mirror = Mirror(Number("0", 64), Number("2", 64))
    with pytest.raises(ValueError, match="reflecting side"):
        mirror.validate(_point((0, 0, 1), number_inputs),
                        origin=_point((origin_x, 0, 0), number_inputs),
                        direction=_point((direction_x, 0, 1), number_inputs))


@pytest.mark.parametrize("point,allowed", [((1, 0.2, 2), True), ((0, 1, 2), False),
                                         ((1, 1, 2), False), ((1.01, 0.2, 2), False)])
def test_polygon_face_context_preserves_point_validation(point, allowed):
    zero, one = Number("0", 64), Number("1", 64)
    face = PolygonWall((zero, zero), one, 4, zero).face(0)
    for context in ({}, {"origin": (0, 0, 1), "direction": (1, 0.2, 1)}):
        if allowed:
            face.validate(point, **context)
        else:
            with pytest.raises(ValueError):
                face.validate(point, **context)


def test_polished_mirror_hit_cannot_enter_the_edge_margin():
    tolerance = _RAY_POSITION_TOL_M
    mirror = Mirror(Number("0", 64), Number("2", 64))
    true_z = 0.75 * tolerance
    entrance, source = (1 + true_z, 0), (2 + true_z, 0, -2)
    archived = (-0.5 * tolerance, 0, 1.25 * tolerance)
    mirror.validate(archived, origin=(*entrance, -1), direction=(-1, 0, 1))
    assert math.dist(archived, (0, 0, true_z)) < tolerance
    with pytest.raises(ValueError, match="reflecting span"):
        transport_ray(entrance, source, -1, 1.5, [Hit(mirror, archived)], order=1)


class _CoatedHalfPlane(Surface):
    def __init__(self):
        super().__init__("x-1")

    def validate(self, point, *, origin, direction):
        if point[1] <= 0 or origin[0] >= 1 or direction[0] <= 0:
            raise ValueError("hit is outside the illuminated coating")


def test_custom_surface_validation_limits_the_reflecting_region():
    surface = _CoatedHalfPlane()
    accepted = transport_ray((0.2, 0.1), (0, 0, -2), 0, 10,
                             [Hit(surface, (1, 0.5, 8))], order=1)
    assert accepted.position[2][0] == pytest.approx(10)
    with pytest.raises(ValueError, match="illuminated coating"):
        transport_ray((0.2, -0.1), (0, 0, -2), 0, 10,
                      [Hit(surface, (1, -0.5, 8))], order=1)

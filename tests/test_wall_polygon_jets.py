"""Fixed polygon faces for scalar geometry and composed ray jets."""

from decimal import Decimal
import math

import pytest

from formula import Formula, Number
from formula.backend import MP_TYPES
from formula.capsysred.shared.types import _WALL_SURFACE_TOL_M
from formula.capsysred.stages._ray_jets import _Algebra
from formula.capsysred.walls.wall_polygon import PolygonFace, PolygonWall


def _square():
    return PolygonWall((Number("0", 96), Number("0", 96)), Number("1", 96),
                       4, Number("0", 96))


@pytest.mark.parametrize("number_inputs", [False, True])
def test_rotated_unequal_faces_keep_the_selected_plane(number_inputs):
    center = (Number("1.25", 96), Number("-0.5", 96))
    apothems = tuple(Number(v, 96) for v in ("0.8", "1.1", "1.4", "1.7"))
    wall = PolygonWall(center, apothems, 4, Number("atan(4/3)", 96))
    for index, ((nx, ny), apothem) in enumerate(zip(wall.faces, apothems, strict=True)):
        point = (center[0] + apothem * nx - Number("0.1", 96) * ny,
                 center[1] + apothem * ny + Number("0.1", 96) * nx, Number("7", 96))
        if not number_inputs:
            point = tuple(map(float, point))
        face = wall.surface_at(point)
        assert face.index == index
        assert face.precision == center[0].precision
        face.validate(point)
        assert abs(float(face.value(point))) < 1e-14
        angle = math.atan2(4, 3) + index * math.pi / 2
        assert tuple(map(float, face.gradient(point))) == pytest.approx(
            (math.cos(angle), math.sin(angle), 0))


@pytest.mark.parametrize("index, error", [(-1, IndexError), (4, IndexError),
                                         (0.5, TypeError), ("0", TypeError), (True, TypeError)])
def test_face_index_is_explicit_and_validated(index, error):
    wall = _square()
    with pytest.raises(error):
        wall.face(index)
    with pytest.raises(error):
        PolygonFace(wall, index)


@pytest.mark.parametrize("point", [(1, 1, 0), (1, -1, 2),
                                    (1, 1 - _WALL_SURFACE_TOL_M / 2, 0),
                                    (1, 1 + _WALL_SURFACE_TOL_M / 2, 0),
                                    (1, 2, 0), (0, 0, 0),
                                    (1 + 2 * _WALL_SURFACE_TOL_M, 0, 0)])
def test_face_selection_rejects_edges_and_points_outside_the_finite_face(point):
    wall = _square()
    with pytest.raises(ValueError):
        wall.surface_at(point)
    with pytest.raises(ValueError):
        wall.face(0).validate(point)


def test_explicit_face_rejects_a_hit_on_another_face():
    wall = _square()
    assert wall.surface_at((0, 1, 0)).index == 1
    with pytest.raises(ValueError, match="selected polygon face"):
        wall.face(0).validate((0, 1, 0))
    assert wall.face(0).value((1, 2, 0)) == 0
    assert wall.surface_at((1, 1 - 2 * _WALL_SURFACE_TOL_M, 0)).index == 0


@pytest.mark.parametrize("jet_inputs", [False, True])
def test_face_parameters_keep_precision_beyond_float(jet_inputs):
    wall = PolygonWall((Number("1.000000000000000000000000000001", 96), Number("-2", 96)),
                       Number("1.000000000000000000000000000002", 96), 4, Number("0", 96))
    point = (Number("2.000000000000000000000000000006", 96), Number("-2", 96), Number("0", 96))
    face = wall.surface_at(point)
    assert face.index == 0
    if jet_inputs:
        alg = _Algebra(4, 96)
        point = tuple(alg.constant(value) for value in point)
    value = face.value(point)
    result = value.c[0] if jet_inputs else str(value)
    assert abs(Decimal(result) - Decimal("3e-30")) < Decimal("1e-85")
    assert all(isinstance(value, (Number, *MP_TYPES)) and value.precision == face.precision
               for value in face.gradient(point))


@pytest.mark.parametrize("order", [3, 4])
def test_linear_face_composes_higher_derivatives_without_changing_faces(order):
    alg = _Algebra(order, 96)
    seeds = (0, 1, 0, 1)[:order]
    names = tuple("uv"[axis] for axis in seeds)
    u, v = alg.variable("0.3", seeds, 0), alg.variable("0.7", seeds, 1)
    point = tuple(alg.evaluate(expression, u=u, v=v)
                  for expression in ("u^3*v+u*v^2+1", "u^2*v^2-u+2", "u+v"))
    wall = PolygonWall((Number("1.25", 96), Number("-0.5", 96)), Number("0.8", 96),
                       4, Number("atan(4/3)", 96))
    face = wall.face(0)
    result = face.value(point)
    formula = Formula("0.6*(u^3*v+u*v^2-0.25)+0.8*(u^2*v^2-u+2.5)-0.8", precision=96)
    values = {"u": "0.3", "v": "0.7"}
    for mask, coefficient in enumerate(result.c):
        variables = [name for bit, name in enumerate(names) if mask & (1 << bit)]
        expected = (formula.get_partial_derivative(variables, values)
                    if variables else formula.get(values))
        assert abs(Decimal(coefficient) - Decimal(expected)) < Decimal("1e-80")
    assert abs(Decimal(result.c[-1])) > Decimal("1")
    assert tuple(str(Number(v, 96)) for v in face.gradient(point)) == ("0.6", "0.8", "0")

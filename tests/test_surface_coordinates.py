"""Coordinate precision and mixed scalar/jet inputs across smooth surfaces."""

from decimal import Decimal

import pytest

from formula import Number
from formula.capsysred.shared.implicit_surface import Surface
from formula.capsysred.shared.jets import _Algebra, _Jet
from formula.capsysred.surfaces import ImplicitWall, Mirror
from formula.capsysred.walls.wall_cylinder import CylinderWall
from formula.capsysred.walls.wall_funnel import FunnelWall
from formula.capsysred.walls.wall_polygon import PolygonWall
from formula.capsysred.walls.wall_revolution import RevolutionWall
from formula.capsysred.walls.wall_torus import TorusWall


_KINDS = ("surface", "implicit", "cylinder", "revolution", "funnel", "torus", "polygon_face", "mirror")


def _surface(kind, precision=64):
    def number(value):
        return Number(value, precision)

    center = (number("0.1"), number("-0.2"))
    if kind == "surface":
        return Surface("x^2+y^2-1", precision=precision)
    if kind == "implicit":
        return ImplicitWall("(x/1000000)^2+(y/1000000)^2-1", center, number("1"))
    if kind == "cylinder":
        return CylinderWall(center, number("1"))
    if kind == "revolution":
        return RevolutionWall(kind, center, (number("1"), number("0.3"), number("-0.2")))
    if kind == "funnel":
        return FunnelWall(center, number("1"), (number("0.2"), number("-0.08")),
                          (number("-0.1"), number("0.03")), number("0.3"))
    if kind == "torus":
        return TorusWall(center, number("1"), number("4"),
                         (number("3"), number("4")), number("0.3"))
    if kind == "polygon_face":
        return PolygonWall(center, number("0.8"), 4, number("atan(4/3)")).face(0)
    return Mirror(number("0"), number("2"))


def _coefficients(value):
    return tuple(map(Decimal, value.c if isinstance(value, _Jet) else (str(value), "0", "0", "0")))


def _assert_equal(actual, expected):
    for got, want in zip(_coefficients(actual), _coefficients(expected), strict=True):
        assert abs(got - want) < Decimal("1e-55")


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("axis", range(3), ids=("jet_x", "jet_y", "jet_z"))
def test_mixed_coordinates_match_all_jet_inputs(kind, axis):
    surface, alg = _surface(kind), _Algebra(2, 64)
    point = [Number(value, 64) for value in ("0.6", "0.25", "0.5")]
    point[axis] = alg.variable(point[axis], (0, 0), 0)
    all_jets = tuple(value if isinstance(value, _Jet) else alg.constant(value) for value in point)
    value = surface.value(point)
    assert isinstance(value, _Jet)
    _assert_equal(value, surface.value(all_jets))
    for actual, expected in zip(surface.gradient(point), surface.gradient(all_jets), strict=True):
        _assert_equal(actual, expected)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("failure", ("algebra", "jet_precision", "number_precision", "mixed_number_precision"))
def test_value_and_gradient_reject_incompatible_coordinates(kind, failure):
    surface, alg = _surface(kind), _Algebra(2, 64)
    point = [Number(value, 64) for value in ("0.6", "0.25", "0.5")]
    if failure == "algebra":
        point[0], point[2] = alg.constant(point[0]), _Algebra(2, 64).constant(point[2])
    elif failure == "jet_precision":
        point[2] = _Algebra(2, 32).constant(point[2])
    else:
        point[2] = Number("0.500000000000000000000000000001", 128)
        if failure == "mixed_number_precision":
            point[0] = alg.constant(point[0])
    message = "algebra" if failure == "algebra" else "precision"
    for method in (surface.value, surface.gradient):
        with pytest.raises(ValueError, match=message):
            method(point)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("surface_precision,jet_precision", [(50, 64), (64, 50)])
def test_equivalent_rounded_precisions_are_accepted(kind, surface_precision, jet_precision):
    surface, alg = _surface(kind, surface_precision), _Algebra(2, jet_precision)
    point = tuple(Number(value, 50) for value in ("0.6", "0.25", "0.5"))
    mixed = (point[0], alg.constant(point[1]), point[2])
    value = surface.value(mixed)
    assert isinstance(value, _Jet)
    _assert_equal(value, surface.value(point))
    for actual, expected in zip(surface.gradient(mixed), surface.gradient(point), strict=True):
        _assert_equal(actual, expected)


@pytest.mark.parametrize("kind", _KINDS)
def test_mixed_coordinates_retain_small_number_digits(kind):
    surface, alg = _surface(kind), _Algebra(2, 64)
    point = (Number("1.250000000000000000000000000001", 64),
             alg.variable("0.25", (0, 0), 0), Number("0.5", 64))
    all_jets = (alg.constant(point[0]), point[1], alg.constant(point[2]))
    actual = surface.value(point)
    _assert_equal(actual, surface.value(all_jets))
    rounded = surface.value((Number("1.25", 64), point[1], point[2]))
    assert abs(Decimal(actual.c[0]) - Decimal(rounded.c[0])) > Decimal("1e-32")
    for component, expected in zip(surface.gradient(point), surface.gradient(all_jets), strict=True):
        _assert_equal(component, expected)

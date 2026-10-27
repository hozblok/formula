"""Implicit wall derivatives convert metre inputs to micrometre expressions."""

from decimal import Decimal

import pytest

from formula import Formula, Number
from formula.capsysred.shared.nums import vunit
from formula.capsysred.stages._ray_jets import _Algebra
from formula.capsysred.surfaces import ImplicitWall


_EXPRESSION = "(x-1.25)^2+(y+0.5)^2+0.2*(z-2.5)^3-0.16"
_WORLD_EXPRESSION = "(1000000*x-1.25)^2+(1000000*y+0.5)^2+0.2*(1000000*z-2.5)^3-0.16"


def _wall(expression):
    return ImplicitWall(expression, (Number("0", 96), Number("0", 96)), Number("1e-6", 96))


@pytest.mark.parametrize("number_inputs", [False, True])
def test_implicit_scalar_values_and_world_gradient(number_inputs):
    wall = _wall(_EXPRESSION)
    formula = Formula(_WORLD_EXPRESSION, precision=96)
    assert wall.precision == Number("0", 96).precision
    for point in [(1.25e-6, -0.5e-6, 2.5e-6), (1.65e-6, -0.5e-6, 2.5e-6),
                  (1.7e-6, 0.2e-6, 3.2e-6)]:
        coordinates = tuple(Number(str(v), 96) for v in point) if number_inputs else point
        values = dict(zip("xyz", map(str, point)))
        assert float(wall.value(coordinates)) == pytest.approx(float(formula.get(values)), abs=1e-15)
        expected = [float(formula.get_derivative(axis, values)) for axis in "xyz"]
        assert tuple(map(float, wall.gradient(coordinates))) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("jet_inputs", [False, True])
def test_implicit_retains_coordinates_beyond_float_precision(jet_inputs):
    wall = _wall("(x-1.000000000000000000000000000001)^2"
                 "+(y+0.500000000000000000000000000002)^2"
                 "+(z-2.500000000000000000000000000003)^2-4e-60")
    point = tuple(Number(value, 96) for value in
                  ("1.000000000000000000000000000004e-6",
                   "-0.500000000000000000000000000006e-6",
                   "2.500000000000000000000000000008e-6"))
    if jet_inputs:
        alg = _Algebra(4, 96)
        point = tuple(alg.constant(value) for value in point)
    actual = (wall.value(point), *wall.gradient(point))
    for value, expected in zip(actual, ("4.6e-59", "6e-24", "-8e-24", "1e-23"), strict=True):
        text = value.c[0] if jet_inputs else str(value)
        assert abs(Decimal(text) - Decimal(expected)) < Decimal("1e-85")


@pytest.mark.parametrize("active_axis", ["x", "y", "z", None])
def test_implicit_inactive_coordinates_have_exact_zero_gradient(active_axis):
    wall = _wall(f"{active_axis}^2-1" if active_axis else "7")
    alg = _Algebra(4, 96)
    seeds = (0, 1, 0, 1)
    u, v = alg.variable("0.3", seeds, 0), alg.variable("0.7", seeds, 1)
    jets = tuple(alg.evaluate(expression, u=u, v=v)
                 for expression in ("(2+u)/1000000", "(3+v)/1000000", "(4+u*v)/1000000"))
    scalars = tuple(Number(jet.c[0], 96) for jet in jets)
    for point in (scalars, jets):
        gradient = wall.gradient(point)
        for axis, component in zip("xyz", gradient, strict=True):
            if axis != active_axis:
                coefficients = component.c if point is jets else (str(component),)
                assert all(Decimal(value) == 0 for value in coefficients)
        if active_axis is None:
            value = wall.value(point)
            assert Decimal(value.c[0] if point is jets else str(value)) == 7
        else:
            component = gradient["xyz".index(active_axis)]
            assert Decimal(component.c[0] if point is jets else str(component)) != 0


def _assert_mixed_derivatives(jet, expression, names):
    formula = Formula(expression, precision=96)
    values = {"u": "0.3", "v": "0.7"}
    for mask, coefficient in enumerate(jet.c):
        variables = [name for bit, name in enumerate(names) if mask & (1 << bit)]
        expected = (formula.get_partial_derivative(variables, values)
                    if variables else formula.get(values))
        assert abs(Decimal(coefficient) - Decimal(expected)) < Decimal("1e-80")


@pytest.mark.parametrize("order", [3, 4])
def test_implicit_composed_world_derivatives_and_normal(order):
    alg = _Algebra(order, 96)
    seeds = (0, 1, 0, 1)[:order]
    names = tuple("uv"[axis] for axis in seeds)
    u, v = alg.variable("0.3", seeds, 0), alg.variable("0.7", seeds, 1)
    x, y, z = "((1+u^2+u*v)/1000000)", "((2+v^2-u)/1000000)", "((2+u^2+u*v+v^2)/1000000)"
    point = tuple(alg.evaluate(expression, u=u, v=v) for expression in (x, y, z))
    wall = _wall(_EXPRESSION)
    dx, dy, dz = f"(1000000*{x}-1.25)", f"(1000000*{y}+0.5)", f"(1000000*{z}-2.5)"
    value, gradient = wall.value(point), wall.gradient(point)
    _assert_mixed_derivatives(value, f"{dx}^2+{dy}^2+0.2*{dz}^3-0.16", names)
    assert abs(Decimal(value.c[-1])) > Decimal("1e-4")
    expressions = (f"(2000000*{dx})", f"(2000000*{dy})", f"(600000*{dz}^2)")
    length = "sqrt(" + "+".join(f"{expression}^2" for expression in expressions) + ")"
    for component, normal, expression in zip(gradient, vunit(gradient), expressions, strict=True):
        _assert_mixed_derivatives(component, expression, names)
        _assert_mixed_derivatives(normal, f"{expression}/{length}", names)

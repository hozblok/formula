"""Cylinder surface values and gradients for numbers and composed ray jets."""

from decimal import Decimal, localcontext

import pytest

from formula import Formula, Number
from formula.capsysred.shared.nums import vunit
from formula.capsysred.stages._ray_jets import _Algebra
from formula.capsysred.walls.wall_cylinder import CylinderWall


@pytest.mark.parametrize("number_inputs", [False, True])
def test_shifted_cylinder_value_and_gradient(number_inputs):
    wall = CylinderWall((Number("1.25", 64), Number("-0.5", 64)), Number("2", 64))
    for point, expected in [((1.25, -0.5, -7.0), -2.0),
                            ((3.25, -0.5, 0.0), 0.0),
                            ((4.25, -0.5, 9.0), 2.5)]:
        coordinates = tuple(Number(str(v), 64) for v in point) if number_inputs else point
        value = wall.value(coordinates)
        assert float(value) == pytest.approx(expected)
        assert value.precision == 64
        expected_gradient, _ = wall.derivatives_at(*point)
        assert tuple(map(float, wall.gradient(coordinates))) == pytest.approx(expected_gradient)


@pytest.mark.parametrize("jet_inputs", [False, True])
def test_cylinder_parameters_keep_precision_beyond_float(jet_inputs):
    alg = _Algebra(4, 96)
    with localcontext() as context:
        context.prec = 110
        cases = [
            ("1.000000000000000000000000000001", "2e-30",
             "1.000000000000000000000000000004", Decimal("2.5e-60"), Decimal("3e-30")),
            ("0", "1.000000000000000000000000000002", "1",
             -Decimal("2e-30") - Decimal("2e-60"), Decimal("1")),
        ]
        for center, radius, x, expected_value, expected_gradient in cases:
            wall = CylinderWall((Number(center, 96), Number("0", 96)), Number(radius, 96))
            point = tuple(Number(v, 96) for v in (x, "0", "0"))
            if jet_inputs:
                point = tuple(alg.constant(v) for v in point)
            value, gradient = wall.value(point), wall.gradient(point)
            value_text = value.c[0] if jet_inputs else str(value)
            gradient_text = gradient[0].c[0] if jet_inputs else str(gradient[0])
            assert abs(Decimal(value_text) - expected_value) < Decimal("1e-85")
            assert abs(Decimal(gradient_text) - expected_gradient) < Decimal("1e-85")


def _assert_mixed_derivatives(jet, expression, names):
    formula = Formula(expression, precision=96)
    values = {"u": "0.3", "v": "0.7"}
    for mask, coefficient in enumerate(jet.c):
        variables = [name for bit, name in enumerate(names) if mask & (1 << bit)]
        expected = (formula.get_partial_derivative(variables, values)
                    if variables else formula.get(values))
        assert abs(Decimal(coefficient) - Decimal(expected)) < Decimal("1e-80")


@pytest.mark.parametrize("order", [3, 4])
def test_cylinder_composes_higher_derivatives_and_normal(order):
    alg = _Algebra(order, 96)
    seeds = (0, 1, 0, 1)[:order]
    names = tuple("uv"[axis] for axis in seeds)
    u, v = alg.variable("0.3", seeds, 0), alg.variable("0.7", seeds, 1)
    x = alg.evaluate("u^2+u*v+1", u=u, v=v)
    y = alg.evaluate("v^2-u+2", u=u, v=v)
    wall = CylinderWall((Number("1.25", 96), Number("-0.5", 96)), Number("2", 96))
    point = (x, y, 0)
    value, gradient = wall.value(point), wall.gradient(point)
    dx, dy = "(u^2+u*v-0.25)", "(v^2-u+2.5)"
    _assert_mixed_derivatives(value, f"({dx}^2+{dy}^2-4)/2", names)
    assert abs(Decimal(value.c[-1])) > 1
    _assert_mixed_derivatives(gradient[0], dx, names)
    _assert_mixed_derivatives(gradient[1], dy, names)
    assert gradient[2] == 0

    normal = vunit(gradient)
    _assert_mixed_derivatives(normal[0], f"{dx}/sqrt({dx}^2+{dy}^2)", names)
    _assert_mixed_derivatives(normal[1], f"{dy}/sqrt({dx}^2+{dy}^2)", names)
    assert all(Decimal(coefficient) == 0 for coefficient in normal[2].c)

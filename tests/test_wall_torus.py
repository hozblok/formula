"""Torus surface values and gradients for numbers and composed ray jets."""

from decimal import Decimal, localcontext
import math

import pytest

from formula import Formula, Number
from formula.capsysred.shared.nums import vunit
from formula.capsysred.stages._ray_jets import _Algebra
from formula.capsysred.walls.wall_torus import TorusWall


@pytest.mark.parametrize("number_inputs", [False, True])
def test_shifted_rotated_torus_value_and_gradient(number_inputs):
    wall = TorusWall((Number("1.25", 64), Number("-0.5", 64)),
                     Number("0.4", 64), Number("5", 64),
                     (Number("3", 64), Number("4", 64)), Number("2.5", 64))
    assert wall.precision == 64
    for xi, eta, zr in [(0, 0, 0), (0, 0.4, 0), (0, 0.7, 0), (0.3, -0.2, 1.1)]:
        point = (1.25 + 0.6 * xi - 0.8 * eta,
                 -0.5 + 0.8 * xi + 0.6 * eta, 2.5 + zr)
        coordinates = tuple(Number(str(v), 64) for v in point) if number_inputs else point
        expected = ((math.hypot(xi - 5, zr) - 5) ** 2 + eta ** 2 - 0.4 ** 2) / 2
        value = wall.value(coordinates)
        assert float(value) == pytest.approx(expected, abs=1e-14)
        assert value.precision == 64
        expected_gradient, _ = wall.derivatives_at(*point)
        assert tuple(map(float, wall.gradient(coordinates))) == pytest.approx(expected_gradient)


@pytest.mark.parametrize("jet_inputs", [False, True])
def test_torus_keeps_tiny_offsets_with_large_bend_radius(jet_inputs):
    alg = _Algebra(4, 96)
    with localcontext() as context:
        context.prec = 160
        cx, cy, z0 = ("1.000000000000000000000000000001",
                      "-2.000000000000000000000000000002",
                      "3.000000000000000000000000000003")
        bend_radius = "1.000000000000000000000000000002e40"
        wall = TorusWall((Number(cx, 96), Number(cy, 96)),
                         Number("2e-30", 96), Number(bend_radius, 96),
                         (Number("1", 96), Number("0", 96)), Number(z0, 96))
        xi, eta, zr = Decimal("3e-30"), Decimal("4e-30"), Decimal("1e5")
        point = tuple(Number(str(v), 96) for v in
                      (Decimal(cx) + xi, Decimal(cy) + eta, Decimal(z0) + zr))
        if jet_inputs:
            point = tuple(alg.constant(v) for v in point)
        value, gradient = wall.value(point), wall.gradient(point)
        actual = [v.c[0] if jet_inputs else str(v) for v in (value, *gradient)]
        radius = Decimal(bend_radius)
        rho = ((xi - radius) ** 2 + zr ** 2).sqrt()
        dr = rho - radius
        expected = ((dr ** 2 + eta ** 2 - Decimal("2e-30") ** 2) / 2,
                    dr * (xi - radius) / rho, eta, dr * zr / rho)
        for result, target in zip(actual, expected, strict=True):
            assert abs(Decimal(result) - Decimal(target)) < Decimal("1e-85")


def _assert_mixed_derivatives(jet, expression, names):
    formula = Formula(expression, precision=96)
    values = {"u": "0.3", "v": "0.7"}
    for mask, coefficient in enumerate(jet.c):
        variables = [name for bit, name in enumerate(names) if mask & (1 << bit)]
        expected = (formula.get_partial_derivative(variables, values)
                    if variables else formula.get(values))
        assert abs(Decimal(coefficient) - Decimal(expected)) < Decimal("1e-80")


@pytest.mark.parametrize("order", [3, 4])
def test_torus_composes_higher_derivatives_and_normal(order):
    alg = _Algebra(order, 96)
    seeds = (0, 1, 0, 1)[:order]
    names = tuple("uv"[axis] for axis in seeds)
    u, v = alg.variable("0.3", seeds, 0), alg.variable("0.7", seeds, 1)
    point = tuple(alg.evaluate(expression, u=u, v=v)
                  for expression in ("1+u^2+u*v", "v^2-u+2", "2+u+v^2"))
    wall = TorusWall((Number("1.25", 96), Number("-0.5", 96)),
                     Number("0.4", 96), Number("5", 96),
                     (Number("3", 96), Number("4", 96)), Number("2.5", 96))
    dx, dy, zr = "(u^2+u*v-0.25)", "(v^2-u+2.5)", "(u+v^2-0.5)"
    xi, eta = f"(0.6*{dx}+0.8*{dy})", f"(-0.8*{dx}+0.6*{dy})"
    rho = f"sqrt(({xi}-5)^2+{zr}^2)"
    dr = f"({rho}-5)"
    value, gradient = wall.value(point), wall.gradient(point)
    _assert_mixed_derivatives(value, f"({dr}^2+{eta}^2-0.16)/2", names)
    assert abs(Decimal(value.c[-1])) > Decimal("1e-4")
    expressions = (f"({dr}/{rho}*({xi}-5)*0.6-{eta}*0.8)",
                   f"({dr}/{rho}*({xi}-5)*0.8+{eta}*0.6)",
                   f"({dr}/{rho}*{zr})")
    normal_length = f"sqrt({dr}^2+{eta}^2)"
    for component, normal_component, expression in zip(gradient, vunit(gradient), expressions, strict=True):
        _assert_mixed_derivatives(component, expression, names)
        _assert_mixed_derivatives(normal_component, f"{expression}/{normal_length}", names)

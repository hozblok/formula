"""Funnel formulas on Number and higher-order ray jets."""

from decimal import Decimal

import pytest

from formula import Formula, Number
from formula.capsysred.shared.nums import vunit
from formula.capsysred.stages._ray_jets import _Algebra
from formula.capsysred.walls.wall_funnel import FunnelWall


_PARAMS = {"cx": "1.25", "cy": "-0.5", "r0": "0.4", "z0": "2.5",
           "ag": "0.2", "bg": "-0.08", "af": "-0.1", "bf": "0.03"}
_G, _F = "(1+ag*(z-z0)+bg*(z-z0)^2)", "(1+af*(z-z0)+bf*(z-z0)^2)"
_EXPRESSION = f"((x-cx*{_G})^2+(y-cy*{_G})^2-r0^2*{_F}^2)/2"


def _wall(parameters):
    p = {name: Number(value, 96) for name, value in parameters.items()}
    return FunnelWall((p["cx"], p["cy"]), p["r0"], (p["ag"], p["bg"]),
                      (p["af"], p["bf"]), p["z0"])


@pytest.mark.parametrize("number_inputs", [False, True])
def test_funnel_shifted_scalar_geometry(number_inputs):
    wall = _wall(_PARAMS)
    formula = Formula(_EXPRESSION, precision=96)
    assert wall.precision == Number("0", 96).precision
    for point in [(1.25, -0.5, 2.5), (1.65, -0.5, 2.5), (1.7, 0.2, 3.2)]:
        coordinates = tuple(Number(str(v), 96) for v in point) if number_inputs else point
        values = dict(_PARAMS, **dict(zip("xyz", map(str, point))))
        assert float(wall.value(coordinates)) == pytest.approx(float(formula.get(values)), abs=1e-15)
        gradient = tuple(map(float, wall.gradient(coordinates)))
        assert gradient == pytest.approx(wall.derivatives_at(*point)[0])
        assert gradient == pytest.approx([float(formula.get_derivative(axis, values)) for axis in "xyz"])


@pytest.mark.parametrize("jet_inputs", [False, True])
def test_funnel_retains_parameter_precision(jet_inputs):
    parameters = {"cx": "1.250000000000000000000000000001",
                  "cy": "-0.500000000000000000000000000002",
                  "r0": "0.400000000000000000000000000003",
                  "z0": "2.500000000000000000000000000004",
                  "ag": "0.200000000000000000000000000005",
                  "bg": "-0.080000000000000000000000000006",
                  "af": "-0.100000000000000000000000000007",
                  "bf": "0.030000000000000000000000000008"}
    wall, alg = _wall(parameters), _Algebra(4, 96)
    coordinates = ("1.7", "0.2", "3.2")
    point = tuple(Number(v, 96) for v in coordinates)
    if jet_inputs:
        point = tuple(alg.constant(v) for v in point)
    formula = Formula(_EXPRESSION, precision=96)
    values = dict(parameters, **dict(zip("xyz", coordinates)))
    expected = [formula.get(values)] + [formula.get_derivative(axis, values) for axis in "xyz"]
    for result, target in zip((wall.value(point), *wall.gradient(point)), expected, strict=True):
        text = result.c[0] if jet_inputs else str(result)
        assert abs(Decimal(text) - Decimal(target)) < Decimal("1e-85")


def _assert_mixed_derivatives(jet, expression, names):
    formula = Formula(expression, precision=96)
    values = {"u": "0.3", "v": "0.7"}
    for mask, coefficient in enumerate(jet.c):
        variables = [name for bit, name in enumerate(names) if mask & (1 << bit)]
        expected = (formula.get_partial_derivative(variables, values)
                    if variables else formula.get(values))
        assert abs(Decimal(coefficient) - Decimal(expected)) < Decimal("1e-80")


@pytest.mark.parametrize("order", [3, 4])
def test_funnel_composed_derivatives_and_normal(order):
    alg = _Algebra(order, 96)
    seeds = (0, 1, 0, 1)[:order]
    names = tuple("uv"[axis] for axis in seeds)
    u, v = alg.variable("0.3", seeds, 0), alg.variable("0.7", seeds, 1)
    point = tuple(alg.evaluate(expression, u=u, v=v)
                  for expression in ("1+u^2+u*v", "v^2-u+2", "2+u+v^2"))
    wall = _wall(_PARAMS)
    z = "(u+v^2-0.5)"
    g, f = f"(1+0.2*{z}-0.08*{z}^2)", f"(1-0.1*{z}+0.03*{z}^2)"
    dx, dy = f"(1+u^2+u*v-1.25*{g})", f"(v^2-u+2+0.5*{g})"
    value, gradient = wall.value(point), wall.gradient(point)
    _assert_mixed_derivatives(value, f"({dx}^2+{dy}^2-0.16*{f}^2)/2", names)
    assert abs(Decimal(value.c[-1])) > Decimal("1e-4")
    gz = f"(-(1.25*{dx}-0.5*{dy})*(0.2-0.16*{z})-0.16*{f}*(-0.1+0.06*{z}))"
    length = f"sqrt({dx}^2+{dy}^2+{gz}^2)"
    for component, normal, expression in zip(gradient, vunit(gradient), (dx, dy, gz), strict=True):
        _assert_mixed_derivatives(component, expression, names)
        _assert_mixed_derivatives(normal, f"{expression}/{length}", names)

"""Compose hyper-dual inputs without dropping structural zero coefficients."""

from decimal import Decimal

import pytest

from formula import Formula
from tests.decimal_pairs import parse


def seeded(names, values):
    size = 1 << len(names)
    coefficients = {name: [str(value)] + ["0"] * (size - 1)
                    for name, value in values.items()}
    supports = {name: 1 for name in values}
    for bit, name in enumerate(names):
        subset = 1 << bit
        coefficients[name][subset] = "1"
        supports[name] |= 1 << subset
    return coefficients, supports


@pytest.mark.parametrize("complex_formula", [False, True])
def test_composition_matches_direct_mixed_derivatives(complex_formula):
    names = ["u", "v", "u", "v"]
    values = {"u": "0.3", "v": "0.7"}
    coefficients, supports = seeded(names, values)
    inner = {"x": "sin(u*v)+u^2", "y": "exp(u-v)/(2+v)"}
    if complex_formula:
        inner["x"] += "+i*cos(v)"
        inner["y"] += "+i*u"
    composed, inherited = {}, {}
    for name, expression in inner.items():
        composed[name], inherited[name] = Formula(expression, precision=96)._evaluate_jet(
            4, coefficients, supports)
    suffix = "+0*i" if complex_formula else ""
    result, mask = Formula("sqrt(2+x^2)+x/y+sin(y)" + suffix, precision=96)._evaluate_jet(
        4, composed, inherited)
    x, y = inner["x"], inner["y"]
    direct = Formula(f"sqrt(2+({x})^2)+({x})/({y})+sin({y})", precision=96)
    for subset, value in enumerate(result):
        variables = [name for bit, name in enumerate(names) if subset & (1 << bit)]
        expected = direct.get_partial_derivative(variables, values) if variables else direct.get(values)
        for got, wanted in zip(parse(value), parse(expected)):
            assert abs(got - wanted) < Decimal("1e-80") * max(1, abs(wanted))
    assert mask == (1 << 16) - 1


def test_structural_zero_survives_a_composition_boundary():
    coefficients, supports = seeded(["x", "x"], {"x": "0"})
    square, mask = Formula("x^2", precision=64)._evaluate_jet(2, coefficients, supports)
    assert [Decimal(value) for value in square] == [0, 0, 0, 2]
    assert mask == 15
    with pytest.raises(ValueError):
        Formula("sqrt(y)", precision=64)._evaluate_jet(2, {"y": square}, {"y": mask})
    zero, constant = Formula("sqrt(y)")._evaluate_jet(
        2, {"y": ["0"] * 4}, {"y": 1})
    assert list(map(Decimal, zero)) == [0, 0, 0, 0]
    assert constant == 1


def test_mixed_only_input_and_constant_support_are_preserved():
    result, support = Formula("exp(x)+y", precision=64)._evaluate_jet(
        2, {"x": ["0", "0", "0", "3"], "y": ["2", "0", "0", "0"]},
        {"x": 9, "y": 1})
    assert list(map(Decimal, result)) == [3, 0, 0, 3]
    assert support == 9


@pytest.mark.parametrize("expression", ["sin(x)*exp(x)", "sqrt(x)/(1+x)", "x^2+i*x"])
def test_first_derivative_keeps_the_existing_arithmetic(expression):
    formula = Formula(expression, precision=64)
    result, _ = formula._evaluate_jet(1, {"x": ["0.3", "1"]}, {"x": 3})
    assert result[0] == formula.get({"x": "0.3"})
    assert result[1] == formula.get_derivative("x", {"x": "0.3"})
    assert result[1] == formula.get_partial_derivative(["x"], {"x": "0.3"})


def test_order_five_returns_all_subsets():
    coefficients, supports = seeded(["x"] * 5, {"x": "0"})
    result, support = Formula("exp(x)", precision=64)._evaluate_jet(5, coefficients, supports)
    assert list(map(Decimal, result)) == [1] * 32
    assert support == (1 << 32) - 1


@pytest.mark.parametrize("order,coefficients,supports", [
    (0, {"x": ["1"]}, {"x": 1}),
    (6, {"x": ["0"] * 64}, {"x": 1}),
    (2, {"x": ["0"] * 3}, {"x": 1}),
    (2, {"x": ["0"] * 4}, {"x": 0}),
    (2, {"x": ["0"] * 4}, {"x": 17}),
    (2, {"x": ["0"] * 4}, {}),
    (2, {"x": ["0"] * 4}, {"y": 1}),
    (2, {"x": ["0", "1", "0", "0"]}, {"x": 1}),
    (1, {"x": ["nan", "0"]}, {"x": 1}),
    (1, {"x": ["0", "1/3"]}, {"x": 3}),
])
def test_invalid_jet_inputs_are_rejected(order, coefficients, supports):
    with pytest.raises(ValueError):
        Formula("x")._evaluate_jet(order, coefficients, supports)


def test_missing_formula_variable_is_rejected():
    with pytest.raises(ValueError):
        Formula("x+y")._evaluate_jet(1, {"x": ["0", "0"]}, {"x": 1})

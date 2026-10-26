"""Regression: a custom imaginary unit works in composite expressions.

The cseval_complex parser did not pass imaginary_unit_ down to child
nodes, so anything beyond a bare "j" parsed the unit as a variable and
failed with "The required value is not found". The unit also leaked as
a hardcoded 'i' into get() output.
"""

import pytest

from formula import Formula, Solver


def test_bare_custom_unit():
    assert Solver("j", imaginary_unit="j")() == "0+j*(1)"


def test_custom_unit_in_product():
    assert Solver("2*j", imaginary_unit="j")() == "0+j*(2)"


def test_custom_unit_with_variable():
    assert Solver("x+j*x", imaginary_unit="j")({"x": "3"}) == "3+j*(3)"


def test_custom_unit_inside_parentheses_and_function():
    assert Solver("(1+j)*(1-j)", imaginary_unit="j")() == "2+j*(0)"
    assert Solver("abs(3+4*j)", imaginary_unit="j")() == "5+j*(0)"


def test_custom_unit_negation():
    assert Solver("-j", imaginary_unit="j")() == "0+j*(-1)"


def test_default_unit_output_unchanged():
    assert Solver("2*i")() == "0+i*(2)"


def test_unit_letter_stays_a_variable_under_other_unit():
    # With unit 'j', plain 'i' is an ordinary variable.
    assert Solver("i+j", imaginary_unit="j")({"i": "1"}) == "1+j*(1)"


def test_custom_unit_derivative():
    s = Solver("j*x^2", imaginary_unit="j")
    assert s({"x": "2"}, derivative="x") == "0+j*(4)"
    assert s.get_partial_derivative(["x"], {"x": "2"}, s.precision) == "0+j*(4)"


def test_pair_with_custom_unit():
    assert Solver("2*j", imaginary_unit="j").pair() == ("0", "2")


@pytest.mark.parametrize("unit", ["j", "J", "q"])
@pytest.mark.parametrize("digits", [0, 7, 25])
def test_custom_unit_on_higher_derivatives(unit, digits):
    # d^5/(d alpha^3 d beta_2^2) of (unit alpha^3) (beta_2^2 / 7) is (unit 6) (2/7).
    names = ["alpha"] * 3 + ["beta_2"] * 2
    f = Formula(f"{unit}*alpha^3*beta_2^2/7", 24, unit)
    got = f.get_partial_derivative(names, {"alpha": "1.5", "beta_2": "-2"}, digits)
    assert got == Formula(f"({unit}*6)*(2/7)", 24, unit).get({}, digits)
    assert got.startswith(f"0+{unit}*(")


def test_letter_i_stays_a_variable_in_higher_derivatives_under_unit_j():
    # j i^2 x^3: d^2/di^2 d^3/dx^3 = 2 * 6 j.
    f = Formula("j*i^2*x^3", 24, "j")
    assert f.get_partial_derivative(["i", "i", "x", "x", "x"], {"i": "5", "x": "7"}) == "0+j*(12)"

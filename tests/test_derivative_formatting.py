"""Regression: get_derivative honors format_digits / format_flags.

The derivative visitor called .str() with no arguments, dumping all
internal digits regardless of what the caller asked for. Complex
derivatives also leaked boost's "(re,im)" form instead of the
"re+i*(im)" shape every other output uses.
"""

import pytest

from formula import FmtFlags, Formula, Solver


def test_derivative_respects_format_digits():
    s = Solver("1/x", precision=24)
    assert s({"x": "3"}, derivative="x", format_digits=5) == "-0.11111"
    assert s.get_partial_derivative(["x"], {"x": "3"}, 5) == "-0.11111"


def test_derivative_respects_format_flags():
    s = Solver("x^2", precision=24)
    out = s({"x": "10"}, derivative="x", format_digits=5, format_flags=FmtFlags.scientific)
    assert out == "2.00000e+01"
    assert s.get_partial_derivative(["x"], {"x": "10"}, 5, FmtFlags.scientific) == out


def test_derivative_default_digits_match_value_formatting():
    # Same number via get() and via get_derivative() must format the same.
    s = Solver("1/x", precision=24)
    value = s({"x": "3"}, format_digits=10)
    derivative = Solver("0-1/(x^2)", precision=24)({"x": "3"}, format_digits=10)
    assert s({"x": "3"}, derivative="x", format_digits=10) == derivative
    assert s.get_partial_derivative(["x"], {"x": "3"}, 10) == derivative
    assert value == "0.3333333333"


def test_complex_derivative_uses_value_output_shape():
    # Was "(0,4)"; must match get()'s "re+i*(im)" shape.
    s = Solver("i*x^2")
    assert s({"x": "2"}, derivative="x") == "0+i*(4)"
    assert s.get_partial_derivative(["x"], {"x": "2"}, s.precision) == "0+i*(4)"


FORMATS = [FmtFlags.default, FmtFlags.fixed, FmtFlags.scientific, FmtFlags.showpos]
FORMAT_IDS = ["default", "fixed", "scientific", "showpos"]
MIXED = ["alpha"] * 3 + ["beta_2"] * 2
MIXED_POINT = {"alpha": "1.5", "beta_2": "-2"}


@pytest.mark.parametrize("fmt", FORMATS, ids=FORMAT_IDS)
@pytest.mark.parametrize("digits", [0, 7, 25])
@pytest.mark.parametrize("precision", [24, 128])
def test_higher_derivative_formats_like_the_value_of_the_derivative(precision, digits, fmt):
    # d^5/(d alpha^3 d beta_2^2) of alpha^3 (beta_2^2 / 7) is the constant 6 (2/7), the
    # reference written in the same order of operations.
    f = Formula("alpha^3*beta_2^2/7", precision)
    got = f.get_partial_derivative(MIXED, MIXED_POINT, digits, fmt)
    assert got == Formula("6*(2/7)", precision).get({}, digits, fmt)


@pytest.mark.parametrize("fmt", FORMATS, ids=FORMAT_IDS)
@pytest.mark.parametrize("digits", [0, 7])
@pytest.mark.parametrize("expr,reference", [
    ("i*alpha^3*beta_2^2/7", "(i*6)*(2/7)"),            # real part 0
    ("alpha^3*beta_2^2/7+i*alpha^2", "6*(2/7)+0*i"),    # imaginary part 0
])
def test_complex_higher_derivative_formats_like_its_value(expr, reference, digits, fmt):
    got = Formula(expr, 24).get_partial_derivative(MIXED, MIXED_POINT, digits, fmt)
    assert got == Formula(reference, 24).get({}, digits, fmt)

"""Regression: malformed variable values raise instead of silently truncating.

boost's cpp_dec_float("1/3") parses the leading "1" and stops, so
Solver("x^2")({"x": "1/3"}) silently computed 1. Values must be plain
numerals; expressions belong in the formula string.
"""

import pytest

from formula import Formula, Solver


@pytest.mark.parametrize("bad", ["1/3", "1+2", "2x", "1j", "(1,2)", "one"])
def test_malformed_value_raises(bad):
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("x*2")({"x": bad})


def test_error_names_variable_and_value():
    with pytest.raises(ValueError, match=r"'1/3'.*'x'"):
        Solver("x*2")({"x": "1/3"})


def test_python_complex_value_rejected():
    # str(1j) == "1j" used to silently become 1.
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("x*2")({"x": 1j})


def test_malformed_value_raises_for_derivative():
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("x^2")({"x": "1/3"}, derivative="x")
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("x^2").get_partial_derivative(["x"], {"x": "1/3"})


def test_malformed_value_raises_for_complex_expression():
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("i*x")({"x": "1/3"})
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("i*x")({"x": "1/3"}, derivative="x")
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("i*x").get_partial_derivative(["x"], {"x": "1/3"})


@pytest.mark.parametrize(
    "good", ["1", "-1", "+3", ".5", "0009.", "1e-15", "-100.0e10000", "+0e+0"]
)
def test_plain_numerals_still_accepted(good):
    Solver("x*2")({"x": good})  # must not raise


def test_int_and_float_values_still_accepted():
    assert Solver("x*2")({"x": 21}) == "42"
    assert Solver("x*2")({"x": 0.5}) == "1"
    assert Solver("x*2")(1.5e-5) == "3e-05"


@pytest.mark.parametrize("k", [2, 3, 4, 5])
@pytest.mark.parametrize("expr", ["x^6*y", "x^6*y+0*i"])
def test_higher_derivative_needs_every_value_and_ignores_extra_ones(expr, k):
    f = Formula(expr, 64)
    for point in ({"x": "2"}, {"y": "3"}, {}):
        with pytest.raises(ValueError):
            f.get_partial_derivative(["x"] * k, point)
    full = {"x": "2", "y": "3"}
    assert f.get_partial_derivative(["x"] * k, dict(full, unused="9")) == \
        f.get_partial_derivative(["x"] * k, full)


@pytest.mark.parametrize("bad", ["1/3", "nan", "inf", "-inf", "NaN"])
@pytest.mark.parametrize("expr", ["x^6*y", "x^6*y+0*i"])
def test_higher_derivative_rejects_malformed_values(expr, bad):
    with pytest.raises(ValueError, match="Invalid number value"):
        Formula(expr, 64).get_partial_derivative(["x", "x", "y"], {"x": bad, "y": "1"})


@pytest.mark.parametrize("names", [["x", None], "xx", 5, [1, 2]])
@pytest.mark.parametrize("expr", ["x^6*y", "x^6*y+0*i"])
def test_higher_derivative_rejects_names_of_the_wrong_type(expr, names):
    with pytest.raises(TypeError):
        Formula(expr, 64).get_partial_derivative(names, {"x": "2", "y": "3"})


def test_case_insensitive_names_follow_get_derivative():
    # Case folding of derivative names is not documented: both methods agree.
    f = Formula("X^3*y", 64, "i", True)
    at = {"x": "2", "y": "3"}
    for name in ("x", "X", "y", "Y"):
        assert f.get_partial_derivative([name], at) == f.get_derivative(name, at)
    assert f.get_partial_derivative(["x", "x", "y"], at) == "12"

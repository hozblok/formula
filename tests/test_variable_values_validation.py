"""Regression: malformed variable values raise instead of silently truncating.

boost's cpp_dec_float("1/3") parses the leading "1" and stops, so
Solver("x^2")({"x": "1/3"}) silently computed 1. Values must be plain
numerals; expressions belong in the formula string.
"""

import pytest

from formula import Solver


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


def test_malformed_value_raises_for_complex_expression():
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("i*x")({"x": "1/3"})
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("i*x")({"x": "1/3"}, derivative="x")


@pytest.mark.parametrize(
    "good", ["1", "-1", "+3", ".5", "0009.", "1e-15", "-100.0e10000", "+0e+0"]
)
def test_plain_numerals_still_accepted(good):
    Solver("x*2")({"x": good})  # must not raise


def test_int_and_float_values_still_accepted():
    assert Solver("x*2")({"x": 21}) == "42"
    assert Solver("x*2")({"x": 0.5}) == "1"
    assert Solver("x*2")(1.5e-5) == "3e-05"

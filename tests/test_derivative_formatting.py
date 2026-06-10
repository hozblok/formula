"""Regression: get_derivative honors format_digits / format_flags.

The derivative visitor called .str() with no arguments, dumping all
internal digits regardless of what the caller asked for. Complex
derivatives also leaked boost's "(re,im)" form instead of the
"re+i*(im)" shape every other output uses.
"""

from formula import FmtFlags, Solver


def test_derivative_respects_format_digits():
    s = Solver("1/x", precision=24)
    assert s({"x": "3"}, derivative="x", format_digits=5) == "-0.11111"


def test_derivative_respects_format_flags():
    s = Solver("x^2", precision=24)
    out = s({"x": "10"}, derivative="x", format_digits=5, format_flags=FmtFlags.scientific)
    assert out == "2.00000e+01"


def test_derivative_default_digits_match_value_formatting():
    # Same number via get() and via get_derivative() must format the same.
    s = Solver("1/x", precision=24)
    value = s({"x": "3"}, format_digits=10)
    derivative = Solver("0-1/(x^2)", precision=24)({"x": "3"}, format_digits=10)
    assert s({"x": "3"}, derivative="x", format_digits=10) == derivative
    assert value == "0.3333333333"


def test_complex_derivative_uses_value_output_shape():
    # Was "(0,4)"; must match get()'s "re+i*(im)" shape.
    assert Solver("i*x^2")({"x": "2"}, derivative="x") == "0+i*(4)"

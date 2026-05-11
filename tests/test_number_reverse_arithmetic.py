"""Regression: Number supports reverse arithmetic with builtin numerics.

See ai/improvements_2026-05-09.md item #4.

Before the fix, `1 + Number("2")` raised TypeError because Number had no
__radd__ (and likewise for sub/mul/truediv/pow).
"""

from formula import Number


def test_int_plus_number():
    result = 1 + Number("2")
    assert isinstance(result, Number)
    assert result == Number("3")


def test_int_minus_number():
    result = 5 - Number("2")
    assert result == Number("3")


def test_int_times_number():
    result = 2 * Number("3")
    assert result == Number("6")


def test_int_div_number():
    result = 10 / Number("2")
    assert result == Number("5")


def test_int_pow_number():
    result = 2 ** Number("3")
    assert result == Number("8")


def test_str_plus_number():
    result = "1.5" + Number("2.5")
    assert result == Number("4")


def test_float_div_number():
    result = 1.0 / Number("4")
    assert result == Number("0.25")

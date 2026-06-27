"""Tests for atanh() (inverse hyperbolic tangent)."""

import pytest

from formula import Formula, Solver


def test_atanh_zero():
    assert Formula("atanh(0)").get() == "0"


def test_atanh_derivative_at_zero():
    # d(atanh x)/dx = 1/(1 - x^2); at x=0 → 1.
    assert Solver("atanh(x)").get_derivative("x", {"x": "0"}) == "1"


def test_atanh_derivative_at_one_diverges():
    with pytest.raises(ValueError, match="atanh"):
        Solver("atanh(x)").get_derivative("x", {"x": "1"})


def test_atanh_derivative_at_minus_one_diverges():
    # The other pole of 1/(1 - x^2), at x=-1.
    with pytest.raises(ValueError, match="atanh"):
        Solver("atanh(x)").get_derivative("x", {"x": "-1"})


def test_atanh_value_outside_domain_is_nan():
    # atanh is real only on (-1, 1); outside, the value is NaN.
    assert Formula("atanh(2)").get() == "nan"

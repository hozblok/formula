"""Tests for acosh() (inverse hyperbolic cosine)."""

import pytest

from formula import Formula, Solver


def test_acosh_one_is_zero():
    assert Formula("acosh(1)").get() == "0"


def test_acosh_derivative_at_one_diverges():
    # d(acosh x)/dx = 1/sqrt(x^2 - 1); diverges at x=1.
    with pytest.raises(ValueError, match="acosh"):
        Solver("acosh(x)").get_derivative("x", {"x": "1"})


def test_acosh_derivative_at_regular_point():
    # d(acosh x)/dx = 1/sqrt(x^2 - 1); at x=2 -> 1/sqrt(3) ≈ 0.5773502692.
    result = Solver("acosh(x)").get_derivative("x", {"x": "2"})
    assert abs(float(result) - 1 / 3 ** 0.5) < 1e-9


def test_acosh_value_below_domain_is_nan():
    # acosh is real only for x >= 1; below, the value is NaN.
    assert Formula("acosh(0.5)").get() == "nan"

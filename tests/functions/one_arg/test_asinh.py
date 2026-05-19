"""Tests for asinh() (inverse hyperbolic sine)."""

from formula import Formula, Solver


def test_asinh_zero():
    assert Formula("asinh(0)").get() == "0"


def test_asinh_derivative_at_zero():
    # d(asinh x)/dx = 1/sqrt(1 + x^2); at x=0 → 1.
    assert Solver("asinh(x)").get_derivative("x", {"x": "0"}) == "1"

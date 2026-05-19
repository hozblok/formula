"""Tests for tanh() (hyperbolic tangent)."""

from formula import Formula, Solver


def test_tanh_zero():
    assert Formula("tanh(0)").get() == "0"


def test_tanh_one_inverse():
    # tanh(atanh(0.5)) == 0.5 — round-trip with the inverse.
    assert Formula("tanh(atanh(0.5))").get().startswith("0.5")


def test_tanh_derivative_at_zero():
    # d(tanh x)/dx = 1/cosh(x)^2; at x=0 → 1/1 = 1.
    assert Solver("tanh(x)").get_derivative("x", {"x": "0"}) == "1"

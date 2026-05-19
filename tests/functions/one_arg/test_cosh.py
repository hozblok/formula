"""Tests for cosh() (hyperbolic cosine)."""

from formula import Formula, Solver


def test_cosh_zero():
    assert Formula("cosh(0)").get() == "1"


def test_cosh_one_inverse():
    # cosh(acosh(1)) == 1 — round-trip with the inverse.
    assert Formula("cosh(acosh(1))").get().startswith("1")


def test_cosh_derivative_is_sinh():
    # d(cosh x)/dx = sinh x; at x=0 → sinh(0) = 0.
    assert Solver("cosh(x)").get_derivative("x", {"x": "0"}) == "0"

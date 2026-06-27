"""Tests for sinh() (hyperbolic sine)."""

from formula import Formula, Solver


def test_sinh_zero():
    assert Formula("sinh(0)").get() == "0"


def test_sinh_one_inverse():
    # sinh(asinh(1)) ≈ 1 — round-trip with the inverse (last-digit drift OK).
    result = Formula("sinh(asinh(1))").get()
    assert result.startswith("1") or result.startswith("0.9999")


def test_sinh_derivative_is_cosh():
    # d(sinh x)/dx = cosh x; at x=0 → cosh(0) = 1.
    assert Solver("sinh(x)").get_derivative("x", {"x": "0"}) == "1"


def test_sinh_is_odd():
    # sinh(-x) = -sinh(x).
    assert Formula("sinh(-2)").get() == "-" + Formula("sinh(2)").get()

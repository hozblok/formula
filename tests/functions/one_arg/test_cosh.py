"""Tests for cosh() (hyperbolic cosine)."""

from formula import Formula, Solver
from tests.derivatives import first_derivative


def test_cosh_zero():
    assert Formula("cosh(0)").get() == "1"


def test_cosh_one_inverse():
    # cosh(acosh(1)) == 1 — round-trip with the inverse.
    assert Formula("cosh(acosh(1))").get().startswith("1")


def test_cosh_derivative_is_sinh():
    # d(cosh x)/dx = sinh x; at x=0 → sinh(0) = 0.
    assert first_derivative(Solver("cosh(x)"), "x", {"x": "0"}) == "0"


def test_cosh_is_even():
    # cosh(-x) = cosh(x).
    assert Formula("cosh(-2)").get() == Formula("cosh(2)").get()

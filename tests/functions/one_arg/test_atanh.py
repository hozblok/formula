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

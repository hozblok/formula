"""Tests for acosh() (inverse hyperbolic cosine)."""

import pytest

from formula import Formula, Solver


def test_acosh_one_is_zero():
    assert Formula("acosh(1)").get() == "0"


def test_acosh_derivative_at_one_diverges():
    # d(acosh x)/dx = 1/sqrt(x^2 - 1); diverges at x=1.
    with pytest.raises(ValueError, match="acosh"):
        Solver("acosh(x)").get_derivative("x", {"x": "1"})

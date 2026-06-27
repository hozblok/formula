"""Tests for asinh() (inverse hyperbolic sine)."""

import pytest

from formula import Formula, Solver


def test_asinh_zero():
    assert Formula("asinh(0)").get() == "0"


def test_asinh_derivative_at_zero():
    # d(asinh x)/dx = 1/sqrt(1 + x^2); at x=0 → 1.
    assert Solver("asinh(x)").get_derivative("x", {"x": "0"}) == "1"


def test_asinh_derivative_at_imaginary_branch_point_diverges():
    # asinh' = 1/sqrt(1 + z^2) is singular at the complex branch points z = +-i.
    with pytest.raises(ValueError, match="asinh"):
        Solver("asinh(i*x)").get_derivative("x", {"x": "1"})

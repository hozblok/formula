"""Tests for asinh() (inverse hyperbolic sine)."""

import pytest

from formula import Formula, Solver
from tests.derivatives import first_derivative


def test_asinh_zero():
    assert Formula("asinh(0)").get() == "0"


def test_asinh_derivative_at_zero():
    # d(asinh x)/dx = 1/sqrt(1 + x^2); at x=0 → 1.
    assert first_derivative(Solver("asinh(x)"), "x", {"x": "0"}) == "1"


def test_asinh_derivative_at_imaginary_branch_point_diverges():
    # asinh' = 1/sqrt(1 + z^2) is singular at the complex branch points z = +-i.
    with pytest.raises(ValueError, match="asinh"):
        first_derivative(Solver("asinh(i*x)"), "x", {"x": "1"})

"""Tests for log10() (base-10 logarithm)."""

import pytest

from formula import Formula, Solver
from tests.derivatives import first_derivative


def test_log10_thousand():
    assert Formula("log10(1000)").get().startswith("3")


def test_log10_one_is_zero():
    assert Formula("log10(1)").get() == "0"


def test_log10_derivative():
    # d/dx[log_10(x)] = 1 / (x * ln(10)); at x=10 → 1/(10 ln 10) ≈ 0.04342944819
    result = first_derivative(Solver("log10(x)"), "x", {"x": "10"})
    assert result.startswith("0.04342944819")


def test_log10_derivative_at_zero_diverges():
    # log_10'(x) = 1/(x ln 10); diverges at x=0.
    with pytest.raises(ValueError, match="log10"):
        first_derivative(Solver("log10(x)"), "x", {"x": "0"})

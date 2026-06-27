"""Tests for log2() (base-2 logarithm)."""

import pytest

from formula import Formula, Solver


def test_log2_eight():
    assert Formula("log2(8)").get().startswith("3")


def test_log2_one_is_zero():
    assert Formula("log2(1)").get() == "0"


def test_log2_derivative():
    # d/dx[log_2(x)] = 1 / (x * ln(2)); at x=2 → 1/(2 ln 2) ≈ 0.7213475204
    result = Solver("log2(x)").get_derivative("x", {"x": "2"})
    assert result.startswith("0.7213475204")


def test_log2_derivative_at_zero_diverges():
    # log_2'(x) = 1/(x ln 2); diverges at x=0.
    with pytest.raises(ValueError, match="log2"):
        Solver("log2(x)").get_derivative("x", {"x": "0"})

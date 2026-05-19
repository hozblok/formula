"""Tests for log10() (base-10 logarithm)."""

from formula import Formula, Solver


def test_log10_thousand():
    assert Formula("log10(1000)").get().startswith("3")


def test_log10_one_is_zero():
    assert Formula("log10(1)").get() == "0"


def test_log10_derivative():
    # d/dx[log_10(x)] = 1 / (x * ln(10)); at x=10 → 1/(10 ln 10) ≈ 0.04342944819
    result = Solver("log10(x)").get_derivative("x", {"x": "10"})
    assert result.startswith("0.04342944819")

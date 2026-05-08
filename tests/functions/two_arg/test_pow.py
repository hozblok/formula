"""Tests for two-argument exponentiation."""

from formula import Formula, FmtFlags


def test_pow_simple():
    assert Formula("2^3").get() == "8"


def test_pow_precision_256():
    expected = "8." + ("0" * 256)
    got = Formula("2^3", 256).get(digits=256, format=FmtFlags.fixed)
    assert got == expected

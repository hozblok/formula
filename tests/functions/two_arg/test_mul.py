"""Tests for two-argument multiplication."""

from formula import Formula, FmtFlags


def test_mul_simple():
    assert Formula("3*4").get() == "12"


def test_mul_precision_256():
    expected = "12." + ("0" * 256)
    got = Formula("3*4", 256).get(digits=256, format=FmtFlags.fixed)
    assert got == expected

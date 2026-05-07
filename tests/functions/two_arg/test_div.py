"""Tests for two-argument division."""

from formula import Formula, FmtFlags


def test_div_simple():
    assert Formula("10/4").get() == "2.5"


def test_div_precision_256():
    expected = "2.5" + ("0" * 255)
    got = Formula("10/4", 256).get(digits=256, format=FmtFlags.fixed)
    assert got == expected

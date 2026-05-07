"""Tests for two-argument addition."""

from formula import Formula, FmtFlags


def test_add_simple():
    assert Formula("1+2").get() == "3"


def test_add_precision_256():
    expected = "3." + ("0" * 256)
    got = Formula("1+2", 256).get(digits=256, format=FmtFlags.fixed)
    assert got == expected

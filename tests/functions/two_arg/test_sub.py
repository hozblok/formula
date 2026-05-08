"""Tests for two-argument subtraction."""

from formula import Formula, FmtFlags


def test_sub_simple():
    assert Formula("5-2").get() == "3"


def test_sub_precision_256():
    expected = "3." + ("0" * 256)
    got = Formula("5-2", 256).get(digits=256, format=FmtFlags.fixed)
    assert got == expected

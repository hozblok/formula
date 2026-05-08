"""Tests for abs()."""

from formula import Formula, FmtFlags


def test_abs_negative_number():
    assert Formula("abs(-5)").get() == "5"


def test_abs_precision_256():
    expected = "0." + ("3" * 256)
    got = Formula("abs(-1/3)", 256).get(digits=256, format=FmtFlags.fixed)
    assert got == expected

"""Tests for sign()."""

from formula import Formula, FmtFlags


def test_sign_negative_number():
    assert Formula("sign(-5)").get() == "-1"


def test_sign_precision_256():
    expected = "-0." + ("3" * 256)
    got = Formula("sign(-2)*(1/3)", 256).get(digits=256, format=FmtFlags.fixed)
    assert got == expected

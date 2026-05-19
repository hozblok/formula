"""Regression: Number ordering must honor the user-supplied precision.
"""

from formula import Number


def test_number_gt_at_high_precision():
    a = Number("1." + "0" * 250 + "1", precision=256)
    b = Number("1.0", precision=256)
    assert a > b
    assert not (b > a)


def test_number_lt_at_high_precision():
    a = Number("1." + "0" * 250 + "1", precision=256)
    b = Number("1.0", precision=256)
    assert b < a
    assert not (a < b)


def test_number_ge_le_at_high_precision():
    a = Number("1." + "0" * 250 + "1", precision=256)
    b = Number("1.0", precision=256)
    assert a >= b
    assert b <= a
    assert not (b >= a)
    assert not (a <= b)

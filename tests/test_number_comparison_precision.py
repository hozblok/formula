"""Regression: Number ordering must honor the user-supplied precision.

See ai/improvements_2026-05-09.md item #3.

Number.__make_comparison previously built its inner Solver with no params,
defaulting to precision=24. That truncated high-precision inputs at parse
time and produced wrong answers for values that differed only beyond the
24th digit.
"""

from formula import Number


def test_number_gt_at_high_precision():
    a = Number("1." + "0" * 50 + "1", precision=256)
    b = Number("1.0", precision=256)
    assert a > b
    assert not (b > a)


def test_number_lt_at_high_precision():
    a = Number("1." + "0" * 50 + "1", precision=256)
    b = Number("1.0", precision=256)
    assert b < a
    assert not (a < b)


def test_number_ge_le_at_high_precision():
    a = Number("1." + "0" * 50 + "1", precision=256)
    b = Number("1.0", precision=256)
    assert a >= b
    assert b <= a
    assert not (b >= a)
    assert not (a <= b)

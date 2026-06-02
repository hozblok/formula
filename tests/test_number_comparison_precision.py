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


def test_ordering_against_str_int_float_value_compare():
    # _cmp whitelists (Number, str, int, float) and routes through
    # _as_number. Existing tests only compare Number vs Number; if a
    # future edit narrowed the whitelist, str/int/float ordering would
    # silently start raising TypeError.
    assert (Number("1") < "2") is True
    assert (Number("1") > "0.5") is True
    assert (Number("1") <= 1) is True
    assert (Number("1") >= 1.0) is True
    assert (Number("2.5") > 2) is True
    assert (Number("2.5") < 3.0) is True

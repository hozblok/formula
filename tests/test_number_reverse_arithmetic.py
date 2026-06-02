"""Regression: Number supports reverse arithmetic with builtin numerics.

See ai/improvements_2026-05-09.md item #4.

Before the fix, `1 + Number("2")` raised TypeError because Number had no
__radd__ (and likewise for sub/mul/truediv/pow).
"""

import operator

import pytest

from formula import Number


def test_int_plus_number():
    result = 1 + Number("2")
    assert isinstance(result, Number)
    assert result == Number("3")


def test_int_minus_number():
    result = 5 - Number("2")
    assert result == Number("3")


def test_int_times_number():
    result = 2 * Number("3")
    assert result == Number("6")


def test_int_div_number():
    result = 10 / Number("2")
    assert result == Number("5")


def test_int_pow_number():
    result = 2 ** Number("3")
    assert result == Number("8")


def test_str_plus_number():
    result = "1.5" + Number("2.5")
    assert result == Number("4")


def test_float_div_number():
    result = 1.0 / Number("4")
    assert result == Number("0.25")


def test_reverse_arithmetic_with_complex_self_preserves_kind_and_precision():
    # _as_number(value) takes self._precision, then __op__ runs _align,
    # which widens to the complex kind. Existing tests only cover real
    # self at default precision — a regression where _as_number used a
    # hard-coded precision or where __rsub__/__rtruediv__ swapped operands
    # would slip through.
    n = Number("1+i", precision=128)
    r = 2 + n
    assert r._is_complex is True
    assert r._precision == 128
    assert r == Number("3+i", precision=128)
    r2 = "1" - n
    assert r2._is_complex is True
    assert r2._precision == 128
    assert r2 == Number("-i", precision=128)


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]
)
def test_bool_lhs_rejected_through_all_reverse_ops(op):
    # int.__op__(True, Number) returns NotImplemented (Number is not int),
    # so Number.__rop__ runs and _as_number(True) must raise TypeError on
    # bool — not silently coerce True to 1.
    with pytest.raises(TypeError, match="bool"):
        op(True, Number("3"))
    with pytest.raises(TypeError, match="bool"):
        op(False, Number("3"))


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]
)
@pytest.mark.parametrize("lhs", [None, [1], {"a": 1}, object()])
def test_reverse_arithmetic_with_foreign_lhs_across_all_ops(op, lhs):
    # Each foreign type's __op__ returns NotImplemented vs Number, so
    # Python invokes Number.__rop__, which routes through _as_number and
    # must raise TypeError naming the offending type.
    with pytest.raises(TypeError, match=type(lhs).__name__):
        op(lhs, Number("3"))

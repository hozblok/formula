"""Regression: Number rejects mixing different storage precisions.

Arithmetic and ordering require both operands at the same mp precision.
Cross-precision equality still compares via parts() and remains allowed.
"""

import operator

import pytest

from formula import Number


def test_eq_across_precisions_still_allowed():
    a = Number("0.5", precision=24)
    b = Number("0.5", precision=256)
    assert a == b


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]
)
def test_forward_arithmetic_rejects_cross_precision_number(op):
    a = Number("1", precision=24)
    b = Number("2", precision=64)
    with pytest.raises(ValueError, match="precision mismatch"):
        op(a, b)


@pytest.mark.parametrize("op_symbol", ["<", "<=", ">", ">="])
def test_ordering_rejects_cross_precision_number(op_symbol):
    a = Number("1", precision=24)
    b = Number("2", precision=64)
    with pytest.raises(ValueError, match="precision mismatch"):
        eval(f"a {op_symbol} b", {"a": a, "b": b})


def test_constructor_rejects_number_at_different_precision():
    inner = Number("1/3", precision=24)
    with pytest.raises(ValueError, match="precision mismatch"):
        Number(inner, precision=64)


def test_same_precision_arithmetic_still_works():
    a = Number("1", precision=64)
    b = Number("2", precision=64)
    assert a + b == Number("3", precision=64)


# The contract forbids mixing *precision*, not real/complex *kind*: two operands
# of different kind at the same precision must combine (real promoted to complex).
@pytest.mark.parametrize("order", ["real_first", "complex_first"])
def test_add_real_and_complex_at_same_precision(order):
    real, imag = Number("2", precision=32), Number("3*i", precision=32)
    result = real + imag if order == "real_first" else imag + real
    assert result.is_complex
    assert result.precision == 32
    assert result == Number("2+3*i", precision=32)


@pytest.mark.parametrize("order", ["real_first", "complex_first"])
def test_multiply_real_and_complex_at_same_precision(order):
    real, imag = Number("2", precision=32), Number("3*i", precision=32)
    result = real * imag if order == "real_first" else imag * real
    assert result.is_complex
    assert result.precision == 32
    assert result == Number("6*i", precision=32)


# A bare str/int/float operand has no precision of its own; it adopts the
# Number's, so this is not a cross-precision mix and must not raise.
def test_scalar_operand_adopts_number_precision():
    n = Number("1", precision=64)
    assert (n + 5).precision == 64
    assert n + 5 == Number("6", precision=64)
    assert (5 + n).precision == 64  # reverse op too


# Re-wrapping a Number keeps its exact value at its own precision -- no string
# round-trip that would truncate digits below the display precision.
def test_rewrap_preserves_precision_and_exact_value():
    n = Number("1/3", precision=256)
    same = Number(n)
    assert same.precision == 256
    assert same.parts == n.parts
    assert same == n

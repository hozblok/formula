"""Python floats retain their binary value in native multiplication."""

from decimal import Decimal

import pytest

from formula import Solver
from formula.backend import MP_TYPES, mp_class


@pytest.mark.parametrize("cls", MP_TYPES)
@pytest.mark.parametrize("scalar", [0.0, -0.0, 0.5, -2.0, 2.5])
def test_float_multiply(cls, scalar):
    value = cls("1.25")
    expected = cls(str(1.25 * scalar))
    for result in (value * scalar, scalar * value):
        assert type(result) is cls
        assert result == expected
    assert 0.5 * (value + value) == value


@pytest.mark.parametrize("is_complex", [False, True])
def test_float_multiply_preserves_binary_value(is_complex):
    cls = mp_class(64, is_complex)
    one = cls("1")
    expected = cls(str(Decimal.from_float(0.1)))
    assert one * 0.1 == expected
    assert 0.1 * one == expected
    assert expected != cls("0.1")


@pytest.mark.parametrize("expression", ["1/3", "1+2*i"])
def test_solver_result_float_multiply(expression):
    value = Solver(expression, precision=64).evaluate()
    assert value * 2.0 == value + value
    assert 2.0 * value == value + value
    assert 0.5 * (value * 2.0) == value

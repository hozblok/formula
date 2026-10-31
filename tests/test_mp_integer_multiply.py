"""Native integer multiplication keeps working across arithmetic results."""

import pytest

from formula import Solver
from formula.backend import MP_TYPES


@pytest.mark.parametrize("cls", MP_TYPES)
@pytest.mark.parametrize("scalar", [0, 2, -3, -(2**31), 2**31 - 1, 2**31,
                                   -(2**63), 2**63 - 1, 2**63,
                                   -(2**63) - 1, 10**40 + 1])
def test_integer_multiply(cls, scalar):
    value = cls("1.25")
    expected = value * cls(str(scalar))
    for result in (value * scalar, scalar * value):
        assert type(result) is cls
        assert result == expected
    assert 2 * (value + value) == cls("5")


@pytest.mark.parametrize("expression", ["1/3", "1+2*i"])
def test_solver_result_integer_multiply(expression):
    value = Solver(expression, precision=64).evaluate()
    expected = value + value
    assert 2 * value == expected
    assert value * 2 == expected
    assert 2 * (value * 2) == expected + expected


@pytest.mark.parametrize("cls", MP_TYPES)
@pytest.mark.parametrize("scalar", [True, False, "2"])
def test_integer_multiply_rejects_other_scalars(cls, scalar):
    value = cls("1")
    with pytest.raises(TypeError):
        value * scalar
    with pytest.raises(TypeError):
        scalar * value

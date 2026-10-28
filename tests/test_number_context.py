"""Fixed-precision factories remain independent and return backend values."""

from dataclasses import FrozenInstanceError

import pytest

from formula import MAX_PRECISION, Number, NumberContext, mp_class
from formula.backend import MP_TYPES, round_up_precision


@pytest.mark.parametrize("precision", [1, 24, 64, 100, MAX_PRECISION])
def test_storage_precision_is_rounded_once(precision):
    N = NumberContext(precision)
    assert N.precision == round_up_precision(precision)
    assert N("0.125").precision == N.precision


@pytest.mark.parametrize("value", [0, -7, 0.125, "1e-80", "1/3", "3+4*i"])
def test_factory_matches_the_number_constructor(value):
    N = NumberContext(64)
    actual = N(value)
    expected = Number(value, 64)
    assert isinstance(actual, MP_TYPES)
    assert actual == expected._value
    assert actual.is_complex == expected.is_complex


def test_contexts_do_not_change_each_other_or_the_default():
    low = NumberContext(24)
    x = low("1/7")
    high = NumberContext(256)
    y = high("1/7")
    assert (2 * x + low(1)).precision == low.precision
    assert (2 * y + high(1)).precision == high.precision
    assert NumberContext().precision == Number.DEFAULT_PRECISION
    with pytest.raises(TypeError, match="precision mismatch"):
        _ = x + y


def test_equal_storage_precisions_are_compatible():
    a = NumberContext(63)
    b = NumberContext(64)
    assert a(1) + b(2) == b(3)
    assert a(b(2)).precision == 64


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("expression", ["1/7", "(1+i)/7"])
def test_wrapping_keeps_the_backend_value(raw, expression):
    N = NumberContext(64)
    original = Number(expression, 64)
    wrapped = N(original._value if raw else original)
    assert wrapped is original._value


@pytest.mark.parametrize("raw", [False, True])
def test_wrapping_rejects_different_storage_precision(raw):
    value = mp_class(256)("0.125") if raw else Number("0.125", 256)
    with pytest.raises(ValueError, match="precision mismatch"):
        NumberContext(64)(value)


def test_constants_follow_the_context():
    for precision in (32, 256):
        N = NumberContext(precision)
        assert N.i.precision == N.pi.precision == precision
        assert N.i.is_complex and not N.pi.is_complex
        assert N.i * N.i == N(-1)
        assert N.pi == Number("pi", precision)._value


def test_context_precision_is_immutable():
    N = NumberContext(64)
    with pytest.raises(FrozenInstanceError):
        N.precision = 256
    with pytest.raises(FrozenInstanceError):
        del N.precision
    assert N(1).precision == 64


@pytest.mark.parametrize("precision", [0, -1, MAX_PRECISION + 1])
def test_invalid_precision_range(precision):
    with pytest.raises(ValueError, match="precision must be"):
        NumberContext(precision)


@pytest.mark.parametrize("precision", [True, False, None, "64", 64.0])
def test_invalid_precision_type(precision):
    with pytest.raises(TypeError, match="precision must be an int"):
        NumberContext(precision)


@pytest.mark.parametrize("value", [True, None, [], {}])
def test_invalid_values_are_rejected(value):
    with pytest.raises(TypeError):
        NumberContext(64)(value)

"""Number API audit: precision=0 rejection, Python-number protocol
(float/complex/bool), real/imag parts, raw-mp equality."""

import pytest

from formula import Number


# --------------------------------------------------------------------------- #
# Number constructor
# --------------------------------------------------------------------------- #

def test_precision_zero_rejected():
    # Used to silently fall back to the default precision.
    with pytest.raises(ValueError, match="precision must be in"):
        Number("2", 0)


def test_precision_negative_rejected():
    with pytest.raises(ValueError, match="precision must be in"):
        Number("2", -5)


# --------------------------------------------------------------------------- #
# Python number protocol
# --------------------------------------------------------------------------- #

def test_float_conversion():
    assert float(Number("1.5", 32)) == 1.5
    assert float(Number("-2", 32)) == -2.0


def test_float_of_complex_raises():
    with pytest.raises(TypeError, match="complex"):
        float(Number("3+4*i", 32))


def test_float_of_complex_with_zero_imag_works():
    assert float(Number("2 + 0*i", 32)) == 2.0


def test_complex_conversion():
    assert complex(Number("3+4*i", 32)) == 3 + 4j
    assert complex(Number("1.5", 32)) == 1.5 + 0j


def test_bool_zero_is_false():
    assert not Number("0", 32)
    assert not Number("0*i", 32)
    assert not Number("-0", 32)


def test_bool_nonzero_is_true():
    assert Number("2", 32)
    assert Number("1e-300", 32)
    assert Number("i", 32)


def test_bool_consistent_with_eq_zero():
    for expr in ("0", "1e-300", "3+4*i", "0*i"):
        n = Number(expr, 32)
        assert bool(n) == (n != Number("0", 32))


# --------------------------------------------------------------------------- #
# real / imag
# --------------------------------------------------------------------------- #

def test_real_imag_of_complex():
    z = Number("3+4*i", 32)
    assert z.real == Number("3", 32)
    assert z.imag == Number("4", 32)
    assert not z.real.is_complex
    assert not z.imag.is_complex
    assert z.real.precision == z.precision


def test_real_imag_of_real():
    x = Number("1/3", 32)
    assert x.real == x
    assert x.imag == Number("0", 32)
    assert x.imag.precision == x.precision


def test_real_imag_roundtrip_exact():
    # Parts extracted with all digits, not truncated to display precision.
    z = Number("(1/3) + (1/7)*i", 32)
    assert z.real == Number("1/3", 32)
    assert z.imag == Number("1/7", 32)


# --------------------------------------------------------------------------- #
# Equality
# --------------------------------------------------------------------------- #

def test_eq_with_raw_mp_value_cross_precision():
    with pytest.raises(ValueError, match="precision mismatch"):
        _ = Number("1", 24) == Number("1", 64)._value
    with pytest.raises(ValueError, match="precision mismatch"):
        _ = Number("2", 24) != Number("1", 64)._value

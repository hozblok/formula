"""Number/Solver API audit: precision=0 rejection, Python-number protocol
(float/complex/bool), real/imag parts, raw-mp equality, Solver.number()."""

import pytest

from formula import Number, Solver


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
    # Used to raise ValueError; now compares by parts like Number-vs-Number.
    assert Number("1", 24) == Number("1", 64)._value
    assert Number("2", 24) != Number("1", 64)._value


# --------------------------------------------------------------------------- #
# Solver.number
# --------------------------------------------------------------------------- #

def test_solver_number_returns_number_at_solver_precision():
    s = Solver("x*acos(0)", precision=48)
    n = s.number(2)
    assert isinstance(n, Number)
    assert n.precision == s.precision
    assert n == Number("pi", s.precision)


def test_solver_number_no_variables():
    assert Solver("2^0.5", precision=32).number() == Number("2^0.5", 32)


def test_solver_number_dict_form():
    s = Solver("x + y", precision=32)
    assert s.number({"x": 1, "y": "2"}) == Number("3", 32)


def test_solver_number_complex_result():
    n = Solver("x + 2*i", precision=32).number(1)
    assert n.is_complex
    assert n == Number("1 + 2*i", 32)


def test_solver_number_missing_values_raises():
    with pytest.raises(ValueError, match="Missing values"):
        Solver("x + y", precision=32).number()


def test_solver_number_malformed_value_raises():
    with pytest.raises(ValueError, match="Invalid number value"):
        Solver("x*2", precision=32).number("1/3")


# --------------------------------------------------------------------------- #
# Solver.__call__ derivative argument
# --------------------------------------------------------------------------- #

def test_derivative_non_iterable_raises_value_error():
    with pytest.raises(ValueError, match="not a string or iterable"):
        Solver("x^2")(2, derivative=123)


def test_derivative_generator_accepted():
    s = Solver("x^2 + y")
    assert s({"x": 2, "y": 1}, derivative=(v for v in ("x", "y"))) == ["4", "1"]


def test_derivative_element_error_not_masked_as_iterable_error():
    # A TypeError from get_derivative itself must surface, not turn into
    # the misleading "not a string or iterable" ValueError.
    with pytest.raises(TypeError) as ex:
        Solver("x^2")(2, derivative=[None])
    assert "not a string or iterable" not in str(ex.value)

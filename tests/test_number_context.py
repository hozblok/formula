"""Fixed-precision factories remain independent and return backend values."""

from dataclasses import FrozenInstanceError

import pytest

from formula import MAX_PRECISION, NumberContext, Solver, _formula
from formula.backend import MP_TYPES, mp_number, round_up_precision
from formula.constants import DEFAULT_PRECISION


def E(context, text):
    """Expression text evaluated at the context's precision."""
    return Solver(text, context.precision).evaluate()


@pytest.mark.parametrize("precision", [1, 24, 64, 100, MAX_PRECISION])
def test_storage_precision_is_rounded_once(precision):
    N = NumberContext(precision)
    assert N.precision == round_up_precision(precision)
    assert N("0.125").precision == N.precision


@pytest.mark.parametrize("value, expression", [
    (0, "0"), (-7, "-7"), (0.125, "0.125"), ("1e-80", "1e-80"), ("(3,4)", "3+4*i"),
])
def test_literals_match_the_evaluated_expression(value, expression):
    N = NumberContext(64)
    actual = N(value)
    expected = E(N, expression)
    assert isinstance(actual, MP_TYPES)
    assert actual == expected
    assert actual.is_complex == expected.is_complex


@pytest.mark.parametrize("text", ["1/3", "3+4*i", "pi", "(1+i)/7", "abc", ""])
def test_expressions_are_not_literals(text):
    # Only literals: anything else is a Solver expression, never a silent parse.
    with pytest.raises(ValueError, match="not a numeric literal"):
        NumberContext(64)(text)


@pytest.mark.parametrize("text, is_complex", [("1.5", False), ("(1.5,0)", True), ("-1e-80", False)])
def test_the_backend_factory_picks_the_kind_from_the_text(text, is_complex):
    value = mp_number(64, text)
    assert type(value) is (_formula.mp_complex_64 if is_complex else _formula.mp_real_64)
    with pytest.raises(ValueError, match="unsupported precision"):
        mp_number(65, text)


def test_contexts_do_not_change_each_other_or_the_default():
    low = NumberContext(24)
    x = E(low, "1/7")
    high = NumberContext(256)
    y = E(high, "1/7")
    assert (2 * x + low(1)).precision == low.precision
    assert (2 * y + high(1)).precision == high.precision
    assert NumberContext().precision == round_up_precision(DEFAULT_PRECISION)
    with pytest.raises(TypeError, match="precision mismatch"):
        _ = x + y


def test_equal_storage_precisions_are_compatible():
    a = NumberContext(63)
    b = NumberContext(64)
    assert a(1) + b(2) == b(3)
    assert a(b(2)).precision == 64


@pytest.mark.parametrize("expression", ["1/7", "(1+i)/7"])
def test_a_backend_value_passes_through_unchanged(expression):
    N = NumberContext(64)
    value = E(N, expression)
    assert N(value) is value


def test_a_backend_value_of_another_storage_precision_is_rejected():
    with pytest.raises(ValueError, match="precision mismatch"):
        NumberContext(64)(NumberContext(256)("0.125"))


@pytest.mark.parametrize("precision", [16, 64, 256])
def test_eval_is_a_cached_solver_at_the_context_precision(precision):
    N = NumberContext(precision)
    assert N.eval("1/3") == E(N, "1/3") and N.eval("1/3").precision == N.precision
    assert N.eval("cos(a)", a="0") == 1 and N.eval("x*x", x=N("1.5")) == N("2.25")
    assert N.eval("asin(s)", s="0.5") == Solver("asin(0.5)", precision).evaluate()
    with pytest.raises(ValueError, match="Missing values"):
        N.eval("x+1")


def test_constants_follow_the_context():
    for precision in (32, 256):
        N = NumberContext(precision)
        assert N.i.precision == N.pi.precision == precision
        assert N.i.is_complex and not N.pi.is_complex
        assert N.i * N.i == -1
        assert N.pi == Solver("pi", precision).evaluate()


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


@pytest.mark.parametrize("precision", [16, 64, 256])
@pytest.mark.parametrize("expression", ["1/3", "-1e-80", "1+2*i", "1/3-2*i", "1e-100000000*i",
                                        "1+0*i", "1e-100000000+0*i"])
def test_str_of_a_value_round_trips_through_the_factory(precision, expression):
    # str() is the backend text, all stored digits: a real literal or "(re,im)".
    N = NumberContext(precision)
    value = E(N, expression)
    again = N(str(value))
    assert again == value and type(again) is type(value)
    assert N("(1,2)") == E(N, "1+2*i") and N("(1.5,-2e-3)") == E(N, "1.5-2e-3*i")

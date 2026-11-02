"""Solver API audit: Solver.number() returns backend values and keeps every stored
digit of its variables; the derivative argument and partial derivatives."""

import pytest

from formula import FmtFlags, Formula, NumberContext, Solver
from formula.backend import MP_TYPES


# --------------------------------------------------------------------------- #
# Solver.number
# --------------------------------------------------------------------------- #

def test_solver_number_returns_a_backend_value_at_solver_precision():
    s = Solver("x*acos(0)", precision=48)
    n = s.number(2)
    assert isinstance(n, MP_TYPES)
    assert n.precision == s.precision
    p = s.precision
    # equal at the display precision
    assert n.str(p, FmtFlags.default) == NumberContext(p).pi.str(p, FmtFlags.default)


def test_solver_number_no_variables():
    assert Solver("2^0.5", precision=32).number() == NumberContext(32).eval("2^0.5")


def test_solver_number_dict_form():
    s = Solver("x + y", precision=32)
    assert s.number({"x": 1, "y": "2"}) == 3


@pytest.mark.parametrize("precision", [32, 64, 256])
@pytest.mark.parametrize("mapping", [False, True])
def test_solver_preserves_real_number_guard_digits(precision, mapping):
    x = NumberContext(precision)("1." + "0" * precision + "1")
    values = {"x": x} if mapping else x
    expected = x - 1
    text = expected.str(precision, FmtFlags.default)
    assert x.str(precision, FmtFlags.default) == "1"
    assert expected != 0
    s = Solver("x-1", precision)
    assert s.number(values) == expected
    assert s(values) == text
    assert s.pair(values) == (text, "0")
    assert Solver("x^2/2-x", precision)(values, derivative="x") == text


def test_solver_identity_preserves_all_stored_digits():
    x = Solver("1/7", 64).evaluate()
    assert Solver("x", 64).number(x) == x


def test_solver_number_complex_result():
    n = Solver("x + 2*i", precision=32).number(1)
    assert n.is_complex
    assert n == NumberContext(32)("(1,2)")


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
    values = {"x": "2", "y": "1"}
    assert [s.get_partial_derivative([v], values, s.precision) for v in "xy"] == ["4", "1"]


def test_derivative_element_error_not_masked_as_iterable_error():
    # A TypeError from get_derivative itself must surface, not turn into
    # the misleading "not a string or iterable" ValueError.
    with pytest.raises(TypeError) as ex:
        Solver("x^2")(2, derivative=[None])
    assert "not a string or iterable" not in str(ex.value)
    with pytest.raises(TypeError):
        Solver("x^2").get_partial_derivative([None], {"x": "2"})


# get_partial_derivative next to get_derivative and the Solver call.
FORMATS = [FmtFlags.default, FmtFlags.fixed, FmtFlags.scientific, FmtFlags.showpos]


@pytest.mark.parametrize("expr", ["x^3*sin(x)", "x^3*sin(x)+i*x^2"])
def test_first_partial_derivative_equals_get_derivative_in_every_format(expr):
    f, at = Formula(expr, 32), {"x": "0.7"}
    for digits in (0, 7, 25):
        for fmt in FORMATS:
            assert f.get_partial_derivative(["x"], at, digits, fmt) == \
                f.get_derivative("x", at, digits, fmt)


@pytest.mark.parametrize("expr", ["x^4*y", "x^4*y+i*x"])
def test_solver_partial_derivative_equals_formula(expr):
    s, f, at = Solver(expr, precision=32), Formula(expr, 32), {"x": "2", "y": "3"}
    for names in (["x"], ["x", "y"], ["x", "x", "x", "y"], ["x"] * 5):
        assert s.get_partial_derivative(names, at) == f.get_partial_derivative(names, at)


def test_solver_derivative_list_keeps_separate_first_derivatives():
    s = Solver("x^4")
    assert s(2, derivative=["x", "x"]) == ["32", "32"]
    assert s.get_partial_derivative(["x", "x"], {"x": "2"}) == "48"


@pytest.mark.parametrize("unit", ["i", "j"])
def test_precision_round_up_and_custom_unit_on_higher_orders(unit):
    expr = f"{unit}*x^5+x^3"
    rounded, exact = Formula(expr, 30, unit), Formula(expr, 32, unit)
    assert rounded.precision == 32
    for k in range(2, 6):
        assert rounded.get_partial_derivative(["x"] * k, {"x": "1.5"}) == \
            exact.get_partial_derivative(["x"] * k, {"x": "1.5"})


def test_typed_variables_must_match_the_solver_precision():
    # A backend value of another storage precision is refused; text is the explicit way.
    s = Solver("x+1", 64)
    value = NumberContext(128)(2)
    with pytest.raises(ValueError, match="precision mismatch"):
        s.number(value)
    assert s.number(str(value)) == 3
    assert s.number(NumberContext(64)(2)) == 3 and s.number(NumberContext(63)(2)) == 3


def test_complex_values_need_a_complex_formula():
    # The author's contract of 2026-11-02: the evaluator kind follows the expression
    # text; a real formula refuses a complex value with a hint instead of promoting.
    N = NumberContext(64)
    with pytest.raises(TypeError, match="imaginary unit"):
        Solver("z+1", 64).number({"z": N.i})
    s = Solver("(z+1)+0*i", 64)
    assert str(s.number({"z": N.i})) == "(1,1)"
    assert str(s.number({"z": N(2)})) == "(3,0)"
    s.expression = "(z+2)+0*i"
    assert str(s.number({"z": N.i})) == "(2,1)"
    assert str(s.number({"z": N("(1,2)")})) == "(3,2)"
    assert str(Solver("z*z", 64).number({"z": N(2)})) == "4"

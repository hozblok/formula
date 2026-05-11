"""Regression: Number arithmetic accumulates symbolically; ops do not lose
precision through formatted-string round-trips.

See ai/improvements_2026-05-09.md items #9 and #10.

Before, every arithmetic op evaluated the expression eagerly and stored the
formatted result string as the new Number's expression. Each subsequent op
re-parsed that rounded string. This test pins the new behavior: arithmetic
builds a symbolic expression that is evaluated only when str() / __eq__ is
called.
"""

from formula import Number


def test_expression_is_symbolic_after_addition():
    n = Number("1.1") + "1.1"
    # The combined symbolic form is kept; we do not see the eager "2.2" eval.
    assert "1.1" in n.expression
    assert "+" in n.expression


def test_str_still_evaluates_the_symbolic_expression():
    # The user-facing str() output remains the *value*, not the symbolic form.
    assert str(Number("1.1") + "1.1") == "2.2"
    assert str(Number("1.1") - "1.2") == "-0.1"
    assert str(Number("3") * "4") == "12"


def test_chained_arithmetic_evaluates_correctly():
    # Each step builds on the previous symbolic expression; final eval reads
    # the whole tree at the configured precision.
    result = (Number("1.1") * Number("2") + Number("0.8")) ** 2
    assert result == "9"


def test_abs_result_evaluates_correctly():
    # __abs__ stays eager (see comment in formula.py); the user-facing
    # str() output is still the canonical complex magnitude.
    n = abs(Number("4-3*i"))
    assert str(n) == "5+i*(0)"

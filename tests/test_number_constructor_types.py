"""Regression: Number.__init__ accepts str/int/float and evaluates eagerly.

See ai/improvements_2026-05-09.md item #4 and CLAUDE.md
("Prefer obvious failures over silent acceptance of wrong-shaped input").

Design (post owner review on PR #18):

  - Number represents a *concrete numeric value*, not a symbolic expression.
  - Constructor accepts str/int/float only. Number-as-input is rejected
    (use the inner Number's expression/value directly if you need that).
  - bool is rejected explicitly: it's an int subclass, and Number(True)
    silently becoming Number("True") is exactly the silent-failure
    pattern CLAUDE.md tells us to refuse.
  - The expression is evaluated by Solver at construction time and the
    result cached as self._value. Expressions with unbound variables
    cannot represent a concrete value and are rejected with a ValueError
    whose message names the offending input.
"""

import pytest

from formula import Number


def test_int_expression():
    n = Number(5)
    assert n.expression == "5"
    assert str(n) == "5"


def test_negative_int_expression():
    n = Number(-7)
    assert n.expression == "-7"
    assert str(n) == "-7"


def test_float_expression():
    n = Number(3.14)
    assert n.expression == "3.14"
    # str(n) goes through Solver evaluation; 3.14 evaluates to 3.14 at
    # default precision.
    assert str(n).startswith("3.14")


def test_str_expression_still_works():
    n = Number("1 + 2")
    assert n.expression == "1 + 2"


def test_bool_rejected_explicitly():
    # bool is an int subclass; Number(True) used to become Number("True")
    # which then crashed downstream. Now it's rejected at the boundary.
    with pytest.raises(TypeError, match="bool"):
        Number(True)
    with pytest.raises(TypeError, match="bool"):
        Number(False)


def test_none_rejected():
    with pytest.raises(TypeError, match="NoneType"):
        Number(None)


def test_list_rejected():
    with pytest.raises(TypeError, match="list"):
        Number([1, 2, 3])


def test_dict_rejected():
    with pytest.raises(TypeError, match="dict"):
        Number({"x": 1})


def test_number_as_input_rejected():
    # Per the post-review design, Number wraps primitives only. Wrapping a
    # Number is no longer supported — read the inner one's .expression or
    # ._value if you really need to round-trip.
    inner = Number("1 + 2")
    with pytest.raises(TypeError, match="Number"):
        Number(inner)


def test_error_message_names_offending_type_concretely():
    with pytest.raises(TypeError, match="str, int, or float"):
        Number(object())


def test_unbound_variable_rejected():
    # A Number must evaluate to a concrete numeric value at construction.
    # Expressions with unbound variables can't, so they're rejected with a
    # ValueError that names the input and the constraint.
    with pytest.raises(ValueError) as excinfo:
        Number("x + 1")
    msg = str(excinfo.value)
    assert "Number" in msg
    assert "x + 1" in msg


def test_garbage_expression_rejected():
    # Same path: anything Solver can't parse as a concrete value bubbles up
    # as a ValueError naming the offending input.
    with pytest.raises(ValueError) as excinfo:
        Number("foo bar")
    msg = str(excinfo.value)
    assert "Number" in msg
    assert "foo bar" in msg


def test_unbound_variable_error_preserves_chain():
    # The from-clause keeps the Solver-level ValueError reachable via
    # __cause__ for debuggers and post-mortem inspection.
    with pytest.raises(ValueError) as excinfo:
        Number("x + 1")
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_coerce_expression_callable_directly_for_int():
    # _coerce_expression is the documented (single-underscore "protected")
    # entry point for the type-dispatch — subclasses can override it, and
    # callers in advanced cases can reach it directly without going through
    # __init__.
    assert Number._coerce_expression(5) == "5"


def test_coerce_expression_callable_directly_for_str():
    assert Number._coerce_expression("1 + 2") == "1 + 2"


def test_coerce_expression_rejects_bool_at_method_level():
    # The same rejection applies whether the dispatch is reached via the
    # constructor or called directly.
    with pytest.raises(TypeError, match="bool"):
        Number._coerce_expression(True)


def test_coerce_expression_rejects_number_at_method_level():
    inner = Number("1 + 2")
    with pytest.raises(TypeError, match="Number"):
        Number._coerce_expression(inner)

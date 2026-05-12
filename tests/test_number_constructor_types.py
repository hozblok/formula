"""Regression: Number.__init__ accepts Number/str/int/float; rejects everything else.

See ai/improvements_2026-05-09.md item #4 and CLAUDE.md
("Prefer obvious failures over silent acceptance of wrong-shaped input").

Previously Number stored whatever `expression` was passed verbatim, then
fed it into a Solver later. Bad inputs (None, list, dict, bool — bool
being especially subtle as a Python int subclass) survived construction
and exploded inside the C++ parser with an opaque error. The
constructor now validates at the boundary and converts numeric values
to their string form once, up front.
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


def test_number_expression_unwraps_inner():
    inner = Number("x + 1")
    outer = Number(inner)
    assert outer.expression == "x + 1"
    # And the precision defaults independently — it's not copied from inner
    # unless explicitly passed.
    inner_hi = Number("x + 1", precision=128)
    wrapped = Number(inner_hi, precision=64)
    assert wrapped.expression == "x + 1"
    assert wrapped.params == {"precision": 64}


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


def test_error_message_names_offending_type_concretely():
    with pytest.raises(TypeError, match="Number, str, int, or float"):
        Number(object())


def test_coerce_expression_callable_directly_for_int():
    # _coerce_expression is the documented (single-underscore "protected")
    # entry point for the type-dispatch — subclasses can override it, and
    # callers in advanced cases can reach it directly without going through
    # __init__.
    assert Number._coerce_expression(5) == "5"


def test_coerce_expression_unwraps_inner_number():
    inner = Number("x + 1")
    assert Number._coerce_expression(inner) == "x + 1"


def test_coerce_expression_rejects_bool_at_method_level():
    # The same rejection applies whether the dispatch is reached via the
    # constructor or called directly.
    with pytest.raises(TypeError, match="bool"):
        Number._coerce_expression(True)

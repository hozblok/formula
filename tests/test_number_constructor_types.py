"""Regression: Number.__init__ accepts Number/str/int/float.

  - Constructor accepts Number, str, int, or float. A Number wrapping
    another Number unwraps to the inner expression.
  - bool is rejected explicitly: it's an int subclass, and Number(True)
    silently becoming Number("True") is exactly the silent-failure
    pattern CLAUDE.md tells us to refuse.
  - None, list, dict, and arbitrary objects are rejected with a TypeError
    whose message names the offending type.
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
    assert str(n) == "3.14"


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


def test_error_message_names_offending_type_concretely():
    with pytest.raises(TypeError, match="str, int, or float"):
        Number(object())

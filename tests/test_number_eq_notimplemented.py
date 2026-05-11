"""Regression: Number rich comparisons return NotImplemented for foreign types.

See ai/improvements_2026-05-09.md item #5.

Before the fix, `Number("1") == None` (or against a list, an object, etc.)
raised ValueError from the inner expression parser, because str(other) was
fed into a Solver. Python's data model expects unknown types to be handled
by returning NotImplemented.
"""

import pytest

from formula import Number


def test_eq_against_none_is_false():
    assert (Number("1") == None) is False  # noqa: E711 - we want the actual `==`


def test_ne_against_none_is_true():
    assert (Number("1") != None) is True  # noqa: E711


def test_eq_against_list_is_false():
    assert (Number("1") == [1, 2, 3]) is False


def test_eq_against_object_is_false():
    assert (Number("1") == object()) is False


def test_eq_with_number_still_works():
    assert Number("1") == Number("1")
    assert not (Number("1") == Number("2"))


def test_eq_with_str_int_float_still_works():
    assert Number("1") == "1"
    assert Number("1") == 1
    assert Number("1.5") == 1.5


def test_lt_against_none_raises_typeerror():
    with pytest.raises(TypeError):
        _ = Number("1") < None


def test_gt_against_list_raises_typeerror():
    with pytest.raises(TypeError):
        _ = Number("1") > [1]


def test_le_against_object_raises_typeerror():
    with pytest.raises(TypeError):
        _ = Number("1") <= object()

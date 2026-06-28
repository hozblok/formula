"""Regression: derivative-arg type validation is up front, not error-swallowing.

See ai/improvements_2026-05-09.md item #21.

Old code wrapped the entire `[self.get_derivative(d) for d in derivative]`
comprehension in `except TypeError` and reported any failure as "derivative
is not iterable". A TypeError raised inside get_derivative for unrelated
reasons (bad format_flags type, binding bug, etc.) would be re-labeled as
an iteration problem.

Now the iter-check happens once up front; downstream TypeErrors propagate
with their real source.
"""

import pytest

from formula import Solver


def test_invalid_derivative_type_int():
    solver = Solver("x*y")
    with pytest.raises(ValueError, match="must be a str or iterable"):
        solver(values={"x": "1", "y": "2"}, derivative=42)


def test_invalid_derivative_type_object():
    solver = Solver("x*y")
    with pytest.raises(ValueError, match="must be a str or iterable"):
        solver(values={"x": "1", "y": "2"}, derivative=object())


def test_derivative_error_message_names_actual_type():
    solver = Solver("x*y")
    with pytest.raises(ValueError, match="int"):
        solver(values={"x": "1", "y": "2"}, derivative=123)


def test_derivative_str_still_works():
    solver = Solver("x*y")
    result = solver(values={"x": "1", "y": "2"}, derivative="x")
    # d(x*y)/dx = y = 2
    assert result == "2"


def test_derivative_list_still_works():
    solver = Solver("x*y")
    result = solver(values={"x": "1", "y": "2"}, derivative=["x", "y"])
    assert isinstance(result, list)
    assert len(result) == 2

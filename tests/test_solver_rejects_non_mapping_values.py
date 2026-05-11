"""Regression: Solver.__call__ rejects non-Mapping `values` with a clear TypeError.

See ai/improvements_2026-05-09.md item #2 and CLAUDE.md ("Prefer obvious
failures over silent acceptance of wrong-shaped input").

The wrapper used to (a) silently iterate over a scalar as if it were a
dict (raising `TypeError: 'int' object is not iterable` from deep inside
the loop), and (b) coerce a single scalar into a `{only_var: str(scalar)}`
shortcut when the formula had exactly one variable. Both paths obscured
the user's actual mistake. Now the wrapper raises a clean `TypeError`
at the boundary naming the parameter and the offending type.

`values=None` continues to mean "no bindings" — empty dict passes through
to the C++ side as before.
"""

import pytest

from formula import Solver


def test_rejects_scalar_int_no_variables():
    solver = Solver("2 + 2")
    with pytest.raises(TypeError, match="must be a Mapping"):
        solver(5)


def test_rejects_scalar_int_one_variable():
    """The previous 1-variable scalar-coercion shortcut is gone too."""
    solver = Solver("2 * x")
    with pytest.raises(TypeError, match="must be a Mapping"):
        solver(5)


def test_rejects_scalar_str():
    solver = Solver("2 * x")
    with pytest.raises(TypeError, match="must be a Mapping"):
        solver("ignored")


def test_rejects_scalar_float():
    solver = Solver("2 * x")
    with pytest.raises(TypeError, match="must be a Mapping"):
        solver(3.14)


def test_rejects_list():
    solver = Solver("x + y")
    with pytest.raises(TypeError, match="must be a Mapping"):
        solver([1, 2])


def test_error_message_names_offending_type():
    solver = Solver("x")
    with pytest.raises(TypeError, match="int"):
        solver(7)


def test_none_still_means_no_bindings():
    solver = Solver("2 + 2")
    assert solver() == "4"
    assert solver(None) == "4"


def test_dict_values_still_work():
    solver = Solver("x + y")
    assert solver({"x": "1", "y": "2"}) == "3"

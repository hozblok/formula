"""Regression: Solver.__call__ must not mutate the caller's values dict.

See ai/improvements_2026-05-09.md item #1.
"""

from formula import Solver


def test_solver_does_not_mutate_caller_dict_int_values():
    solver = Solver("x + 1")
    values = {"x": 1}
    snapshot = dict(values)
    solver(values)
    assert values == snapshot


def test_solver_does_not_mutate_caller_dict_float_values():
    solver = Solver("a + b")
    values = {"a": 1.5, "b": 2.5}
    snapshot = dict(values)
    solver(values)
    assert values == snapshot


def test_solver_does_not_mutate_caller_dict_string_values_passthrough():
    solver = Solver("a + b")
    values = {"a": "1", "b": "2"}
    snapshot = dict(values)
    result = solver(values)
    assert values == snapshot
    assert result == "3"

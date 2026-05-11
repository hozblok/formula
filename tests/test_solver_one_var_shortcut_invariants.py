"""Regression: the one-variable scalar shortcut must not mutate variables().

See ai/improvements_2026-05-09.md item #8.

The old code popped from the set returned by self.variables(). That set is
currently constructed fresh per call, but the mutation was a latent bug
waiting on any binding optimization that caches or returns a reference.
This test pins the invariant: variables() returns the same set after the
shortcut path.
"""

from formula import Solver


def test_one_var_shortcut_leaves_variables_intact():
    solver = Solver("x * 2")
    before = solver.variables()
    assert before == {"x"}
    assert solver(5) == "10"
    after = solver.variables()
    assert after == {"x"}
    assert before == after


def test_one_var_shortcut_repeatable():
    solver = Solver("x * 2")
    assert solver(3) == "6"
    assert solver(5) == "10"
    assert solver(7) == "14"
    assert solver.variables() == {"x"}


def test_one_var_shortcut_with_dict_form_also_repeatable():
    solver = Solver("x * 2")
    assert solver({"x": "3"}) == "6"
    assert solver(5) == "10"
    assert solver({"x": "7"}) == "14"
    assert solver.variables() == {"x"}

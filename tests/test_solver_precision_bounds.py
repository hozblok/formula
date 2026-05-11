"""Regression: Solver enforces the documented precision bounds.

See ai/improvements_2026-05-09.md item #19.

`MAX_PRECISION = 8192` is exported from the package but used to be
purely decorative — the wrapper forwarded any precision value straight
to the C++ extension. Now Solver.__init__ rejects values outside
[0, MAX_PRECISION] with a clear ValueError.
"""

import pytest

from formula import MAX_PRECISION, Solver


def test_solver_accepts_max_precision():
    # Boundary: exactly MAX_PRECISION is allowed.
    solver = Solver("2 + 2", precision=MAX_PRECISION)
    assert solver() == "4"


def test_solver_accepts_zero_precision():
    # Lower bound: 0 is allowed (used as the "all digits" sentinel).
    solver = Solver("2 + 2", precision=0)
    assert solver() == "4"


def test_solver_rejects_negative_precision():
    with pytest.raises(ValueError, match="precision must be in"):
        Solver("x", precision=-1)


def test_solver_rejects_precision_above_max():
    with pytest.raises(ValueError, match="precision must be in"):
        Solver("x", precision=MAX_PRECISION + 1)


def test_max_precision_constant_value():
    # If this ever changes, the error message and tests above need to follow.
    assert MAX_PRECISION == 8192

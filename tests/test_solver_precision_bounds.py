"""Regression: Solver enforces the documented precision bounds.

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
    # Capped by M_PI_STR's 8198 digits: a higher rung would lie about pi.
    assert MAX_PRECISION == 8192


def test_pi_has_full_precision_at_max():
    # Regression: with rungs above 8192 the baked-in pi constant ran out
    # of digits and the tail was silently zero-padded.
    pi = Solver("pi", precision=MAX_PRECISION)()
    assert len(pi.replace(".", "")) >= MAX_PRECISION
    assert pi.startswith("3.14159265358979323846")

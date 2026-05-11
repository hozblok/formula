"""Regression: Solver accepts the legacy `format=` kwarg with a deprecation
warning, while `format_flags=` is the preferred spelling going forward.

See ai/improvements_2026-05-09.md item #25.

Background: `Formula.get` (the C++ binding) uses the keyword `format=`,
whereas the Python `Solver.__call__` wrapper introduced `format_flags=`.
Users moving between the two APIs got keyword-argument errors. The
wrapper now accepts both, emitting a DeprecationWarning for `format=`.
"""

import warnings

import pytest

from formula import FmtFlags, Solver


def test_format_flags_preferred_no_warning():
    solver = Solver("2*asin(x)", precision=24)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an exception
        result = solver({"x": "1"}, format_flags=FmtFlags.fixed)
    assert result.startswith("3.14")


def test_format_legacy_kwarg_emits_deprecation_warning():
    solver = Solver("2*asin(x)", precision=24)
    with pytest.warns(DeprecationWarning, match="format_flags"):
        result = solver({"x": "1"}, format=FmtFlags.fixed)
    assert result.startswith("3.14")


def test_format_and_format_flags_format_wins_and_warns():
    # Both supplied: the legacy alias wins (to preserve old caller intent)
    # and the deprecation warning still fires so the caller fixes it.
    solver = Solver("2*asin(x)", precision=24)
    with pytest.warns(DeprecationWarning):
        result = solver(
            {"x": "1"},
            format_flags=FmtFlags.scientific,
            format=FmtFlags.fixed,
        )
    # `format=FmtFlags.fixed` should override -> output begins "3.14..." not "3.14...e+00"
    assert "e" not in result

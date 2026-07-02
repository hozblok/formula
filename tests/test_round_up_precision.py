"""Regression: round_up_precision is the engine's single rounding rule.

The C++ engine picks the smallest supported mp precision >= the request; Python
must call that exact rule (not reimplement it), so precision reported by Number
never drifts from the mp type its value is actually stored in.
"""

import pytest

from formula import Number, _formula
from formula.backend import MAX_PRECISION, round_up_precision


def test_python_uses_the_cpp_binding_directly():
    # Single source of truth: no Python reimplementation may shadow it.
    assert round_up_precision is _formula.round_up_precision


@pytest.mark.parametrize("request_p", list(range(1, 130)) + [200, 500, 1000, 4096])
def test_matches_the_precision_values_are_stored_at(request_p):
    # The rounded value must equal the precision the engine actually stores in.
    assert round_up_precision(request_p) == Number("1", precision=request_p).precision


@pytest.mark.parametrize(
    "request_p, rounded",
    [(1, 16), (15, 16), (16, 16), (17, 24), (24, 24), (25, 32), (MAX_PRECISION, MAX_PRECISION)],
)
def test_rounds_up_to_the_next_supported_precision(request_p, rounded):
    assert round_up_precision(request_p) == rounded


def test_request_above_maximum_returns_zero_sentinel():
    assert round_up_precision(MAX_PRECISION + 1) == 0


def test_monotonic_and_idempotent_on_supported_values():
    values = [round_up_precision(p) for p in range(1, MAX_PRECISION + 1)]
    assert values == sorted(values)  # non-decreasing
    supported = sorted(set(values))
    assert all(round_up_precision(p) == p for p in supported)  # fixed points

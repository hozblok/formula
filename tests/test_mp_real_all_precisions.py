"""Regression: every AllowedPrecisions value has its own mp_real_<P> binding.

Closes TODO `src/cpp/main.cpp:145` (// TODO support all mp_real).

Before: only mp_real_24 was bound. Now: one py::class_<mp_real<P>> per
AllowedPrecisions value, named mp_real_16 ... mp_real_8192.
"""

import pytest

from formula import _formula


ALLOWED_PRECISIONS = (
    16, 24, 32, 48, 64, 96, 128, 192, 256,
    384, 512, 768, 1024, 2048, 3072, 4096, 6144, 8192,
)


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_mp_real_class_exists(p):
    assert hasattr(_formula, f"mp_real_{p}")


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_mp_real_str_roundtrip(p):
    cls = getattr(_formula, f"mp_real_{p}")
    assert cls("3.14").str(10) == "3.14"


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_mp_real_str_default_args(p):
    cls = getattr(_formula, f"mp_real_{p}")
    # digits=0 / fmtflags=0 → default formatting; must not crash and must
    # contain the integer part.
    assert "5" in cls("5").str()

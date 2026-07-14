"""Every AllowedPrecisions value has a usable mp_real_<P> binding.

Closes the old `// TODO support all mp_real` in src/cpp/main.cpp. Each
mp_real_<P> is a thin binding over boost cpp_dec_float<P>: construct from a
string, do arithmetic / comparison, hash, repr, and format via .str().
"""

import pytest

from formula import _formula

# Hardcoded, not read from the extension, so the test catches an accidental
# change to the bound set.
ALLOWED_PRECISIONS = (
    16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
)


def cls(p):
    return getattr(_formula, f"mp_real_{p}")


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_class_exists(p):
    assert hasattr(_formula, f"mp_real_{p}")


def test_bound_set_is_exactly_allowed_precisions():
    bound = {
        int(name[len("mp_real_"):])
        for name in dir(_formula)
        if name.startswith("mp_real_")
    }
    assert bound == set(ALLOWED_PRECISIONS)


def test_unknown_precision_absent():
    assert not hasattr(_formula, "mp_real_20")


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_str_roundtrip(p):
    assert cls(p)("3.14").str(10) == "3.14"


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_str_default_args(p):
    assert cls(p)("5").str() == "5"


def test_higher_precision_retains_more_digits():
    # Only test that proves P actually selects a backend width: a value with
    # 50 fractional digits is truncated at p_16 but kept at p_64.
    long = "1." + "1234567890" * 5
    assert len(cls(64)(long).str(0)) > len(cls(16)(long).str(0))


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_arithmetic(p):
    r = cls(p)
    assert (r("2") + r("3")).str(10) == "5"
    assert (r("2") - r("3")).str(10) == "-1"
    assert (r("2") * r("3")).str(10) == "6"
    assert (r("6") / r("4")).str(10) == "1.5"
    assert (r("2") ** r("10")).str(10) == "1024"
    assert (-r("2")).str(10) == "-2"
    assert abs(r("-2")).str(10) == "2"


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_sqrt(p):
    r = cls(p)
    assert r("9").sqrt().str(10) == "3"
    assert r("-1").sqrt().str(10) == "nan"  # same as the engine's real sqrt


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_comparison_is_by_value(p):
    r = cls(p)
    assert r("1") == r("1.0")  # value equality, not object identity
    assert r("1") != r("2")
    assert r("1") < r("2") <= r("2")
    assert r("2") > r("1") >= r("1")


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_hash_agrees_with_eq(p):
    r = cls(p)
    assert hash(r("1")) == hash(r("1.0"))
    assert len({r("1"), r("1.0"), r("2")}) == 2


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_repr_roundtrips(p):
    assert repr(cls(p)("3.5")) == f"mp_real_{p}('3.5')"

"""Ordinary attribute changes cannot corrupt Number values or cached comparisons."""

import pytest

from formula import Number


@pytest.mark.parametrize("name", ["parts", "cmp_key"])
@pytest.mark.parametrize("cached", [False, True])
def test_comparison_properties_cannot_be_overwritten(name, cached):
    n = Number("7/3", 64)
    if cached:
        getattr(n, name)
    with pytest.raises(AttributeError, match="read-only"):
        setattr(n, name, None)
    assert n == Number("7/3", 64)
    assert 2 < n < 3
    assert hash(n) == hash(Number("7/3", 64))


@pytest.mark.parametrize("name", ["_value", "_precision", "_is_complex", "parts", "cmp_key"])
def test_numeric_state_cannot_be_deleted(name):
    n = Number("7/3", 64)
    getattr(n, name)
    with pytest.raises(AttributeError, match="read-only"):
        delattr(n, name)
    assert n.precision == 64
    assert not n.is_complex
    assert n == Number("7/3", 64)
    assert 2 < n < 3


def test_shared_constant_keeps_display_and_arithmetic_consistent(monkeypatch):
    monkeypatch.setattr(Number, "_CONST_CACHE", {})
    n = Number(1, 64)
    assert Number(1, 64) is n
    with pytest.raises(AttributeError, match="read-only"):
        n.parts = ("999", "0")
    assert str(Number(1, 64)) == "1"
    assert Number(1, 64) + 1 == Number(2, 64)

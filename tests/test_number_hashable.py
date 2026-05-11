"""Regression: Number must be hashable so it works in sets / as dict keys.

See ai/improvements_2026-05-09.md item #6.

Defining __eq__ without __hash__ silently sets __hash__ = None, making
the class unhashable. The hash is built on the canonical fixed-point
string so it agrees with __eq__.
"""

from formula import Number


def test_number_is_hashable():
    assert isinstance(hash(Number("1")), int)


def test_number_in_set_deduplicates_equal_values():
    s = {Number("1"), Number("2"), Number("1")}
    assert len(s) == 2


def test_number_as_dict_key():
    d = {Number("1"): "one", Number("2"): "two"}
    assert d[Number("1")] == "one"
    assert d[Number("2")] == "two"


def test_hash_consistent_with_eq_for_equivalent_forms():
    a = Number("1.0")
    b = Number("1")
    assert a == b
    assert hash(a) == hash(b)

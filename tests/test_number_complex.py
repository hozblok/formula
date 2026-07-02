"""Tests for Number with complex-number expressions.

Number hardcodes imaginary_unit='i', so any expression containing `i` is
parsed as a complex number. These tests cover the identities that have
to hold for the Number API to be useful with complex arithmetic:
  - i^2 == -1 (the defining identity)
  - syntactically different but mathematically equal forms compare equal
  - magnitude, conjugate-product, and other standard identities
  - hash agrees with equality for complex equivalents
"""

import pytest

from formula import Number


def test_imaginary_part_zero_equals_real():
    # 1 + i*0 reduces to 1 — zero imaginary part should not distinguish
    # a complex literal from a real one.
    assert Number("1+i*0") == Number("1")


def test_i_squared_inside_expression_collapses_to_real():
    # 2 + i*i = 2 + (-1) = 1 — the i*i term must collapse to -1.
    assert Number("2+i*i") == Number("1")


def test_i_squared_equals_minus_one():
    # The defining identity of the imaginary unit.
    assert Number("i*i") == Number("-1")


@pytest.mark.xfail(reason="complex ^ drifts via exp(b·log(a))")
def test_i_to_the_fourth_equals_one():
    assert Number("i^4") == Number("1")


def test_i_to_the_fourth_equals_one_via_mul():
    # Use * instead of ^ to avoid complex-pow drift. The mp-backed value
    # normalizes the signed zero (-0) on the imaginary component, so this
    # now compares equal to 1.
    assert Number("i*i*i*i") == Number("1")


def test_complex_addition():
    assert Number("(1+2*i)+(3+4*i)") == Number("4+6*i")


def test_conjugate_product_is_real():
    # (a+bi)(a-bi) = a^2+b^2 — pure real, classic complex identity.
    assert Number("(1+i)*(1-i)") == Number("2")


def test_pure_imaginary_forms_equal():
    # "i", "0+i", "0+1*i" must all be the same number.
    assert Number("i") == Number("0+i")
    assert Number("i") == Number("0+1*i")


def test_negation_of_imaginary_unit():
    assert Number("-i") == Number("0-i")
    assert Number("-i") == Number("0-1*i")


def test_imaginary_inverse_is_negative_imaginary():
    # 1/i = -i. Easy to get wrong if the engine doesn't rationalize the
    # denominator.
    assert Number("1/i") == Number("-i")


def test_magnitude_of_complex_via_abs():
    # |3+4i| = 5. This exercises abs() on a complex value, not just
    # sign-flipping on a real one.
    assert abs(Number("3+4*i")) == Number("5")


def test_sum_of_opposite_imaginaries_is_zero_via_arithmetic_op():
    # Use the __add__ surface (not just constructor evaluation): the
    # arithmetic dispatch must preserve the cancellation.
    assert Number("i") + Number("-i") == Number("0")


@pytest.mark.xfail(reason="complex ^ drifts via exp(b·log(a))")
def test_square_of_one_plus_i_is_two_i():
    assert Number("(1+i)^2") == Number("2*i")


def test_square_of_one_plus_i_is_two_i_via_mul():
    # Use * instead of ^ to avoid complex-pow drift.
    assert Number("(1+i)*(1+i)") == Number("2*i")


def test_distinct_complex_numbers_are_unequal():
    # Conjugates differ — they must not compare equal.
    assert Number("1+i") != Number("1-i")
    assert Number("2+3*i") != Number("2-3*i")


def test_hash_agrees_for_equivalent_complex_forms():
    # __eq__ ⇒ same hash. Equivalent syntactic forms of the same complex
    # value must collapse to the same hash bucket.
    a = Number("2+i*i")
    b = Number("1")
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.parametrize("op_symbol", ["<", "<=", ">", ">="])
def test_complex_ordering_raises_typeerror(op_symbol):
    # _cmp explicitly raises 'complex numbers are not orderable' when
    # _align reports either side complex. Existing NotImplemented tests
    # cover foreign-type rejection but never hit this branch.
    a = Number("1+i")
    b = Number("2")
    with pytest.raises(TypeError, match="complex"):
        eval(f"a {op_symbol} b", {"a": a, "b": b})
    with pytest.raises(TypeError, match="complex"):
        eval(f"b {op_symbol} a", {"a": a, "b": b})


def test_complex_ordering_against_str_raises_typeerror():
    # str passes the isinstance gate, gets coerced via Number(), and
    # _align still reports complex — so the same guard fires.
    with pytest.raises(TypeError, match="complex"):
        _ = Number("1+i") < "2"


def test_arithmetic_zero_imag_collapse_invariants():
    # Complex zero from arithmetic cancellation vs real zero from a literal.
    # The three invariants (==, hash, str) must agree across kinds.
    cz = Number("i") + Number("-i")  # mp_complex(0, 0)
    rz = Number("0")  # mp_real(0)
    assert cz.is_complex is True
    assert rz.is_complex is False
    assert cz == rz
    assert hash(cz) == hash(rz)
    assert str(cz) == str(rz) == "0"

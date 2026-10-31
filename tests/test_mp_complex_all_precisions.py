"""Every AllowedPrecisions value has a usable mp_complex_<P> binding.

Mirrors mp_real_<P>, but over boost cpp_complex<P>: construct from a string
("(re,im)" or "re") or from a (real, imag) pair, do arithmetic / equality,
hash, repr, decompose via .real/.imag, and format via .str(). Complex is
unordered, so there are no <, <=, >, >= operators.
"""

import pytest

from formula import NumberContext, _formula

# Hardcoded, not read from the extension, so the test catches an accidental
# change to the bound set.
ALLOWED_PRECISIONS = (
    16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
)


def cls(p):
    return getattr(_formula, f"mp_complex_{p}")


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_class_exists(p):
    assert hasattr(_formula, f"mp_complex_{p}")


def test_bound_set_is_exactly_allowed_precisions():
    bound = {
        int(name[len("mp_complex_"):])
        for name in dir(_formula)
        if name.startswith("mp_complex_")
    }
    assert bound == set(ALLOWED_PRECISIONS)


def test_unknown_precision_absent():
    assert not hasattr(_formula, "mp_complex_20")


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_str_roundtrip(p):
    assert cls(p)("(3.14,2.5)").str(10) == "(3.14,2.5)"


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_zero_imag_renders_as_real(p):
    assert cls(p)("3.14").str(10) == "3.14"
    assert cls(p)("(3.14,0)").str(10) == "3.14"


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_pair_constructor(p):
    assert cls(p)("1.5", "2.5").str(10) == "(1.5,2.5)"


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_real_imag(p):
    z = cls(p)("1.5", "2.5")
    assert z.real.str(10) == "1.5"
    assert z.imag.str(10) == "2.5"


def test_higher_precision_retains_more_digits():
    long = "1." + "1234567890" * 5
    assert len(cls(64)(long).real.str(0)) > len(cls(16)(long).real.str(0))


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_arithmetic(p):
    c = cls(p)
    assert (c("1", "2") + c("3", "4")).str(10) == "(4,6)"
    assert (c("3", "4") - c("1", "2")).str(10) == "(2,2)"
    assert (-c("2", "3")).str(10) == "(-2,-3)"
    # i * i == -1
    assert (c("0", "1") * c("0", "1")).str(10) == "-1"
    # (1+i)(1-i) == 2
    assert (c("1", "1") * c("1", "-1")).str(10) == "2"
    # |3+4i| == 5
    assert abs(c("3", "4")).str(10) == "5"


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_no_ordering_operators(p):
    a, b = cls(p)("1", "2"), cls(p)("3", "4")
    with pytest.raises(TypeError):
        a < b


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_equality_is_by_value(p):
    c = cls(p)
    assert c("1", "2") == c("1.0", "2.0")  # value equality, not identity
    assert c("1", "2") != c("1", "3")
    assert c("1", "1") != c("1", "-1")  # conjugates differ


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_hash_agrees_with_eq(p):
    c = cls(p)
    assert hash(c("1", "2")) == hash(c("1.0", "2.0"))
    assert len({c("1", "2"), c("1.0", "2.0"), c("1", "3")}) == 2


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_repr_roundtrips(p):
    assert repr(cls(p)("(3.5,1.5)")) == f"mp_complex_{p}('(3.5,1.5)')"
    assert repr(cls(p)("3.5")) == f"mp_complex_{p}('3.5')"


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_precision_and_kind_on_class_and_value(p):
    # Storage precision travels inside the value and survives every operation.
    z = NumberContext(p)("1.5-2*i")
    assert cls(p).precision == z.precision == p
    assert cls(p).is_complex is True and z.is_complex is True
    for w in (z + z, z * z, 2 * z, 0.5 * z, -z):
        assert w.precision == p and w.is_complex is True
    assert abs(z).precision == p and abs(z).is_complex is False


@pytest.mark.parametrize("p", ALLOWED_PRECISIONS)
def test_precision_and_kind_are_read_only(p):
    z = NumberContext(p)("1.5-2*i")
    for name in ("precision", "is_complex"):
        with pytest.raises(AttributeError):
            setattr(z, name, 7)
        with pytest.raises(AttributeError):
            delattr(z, name)
    assert z.precision == p

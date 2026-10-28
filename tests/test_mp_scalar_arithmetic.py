"""Python scalars, real/complex promotion and the Python protocol on the mp bindings.

int and float operands are converted through their decimal text (repr for
floats), so 0.1 means the decimal 0.1 and not the double. Another storage
precision is refused with TypeError; strings and other objects fall back to
Python's NotImplemented handling.
"""

import operator

import numpy as np
import pytest

from formula import NumberContext, mp_class

PRECISIONS = (16, 64, 256)
BINARY = [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]


@pytest.mark.parametrize("p", PRECISIONS)
@pytest.mark.parametrize("op", BINARY)
@pytest.mark.parametrize("scalar", [2, -3, 2.5, 0.1, 1e-6, np.int64(2), np.float64(0.25)])
def test_scalars_act_as_their_decimal_text(p, op, scalar):
    N = NumberContext(p)
    x, z = N("1.5"), N("1.5+2*i")
    s = N(repr(float(scalar)) if isinstance(scalar, (float, np.floating)) else int(scalar))
    for a in (x, z):
        assert op(a, scalar) == op(a, s)
        assert op(a, scalar).precision == p and op(scalar, a).precision == p
        if op is operator.pow and scalar < 0 and a is x:
            continue  # (-3) ** 1.5 is nan, and nan != nan
        assert op(scalar, a) == op(s, a)


@pytest.mark.parametrize("p", PRECISIONS)
def test_float_is_decimal_not_binary(p):
    x = NumberContext(p)("3")
    assert x * 0.1 == x * NumberContext(p)("0.1")
    assert str(x * 0.1) == "0.3"


@pytest.mark.parametrize("op", BINARY + [operator.eq, operator.lt])
def test_bool_is_refused(op):
    x = NumberContext(64)("1.5")
    with pytest.raises(TypeError):
        op(x, True)
    with pytest.raises(TypeError):
        op(True, x)


@pytest.mark.parametrize("p", PRECISIONS)
def test_real_and_complex_promote(p):
    N = NumberContext(p)
    x, z = N("1.5"), N("0.5+2*i")
    C = mp_class(p, is_complex=True)
    for op in BINARY:
        assert op(x, z) == op(C("1.5"), z) and op(x, z).is_complex
        assert op(z, x) == op(z, C("1.5")) and op(z, x).is_complex
    assert N("1.5+0*i") == x and x == N("1.5+0*i")
    assert N("1.5+1*i") != x
    with pytest.raises(TypeError, match="orderable"):
        x < z


@pytest.mark.parametrize("p", PRECISIONS)
def test_ordering_against_scalars(p):
    x = NumberContext(p)("1.5")
    assert x < 2 and 1 < x and x <= 1.5 and x >= 1.5 and x > 1.25 and 3 > x
    assert x == 1.5 and 1.5 == x and x != 1 and sorted([x, 1, 2.0]) == [1, x, 2.0]


@pytest.mark.parametrize("other", ["1.5", None, [1.5], object()])
def test_other_objects_leave_the_decision_to_python(other):
    x = NumberContext(64)("1.5")
    assert not (x == other) and x != other
    with pytest.raises(TypeError):
        x + other
    with pytest.raises(TypeError):
        x < other


@pytest.mark.parametrize("kind", [False, True])
def test_other_precision_is_a_type_error_everywhere(kind):
    a = NumberContext(64)("1.5")
    b = mp_class(128, is_complex=kind)("1.5")
    for op in BINARY + [operator.eq, operator.ne]:
        with pytest.raises(TypeError, match="precision mismatch"):
            op(a, b)
        with pytest.raises(TypeError, match="precision mismatch"):
            op(b, a)


@pytest.mark.parametrize("p", PRECISIONS)
def test_truthiness(p):
    N = NumberContext(p)
    assert not N(0) and not N("0*i") and not N("-0.0")
    assert N("1e-1000") and N("1e-1000*i") and N(-1)


@pytest.mark.parametrize("p", PRECISIONS)
def test_str_prints_the_stored_digits(p):
    N = NumberContext(p)
    assert str(N("1.5")) == f"{N('1.5')}" == "1.5"
    assert str(N("-0")) == "0"
    assert str(N("1.5+2*i")) == "(1.5,2)" and str(N("1.5+0*i")) == "1.5"
    third = N("1/3")
    assert len(str(third)) >= p + 1 and N(str(third)) == third


@pytest.mark.parametrize("p", PRECISIONS)
def test_float_and_complex_conversions(p):
    N = NumberContext(p)
    assert float(N("2.5")) == 2.5 and float(N("2.5+0*i")) == 2.5
    assert complex(N("1.5+2*i")) == 1.5 + 2j and complex(N("1.5")) == 1.5 + 0j
    with pytest.raises(TypeError):
        float(N("1+i"))


@pytest.mark.parametrize("p", PRECISIONS)
def test_real_imag_abs_sqrt(p):
    N = NumberContext(p)
    R = mp_class(p)
    z = N("3-4*i")
    assert z.real == R("3") and z.imag == R("-4") and not z.real.is_complex
    assert z.real.precision == p and abs(z) == R("5") and not abs(z).is_complex
    assert N("1.5").real == N("1.5") and N("1.5").imag == 0
    assert z.sqrt() == z ** N("0.5")
    assert abs(N("-4+0*i").sqrt() - N("2*i")) < 10 ** -(p - 2)
    assert N("2.25").sqrt() == N("1.5")


@pytest.mark.parametrize("p", PRECISIONS)
def test_sum_and_hash(p):
    N = NumberContext(p)
    values = [N("1.5"), N("2.5"), N(1)]
    assert sum(values) == 5 and sum(values) == N(5)
    assert hash(N("1/3")) == hash(N("1/3")) and len({N("1/3"), N("1/3")}) == 1


def test_context_returns_backend_values_and_constants():
    N = NumberContext(64)
    assert type(N("1.5")) is mp_class(64) and type(N("1+i")) is mp_class(64, True)
    assert N.i * N.i == -1 and N.i is N.i and N.pi == N("pi")
    assert N(N("1.5")) is not None and N(mp_class(64)("2")).precision == 64
    with pytest.raises(ValueError, match="precision mismatch"):
        N(mp_class(128)("2"))


@pytest.mark.parametrize("p", PRECISIONS)
def test_real_complex_equality_is_transitive(p):
    # A real with a guard digit beyond p is not equal to the complex 1, which
    # rounds R -> C; equality runs in R instead.
    R, C = mp_class(p), mp_class(p, True)
    a, b, c = R("1." + "0" * p + "1"), R("1"), C("1")
    assert a != b and b == c and c == b
    assert a != c and c != a
    assert C("1", "1e-300") != b and b != C("1", "1e-300")

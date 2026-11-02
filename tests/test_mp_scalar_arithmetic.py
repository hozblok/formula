"""Python scalars, real/complex promotion and the Python protocol on the mp bindings.

In arithmetic int and float operands are converted through their decimal
text (repr for floats), so x * 0.1 means the decimal 0.1 and not the double.
Comparisons are exact and strict: mp values of one type and Python ints
within 2**53; real against complex and any other Python number raise
TypeError (the binary exponent range exceeds the decimal one, so no
conversion compares the kinds faithfully). Another storage
precision is refused with TypeError; strings and other objects fall back to
Python's NotImplemented handling.
"""

import operator

import numpy as np
import pytest

from formula import Solver, NumberContext, _formula

PRECISIONS = (16, 64, 256)
BINARY = [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]


def E(context, text):
    """Expression text evaluated at the context's precision."""
    return Solver(text, context.precision).evaluate()


@pytest.mark.parametrize("p", PRECISIONS)
@pytest.mark.parametrize("op", BINARY)
@pytest.mark.parametrize("scalar", [2, -3, 2.5, 0.1, 1e-6, np.int64(2), np.float64(0.25)])
def test_scalars_act_as_their_decimal_text(p, op, scalar):
    N = NumberContext(p)
    x, z = N("1.5"), N("(1.5,2)")
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
    x, z = N("1.5"), N("(0.5,2)")
    C = type(z)
    for op in BINARY:
        assert op(x, z) == op(C("1.5"), z) and op(x, z).is_complex
        assert op(z, x) == op(z, C("1.5")) and op(z, x).is_complex
    for op in (operator.eq, operator.ne, operator.lt):
        with pytest.raises(TypeError, match="do not compare"):
            op(x, z)
        with pytest.raises(TypeError, match="do not compare"):
            op(z, x)


@pytest.mark.parametrize("p", PRECISIONS)
def test_ordering_against_scalars(p):
    x = NumberContext(p)("1.5")
    assert x < 2 and 1 < x and x <= 2 and x >= 1 and x > 1 and 3 > x
    assert x != 1 and 1 != x and NumberContext(p)(2) == 2 and sorted([x, 1, 2]) == [1, x, 2]


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
    b = NumberContext(128)("(1.5,0)" if kind else "1.5")
    for op in BINARY + [operator.eq, operator.ne]:
        with pytest.raises(TypeError, match="precision mismatch"):
            op(a, b)
        with pytest.raises(TypeError, match="precision mismatch"):
            op(b, a)


@pytest.mark.parametrize("p", PRECISIONS)
def test_truthiness(p):
    N = NumberContext(p)
    assert not N(0) and not N("(0,0)") and not N("-0.0")
    assert N("1e-1000") and N("(0,1e-1000)") and N(-1)


@pytest.mark.parametrize("p", PRECISIONS)
def test_str_prints_the_stored_digits(p):
    N = NumberContext(p)
    assert str(N("1.5")) == f"{N('1.5')}" == "1.5"
    assert str(N("-0")) == "0"
    assert str(N("(1.5,2)")) == "(1.5,2)" and str(N("(1.5,0)")) == "(1.5,0)"
    third = E(N, "1/3")
    assert len(str(third)) >= p + 1 and N(str(third)) == third


@pytest.mark.parametrize("p", PRECISIONS)
def test_float_and_complex_conversions(p):
    N = NumberContext(p)
    assert float(N("2.5")) == 2.5 and float(N("(2.5,0)")) == 2.5
    assert complex(N("(1.5,2)")) == 1.5 + 2j and complex(N("1.5")) == 1.5 + 0j
    with pytest.raises(TypeError):
        float(E(N, "1+i"))


@pytest.mark.parametrize("p", PRECISIONS)
def test_real_imag_abs_sqrt(p):
    N = NumberContext(p)
    z = N("(3,-4)")
    assert z.real == N("3") and z.imag == N("-4") and not z.real.is_complex
    assert z.real.precision == p and abs(z) == N("5") and not abs(z).is_complex
    assert N("1.5").real == N("1.5") and N("1.5").imag == 0
    # The named operation is Solver's sqrt(), not ** 0.5: the same algorithm, the same bits.
    assert z.sqrt() == E(N, "sqrt(3-4*i)") and N("2").sqrt() == E(N, "sqrt(2)")
    assert abs(N("(-4,0)").sqrt() - N("(0,2)")) < N(f"1e-{p - 2}")
    assert N("2.25").sqrt() == N("1.5")
    tiny = N("(1e-400000000,1e-400000000)")
    assert "nan" not in str(tiny.sqrt()) and tiny.sqrt() == E(N, "sqrt(1e-400000000+1e-400000000*i)")


@pytest.mark.parametrize("p", PRECISIONS)
def test_sum_and_hash(p):
    N = NumberContext(p)
    values = [N("1.5"), N("2.5"), N(1)]
    assert sum(values) == 5 and sum(values) == N(5)
    assert hash(E(N, "1/3")) == hash(E(N, "1/3")) and len({E(N, "1/3"), E(N, "1/3")}) == 1


def test_context_returns_backend_values_and_constants():
    N = NumberContext(64)
    assert type(N("1.5")) is _formula.mp_real_64 and type(E(N, "1+i")) is _formula.mp_complex_64
    assert N.i * N.i == -1 and N.i is N.i and N.pi == E(N, "pi")
    assert N(N("1.5")) is not None and N(_formula.mp_real_64("2")).precision == 64
    with pytest.raises(ValueError, match="precision mismatch"):
        N(_formula.mp_real_128("2"))


@pytest.mark.parametrize("p", PRECISIONS)
def test_real_and_complex_never_compare(p):
    # The binary exponent range is wider than the decimal one: 1e-100000000
    # and 2e-100000000 differ, yet neither fits in mp_real, so no conversion
    # compares the kinds faithfully.
    R, C = getattr(_formula, f"mp_real_{p}"), getattr(_formula, f"mp_complex_{p}")
    a, b = C("1e-100000000"), C("2e-100000000")
    assert a != b
    with pytest.raises(ValueError, match="exponent range"):
        a.real
    for v in (a, C("0"), C("1", "1")):
        with pytest.raises(TypeError, match="do not compare"):
            v == R("0")
        with pytest.raises(TypeError, match="do not compare"):
            R("0") != v
    assert C("0").real == R("0") and C("1.5", "0").real == R("1.5")


@pytest.mark.parametrize("p", PRECISIONS)
def test_small_ints_compare_exactly_other_python_numbers_raise(p):
    # An int within 2**53 is exact in every mp type; floats, complex, bool and
    # larger ints need an explicit N(...): TypeError, never a silent False.
    N = NumberContext(p)
    x, z = N("1.5"), N("(1.5,0)")
    assert x != 1 and x > 1 and x < 2 and 1 < x and z != 1 and N(3) == 3 == N("(3,0)")
    assert N(2 ** 53) == 2 ** 53 and N(-(2 ** 53)) == -(2 ** 53) and N(2 ** 53) != 2 ** 53 - 1
    for other in (2 ** 53 + 1, -(2 ** 53) - 1, 1.5, 1 + 0j, True):
        for v in (x, z):
            with pytest.raises(TypeError):
                v == other
            with pytest.raises(TypeError):
                other != v
        with pytest.raises(TypeError):
            x < other
    assert x * 1.5 == N("2.25")  # a float still multiplies


@pytest.mark.parametrize("p", PRECISIONS)
def test_hash_agrees_with_equality(p):
    N = NumberContext(p)
    one, big = N(1), N(2 ** 53)
    assert hash(one) == hash(1) == hash(N("(1,0)")) and hash(big) == hash(2 ** 53)
    assert {one: "found"}.get(1) == "found" and len({one, 1, N("1.0")}) == 1
    assert len({N("(1,0)"), 1, N("(1.0,0)")}) == 1 and hash(N(f"({2 ** 53},0)")) == hash(2 ** 53)
    assert hash(N("-0")) == hash(0) and hash(N(-1)) == hash(-1)
    # Python's own int hash, which is not n itself on a 32-bit build past 2**31 - 1.
    for n in (2 ** 31 - 1, 2 ** 31, 2 ** 31 + 1, -2 ** 31, -2 ** 53):
        assert hash(N(n)) == hash(n) == hash(N(f"({n},0)"))
    third = E(N, "1/3")
    assert hash(third) == hash(third + 0) and len({third, third * 1, E(N, "1/3") + N(0)}) == 1
    assert hash(N("1.5")) == hash(E(N, "3/2")) and hash(N("(1.5,0)")) == hash(E(N, "3/2+0*i"))
    assert hash(N("(1,2)")) == hash(N("(1,2)") * 1) and len({N("(1,2)"), N("(1,2)") + 0}) == 1


def test_only_registered_backend_classes_mismatch_precision():
    # A look-alike name is not a backend value: NotImplemented, never a precision error.
    x = NumberContext(64)("1.5")
    look_alike = type("mp_real_128", (), {})()
    assert (x == look_alike) is False and (x != look_alike) is True
    with pytest.raises(TypeError, match="unsupported operand"):
        x + look_alike
    with pytest.raises(TypeError, match="precision mismatch"):
        x == NumberContext(128)("1.5")


@pytest.mark.parametrize("kind", [False, True])
def test_backend_classes_are_final(kind):
    with pytest.raises(TypeError, match="not an acceptable base type"):
        type("Sub", (_formula.mp_complex_64 if kind else _formula.mp_real_64,), {})


@pytest.mark.xfail(strict=True, reason="the native complex sqrt loses the argument of a tiny "
                   "base (its re^2+im^2 underflows); Solver's sqrt() is the same algorithm, so "
                   "it agrees and is as wrong. A scaled fix existed and was withdrawn "
                   "(2026-11-02): physical arguments never leave exponents -7..0, and "
                   "simplicity wins over an unreachable edge")
def test_sqrt_of_a_tiny_base_scales():
    # sqrt(z) = sqrt(z s^2) / s with s = 1e200000000: the reference is an O(1) sqrt, (1.0987, 0.4551).
    N = NumberContext(64)
    tiny, s = N("(1e-400000000,1e-400000000)"), N("(1e200000000,0)")
    assert abs(tiny.sqrt() * s - N("(1,1)").sqrt()) < N("1e-60")


@pytest.mark.xfail(strict=True, reason="squares of the parts overflow inside Boost's eval_sqrt "
                   "near the exponent ceiling: (nan,nan) for a finite representable input; "
                   "withdrawn with the same fix, an unreachable edge")
def test_sqrt_of_a_huge_negative_base_stays_finite():
    N = NumberContext(64)
    z, s = N("(-2.25e646456864,0)"), N("(1e323228432,0)")
    assert z - z == N("(0,0)")
    root = z.sqrt()
    assert abs(root / s - N("(0,1.5)")) < N("1e-60")


@pytest.mark.parametrize("w, root", [("(-3,4)", "(1,2)"), ("(-3,-4)", "(1,-2)"), ("(0,-2)", "(1,-1)"),
                                     ("(-4,0)", "(0,2)"), ("(9,0)", "(3,0)")])
def test_sqrt_takes_the_principal_branch(w, root):
    N = NumberContext(64)
    assert N(w).sqrt() == N(root)
    assert N.eval(f"sqrt(({w[1:-1].split(',')[0]})+({w[1:-1].split(',')[1]})*i)") == N(root)


def test_float_of_a_complex_rounds_its_own_binary_part():
    # float(z) parses the binary part's 16-digit text; z.real rounds that part once more into a
    # decimal mp_real, so here the two doubles differ by an ulp (the documented contract).
    z = NumberContext(16)("(88395772929158265e-16,0)")
    assert float(z) == float(z.str(16)[1:-1].split(",")[0]) == complex(z).real
    assert (float(z), float(z.real)) == (8.839577292915827, 8.839577292915825)


@pytest.mark.parametrize("p", PRECISIONS)
def test_negative_infinity_keeps_its_sign(p):
    N = NumberContext(p)
    neg_inf = -N(1) / N(0)
    assert str(neg_inf) == "-inf" and float(neg_inf) == float("-inf") and str(N(1) / N(0)) == "inf"
    z = N("(-1,1)") / N("(0,0)")
    assert str(z) == "(-inf,inf)" and complex(z) == complex("-inf+infj") and str(z.real) == "-inf"
    assert str(N("(1,-1)") / N("(0,0)")) == "(inf,-inf)"
    assert str(-N(0)) == "0" and str(N("(-0,-0)")) == "(0,0)"


def test_components_beyond_the_decimal_range_raise():
    # mp_complex holds 1e-100000000, mp_real cannot: an error, never a silent 0 or inf.
    N = NumberContext(64)
    tiny, huge = N("(1e-100000000,0)"), N("(1e100000000,2)")
    assert bool(tiny) and tiny.imag == 0 and huge.imag == 2
    for get in (lambda: tiny.real, lambda: abs(tiny), lambda: huge.real, lambda: abs(huge),
                lambda: N("(0,1e-100000000)").imag):
        with pytest.raises(ValueError, match="exponent range"):
            get()
    assert N("(1e-67000000,0)").real != 0


@pytest.mark.parametrize("p", PRECISIONS)
def test_float_is_the_correctly_rounded_parse_of_the_stored_digits(p):
    # float(x) == float(x.str(precision)); Boost's convert_to<double>
    # is off by an ulp in about one computed value per thousand.
    import random
    N = NumberContext(p)
    rng = random.Random(p)
    for _ in range(3000):
        x = N(repr(rng.uniform(0.5, 2))) / N(repr(rng.uniform(0.5, 2))) * N(f"1e{rng.randint(-30, 30)}")
        assert float(x) == float(x.str(p)) and complex(x) == complex(float(x), 0)
        z = N(f"({x.str(0)},{(-x).str(0)})")
        re_text, im_text = z.str(p)[1:-1].split(",")  # the components' own p-digit text
        assert complex(z) == complex(float(re_text), float(im_text))
    assert float(N(1) / N(0)) == float("inf") and float(-N(1) / N(0)) == float("-inf")
    assert float(N(0) / N(0)) != float(N(0) / N(0))  # nan

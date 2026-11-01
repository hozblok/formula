"""Integer powers of mp_complex take the integral pow (repeated squaring):
exact against repeated products, and the base itself for a tiny base, where
the general exp/log pow returned nan."""

import pytest

from formula import NumberContext, Solver

PRECISIONS = (16, 64, 256)


@pytest.mark.parametrize("p", PRECISIONS)
@pytest.mark.parametrize("exponent", [1, 2, -1])
def test_integer_powers_of_i_equal_repeated_products(p, exponent):
    i = NumberContext(p)("(0,1)")
    expected = {1: i, 2: i * i, -1: 1 / i}[exponent]
    assert i ** exponent == expected


@pytest.mark.parametrize("p", PRECISIONS)
@pytest.mark.parametrize("exponent", [3, -2])
def test_integer_powers_equal_repeated_products(p, exponent):
    w = NumberContext(p)("(1.5,-2)")
    expected = {3: w * w * w, -2: 1 / (w * w)}[exponent]
    assert w ** exponent == expected


def test_first_power_of_a_tiny_base_is_the_base():
    tiny = NumberContext(64)("(1e-400000000,1e-400000000)")
    assert tiny ** 1 == tiny


@pytest.mark.parametrize("p", PRECISIONS)
def test_zeroth_power_is_one(p):
    N = NumberContext(p)
    assert N("(0,1)") ** 0 == 1 and N("(1.5,-2)") ** 0 == 1


@pytest.mark.parametrize("p", PRECISIONS)
def test_integral_backend_exponents_take_the_same_path(p):
    # The rule is the exponent's value, not its Python type.
    N = NumberContext(p)
    w = N("(1.5,-2)")
    assert w ** N("(3,0)") == w ** N("3") == w ** 3 == w * w * w
    assert w ** N("(-2,0)") == w ** -2 == 1 / (w * w)


def test_other_exponents_take_the_general_pow():
    # Beyond 2**62 the exponent stays on the general path; bool is no exponent at all.
    N = NumberContext(64)
    assert N("(1,0)") ** (2 ** 70) == 1
    with pytest.raises(TypeError):
        N("(0,1)") ** True


def test_the_power_rule_is_shared_with_solver():
    # One helper behind R ** C, C ** C and Solver's ^: the same bounds, the same zero base.
    N = NumberContext(64)
    E = lambda text: Solver(text, 64).evaluate()
    assert N(-1) ** N("(2,0)") == N("(-1,0)") ** N("(2,0)") == E("(-1+0*i)^2") == 1
    assert N("(0,1)") ** 2147483648 == E("i^2147483648") == 1
    assert N("(0,0)") ** -1 == E("(0*i)^(-1)") and N("(0,0)") ** 0 == E("(0*i)^0") == 1
    z = N("(0.6,-0.8)")
    for k in (3, -7, 2 ** 40, 2 ** 62, 2 ** 62 + 1, 2 ** 70):
        assert z ** k == E(f"(0.6-0.8*i)^{k}") == z ** N(f"({k},0)")

"""Integer powers of mp_complex go through the general pow (log/exp): a rounding
error against repeated multiplication, and nan for a representable tiny base.
Strict xfails; they turn green once an integral-exponent path exists."""

import pytest

from formula import NumberContext

PRECISIONS = (16, 64, 256)
REASON = "complex ** int goes through log/exp, not an integral pow"


@pytest.mark.xfail(strict=True, reason=REASON)
@pytest.mark.parametrize("p", PRECISIONS)
@pytest.mark.parametrize("exponent", [1, 2, -1])
def test_integer_powers_of_i_equal_repeated_products(p, exponent):
    i = NumberContext(p)("(0,1)")
    expected = {1: i, 2: i * i, -1: 1 / i}[exponent]
    assert i ** exponent == expected


@pytest.mark.xfail(strict=True, reason=REASON)
@pytest.mark.parametrize("p", PRECISIONS)
@pytest.mark.parametrize("exponent", [3, -2])
def test_integer_powers_equal_repeated_products(p, exponent):
    w = NumberContext(p)("(1.5,-2)")
    expected = {3: w * w * w, -2: 1 / (w * w)}[exponent]
    assert w ** exponent == expected


@pytest.mark.xfail(strict=True, reason=REASON + "; log/exp gives nan for a tiny base")
def test_first_power_of_a_tiny_base_is_the_base():
    tiny = NumberContext(64)("(1e-400000000,1e-400000000)")
    assert tiny ** 1 == tiny


@pytest.mark.parametrize("p", PRECISIONS)
def test_zeroth_power_is_already_exact(p):
    N = NumberContext(p)
    assert N("(0,1)") ** 0 == 1 and N("(1.5,-2)") ** 0 == 1

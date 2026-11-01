"""A jet constant carries every stored digit of its value, whichever wrapper brings it."""

from formula import Number, NumberContext
from formula.capsysred.shared.jets import _Algebra


def test_a_constant_keeps_the_guard_digits_of_a_number_and_a_backend_value():
    N = NumberContext(64)
    v = N(1) + N("1e-70")  # 1 + 1e-70 lives in the guard digits: str(Number(v)) is "1"
    alg = _Algebra(1, 64)
    assert alg.evaluate("x-1", x=v).c[0] == alg.evaluate("x-1", x=Number(v)).c[0] == "1e-70"
    assert N(alg.evaluate("1/(x-1)", x=Number(v)).c[0]) == 1 / (v - 1)
    assert alg.constant(v).c[0] == alg.constant(Number(v)).c[0] == str(v)
    assert alg.constant(2).c[0] == "2" and alg.constant(0.5).c[0] == "0.5" and alg.constant("1/3").c[0] == "1/3"


def test_a_constant_of_another_precision_is_refused():
    import pytest
    alg = _Algebra(1, 64)
    for value in (NumberContext(128)(2), Number("2", 128)):
        with pytest.raises(ValueError, match="precision mismatch"):
            alg.constant(value)
        with pytest.raises(ValueError, match="precision mismatch"):
            alg.evaluate("x+1", x=value)
    assert alg.constant(NumberContext(64)(2)).c[0] == "2" and alg.evaluate("x+1", x="2").c[0] == "3"
    assert _Algebra(1, 63).constant(NumberContext(64)(2)).c[0] == "2"  # 63 rounds up to the same storage

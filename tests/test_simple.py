# -*- coding: utf-8 -*-
# <-- support python2.7. It's not necessary in python3+

import pytest

from formula import FmtFlags, Formula


def test_evaluation():
    """Evaluate several simple expressions."""
    assert Formula("2--1").get() == "3"
    assert Formula("2-0-1").get() == "1"
    assert Formula("2-(0-1)").get() == "3"
    assert Formula("(2-0)-1").get() == "1"
    assert Formula("1 + .0e0 + 1").get() == "2"
    assert Formula("1 + .0e0 + 1").get() == "2"
    assert Formula("1 + .0e+0 + 1").get() == "2"
    assert Formula("1 + +.0e0 + +1").get() == "2"
    assert Formula("1 + 0.e+0 + +1").get() == "2"
    assert Formula("1 + +1.e+0").get() == "2"
    assert Formula("1 + +1.e+0 + x").get({"x": "1"}) == "3"
    assert Formula("1 - -1.e-1 - x").get({"x": "1"}) == "0.1"
    assert Formula("1 + -1.e0 + x").get({"x": "1"}) == "1"
    assert Formula("-1 - 0.e-0 - x").get({"x": "1"}) == "-2"
    assert Formula("1 - +0.e-0 - x").get({"x": "1"}) == "0"
    assert Formula("-e - +0.e-0 - e").get({"e": "1"}) == "-2"
    assert Formula("-e - +.9e-0 - - e").get({"e": "1"}) == "-0.9"
    assert Formula("0-+0.9e-0").get() == "-0.9"
    assert Formula("1-0-1").get() == "0"
    assert Formula("1-0--1").get() == "2"
    assert Formula("1--0-1").get() == "0"
    assert Formula("1--0+-1").get() == "0"
    assert Formula("1-+0e-0-1--1").get() == "1"
    assert Formula("2^5/2^2/2^2").get() == "2"
    assert Formula("00000000000000000000000000000000009").get() == "9"
    assert Formula("0000000000000000+0000000000000000008").get() == "8"
    assert Formula("0000000000000000+.08").get() == "0.08"
    assert Formula("000000000000000008.").get() == "8"
    assert (
        Formula("--000000000000000008.+0+aA.dF_gH+(.0+0.-0e0)").get({"aA.dF_gH": "1"})
        == "9"
    )
    with pytest.raises(ValueError):
        Formula("-e - +.0e*0 - e").get()
    with pytest.raises(ValueError):
        Formula(".e - 1").get()
    with pytest.raises(ValueError):
        Formula("0eb - 1").get()
    with pytest.raises(ValueError):
        Formula("1e - .1").get()
    with pytest.raises(ValueError):
        Formula("2eq - 1e").get()
    with pytest.raises(ValueError):
        Formula("3eq - 1").get()
    with pytest.raises(ValueError):
        Formula("4eq - 1").get()
    with pytest.raises(ValueError):
        Formula("5eq - 1").get()
    with pytest.raises(ValueError):
        Formula("6eq - 1").get()
    with pytest.raises(ValueError):
        Formula("7eq - 1").get()
    with pytest.raises(ValueError):
        Formula("8eq - 1").get()
    with pytest.raises(ValueError):
        Formula("9eq - 1").get()
    with pytest.raises(ValueError):
        Formula(".0.e - 1").get()
    with pytest.raises(ValueError):
        Formula("(((((0))))").get()
    assert Formula("(0+(1+2)/3)*((3+-5)^3)").get() == "-8"


def test_getting_variables():
    f = Formula("a*b+c+D/qwe+s_s_s.*16+3^йцу4^f^t+a+xx+x+x+x")
    assert f.variables() == {
        "a",
        "b",
        "c",
        "D",
        "qwe",
        "s_s_s.",
        u"йцу4",  # support python2.7. It's not necessary in python3+
        "f",
        "t",
        "xx",
        "x",
    }


def test_changing_precision():
    """Change precision on the fly."""

    f = Formula("2*asin(x)", 16)
    assert "3.14159265358979" == f.get({"x": "1"})[0:16]
    assert "3.141592654" == f.get({"x": "1"}, 10)
    f.precision = 24
    digits = 18
    format_flags = FmtFlags.fixed
    assert "3.141592653589793238" == f.get({"x": "1"}, digits, format_flags)
    f.set_precision(240)
    assert f.precision == 256
    assert (
        "3.141592653589793238462643383279502884197169399375105820974944"
        "59230781640628620899862803482534211706798214808651328230664709"
        "38446095505822317253594081284811174502841027019385211055596446"
        "22948954930381964428810975665933446128475648233786783165271201"
        "90914564" == f.get({"x": "1"})[0:256]
    )


def test_changing_expression():
    """Change expression and get value."""

    f = Formula("2*asin(x)", 16)
    assert "3.1415926535897932384626433832795" == f.get({"x": "1"})
    assert "2*asin(x)" == f.expression
    f.set_expression("asin(x)")
    assert "1.57079632679489661923132169163975" == f.get({"x": "1"})
    f.expression = "sin(x)"
    assert "0.8414709848078965066609974948209608755664" == f.get({"x": "1"})


def test_formula_object_copy():
    """Compare Formula objects."""

    f = Formula("2*log(1)", 99)
    f2 = f.copy()
    assert f2 != f
    assert f2 is not f
    assert f2.expression == f.expression
    assert f2.precision == f.precision
    assert f2.get() == f.get()
    f.expression = "1"
    assert f.get() == "1"
    assert f2.get() != f.get()

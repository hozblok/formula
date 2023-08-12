"""Tests on various basic module use cases: parenthesized expressions,
copying an object, getting variables, etc."""

import pytest

from formula import FmtFlags, Formula, Solver


def test_evaluation():  # pylint: disable=too-many-statements
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
    assert Formula("-1 - 0.e-0 - x").get({"x": "1", "y": "0000000000.000000"}) == "-2"
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
    """Check that formula return correct variables."""
    formula = Formula("a*b+c+D/qwe+s_s_s.*16+3^йцу4^f^t+a+xx+x+x+x")
    assert formula.variables() == {
        "a",
        "b",
        "c",
        "D",
        "qwe",
        "s_s_s.",
        "йцу4",
        "f",
        "t",
        "xx",
        "x",
    }


def test_changing_precision():
    """Change precision on the fly."""

    formula = Formula("2*asin(x)", 16)
    assert formula.get({"x": "1"})[0:16] == "3.14159265358979"
    assert formula.get({"x": "1"}, 10) == "3.141592654"
    formula.precision = 24
    digits = 18
    format_flags = FmtFlags.fixed
    assert formula.get({"x": "1"}, digits, format_flags) == "3.141592653589793238"
    formula.set_precision(240)
    assert formula.precision == 256
    assert (
        formula.get({"x": "1"})[0:256]
        == "3.141592653589793238462643383279502884197169399375105820974944"
        "59230781640628620899862803482534211706798214808651328230664709"
        "38446095505822317253594081284811174502841027019385211055596446"
        "22948954930381964428810975665933446128475648233786783165271201"
        "90914564"
    )


def test_changing_expression():
    """Change expression and get value."""

    formula = Formula("2*asin(x)", 24)
    assert (
        formula.get({"x": "1"}, digits=23, format=FmtFlags.fixed)
        == "3.14159265358979323846264"
    )
    assert formula.expression == "2*asin(x)"
    formula.set_expression("asin(x)")
    assert (
        formula.get({"x": "1"}, digits=23, format=FmtFlags.fixed)
        == "1.57079632679489661923132"
    )
    formula.expression = "sin(x)"
    assert (
        formula.get({"x": "1"}, digits=23, format=FmtFlags.fixed)
        == "0.84147098480789650665250"
    )


def test_solver_digits():
    """Simple cases with Solver"""
    # just 32 digits:
    assert (
        Solver("2*asin(x)", precision=32)({"x": "1"})
        == "3.1415926535897932384626433832795"
    )
    # by default format_digits is equal to precision:
    assert (
        Solver("2*asin(x)", 32)({"x": "1"}, format_digits=32)
        == "3.1415926535897932384626433832795"
    )
    # let's round in accordance with format_digits:
    assert (
        Solver("2*asin(x)", 32)({"x": "1"}, format_digits=31)
        == "3.14159265358979323846264338328"
    )
    assert (
        Solver("2*asin(x)", 32)({"x": "1"}, format_digits=30)
        == "3.14159265358979323846264338328"
    )
    assert (
        Solver("2*asin(x)", 32)({"x": "1"}, format_digits=29)
        == "3.1415926535897932384626433833"
    )
    assert (
        Solver("2*asin(x)", 32)(1, format_digits=28) == "3.141592653589793238462643383"
    )
    assert Solver("2*asin(x)", 32)(1, format_digits=14) == "3.1415926535898"
    assert Solver("2*asin(x)", 32)(1, format_digits=13) == "3.14159265359"
    assert Solver("2*asin(x)", 32)(1, format_digits=12) == "3.14159265359"
    assert Solver("2*asin(x)", 32)(1, format_digits=11) == "3.1415926536"
    assert Solver("2*asin(x)", 32)(1, format_digits=10) == "3.141592654"
    assert Solver("2*asin(x)", 32)(1, format_digits=9) == "3.14159265"
    assert Solver("2*asin(x)", 32)(1, format_digits=8) == "3.1415927"
    assert Solver("2*asin(x)", 32)(1, format_digits=7) == "3.141593"
    assert Solver("2*asin(x)", 32)(1, format_digits=6) == "3.14159"
    assert Solver("2*asin(x)", 32)(1, format_digits=5) == "3.1416"
    assert Solver("2*asin(x)", 32)(1, format_digits=4) == "3.142"
    assert Solver("2*asin(x)", 32)(1, format_digits=3) == "3.14"
    assert Solver("2*asin(x)", 32)(1, format_digits=2) == "3.1"
    assert Solver("2*asin(x)", 32)(1, format_digits=1) == "3"
    # show the entire chunk of memory, including insignificant digits:
    assert (
        Solver("2*asin(x)", 32)(1, format_digits=0)
        == "3.1415926535897932384626433832795028841971"
    )


def test_complex_numbers_1():
    """Complex numbers."""
    formula = Solver("2*asin(x)*i*i", 24)
    assert (
        formula.get({"x": "1"}, digits=23, format=FmtFlags.fixed)
        == "-3.14159265358979323846264+i*(0.00000000000000000000000)"
    )

    formula = Formula("2*asin(x)*i*i*i^2", 24)
    assert (
        formula.get({"x": "1"}, digits=23, format=FmtFlags.fixed)
        == "3.14159265358979323846264+i*(0.00000000000000000000000)"
    )


def test_complex_numbers_2():
    """Complex numbers."""
    formula = Solver("2*asin(x)*i*i*i", 24)
    assert (
        formula.get({"x": "1"}, digits=23, format=FmtFlags.fixed)
        == "-0.00000000000000000000000+i*(-3.14159265358979323846264)"
    )

    formula = Formula("asin(x)*i*i*i^2*(i+i)", 24)
    assert (
        formula.get({"x": "1"}, digits=23, format=FmtFlags.fixed)
        == "-0.00000000000000000000000+i*(3.14159265358979323846264)"
    )


def test_formula_object_copy():
    """Compare Formula objects."""

    formula = Formula("2*log(1)", 99)
    formula2 = formula.copy()
    assert formula2 != formula
    assert formula2 is not formula
    assert formula2.expression == formula.expression
    assert formula2.precision == formula.precision
    assert formula2.get() == formula.get()
    formula.expression = "1"
    assert formula.get() == "1"
    assert formula2.get() != formula.get()

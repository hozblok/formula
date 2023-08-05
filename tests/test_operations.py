"""Test on calculation with operations: addition, subtraction,
division, multiplication, etc."""

import pytest

from formula import Formula


@pytest.mark.parametrize(
    "var1, var2, expected_result",
    [
        ("0", "0", "0"),
        ("2", "2", "4"),
        ("2.0", "2.0", "4"),
        ("+0e+0", "-0e-0", "0"),
        ("+1", "-1", "0"),
        ("1e-15", "1", "1.000000000000001"),
        ("-100.0e10000", "+450e10000", "3.5e+10002"),
    ],
)
@pytest.mark.parametrize(
    "precision", [0, 16, 24, 32, 64, 128, 256, 1024, 5000, 8000, 8192]
)
def test_simple_sum(var1, var2, expected_result, precision):
    """Calculating the sum of two numbers."""

    assert expected_result == Formula("var1 + var2", precision).get(
        {"var1": var1, "var2": var2}
    )
    assert expected_result == Formula("var1 + var2", precision).get(
        {"var1": var1, "var2": var2, "addition": "1e5000000"}
    )
    assert expected_result == Formula(var1 + "+" + var2, precision).get()
    assert expected_result == Formula(var1 + "+" + var2, precision).get({})
    assert expected_result == Formula(var1 + "+" + var2, precision).get(
        {"qwer": "-1", "й": "0"}
    )


@pytest.mark.parametrize(
    "var1, var2, expected_result",
    [
        ("2", "2", "0"),
        ("2.0", "-2.0", "4"),
        ("-3", "0", "-3"),
        ("+0e+0", "-0e-0", "0"),
        ("+1", "-1", "2"),
        ("1e-15", "1", "-0.999999999999999"),
        ("-100.0e10000", "+450e10000", "-5.5e+10002"),
    ],
)
@pytest.mark.parametrize(
    "precision", [0, 16, 24, 32, 64, 128, 256, 1024, 5000, 8000, 8192]
)
def test_simple_subtract(var1, var2, expected_result, precision):
    """Calculating the subtraction of two numbers."""

    assert expected_result == Formula("var1 - var2", precision).get(
        {"var1": var1, "var2": var2}
    )
    assert expected_result == Formula("var1 - var2", precision).get(
        {"var1": var1, "var2": var2, "addition": "12345"}
    )
    assert expected_result == Formula(var1 + "-" + var2, precision).get()
    assert expected_result == Formula(var1 + "-" + var2, precision).get({})
    assert expected_result == Formula(var1 + "-" + var2, precision).get(
        {"asdf_фыва": "-1", "й": "1e-100500"}
    )


@pytest.mark.parametrize(
    "var1, var2, result",
    [
        (0, 0, "0"),
        (2, 2, "4"),
        (2.0, 2.0, "4"),
        (0e0, -0e-0, "0"),
        (1, -1, "0"),
        (1e-15, 1, "1.000000000000001"),
        (-100.0e10000, 450e10000, "nan"),
    ],
)
@pytest.mark.parametrize(
    "precision", [0, 16, 24, 32, 64, 128, 256, 1024, 5000, 8000, 8192]
)
def test_simple_sum_from_float(var1, var2, result, precision):
    """Calculating the sum of two numbers using 'get_from_float' method."""

    formula = Formula("var1 + var2", precision)
    value = formula.get_from_float({"var1": var1, "var2": var2})
    assert value.startswith(result)

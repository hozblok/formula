import pytest

from formula import Formula


@pytest.mark.parametrize(
    "a, b, expected_result",
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
def test_simple_sum(a, b, expected_result, precision):
    """Calculating the sum of two numbers."""

    assert expected_result == Formula("a + b", precision).get({"a": a, "b": b})
    assert expected_result == Formula("a + b", precision).get(
        {"a": a, "b": b, "addition": "1e5000000"}
    )
    assert expected_result == Formula(a + "+" + b, precision).get()
    assert expected_result == Formula(a + "+" + b, precision).get({})
    assert expected_result == Formula(a + "+" + b, precision).get(
        {"qwer": "-1", "ё": "0"}
    )


@pytest.mark.parametrize(
    "a, b, expected_result",
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
def test_simple_subtract(a, b, expected_result, precision):
    """Calculating the subtraction of two numbers."""

    assert expected_result == Formula("a - b", precision).get({"a": a, "b": b})
    assert expected_result == Formula("a - b", precision).get(
        {"a": a, "b": b, "addition": "12345"}
    )
    assert expected_result == Formula(a + "-" + b, precision).get()
    assert expected_result == Formula(a + "-" + b, precision).get({})
    assert expected_result == Formula(a + "-" + b, precision).get(
        {"dfgh_ф": "-1", "ё": "1e-100500"}
    )


@pytest.mark.parametrize(
    "a, b, result",
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
def test_simple_sum_from_float(a, b, result, precision):
    """Calculating the sum of two numbers using 'get_from_float' method."""

    f = Formula("a + b", precision)
    value = f.get_from_float({"a": a, "b": b})
    assert value.startswith(result)

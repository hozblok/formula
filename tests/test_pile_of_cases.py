import pytest

from formula import Formula


@pytest.mark.parametrize(
    "a, b, result",
    [
        ("2", "2", "4"),
        ("2.0", "2.0", "4"),
        ("0", "0", "0"),
        ("+0e+0", "-0e-0", "0"),
        ("+1", "-1", "0"),
        ("1e-15", "1", "1.000000000000001"),
        ("-100.0e10000", "+450e10000", "3.5e+10002"),
    ],
)
@pytest.mark.parametrize(
    "precision", [0, 16, 24, 32, 64, 128, 256, 1024, 5000, 8000, 8192]
)
def test_sum(a, b, result, precision):
    """Calculating the sum of two numbers."""

    f = Formula("a + b", precision)
    assert result == f.get({"a": a, "b": b})


@pytest.mark.parametrize(
    "a, b, result",
    [
        (2, 2, "4"),
        (2.0, 2.0, "4"),
        (0, 0, "0"),
        (+0e0, -0e-0, "0"),
        (+1, -1, "0"),
        (1e-15, 1, "1.000000000000001"),
        (-100.0e10000, +450e10000, "nan"),
    ],
)
@pytest.mark.parametrize(
    "precision", [0, 16, 24, 32, 64, 128, 256, 1024, 5000, 8000, 8192]
)
def test_sum_float(a, b, result, precision):
    """Calculating the sum of two numbers."""

    f = Formula("a + b", precision)
    value = f.get_from_float({"a": a, "b": b})
    assert value.startswith(result)


@pytest.mark.parametrize(
    "a, b, result",
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
def test_difference(a, b, result, precision):
    """Calculating the sum of two numbers."""

    f = Formula("a - b", precision)
    assert result == f.get({"a": a, "b": b})

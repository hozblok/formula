"""Finite coordinate conversion preserves flat vectors and rejects malformed inputs."""

import pytest

from formula.capsysred.shared.validation import finite_vector


@pytest.mark.parametrize("value", [[1, "2.5", -3], (1, "2.5", -3), iter([1, "2.5", -3])])
def test_finite_vector_returns_float_tuple(value):
    result = finite_vector(value, 3, "source")
    assert result == (1.0, 2.5, -3.0)
    assert isinstance(result, tuple)
    assert all(type(coordinate) is float for coordinate in result)


@pytest.mark.parametrize("value", [
    None, 12, "12", b"12", {1: 10, 2: 20}, {1, 2},
    [], [1], [1, 2, 3], [[1], [2]], [(1,), (2,)],
    [1, None], [1, "bad"], [1, float("nan")], [1, float("inf")],
    [1, -float("inf")], [1, "NaN"], [1, "1e10000"],
    pytest.param([1, 10 ** 400], id="overflow"),
])
def test_finite_vector_rejects_invalid_coordinates(value):
    with pytest.raises(ValueError, match="point must contain 2 finite coordinates"):
        finite_vector(value, 2, "point")


def test_finite_vector_accepts_numpy_scalar_coordinates():
    np = pytest.importorskip("numpy")
    for value in (np.array([1, 2.5, -3]), [np.array(1), np.array(2.5), np.array(-3)]):
        result = finite_vector(value, 3, "source")
        assert result == (1.0, 2.5, -3.0)
        assert all(type(coordinate) is float for coordinate in result)


def test_finite_vector_rejects_numpy_dimensions_without_squeezing():
    np = pytest.importorskip("numpy")
    for value in (np.array(12), np.array([[1, 2]]), np.array([[1], [2]]),
                  [np.array([1]), np.array([2])]):
        with pytest.raises(ValueError, match="point must contain 2 finite coordinates"):
            finite_vector(value, 2, "point")

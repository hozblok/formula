"""Ray derivatives depend on the surface geometry, not its equation's scale."""

import pytest

from formula.capsysred.stages._ray_jets import Hit, Surface, transport_ray


_SCALES = ("1e400", "-1e400", "1e-400", "-1e-400")


def _trace(expression, entrance, anchor, target_z):
    return transport_ray(entrance, (0, 0, -2), 0, target_z,
                         [Hit(Surface(expression), anchor)], order=4)


def _all_coefficients(ray):
    return tuple(value for row in (*ray.position, *ray.direction, ray.path_length) for value in row)


def _assert_same_derivatives(actual, expected):
    assert actual.powers == expected.powers
    assert _all_coefficients(actual) == pytest.approx(_all_coefficients(expected), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("scale", _SCALES)
def test_plane_scaling_preserves_all_fourth_order_ray_derivatives(scale):
    expected = _trace("x-1", (0.2, 0.1), (1, 0.5, 8), 10)
    actual = _trace(f"(x-1)*({scale})", (0.2, 0.1), (1, 0.5, 8), 10)
    assert tuple(row[0] for row in expected.position) == pytest.approx((0.8, 0.6, 10))
    _assert_same_derivatives(actual, expected)


@pytest.mark.parametrize("scale", _SCALES)
def test_cylinder_scaling_preserves_all_fourth_order_ray_derivatives(scale):
    expected = _trace("x^2+y^2-1", (0.4, 0.3), (0.8, 0.6, 2), 5)
    actual = _trace(f"(x^2+y^2-1)*({scale})", (0.4, 0.3), (0.8, 0.6, 2), 5)
    assert tuple(row[0] for row in expected.position) == pytest.approx((0.2, 0.15, 5))
    _assert_same_derivatives(actual, expected)


@pytest.mark.parametrize("scale", ("1", *_SCALES))
def test_scaling_does_not_make_a_zero_gradient_regular(scale):
    surface = Surface(f"((x-1)^2+y^2)*({scale})")
    with pytest.raises(ValueError, match="gradient"):
        transport_ray((1, 0), (1, 0, -1), 0, 2,
                      [Hit(surface, (1, 0, 1))], order=4)


@pytest.mark.parametrize("scale", ("1", *_SCALES))
def test_grazing_cylinder_is_rejected_at_every_equation_scale(scale):
    surface = Surface(f"(x^2+y^2-1)*({scale})")
    with pytest.raises(ValueError, match="grazing"):
        transport_ray((1, -1), (1, -2, -1), 0, 2,
                      [Hit(surface, (1, 0, 1))], order=4)

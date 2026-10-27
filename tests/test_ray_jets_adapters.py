"""Archived implicit walls retain their units; finite mirrors retain their domain."""

import numpy as np
import pytest

from formula import Number
from formula.capsysred.stages._ray_jets import Hit, transport_ray
from formula.capsysred.surfaces import ImplicitWall, Mirror
from formula.capsysred.walls.wall_cylinder import CylinderWall


@pytest.mark.parametrize("precision", [50, 64])
def test_implicit_wall_reuses_micrometre_geometry(precision):
    center = (Number("1.25", precision), Number("-0.5", precision))
    radius = Number("1", precision)
    walls = [ImplicitWall("(x-1250000)^2+(y+500000)^2-1000000000000", center, radius),
             CylinderWall(center, radius)]
    jets = [transport_ray((1.75, -0.5), (1.25, -0.5, -2), 0, 5,
                          [Hit(wall, (2.25, -0.5, 2))], order=4, precision=precision)
            for wall in walls]
    assert jets[0].powers == jets[1].powers
    for attribute in ("position", "direction", "path_length"):
        np.testing.assert_allclose(getattr(jets[0], attribute), getattr(jets[1], attribute),
                                   rtol=1e-12, atol=1e-12)


def test_implicit_wall_precision_mismatch_is_rejected():
    wall = ImplicitWall("x-1000000", (Number("0", 96), Number("0", 96)), Number("1", 96))
    with pytest.raises(ValueError, match="precision"):
        transport_ray((0.2, 0), (0, 0, -2), 0, 10, [Hit(wall, (1, 0, 8))])


@pytest.mark.parametrize("z0,z1", [(2, 4), (0, 2), (3, 4), (0, 1),
                                     (2-5e-10, 3), (0, 2+5e-10)])
def test_mirror_rejects_hits_outside_its_smooth_reflecting_span(z0, z1):
    mirror = Mirror(Number(str(z0), 64), Number(str(z1), 64))
    with pytest.raises(ValueError, match="reflecting span"):
        transport_ray((0.2, 0), (0.4, 0, -2), 0, 4,
                      [Hit(mirror, (0, 0, 2))], order=1)


def test_mirror_rejects_the_back_side():
    mirror = Mirror(Number("0", 64), Number("4", 64))
    with pytest.raises(ValueError, match="reflecting side"):
        transport_ray((-0.2, 0), (-0.4, 0, -2), 0, 4,
                      [Hit(mirror, (0, 0, 2))], order=1)


def test_mirror_keeps_constant_gradient_and_number_precision():
    mirror = Mirror(Number("0", 96), Number("4", 96))
    point = (Number("1.000000000000000000000000000001", 96), 0, 2)
    assert mirror.precision == mirror.z0.precision
    assert mirror.value(point) is point[0]
    assert tuple(map(float, mirror.gradient(point))) == (1, 0, 0)
    assert all(component.precision == mirror.precision for component in mirror.gradient(point))

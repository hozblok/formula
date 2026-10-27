"""Cylinder ray derivatives against an unfolded meridional-ray family."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from formula import Number
from formula.capsysred.stages._ray_jets import Hit, transport_ray
from formula.capsysred.stages.variational_transport import transport_mode
from formula.capsysred.walls.wall_cylinder import CylinderWall


def _family(q, reflections):
    """Unit cylinder, axial source z=-2; the strict interval fixes the hit history."""
    q = np.asarray(q, dtype=float)
    rho = np.linalg.norm(q)
    target_z = 8.0 * reflections - 1.0
    scale = (target_z + 2.0) / 2.0
    assert 0 < rho < 1
    assert 2 * reflections - 1 < scale * rho < 2 * reflections + 1
    sign = (-1) ** reflections
    length = math.sqrt(4.0 + rho * rho)
    point = np.r_[sign * (scale - 2 * reflections / rho) * q, target_z]
    direction = np.r_[sign * q, 2.0] / length
    hits = [np.r_[(-1) ** j * q / rho, 2 * ((2 * j + 1) / rho - 1)]
            for j in range(reflections)]
    return point, direction, scale * length, hits


@pytest.mark.parametrize("reflections", [1, 2, 3])
def test_cylinder_replay_matches_analytic_family_through_fourth_order(reflections):
    source, q = np.array([0.0, 0.0, -2.0]), np.array([0.5, 0.0])
    point, direction, path, hits = _family(q, reflections)
    wall = CylinderWall((Number("0", 64), Number("0", 64)), Number("1", 64))
    jet = transport_ray(q, source, 0.0, point[2],
                        [Hit(wall, hit) for hit in hits], order=4)
    a, sign, c = q[0], (-1) ** reflections, 2.0 * reflections
    scale, length = (point[2] + 2.0) / 2.0, math.sqrt(4.0 + a * a)

    np.testing.assert_allclose(np.asarray(jet.position)[:, 0], point, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jet.direction)[:, 0], direction, atol=1e-12)
    assert jet.path_length[0] == pytest.approx(path, rel=1e-12)
    expected_q = np.diag([sign * scale, sign * (scale - c / a)])
    expected_p = np.diag([sign * 4.0 / length ** 3, sign / length])
    np.testing.assert_allclose(jet.Q, expected_q, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(jet.P, expected_p, rtol=1e-11, atol=1e-12)

    mode = {"origin": source, "points": point[None, :2],
            "directions": direction[None, :2], "refl": [hits]}
    cap = SimpleNamespace(z0=0.0, bores=[{"center": [0.0, 0.0], "radius": 1.0}])
    old = transport_mode(mode, cap, point[2])
    assert old["valid"][0], old["diagnostics"]
    np.testing.assert_allclose(jet.Q, old["Q"][0], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(jet.P, old["P"][0], rtol=1e-10, atol=1e-11)

    # At q=(a,0): X=sign*(scale-2*N/|q|)*q, U=(sign*q,2)/L, path=scale*L.
    derivatives = [
        (jet.position[0], {(0, 2): sign * c / a ** 2,
                           (1, 2): -2 * sign * c / a ** 3,
                           (2, 2): 6 * sign * c / a ** 4,
                           (0, 4): -9 * sign * c / a ** 4}),
        (jet.position[1], {(2, 1): -2 * sign * c / a ** 3,
                           (0, 3): 3 * sign * c / a ** 3,
                           (3, 1): 6 * sign * c / a ** 4,
                           (1, 3): -9 * sign * c / a ** 4}),
        (jet.direction[0], {(1, 2): sign * (-1 / length ** 3 + 3 * a * a / length ** 5),
                            (0, 4): 9 * sign * a / length ** 5}),
        (jet.direction[1], {(0, 3): -3 * sign / length ** 3,
                            (1, 3): 9 * sign * a / length ** 5}),
        (jet.direction[2], {(0, 4): 18 / length ** 5}),
        (jet.path_length, {(3, 0): -12 * scale * a / length ** 5,
                           (1, 2): -scale * a / length ** 3,
                           (4, 0): scale * (-3 / length ** 3 + 18 * a * a / length ** 5
                                            - 15 * a ** 4 / length ** 7),
                           (2, 2): scale * (-1 / length ** 3 + 3 * a * a / length ** 5),
                           (0, 4): -3 * scale / length ** 3}),
    ]
    for coefficients, expected in derivatives:
        for power, value in expected.items():
            assert coefficients[jet.powers.index(power)] == pytest.approx(value, rel=1e-10, abs=1e-11)

    errors = []
    for step in (0.04, 0.02, 0.01):
        delta = step * np.array([0.7, -0.4])
        nearby_point, nearby_direction, nearby_path, nearby_hits = _family(q + delta, reflections)
        assert len(nearby_hits) == len(hits)
        residual = np.r_[np.asarray(jet.predict_position(delta)) - nearby_point,
                         np.asarray(jet.predict_direction(delta)) - nearby_direction,
                         jet.predict_path_length(delta) - nearby_path]
        errors.append(np.linalg.norm(residual))
    assert errors[0] > 1e-10, errors
    assert errors[1] < errors[0] / 2 ** 4.5 + 5e-13, errors
    assert errors[2] < errors[1] / 2 ** 4.5 + 5e-13, errors

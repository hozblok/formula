"""Direct wall replay against independent scalar intersections and nearby rays."""

from types import SimpleNamespace

import numpy as np
import pytest

from formula import Number
from formula.capsysred.shared.nums import vadd, vdot, vscale, vsub, vunit
from formula.capsysred.stages._ray_jets import Hit, transport_ray
from formula.capsysred.stages.variational_transport import transport_mode
from formula.capsysred.surfaces import Mirror
from formula.capsysred.walls.wall_funnel import FunnelWall
from formula.capsysred.walls.wall_polygon import PolygonWall
from formula.capsysred.walls.wall_revolution import RevolutionWall


def _number(value):
    return Number(str(value), 64)


def _wall(kind):
    center = (_number("0.15"), _number("-0.1"))
    if kind == "revolution":
        return RevolutionWall(kind, center, tuple(map(_number, ("1", "-0.02", "0.001"))))
    return FunnelWall(center, _number("1"), tuple(map(_number, ("0.01", "0.001"))),
                      tuple(map(_number, ("-0.01", "0.0005"))), _number("0.3"))


def _trace(wall, q, source, exit_z=10.0, target_z=12.0):
    """Use the existing scalar hit/normal implementations, without jet geometry."""
    point = tuple(map(_number, (*q, 0.0)))
    origin = tuple(map(_number, source))
    direction = vunit(vsub(point, origin))
    length = vdot(vsub(point, origin), vsub(point, origin)).sqrt()
    points = []
    for _ in range(20):
        assert float(direction[2]) > 0.0
        limit = (_number(exit_z) - point[2]) / direction[2]
        result = wall.hit(point, direction, limit)
        if result is None or float(result[0]) >= float(limit):
            distance = (_number(target_z) - point[2]) / direction[2]
            target = vadd(point, vscale(direction, distance))
            return (np.array(list(map(float, target))), np.array(list(map(float, direction))),
                    points, float(length + distance))
        distance, point, normal = result
        length = length + distance
        points.append(np.array(list(map(float, point))))
        direction = vsub(direction, vscale(normal, 2 * vdot(direction, normal)))
    raise AssertionError("scalar reference exceeded its reflection limit")


def _check_nominal(jet, reference):
    point, direction, _, length = reference
    np.testing.assert_allclose(np.asarray(jet.position)[:, 0], point, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(np.asarray(jet.direction)[:, 0], direction, rtol=2e-12, atol=2e-12)
    assert jet.path_length[0] == pytest.approx(length, rel=2e-12, abs=2e-11)
    gradient = [jet.path_length[jet.powers.index(power)] for power in ((1, 0), (0, 1))]
    np.testing.assert_allclose(gradient, direction[:2] @ np.asarray(jet.Q), rtol=1e-11, atol=1e-11)


def _check_first_variations(jet, q, trace):
    nominal = trace(q)
    h = 0.0002
    columns = []
    for axis in np.eye(2):
        samples = []
        for offset in (-2, -1, 1, 2):
            point, direction, hits, _ = trace(q + offset * h * axis)
            assert len(hits) == len(nominal[2])
            samples.append(np.r_[point[:2], direction[:2]])
        columns.append((samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]) / (12 * h))
    derivative = np.column_stack(columns)
    np.testing.assert_allclose(jet.Q, derivative[:2], rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(jet.P, derivative[2:], rtol=2e-9, atol=2e-9)


def _check_neighbours(jet, q, trace, order):
    nominal = trace(q)
    errors = []
    for step in (0.04, 0.02, 0.01):
        delta = step * np.array([0.7, -0.4])
        point, direction, hits, length = trace(q + delta)
        assert len(hits) == len(nominal[2])
        errors.append(np.linalg.norm(np.r_[np.asarray(jet.predict_position(delta)) - point,
                                           np.asarray(jet.predict_direction(delta)) - direction,
                                           jet.predict_path_length(delta) - length]))
    assert errors[0] > 1e-12, errors
    assert errors[1] < errors[0] / 2 ** (order + 0.5) + 5e-13, errors
    assert errors[2] < errors[1] / 2 ** (order + 0.5) + 5e-13, errors


@pytest.mark.parametrize("kind", ["revolution", "funnel"])
@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("q,reflections", [([0.55, 0.0], 1), ([0.8, -0.45], 2)])
def test_curved_wall_replay_matches_independent_scalar_rays(kind, order, q, reflections):
    wall = _wall(kind)
    source, q = np.array([0.23, -0.14, -2.0]), np.array(q)
    trace = lambda entrance: _trace(wall, entrance, source)
    reference = trace(q)
    assert len(reference[2]) == reflections
    jet = transport_ray(q, source, 0.0, 12.0,
                        [Hit(wall, point) for point in reference[2]], order=order)
    _check_nominal(jet, reference)
    _check_first_variations(jet, q, trace)
    _check_neighbours(jet, q, trace, order)


def _mirror_path(q, source):
    v = np.r_[q - source[:2], -source[2]]
    radius = np.linalg.norm(v)
    direction = v / radius
    distance = -q[0] / direction[0]
    point = np.r_[q, 0.0] + distance * direction
    reflected = direction * [-1.0, 1.0, 1.0]
    target = point + (12.0 - point[2]) / reflected[2] * reflected
    return target, reflected, [point], 7 * radius


@pytest.mark.parametrize("order", [3, 4])
def test_direct_mirror_matches_exact_unfolded_ray(order):
    wall = Mirror(_number("0"), _number("10"))
    source, q = np.array([0.6, -0.04, -2.0]), np.array([0.2, 0.1])
    trace = lambda entrance: _mirror_path(entrance, source)
    reference = trace(q)
    jet = transport_ray(q, source, 0.0, 12.0, [Hit(wall, reference[2][0])], order=order)
    _check_nominal(jet, reference)
    np.testing.assert_allclose(jet.Q, np.diag([-7.0, 7.0]), atol=2e-14)
    v = np.r_[q - source[:2], 2.0]
    radius = np.linalg.norm(v)
    direction = v / radius
    expected_p = np.diag([-1.0, 1.0]) @ (np.eye(2) - np.outer(direction[:2], direction[:2])) / radius
    np.testing.assert_allclose(jet.P, expected_p, atol=2e-14)
    for index, power in enumerate(jet.powers):
        if sum(power) >= 2:
            np.testing.assert_allclose(np.asarray(jet.position)[:, index], 0.0, atol=2e-12)
    if order == 4:
        dxxxx = -3 / radius ** 3 + 18 * v[0] ** 2 / radius ** 5 - 15 * v[0] ** 4 / radius ** 7
        dxxyy = -1 / radius ** 3 + 3 * (v[0] ** 2 + v[1] ** 2) / radius ** 5 - 15 * v[0] ** 2 * v[1] ** 2 / radius ** 7
        assert jet.path_length[jet.powers.index((4, 0))] == pytest.approx(7 * dxxxx, rel=1e-12)
        assert jet.path_length[jet.powers.index((2, 2))] == pytest.approx(7 * dxxyy, rel=1e-12)
    _check_neighbours(jet, q, trace, order)


def _polygon_wall():
    return PolygonWall((_number("0.15"), _number("-0.1")), _number("1"), 5, _number("0.2"))


def _face_history(wall, points):
    center = np.array(list(map(float, wall.center)))
    normals = np.array([[float(x), float(y)] for x, y in wall.faces])
    apothems = np.array(list(map(float, wall.apothems)))
    return tuple(int(np.argmax(normals @ (point[:2] - center) - apothems)) for point in points)


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("q,reflections", [([0.55, 0.0], 1), ([0.8, -0.45], 3)])
def test_polygon_replay_freezes_faces_and_matches_first_variations(order, q, reflections):
    wall = _polygon_wall()
    source, q = np.array([0.23, -0.14, -2.0]), np.array(q)
    reference = _trace(wall, q, source)
    assert len(reference[2]) == reflections
    history = _face_history(wall, reference[2])

    def trace(entrance):
        result = _trace(wall, entrance, source)
        assert _face_history(wall, result[2]) == history
        return result

    jet = transport_ray(q, source, 0.0, 12.0,
                        [Hit(wall, point) for point in reference[2]], order=order)
    _check_nominal(jet, reference)
    bore = {"center": [0.15, -0.1], "radius": 1.0, "kind": "polygon", "sides": 5, "rotation": 0.2}
    mode = {"origin": source, "points": np.array([reference[0][:2]]),
            "directions": np.array([reference[1][:2]]), "refl": [reference[2]]}
    old = transport_mode(mode, SimpleNamespace(z0=0.0, bores=[bore]), 12.0)
    assert old["valid"][0], old["diagnostics"]
    np.testing.assert_allclose(jet.Q, old["Q"][0], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(jet.P, old["P"][0], rtol=1e-11, atol=1e-11)
    _check_neighbours(jet, q, trace, order)
    explicit = transport_ray(q, source, 0.0, 12.0,
                             [Hit(wall.face(index), point) for index, point in zip(history, reference[2])],
                             order=order)
    assert explicit == jet

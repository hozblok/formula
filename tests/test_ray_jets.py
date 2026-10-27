"""Higher ray variations against analytic rays and independent neighbouring paths."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

from formula import Number
from formula.capsysred.stages.variational_transport import transport_mode
from formula.capsysred.stages._ray_jets import Hit, Surface, transport_ray
from formula.capsysred.walls.wall_cylinder import CylinderWall
from formula.capsysred.walls.wall_torus import TorusWall


def _cylinder_surface(provider, precision=64):
    if provider == "expression":
        return Surface("x^2+y^2-1", precision=precision)
    zero = Number("0", precision)
    return CylinderWall((zero, zero), Number("1", precision))


def _torus_surface(provider, precision=64):
    if provider == "expression":
        return Surface("(sqrt((x-100)^2+z^2)-100)^2+y^2-1", precision=precision)
    zero, one = Number("0", precision), Number("1", precision)
    return TorusWall((zero, zero), one, Number("100", precision), (one, zero), zero)


def _path_result(source, q, direction, points, target_z):
    last = points[-1] if points else np.r_[q, 0.0]
    target = last + (target_z - last[2]) / direction[2] * direction
    vertices = [source, np.r_[q, 0.0], *points, target]
    length = sum(np.linalg.norm(b - a) for a, b in zip(vertices, vertices[1:]))
    return target, direction, points, length


def _cylinder_path(q, source, exit_z=10.0, target_z=12.0):
    point = np.r_[q, 0.0]
    direction = point - source
    direction /= np.linalg.norm(direction)
    points = []
    for _ in range(20):
        aa = direction[:2] @ direction[:2]
        bb = 2 * (point[:2] @ direction[:2])
        cc = point[:2] @ point[:2] - 1.0
        root = math.sqrt(bb * bb - 4 * aa * cc)
        ts = [t for t in ((-bb - root) / (2 * aa), (-bb + root) / (2 * aa)) if t > 1e-8]
        distance = min(ts, default=np.inf)
        if point[2] + distance * direction[2] > exit_z:
            return _path_result(source, q, direction, points, target_z)
        point = point + distance * direction
        points.append(point.copy())
        normal = np.r_[point[:2], 0.0]
        direction = direction - 2 * (direction @ normal) * normal
    raise AssertionError("cylinder reference exceeded its reflection limit")


def _torus_path(q, source, exit_z=10.0, target_z=12.0):
    point = np.r_[q, 0.0]
    direction = point - source
    direction /= np.linalg.norm(direction)
    points = []

    def wall(position):
        radial = np.array([position[0] - 100.0, 0.0, position[2]])
        rho = np.linalg.norm(radial)
        raw = (rho - 100.0) * radial / rho + np.array([0.0, position[1], 0.0])
        return np.linalg.norm(raw) - 1.0, raw / np.linalg.norm(raw)

    for _ in range(20):
        ts = np.linspace(1e-7, (exit_z - point[2]) / direction[2], 401)
        values = np.array([wall(point + t * direction)[0] for t in ts])
        crossings = np.flatnonzero(values[:-1] * values[1:] < 0)
        if not len(crossings):
            return _path_result(source, q, direction, points, target_z)
        i = crossings[0]
        distance = brentq(lambda t: wall(point + t * direction)[0], ts[i], ts[i + 1], xtol=1e-13)
        point = point + distance * direction
        points.append(point.copy())
        normal = wall(point)[1]
        direction = direction - 2 * (direction @ normal) * normal
    raise AssertionError("torus reference exceeded its reflection limit")


def _plane_path(q, source, target_z=5.0):
    point = np.r_[q, 0.0]
    direction = point - source
    direction /= np.linalg.norm(direction)
    normal = np.array([1.0, 0.0, 0.2])
    distance = (1.0 - normal @ point) / (normal @ direction)
    hit = point + distance * direction
    reflected = direction - 2 * (direction @ normal) / (normal @ normal) * normal
    return _path_result(source, q, reflected, [hit], target_z)


def _paraboloid_path(q, source, target_z=4.0):
    point = np.r_[q, 0.0]
    direction = point - source
    direction /= np.linalg.norm(direction)
    a = 0.1 * direction[1] ** 2 + 0.05 * direction[2] ** 2
    b = direction[0] + 0.2 * point[1] * direction[1]
    c = point[0] + 0.1 * point[1] ** 2 - 1.0
    distance = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    hit = point + distance * direction
    normal = np.array([1.0, 0.2 * hit[1], 0.1 * hit[2]])
    normal /= np.linalg.norm(normal)
    reflected = direction - 2 * (direction @ normal) * normal
    return _path_result(source, q, reflected, [hit], target_z)


def _scalar_prediction(raw, powers, delta):
    return sum(value * delta[0] ** px * delta[1] ** py / (math.factorial(px) * math.factorial(py))
               for value, (px, py) in zip(raw, powers))


def _check_neighbours(jet, q, source, trace, order):
    nominal = trace(q, source)
    errors = []
    for step in (0.04, 0.02, 0.01):
        delta = step * np.array([0.7, -0.4])
        point, direction, hits, length = trace(q + delta, source)
        assert len(hits) == len(nominal[2])
        predicted_length = jet.predict_path_length(delta)
        assert predicted_length == pytest.approx(_scalar_prediction(jet.path_length, jet.powers, delta), rel=2e-14)
        errors.append(np.linalg.norm(np.r_[np.asarray(jet.predict_position(delta)) - point,
                                           np.asarray(jet.predict_direction(delta)) - direction,
                                           predicted_length - length]))
    assert errors[0] > 1e-12, errors
    assert errors[1] < errors[0] / 2 ** (order + 0.5) + 5e-13, errors
    assert errors[2] < errors[1] / 2 ** (order + 0.5) + 5e-13, errors


def _check_nominal_and_fermat(jet, reference):
    point, direction, _, length = reference
    np.testing.assert_allclose(np.asarray(jet.position)[:, 0], point, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(np.asarray(jet.direction)[:, 0], direction, rtol=2e-12, atol=2e-12)
    assert jet.path_length[0] == pytest.approx(length, rel=2e-12, abs=2e-11)
    gradient = [jet.path_length[jet.powers.index(power)] for power in ((1, 0), (0, 1))]
    np.testing.assert_allclose(gradient, direction[:2] @ np.asarray(jet.Q), rtol=1e-11, atol=1e-11)


def test_free_ray_has_analytic_derivatives_and_raw_factorials():
    source = np.array([0.01, -0.02, -2.0])
    q = np.array([0.1, 0.2])
    jet = transport_ray(q, source, 0.0, 5.0, order=4)
    v = np.r_[q - source[:2], 2.0]
    r = np.linalg.norm(v)
    direction = v / r
    np.testing.assert_allclose(jet.Q, 3.5 * np.eye(2), atol=2e-14)
    np.testing.assert_allclose(jet.P, (np.eye(2) - np.outer(direction[:2], direction[:2])) / r, atol=2e-14)
    for i, power in enumerate(jet.powers):
        if sum(power) >= 2:
            np.testing.assert_allclose(np.asarray(jet.position)[:, i], 0.0, atol=1e-12)
    dxxxx_r = -3 / r ** 3 + 18 * v[0] ** 2 / r ** 5 - 15 * v[0] ** 4 / r ** 7
    dxxyy_r = -1 / r ** 3 + 3 * (v[0] ** 2 + v[1] ** 2) / r ** 5 - 15 * v[0] ** 2 * v[1] ** 2 / r ** 7
    assert jet.path_length[jet.powers.index((4, 0))] == pytest.approx(3.5 * dxxxx_r, rel=1e-12)
    assert jet.path_length[jet.powers.index((2, 2))] == pytest.approx(3.5 * dxxyy_r, rel=1e-12)
    assert jet.direction[0][jet.powers.index((3, 0))] == pytest.approx(dxxxx_r, rel=1e-12)
    reference = _path_result(source, q, direction, [], 5.0)
    _check_nominal_and_fermat(jet, reference)
    mode = {"origin": source, "points": np.array([reference[0][:2]]),
            "directions": np.array([direction[:2]]), "refl": [[]]}
    cap = SimpleNamespace(z0=0.0, bores=[{"center": [0.0, 0.0], "radius": 1.0}])
    old = transport_mode(mode, cap, 5.0)
    np.testing.assert_allclose(jet.Q, old["Q"][0], atol=2e-14)
    np.testing.assert_allclose(jet.P, old["P"][0], atol=2e-14)


@pytest.mark.parametrize("order", [1, 2])
def test_lower_orders_preserve_the_first_variations(order):
    jet = transport_ray([0.1, 0.2], [0.0, 0.0, -2.0], 0.0, 5.0, order=order)
    assert len(jet.powers) == (order + 1) * (order + 2) // 2
    assert jet.powers[:3] == ((0, 0), (1, 0), (0, 1))
    assert np.asarray(jet.position).shape == np.asarray(jet.direction).shape == (3, len(jet.powers))
    np.testing.assert_allclose(jet.Q, 3.5 * np.eye(2), atol=2e-14)


def test_tuple_results_support_explicit_numpy_conversion():
    jet = transport_ray([0.1, 0.2], [0.0, 0.0, -2.0], 0.0, 5.0, order=4)
    assert isinstance(jet.powers, tuple)
    assert all(isinstance(power, tuple) for power in jet.powers)
    for rows in (jet.position, jet.direction, jet.Q, jet.P):
        assert isinstance(rows, tuple)
        assert all(isinstance(row, tuple) for row in rows)
        assert all(type(value) is float for row in rows for value in row)
    assert isinstance(jet.path_length, tuple)
    assert all(type(value) is float for value in jet.path_length)
    for prediction in (jet.predict_position([0.01, -0.02]), jet.predict_direction([0.01, -0.02])):
        assert isinstance(prediction, tuple) and len(prediction) == 3
        assert all(type(value) is float for value in prediction)
    assert type(jet.predict_path_length([0.01, -0.02])) is float
    assert np.asarray(jet.path_length).shape == (len(jet.powers),)
    assert np.asarray(jet.Q).shape == np.asarray(jet.P).shape == (2, 2)
    converted = np.asarray(jet.position)
    original = jet.position[0][0]
    converted[0, 0] += 1.0
    assert jet.position[0][0] == original


def test_oblique_plane_reflection_matches_exact_mirror_family():
    source, q = np.array([0.0, 0.0, -2.0]), np.array([0.2, 0.1])
    reference = _plane_path(q, source)
    surface = Surface("x+0.2*z-1")
    jet = transport_ray(q, source, 0.0, 5.0, hits=[Hit(surface, reference[2][0])], order=4)
    _check_nominal_and_fermat(jet, reference)
    _check_neighbours(jet, q, source, _plane_path, 4)


def test_input_iterators_are_reusable_across_seed_sets():
    wall = Surface("x-1")
    jet = transport_ray(iter([0.2, 0.1]), iter([0.0, 0.0, -2.0]), 0.0, 10.0,
                        iter([Hit(wall, iter([1.0, 0.5, 8.0]))]), order=4)
    np.testing.assert_allclose(np.asarray(jet.position)[:, 0], [0.8, 0.6, 10.0], atol=1e-13)
    np.testing.assert_allclose(jet.Q, np.diag([-6.0, 6.0]), atol=1e-13)


@pytest.mark.parametrize("q,reflections", [([0.5, 0.15], 1), ([0.6, -0.4], 2)])
@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("provider", ["expression", "cylinder"])
def test_cylinder_replay_matches_first_variations_and_high_order_neighbours(q, reflections, order, provider):
    source, q = np.array([0.08, -0.04, -2.0]), np.array(q)
    reference = _cylinder_path(q, source)
    assert len(reference[2]) == reflections
    surface = _cylinder_surface(provider)
    jet = transport_ray(q, source, 0.0, 12.0,
                        hits=[Hit(surface, point) for point in reference[2]], order=order)
    _check_nominal_and_fermat(jet, reference)
    mode = {"origin": source, "points": np.array([reference[0][:2]]),
            "directions": np.array([reference[1][:2]]), "refl": [reference[2]]}
    cap = SimpleNamespace(z0=0.0, bores=[{"center": [0.0, 0.0], "radius": 1.0}])
    old = transport_mode(mode, cap, 12.0)
    assert old["valid"][0], old["diagnostics"]
    np.testing.assert_allclose(jet.Q, old["Q"][0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jet.P, old["P"][0], rtol=1e-10, atol=1e-10)
    _check_neighbours(jet, q, source, _cylinder_path, order)


@pytest.mark.parametrize("provider,surface_factory,trace", [
    ("expression", _cylinder_surface, _cylinder_path),
    ("cylinder", _cylinder_surface, _cylinder_path),
    ("torus", _torus_surface, _torus_path),
])
def test_surface_precision_matches_rounded_transport_precision(provider, surface_factory, trace):
    source, q = np.array([0.08, -0.04, -2.0]), np.array([0.5, 0.15])
    reference = trace(q, source)
    surface = surface_factory(provider, precision=50)
    assert surface.precision == (50 if provider == "expression" else 64)
    jet = transport_ray(q, source, 0.0, 12.0,
                        [Hit(surface, point) for point in reference[2]], order=1, precision=50)
    _check_nominal_and_fermat(jet, reference)


@pytest.mark.parametrize("provider,surface_factory,trace", [
    ("expression", _cylinder_surface, _cylinder_path),
    ("cylinder", _cylinder_surface, _cylinder_path),
    ("torus", _torus_surface, _torus_path),
])
def test_surface_precision_mismatch_is_rejected(provider, surface_factory, trace):
    source, q = np.array([0.08, -0.04, -2.0]), np.array([0.5, 0.15])
    reference = trace(q, source)
    surface = surface_factory(provider, precision=96)
    with pytest.raises(ValueError, match="precision"):
        transport_ray(q, source, 0.0, 12.0,
                      [Hit(surface, point) for point in reference[2]], order=1, precision=64)


@pytest.mark.parametrize("provider,surface_factory,trace", [
    ("cylinder", _cylinder_surface, _cylinder_path),
    ("torus", _torus_surface, _torus_path),
])
def test_direct_wall_rejects_bad_and_duplicate_hits(provider, surface_factory, trace):
    source, q = np.array([0.08, -0.04, -2.0]), np.array([0.5, 0.15])
    reference = trace(q, source)
    surface = surface_factory(provider)
    hit = reference[2][0]
    with pytest.raises(ValueError, match="hit must lie"):
        transport_ray(q, source, 0.0, 12.0, [Hit(surface, hit + [0.0, 0.01, 0.0])], order=1)
    with pytest.raises(ValueError, match="position tolerance"):
        transport_ray(q, source, 0.0, 12.0, [Hit(surface, hit), Hit(surface, hit)], order=1)


@pytest.mark.parametrize("q,reflections", [([0.4, 0.1], 1), ([0.6, -0.4], 2)])
@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("provider", ["expression", "torus"])
def test_torus_replay_matches_independent_root_solved_paths(q, reflections, order, provider):
    source, q = np.array([0.08, -0.04, -2.0]), np.array(q)
    reference = _torus_path(q, source)
    assert len(reference[2]) == reflections
    surface = _torus_surface(provider)
    jet = transport_ray(q, source, 0.0, 12.0,
                        hits=[Hit(surface, point) for point in reference[2]], order=order)
    _check_nominal_and_fermat(jet, reference)
    bore = {"center": [0.0, 0.0], "radius": 1.0, "kind": "torus",
            "bend": {"radius": 100.0, "toward": [1.0, 0.0]}}
    mode = {"origin": source, "points": np.array([reference[0][:2]]),
            "directions": np.array([reference[1][:2]]), "refl": [reference[2]]}
    old = transport_mode(mode, SimpleNamespace(z0=0.0, bores=[bore]), 12.0)
    assert old["valid"][0], old["diagnostics"]
    np.testing.assert_allclose(jet.Q, old["Q"][0], rtol=2e-8, atol=1e-8)
    np.testing.assert_allclose(jet.P, old["P"][0], rtol=2e-8, atol=1e-8)
    _check_neighbours(jet, q, source, _torus_path, order)


@pytest.mark.parametrize("order", [3, 4])
def test_generic_paraboloid_supports_higher_variations(order):
    source, q = np.array([0.1, -0.2, -2.0]), np.array([0.4, 0.2])
    reference = _paraboloid_path(q, source)
    surface = Surface("x+0.1*y^2+0.05*z^2-1")
    jet = transport_ray(q, source, 0.0, 4.0, hits=[Hit(surface, reference[2][0])], order=order)
    _check_nominal_and_fermat(jet, reference)
    _check_neighbours(jet, q, source, _paraboloid_path, order)


def test_coordinate_scale_preserves_physical_ray_derivatives():
    source, q = np.array([0.08, -0.04, -2.0]), np.array([0.5, 0.15])
    reference = _cylinder_path(q, source)
    surfaces = [Surface("x^2+y^2-1"),
                Surface("x^2+y^2-1000000000000", coordinate_scale=1e6)]
    jets = [transport_ray(q, source, 0.0, 12.0,
                          hits=[Hit(surface, point) for point in reference[2]], order=3) for surface in surfaces]
    assert jets[0].powers == jets[1].powers
    for attribute in ("position", "direction", "path_length"):
        np.testing.assert_allclose(getattr(jets[0], attribute), getattr(jets[1], attribute), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("provider", ["expression", "cylinder"])
def test_tangent_hit_has_no_smooth_ray_jet(provider):
    surface = _cylinder_surface(provider)
    with pytest.raises(ValueError):
        transport_ray([1.0, -1.0], [1.0, -2.0, -1.0], 0.0, 2.0,
                      hits=[Hit(surface, [1.0, 0.0, 1.0])])


def test_bad_archived_hit_and_backwards_history_are_rejected():
    source, q = np.array([0.0, 0.0, -2.0]), np.array([0.2, 0.1])
    surface = Surface("x+0.2*z-1")
    hit = _plane_path(q, source)[2][0]
    with pytest.raises(ValueError):
        transport_ray(q, source, 0.0, 5.0, hits=[Hit(surface, hit + [0.01, 0.0, 0.0])])
    with pytest.raises(ValueError):
        transport_ray(q, source, 0.0, 5.0, hits=[Hit(surface, hit + [0.0, 0.01, 0.0])])
    with pytest.raises(ValueError):
        transport_ray(q, source, 0.0, 1.0, hits=[Hit(surface, hit)])
    with pytest.raises(ValueError):
        transport_ray(q, source, 0.0, 5.0, hits=[Hit(surface, hit), Hit(surface, hit)])


def test_duplicate_hit_is_rejected_after_root_polishing():
    source, q = np.array([0.0, 0.0, -2.0]), np.array([0.2, 0.13])
    surface = Surface("x+0.2*z-1")
    hit = _plane_path(q, source)[2][0]
    with pytest.raises(ValueError, match="position tolerance"):
        transport_ray(q, source, 0.0, 5.0, [Hit(surface, hit), Hit(surface, hit)], order=1)


@pytest.mark.parametrize("order", [0, 5, 1.5, True])
def test_invalid_order_is_rejected(order):
    with pytest.raises((TypeError, ValueError)):
        transport_ray([0.1, 0.2], [0.0, 0.0, -1.0], 0.0, 1.0, order=order)


@pytest.mark.parametrize("source,z0,target", [([0.0, 0.0, 0.0], 0.0, 1.0),
                                              ([0.0, 0.0, 1.0], 0.0, 2.0),
                                              ([0.0, 0.0, -1.0], 0.0, -0.5)])
def test_nonforward_configuration_is_rejected(source, z0, target):
    with pytest.raises(ValueError):
        transport_ray([0.1, 0.2], source, z0, target)

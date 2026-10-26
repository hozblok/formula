"""Analytic wall derivatives against the Formula expression evaluator."""

import numpy as np
import pytest

from formula import Formula, Number
from formula.capsysred.config import _bore
from formula.capsysred.surfaces import _make_wall
from formula.capsysred.walls.wall_cylinder import CylinderWall
from formula.capsysred.walls.wall_funnel import FunnelWall
from formula.capsysred.walls.wall_polygon import PolygonWall
from formula.capsysred.walls.wall_revolution import RevolutionWall
from formula.capsysred.walls.wall_torus import TorusWall


CASES = {
    "plane": ("scale*(nx*x+ny*y)", {"nx": .6, "ny": .8, "scale": 1.7}),
    "cylinder": ("(x^2+y^2)/2", {}),
    "revolution": ("(x^2+y^2-c1*z-c2*z^2)/2", {"c1": .3, "c2": -.2}),
    "funnel": (
        "((x-cx*(1+ag*z+bg*z^2))^2+(y-cy*(1+ag*z+bg*z^2))^2"
        "-r0^2*(1+af*z+bf*z^2)^2)/2",
        {"cx": .15, "cy": -.12, "r0": .4, "ag": .2, "bg": -.08, "af": -.1, "bf": .03}),
    "torus": (
        "((sqrt((x-nx*(nx*x+ny*y))^2+(y-ny*(nx*x+ny*y))^2+z^2)-R)^2"
        "+(nx*x+ny*y)^2)/2", {"nx": .6, "ny": .8, "R": 2.}),
}


def _analytic(kind, point, p):
    x, y, z = point
    if kind == "plane":
        return PolygonWall.gradient_hessian(p["nx"], p["ny"], p["scale"])
    if kind == "cylinder":
        return CylinderWall.gradient_hessian(x, y)
    if kind == "revolution":
        return RevolutionWall.gradient_hessian(x, y, z, p["c1"], p["c2"])
    if kind == "funnel":
        return FunnelWall.gradient_hessian(x, y, z, **p)
    s = p["nx"] * x + p["ny"] * y
    qx, qy = x - s * p["nx"], y - s * p["ny"]
    rho = np.sqrt(qx * qx + qy * qy + z * z)
    return TorusWall.gradient_hessian(qx, qy, z, s, p["nx"], p["ny"], rho, (rho-p["R"])/rho)


@pytest.mark.parametrize("kind", CASES)
@pytest.mark.parametrize("point", [(2.4, .1, .3), (1.6, -.2, .7), (2.1, .4, -.5)])
def test_gradient_and_mixed_hessian_match_formula(kind, point):
    expr, parameters = CASES[kind]
    formula = Formula(expr, precision=64)
    values = {name: repr(value) for name, value in dict(parameters, **dict(zip("xyz", point))).items()}
    grad, hess = _analytic(kind, point, parameters)
    expected_grad = [float(formula.get_derivative(axis, values)) for axis in "xyz"]
    expected_hess = [float(formula.get_partial_derivative(list(axes), values))
                     for axes in ("xx", "xy", "xz", "yy", "yz", "zz")]
    np.testing.assert_allclose(grad, expected_grad, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(hess, expected_hess, rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize("kind", CASES)
def test_batched_derivatives_match_scalar_evaluation(kind):
    points = np.array([[2.4, .1, .3], [1.6, -.2, .7], [2.1, .4, -.5]])
    parameters = CASES[kind][1]
    batched = _analytic(kind, points.T, parameters)
    scalar = [_analytic(kind, point, parameters) for point in points]
    for group in range(2):
        for component, values in enumerate(batched[group]):
            np.testing.assert_allclose(np.broadcast_to(values, (len(points),)),
                                       [item[group][component] for item in scalar], rtol=2e-15, atol=0.)


@pytest.mark.parametrize("radius", [1., 2156.25, 1e10])
def test_torus_normal_derivative_at_large_radius_matches_formula(radius):
    from formula.capsysred.stages._b5_transport import wall_normal_derivative

    bore = {"kind": "torus", "center": [72e-6, -3e-6], "radius": 24e-6,
            "bend": {"radius": radius, "toward": [1., 0.]}}
    z0 = .07
    point = np.array([80e-6, 17e-6, .25])
    formula = Formula("((sqrt((R-x)^2+z^2)-R)^2+y^2)/2", precision=64)
    values = dict(zip("xyz", map(repr, (point-np.r_[bore["center"], z0]).tolist())), R=repr(radius))
    grad = np.array([float(formula.get_derivative(axis, values)) for axis in "xyz"])
    hess = np.array([[float(formula.get_partial_derivative([a, b], values)) for b in "xyz"]
                     for a in "xyz"])
    length = np.linalg.norm(grad)
    normal = grad/length
    derivative = (np.eye(3)-np.outer(normal, normal))@hess/length
    got_n, got_dn, got_length = wall_normal_derivative(point, bore, z0)
    np.testing.assert_allclose(got_n, normal, rtol=5e-14, atol=2e-15)
    np.testing.assert_allclose(got_dn, derivative, rtol=5e-14, atol=2e-10)
    np.testing.assert_allclose(got_length, length, rtol=5e-14, atol=0.)


WORLD_BORES = {
    "cylinder": {"radius": .4},
    "revolution": {"r2_poly": [.16, .03, -.02]},
    "funnel": {"radius": .4, "funnel": {"g": [.2, -.08], "f": [-.1, .03]}},
    "polygon": {"radius": .4, "sides": 6, "rotation_deg": 23.,
                "apothems": [.4, .48, .43, .5, .39, .46]},
    "torus": {"radius": .4, "bend": {"radius": 2., "toward": [2., -3.]}},
}
WORLD_POINTS = [(.7, -.1, .55), (-.35, .6, .82), (-.4, -.55, -.15)]


def _world_wall(kind):
    raw = dict(WORLD_BORES[kind], center=[.15, -.12])
    return _make_wall(_bore(raw, 64, 0), Number(".37", 64))


def _world_reference(kind, point):
    values = dict(zip("xyz", map(repr, point)), cx=".15", cy="-.12", z0=".37")
    if kind == "cylinder":
        expression = "((x-cx)^2+(y-cy)^2-.4^2)/2"
    elif kind == "revolution":
        expression = "((x-cx)^2+(y-cy)^2-(.16+.03*z-.02*z^2))/2"
    elif kind == "funnel":
        g = "(1+.2*(z-z0)-.08*(z-z0)^2)"
        f = "(1-.1*(z-z0)+.03*(z-z0)^2)"
        expression = f"((x-cx*{g})^2+(y-cy*{g})^2-.4^2*{f}^2)/2"
    elif kind == "polygon":
        expression = "cos(23*pi/180+2*pi*face/6)*(x-cx)+sin(23*pi/180+2*pi*face/6)*(y-cy)-a"
        formula = Formula(expression, precision=64)
        candidates = [dict(values, face=str(k), a=repr(a))
                      for k, a in enumerate(WORLD_BORES[kind]["apothems"])]
        values = max(candidates, key=lambda v: float(Number(formula.evaluate(v), 64)))
    else:
        xi = "(2*(x-cx)-3*(y-cy))/sqrt(13)"
        eta = "(3*(x-cx)+2*(y-cy))/sqrt(13)"
        expression = f"((sqrt((2-({xi}))^2+(z-z0)^2)-2)^2+({eta})^2-.4^2)/2"
    formula = Formula(expression, precision=64)
    grad = [float(formula.get_derivative(axis, values)) for axis in "xyz"]
    hess = [float(formula.get_partial_derivative(list(axes), values))
            for axes in ("xx", "xy", "xz", "yy", "yz", "zz")]
    return grad, hess


@pytest.mark.parametrize("kind", WORLD_BORES)
@pytest.mark.parametrize("point", WORLD_POINTS)
def test_world_derivatives_match_full_formula(kind, point):
    wall = _world_wall(kind)
    actual = wall.derivatives_at(*point)
    expected = _world_reference(kind, point)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=5e-14, atol=3e-15)

    from formula.capsysred.gamma import _curved_lens, _wall_lens
    outgoing = -np.array(expected[0]) / np.linalg.norm(expected[0]) + np.array([0., 0., .3])
    np.testing.assert_allclose(_wall_lens(wall, *point, outgoing),
                               _curved_lens(*expected, outgoing), rtol=3e-13, atol=3e-14)
    if kind in ("cylinder", "polygon", "torus"):
        from formula.capsysred.stages._b5_transport import wall_normal_derivative
        normal = np.array(expected[0]) / np.linalg.norm(expected[0])
        derivative = (np.eye(3) - np.outer(normal, normal)) @ _full_hessian(expected[1]) / np.linalg.norm(expected[0])
        got_n, got_dn, _ = wall_normal_derivative(np.array(point), wall)
        np.testing.assert_allclose(got_n, normal, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(got_dn, derivative, rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize("kind", WORLD_BORES)
def test_world_derivatives_broadcast_coordinates(kind):
    wall = _world_wall(kind)
    x, y, z = np.array([[.7, -.35, -.4]]), np.array([[-.1], [.6]]), .82
    actual = wall.derivatives_at(x, y, z)
    for i in range(2):
        for j in range(3):
            expected = _world_reference(kind, (float(x[0, j]), float(y[i, 0]), z))
            for got, want in zip(actual, expected):
                for component, value in enumerate(got):
                    np.testing.assert_allclose(np.broadcast_to(value, (2, 3))[i, j], want[component],
                                               rtol=5e-14, atol=3e-15)


@pytest.mark.parametrize("face", range(6))
def test_rotated_shifted_polygon_selects_each_unequal_face(face):
    wall = _world_wall("polygon")
    angle = np.deg2rad(23.) + face * 2 * np.pi / 6
    normal = np.array([np.cos(angle), np.sin(angle)])
    point = np.r_[np.array([.15, -.12]) + WORLD_BORES["polygon"]["apothems"][face] * normal, .71]
    grad, hess = wall.derivatives_at(*point)
    np.testing.assert_allclose(grad, np.r_[normal, 0.], rtol=0., atol=1e-15)
    np.testing.assert_array_equal(hess, np.zeros(6))


def _world_surface_point(kind):
    angle, z, radius = .63, .71, .4
    center = np.array([.15, -.12])
    if kind == "revolution":
        radius = np.sqrt(.16 + .03 * z - .02 * z * z)
    elif kind == "funnel":
        zr = z - .37
        center *= 1 + .2 * zr - .08 * zr * zr
        radius *= 1 - .1 * zr + .03 * zr * zr
    elif kind == "polygon":
        angle = np.deg2rad(23.)
    elif kind == "torus":
        tangent = np.array([2., -3., 0.]) / np.sqrt(13.)
        binormal = np.array([3., 2., 0.]) / np.sqrt(13.)
        ring = -tangent * np.cos(.17) + np.array([0., 0., np.sin(.17)])
        return np.r_[center, .37] + 2 * tangent + (2 + radius * np.cos(angle)) * ring + radius * np.sin(angle) * binormal
    return np.r_[center + radius * np.array([np.cos(angle), np.sin(angle)]), z]


def _full_hessian(compact):
    xx, xy, xz, yy, yz, zz = compact
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


@pytest.mark.parametrize("kind", WORLD_BORES)
def test_world_surface_normal_and_curvature_match_stored_formula(kind):
    wall, point = _world_wall(kind), _world_surface_point(kind)
    formula = Formula(wall.expr_um, precision=64)
    values = {axis: str(Number(repr(float(value)), 64) * Number("1000000", 64))
              for axis, value in zip("xyz", point)}
    expected_g = np.array([float(formula.get_derivative(axis, values)) for axis in "xyz"])
    expected_h = np.array([[float(formula.get_partial_derivative([a, b], values)) for b in "xyz"]
                           for a in "xyz"])
    grad, hess = wall.derivatives_at(*point)
    normal = np.array(grad) / np.linalg.norm(grad)
    expected_n = expected_g / np.linalg.norm(expected_g)
    projector = np.eye(3) - np.outer(normal, normal)
    expected_projector = np.eye(3) - np.outer(expected_n, expected_n)
    curvature = projector @ _full_hessian(hess) @ projector / np.linalg.norm(grad)
    expected_curvature = expected_projector @ expected_h @ expected_projector / np.linalg.norm(expected_g) * 1e6
    np.testing.assert_allclose(normal, expected_n, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(curvature, expected_curvature, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("radius", [2156.25, 1e10])
@pytest.mark.parametrize("toward", [(1., 0.), (2., -3.)])
def test_shifted_rotated_torus_world_derivatives_at_large_radius(radius, toward):
    raw = {"center": [72e-6, -3e-6], "radius": 24e-6,
           "bend": {"radius": radius, "toward": toward}}
    wall = _make_wall(_bore(raw, 64, 0), Number(".07", 64))
    point = (80e-6, 17e-6, .25)
    xi = "(tx*(x-cx)+ty*(y-cy))/sqrt(tx^2+ty^2)"
    eta = "(-ty*(x-cx)+tx*(y-cy))/sqrt(tx^2+ty^2)"
    formula = Formula(f"((sqrt((R-({xi}))^2+(z-z0)^2)-R)^2+({eta})^2)/2", precision=64)
    values = dict(zip("xyz", map(repr, point)), R=repr(radius), tx=repr(toward[0]), ty=repr(toward[1]),
                  cx="72e-6", cy="-3e-6", z0=".07")
    grad = [float(formula.get_derivative(axis, values)) for axis in "xyz"]
    hess = [float(formula.get_partial_derivative(list(axes), values))
            for axes in ("xx", "xy", "xz", "yy", "yz", "zz")]
    actual_g, actual_h = wall.derivatives_at(*point)
    np.testing.assert_allclose(actual_g, grad, rtol=5e-14, atol=2e-20)
    np.testing.assert_allclose(actual_h, hess, rtol=5e-14, atol=3e-16)
    distance_formula = Formula(f"sqrt((sqrt((R-({xi}))^2+(z-z0)^2)-R)^2+({eta})^2)-a", precision=64)
    expected_error = abs(float(Number(distance_formula.evaluate(dict(values, a="24e-6")), 64)))
    np.testing.assert_allclose(wall.surface_error_at(*point), expected_error, rtol=5e-14, atol=2e-20)


@pytest.mark.parametrize("kind", ["revolution", "funnel"])
def test_stage17_keeps_unsupported_wall_guard(kind):
    from formula.capsysred.stages._b5_transport import wall_normal_derivative

    with pytest.raises(ValueError, match="supports cylinder, torus and polygon"):
        wall_normal_derivative(np.array(WORLD_POINTS[0]), _world_wall(kind))


@pytest.mark.parametrize("kind", ["cylinder", "polygon", "torus"])
def test_surface_error_matches_normal_offsets_and_ignores_derivative_scale(kind, monkeypatch):
    from formula.capsysred.stages._b5_transport import wall_normal_derivative

    wall, point = _world_wall(kind), _world_surface_point(kind)
    formula = Formula(wall.expr_um, precision=64)
    values = {axis: str(Number(repr(float(value)), 64) * Number("1000000", 64))
              for axis, value in zip("xyz", point)}
    normal = np.array([float(formula.get_derivative(axis, values)) for axis in "xyz"])
    normal /= np.linalg.norm(normal)
    offsets = np.array([-1e-5, 0., 1e-5])
    points = point + offsets[:, None] * normal
    expected = np.abs(offsets)
    actual = wall.surface_error_at(*points.T)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-14)
    for p, error in zip(points, expected):
        assert wall.surface_error_at(*p) == pytest.approx(error, abs=2e-14)

    before_n, before_dn, before_norm = wall_normal_derivative(points, wall)
    original = wall.derivatives_at
    scale = {"cylinder": 1e-9, "polygon": 7., "torus": 1e9}[kind]

    def scaled_derivatives(*coordinates):
        grad, hess = original(*coordinates)
        return tuple(scale * value for value in grad), tuple(scale * value for value in hess)

    monkeypatch.setattr(wall, "derivatives_at", scaled_derivatives)
    after_n, after_dn, after_norm = wall_normal_derivative(points, wall)
    np.testing.assert_allclose(after_n, before_n, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(after_dn, before_dn, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(after_norm, scale * before_norm, rtol=3e-14, atol=0.)
    np.testing.assert_array_equal(wall.surface_error_at(*points.T), actual)


@pytest.mark.parametrize("rotation", [0., 37.])
def test_polygon_surface_error_rejects_face_extensions(rotation):
    raw = {"center": [.3, -.4], "radius": 1., "sides": 4, "rotation_deg": rotation,
           "apothems": [1., 2., 1., 2.]}
    wall = _make_wall(_bore(raw, 64, 0), Number(".37", 64))
    angle = np.deg2rad(rotation)
    transform = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = np.array([[.2, 2.], [.2, 2.00001], [.2, 1.99999], [1., 2.1], [0., 0.]])
    points = np.asarray(raw["center"]) + local @ transform.T
    error = wall.surface_error_at(points[:, 0], points[:, 1], .71)
    np.testing.assert_allclose(error, [0., 1e-5, 1e-5, .1, 1.], rtol=0., atol=2e-15)

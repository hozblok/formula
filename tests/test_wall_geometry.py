"""Analytic wall derivatives against the Formula expression evaluator."""

import numpy as np
import pytest

from formula import Formula
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

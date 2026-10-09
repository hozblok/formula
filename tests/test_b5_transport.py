from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq

from formula.capsysred.stages._b5_transport import (
    count_free_caustics, reflect_variations, transport_mode, wall_normal_derivative,
)


def test_free_caustics_multiplicity_and_endpoints():
    q = np.tile(np.eye(2), (6, 1, 1))
    v = np.array([np.diag([1., 1.]), np.diag([-2., 1.]), np.diag([-2., -3.]),
                  np.diag([-2., -2.]), np.diag([-1., 1.]), np.diag([-1e-14, -2.])])
    count, ambiguous = count_free_caustics(q, v, np.ones(6))
    np.testing.assert_array_equal(count, [0, 1, 2, 2, 0, 1])
    np.testing.assert_array_equal(ambiguous, [False, False, False, False, True, False])


def test_normal_derivatives_and_large_torus_radius():
    torus = {"center": [72e-6, 0.], "radius": 24e-6, "kind": "torus",
             "bend": {"radius": 2156.25, "toward": [-1., 0.]}}
    cylinder = {"center": [0., 0.], "radius": 24e-6}
    for bore, point in [(cylinder, np.array([18e-6, 16e-6, .12])),
                        (torus, np.array([80e-6, 20e-6, .18]))]:
        normal, derivative, _ = wall_normal_derivative(point, bore, 0.)
        h = 2e-10
        finite = np.column_stack([(wall_normal_derivative(point+np.eye(3)[i]*h, bore, 0.)[0]
                                 -wall_normal_derivative(point-np.eye(3)[i]*h, bore, 0.)[0])/(2*h)
                                for i in range(3)])
        np.testing.assert_allclose(derivative, finite, rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(normal@derivative, 0., atol=1e-11)


def test_flat_reflection_flips_geometry_without_caustic():
    u = np.array([[.6, 0., .8]])
    r = np.array([[[1., 0.], [0., 1.], [0., 0.]]])
    du = np.zeros_like(r)
    normal = np.array([[1., 0., 0.]])
    outgoing, reflected, dp = reflect_variations(u, r, du, normal, np.zeros((1, 3, 3)))
    np.testing.assert_allclose(outgoing, [[-.6, 0., .8]])
    np.testing.assert_allclose(reflected, [[[-1., 0.], [0., 1.], [0., 0.]]])
    count, ambiguous = count_free_caustics(reflected[:, :2], dp[:, :2], np.array([1.]))
    np.testing.assert_array_equal(count, [0])
    assert not ambiguous[0]


def _cylinder_trace(q, source, radius, exit_z, target_z):
    origin = np.r_[q, 0.]
    u = origin-source
    u /= np.linalg.norm(u)
    refs = []
    while True:
        aa = np.dot(u[:2], u[:2])
        bb = 2*np.dot(origin[:2], u[:2])
        cc = np.dot(origin[:2], origin[:2])-radius**2
        discriminant = bb*bb-4*aa*cc
        roots = np.array([(-bb-np.sqrt(discriminant))/(2*aa), (-bb+np.sqrt(discriminant))/(2*aa)])
        roots = roots[roots > 1e-8]
        t = min(roots) if len(roots) else np.inf
        if origin[2]+t*u[2] > exit_z:
            end = origin+(target_z-origin[2])/u[2]*u
            return end[:2], u[:2], refs
        origin = origin+t*u
        refs.append(origin.copy().tolist())
        n = np.r_[origin[:2]/radius, 0.]
        u -= 2*np.dot(u, n)*n


def test_cylinder_transport_against_independent_finite_differences():
    source = np.array([.08, -.04, -2.])
    cap = SimpleNamespace(z0=0., bores=[{"center": [0., 0.], "radius": 1.}])
    qs = np.array([[.1, .1], [.5, .15], [.6, -.4]])
    paths = [_cylinder_trace(q, source, 1., 10., 12.) for q in qs]
    assert [len(p[2]) for p in paths] == [0, 1, 2]
    mode = {"origin": source, "points": np.array([p[0] for p in paths]),
            "directions": np.array([p[1] for p in paths]), "refl": [p[2] for p in paths]}
    result = transport_mode(mode, cap, 12.)
    assert np.all(result["valid"]), result["diagnostics"]
    np.testing.assert_allclose(result["entrance"], qs, atol=1e-15)
    h = 1e-6
    for i, q in enumerate(qs):
        dx, dp = [], []
        for delta in np.eye(2)*h:
            plus = _cylinder_trace(q+delta, source, 1., 10., 12.)
            minus = _cylinder_trace(q-delta, source, 1., 10., 12.)
            dx.append((plus[0]-minus[0])/(2*h))
            dp.append((plus[1]-minus[1])/(2*h))
        np.testing.assert_allclose(result["Q"][i], np.array(dx).T, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(result["P"][i], np.array(dp).T, rtol=1e-7, atol=1e-8)
    assert result["maslov"][0] == 0
    assert result["diagnostics"]["lagrangian_defect_max"] < 1e-13


def test_free_transport_has_exact_fixed_source_amplitude_jacobian():
    source = np.array([.01, -.02, -2.])
    q = np.array([[.1, .2], [-.2, .3]])
    target_z = 5.
    u = np.column_stack([q-source[:2], np.full(len(q), 2.)])
    u /= np.linalg.norm(u, axis=1)[:, None]
    points = q+target_z*u[:, :2]/u[:, 2, None]
    mode = {"origin": source, "points": points, "directions": u[:, :2], "refl": [[], []]}
    cap = SimpleNamespace(z0=0., bores=[{"center": [0., 0.], "radius": 1.}])
    result = transport_mode(mode, cap, target_z)
    np.testing.assert_allclose(result["Q"], np.tile(3.5*np.eye(2), (2, 1, 1)), atol=2e-15)
    np.testing.assert_array_equal(result["maslov"], [0, 0])
    assert np.all(result["valid"])


def test_inconsistent_archived_direction_is_invalid():
    mode = {"origin": [0., 0., -1.], "points": np.array([[.1, .2]]),
            "directions": np.array([[0., 0.]]), "refl": [[]]}
    cap = SimpleNamespace(z0=0., bores=[{"center": [0., 0.], "radius": 1.}])
    result = transport_mode(mode, cap, 1.)
    assert not result["valid"][0]


def _torus_trace(q, source, radius, bend_radius, exit_z, target_z):
    origin = np.r_[q, 0.]
    u = origin-source
    u /= np.linalg.norm(u)
    refs = []
    def normal_and_distance(point):
        radial = np.array([point[0]-bend_radius, 0., point[2]])
        rho = np.linalg.norm(radial)
        raw = (rho-bend_radius)*radial/rho+np.array([0., point[1], 0.])
        return raw/np.linalg.norm(raw), np.linalg.norm(raw)-radius
    while True:
        end = (exit_z-origin[2])/u[2]
        ts = np.linspace(1e-7, end, 301)
        values = [normal_and_distance(origin+t*u)[1] for t in ts]
        crossings = np.flatnonzero(np.asarray(values[1:])*values[:-1] < 0)
        if not len(crossings):
            screen = origin+(target_z-origin[2])/u[2]*u
            return screen[:2], u[:2], refs
        i = crossings[0]
        t = brentq(lambda s: normal_and_distance(origin+s*u)[1], ts[i], ts[i+1], xtol=1e-13)
        origin = origin+t*u
        refs.append(origin.copy().tolist())
        normal, _ = normal_and_distance(origin)
        u -= 2*np.dot(u, normal)*normal


def test_torus_transport_against_independent_finite_difference_paths():
    source = np.array([.08, -.04, -2.])
    bore = {"center": [0., 0.], "radius": 1., "kind": "torus",
            "bend": {"radius": 100., "toward": [1., 0.]}}
    cap = SimpleNamespace(z0=0., bores=[bore])
    q = np.array([.6, -.4])
    point, direction, refs = _torus_trace(q, source, 1., 100., 10., 12.)
    assert len(refs) >= 2
    mode = {"origin": source, "points": np.array([point]), "directions": np.array([direction]), "refl": [refs]}
    result = transport_mode(mode, cap, 12.)
    assert result["valid"][0], result["diagnostics"]
    h = 2e-5
    dx, dp = [], []
    for delta in np.eye(2)*h:
        plus = _torus_trace(q+delta, source, 1., 100., 10., 12.)
        minus = _torus_trace(q-delta, source, 1., 100., 10., 12.)
        assert len(plus[2]) == len(minus[2]) == len(refs)
        dx.append((plus[0]-minus[0])/(2*h))
        dp.append((plus[1]-minus[1])/(2*h))
    np.testing.assert_allclose(result["Q"][0], np.array(dx).T, rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(result["P"][0], np.array(dp).T, rtol=2e-7, atol=1e-7)

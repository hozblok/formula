from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

from formula.capsysred.shared.types import _WALL_SURFACE_TOL_M
from formula.capsysred.stages.variational_transport import (
    bore_area, bore_inside, count_free_caustics, det_path_increment, gamma_phase_shift, polygon_faces,
    reflect_variations, transport_mode, wall_normal_derivative,
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


def _square_trace(q, source, apothem, exit_z, target_z, rotation=0.):
    """Straight channel with four flat faces: exact plane hits and mirror reflections."""
    faces = np.array([[np.cos(rotation+np.pi/2*k), np.sin(rotation+np.pi/2*k)] for k in range(4)])
    origin = np.r_[q, 0.]
    u = origin-source
    u /= np.linalg.norm(u)
    refs = []
    while True:
        best = None
        for n in faces:
            md = n@u[:2]
            if md <= 1e-30:
                continue
            t = (apothem-n@origin[:2])/md
            if t > 1e-9 and (best is None or t < best[0]):
                best = (t, n)
        if best is None or origin[2]+best[0]*u[2] > exit_z:
            end = origin+(target_z-origin[2])/u[2]*u
            return end[:2], u[:2], refs
        origin = origin+best[0]*u
        refs.append(origin.copy().tolist())
        n3 = np.r_[best[1], 0.]
        u -= 2*np.dot(u, n3)*n3


def test_polygon_helpers_faces_inside_and_area():
    square = {"center": [1., -2.], "radius": 1., "sides": 4, "rotation_deg": 0}
    np.testing.assert_allclose(polygon_faces(square), [[1, 0], [0, 1], [-1, 0], [0, -1]], atol=1e-15)
    assert bore_inside(np.array([1.9, -1.1]), square) and not bore_inside(np.array([2.1, -2.]), square)
    assert bore_inside(np.array([1.9, -1.1]), {"center": [1., -2.], "radius": 1., "kind": "polygon",
                                               "sides": 4, "rotation": 0.})
    assert not bore_inside(np.array([1.9, -1.1]), {"center": [1., -2.], "radius": 1.})   # inscribed circle
    np.testing.assert_allclose(bore_area(square), 4.)
    np.testing.assert_allclose(bore_area({"center": [0., 0.], "radius": 2.}), 4*np.pi)
    hexagon = {"center": [0., 0.], "radius": 1., "sides": 6}
    np.testing.assert_allclose(bore_area(hexagon), 6*np.tan(np.pi/6))
    rect = {"center": [0., 0.], "radius": 1., "sides": 4, "rotation_deg": 0, "apothems": [1., 2., 1., 2.]}
    assert bore_inside(np.array([.9, 1.9]), rect) and not bore_inside(np.array([1.1, 0.]), rect)
    assert not bore_inside(np.array([0., 2.1]), rect)
    np.testing.assert_allclose(bore_area(rect), 8.)
    np.testing.assert_allclose(wall_normal_derivative(np.array([1., .5, 3.]), rect, 0.)[0], [1., 0., 0.], atol=1e-15)
    np.testing.assert_allclose(wall_normal_derivative(np.array([.2, 2., 3.]), rect, 0.)[0], [0., 1., 0.], atol=1e-15)


def test_polygon_normal_is_the_face_normal_with_zero_derivative():
    for rotation in (0., 45.):
        bore = {"center": [.5, .25], "radius": 1., "sides": 4, "rotation_deg": rotation}
        n = np.array([np.cos(np.radians(rotation)), np.sin(np.radians(rotation))])
        point = np.r_[np.array([.5, .25])+n+.3*np.array([-n[1], n[0]]), 7.]
        normal, derivative, length = wall_normal_derivative(point, bore, 0.)
        np.testing.assert_allclose(normal, np.r_[n, 0.], atol=1e-15)
        np.testing.assert_allclose(derivative, 0., atol=0.)
        np.testing.assert_allclose(length, 1., atol=1e-15)
    stacked = wall_normal_derivative(np.array([[1.5, .25, 1.], [.5, 1.25, 2.]]), {"center": [.5, .25], "radius": 1., "sides": 4}, 0.)
    np.testing.assert_allclose(stacked[0], [[1., 0., 0.], [0., 1., 0.]], atol=1e-15)


@pytest.mark.parametrize("rotation", [0., 37.])
@pytest.mark.parametrize("offset_factor", [0., -.5, .5, -2., 2.])
def test_unequal_polygon_apothems_validate_the_actual_hit_face(rotation, offset_factor):
    from formula.capsysred.stages.method_safe.defect import build_segments

    angle = np.deg2rad(rotation)
    transform = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([.3, -.4])
    offset = offset_factor * _WALL_SURFACE_TOL_M
    height = 2. + offset
    hit = np.r_[center + transform @ np.array([.1 * height, height]), 2 * height - 1]
    screen = center + transform @ np.array([.3, 2 * height - 3])
    direction = np.array([.05, -.5, 1.])
    direction /= np.linalg.norm(direction)
    outgoing = transform @ direction[:2]
    bore = {"center": center.tolist(), "radius": 1., "sides": 4, "rotation_deg": rotation,
            "apothems": [1., 2., 1., 2.]}
    origin = np.r_[center, -1.]
    mode = {"origin": origin, "points": screen[None], "directions": outgoing[None],
            "refl": [[hit.tolist()]]}
    result = transport_mode(mode, SimpleNamespace(z0=0., bores=[bore]), 5.)
    expected_valid = abs(offset) < _WALL_SURFACE_TOL_M
    assert bool(result["valid"][0]) == expected_valid, result["diagnostics"]
    assert bool(result["continuous_valid"][0]) == expected_valid
    assert result["diagnostics"]["position_error_max_m"] < 2e-15
    assert result["diagnostics"]["direction_error_max"] < 2e-15
    assert result["diagnostics"]["wall_error_max_m"] == pytest.approx(abs(offset), abs=2e-15)

    rows = [(screen[0], screen[1], outgoing[0], outgoing[1], 6 / direction[2],
             [abs(direction[1])], [hit.tolist()])]
    segments = build_segments(rows, origin, 5., [bore], 0., 1., (1e-6, 1e-8))
    assert segments.valid.tolist() == [expected_valid, expected_valid]


def test_square_bore_transport_matches_finite_differences_without_caustics():
    source = np.array([.08, -.04, -2.])
    cap = SimpleNamespace(z0=0., bores=[{"center": [0., 0.], "radius": 1., "sides": 4, "rotation_deg": 0}])
    qs = np.array([[.1, .1], [.5, .15], [.9, .9], [.6, -.4]])          # (.9, .9) lies outside the inscribed circle
    paths = [_square_trace(q, source, 1., 10., 12.) for q in qs]
    assert max(len(p[2]) for p in paths) >= 2
    mode = {"origin": source, "points": np.array([p[0] for p in paths]),
            "directions": np.array([p[1] for p in paths]), "refl": [p[2] for p in paths]}
    result = transport_mode(mode, cap, 12.)
    assert np.all(result["valid"]), result["diagnostics"]
    assert result["diagnostics"]["unassigned_bore_count"] == 0
    np.testing.assert_allclose(result["entrance"], qs, atol=1e-15)
    h = 1e-6
    checked = []
    for i, q in enumerate(qs):
        dx, dp = [], []
        for delta in np.eye(2)*h:
            plus = _square_trace(q+delta, source, 1., 10., 12.)
            minus = _square_trace(q-delta, source, 1., 10., 12.)
            if len(plus[2]) != len(paths[i][2]) or len(minus[2]) != len(paths[i][2]):
                break                                           # a reflection at the exit edge: no smooth branch
            dx.append((plus[0]-minus[0])/(2*h))
            dp.append((plus[1]-minus[1])/(2*h))
        else:
            np.testing.assert_allclose(result["Q"][i], np.array(dx).T, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(result["P"][i], np.array(dp).T, rtol=1e-7, atol=1e-8)
            checked.append(len(paths[i][2]))
    assert len(checked) >= 3 and max(checked) >= 1
    np.testing.assert_array_equal(result["maslov"], 0)
    dets = np.abs(result["Q"][:, 0, 0]*result["Q"][:, 1, 1]-result["Q"][:, 0, 1]*result["Q"][:, 1, 0])
    np.testing.assert_allclose(dets, 49., rtol=1e-9)              # mirrors keep the free-flight tube
    assert result["diagnostics"]["lagrangian_defect_max"] < 1e-13


def test_det_path_increment_matches_dense_sampling():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(300):
        A = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
        B = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))*rng.integers(0, 2)
        t = rng.uniform(.1, 3.)
        s = np.linspace(0., t, 20001)
        d = np.linalg.det(A[None]+s[:, None, None]*B[None])
        if np.min(abs(d)) < 1e-3*np.max(abs(d)):
            continue
        increment, hit = det_path_increment(A[None], B[None], np.array([t]))
        assert abs(increment[0]-np.sum(np.angle(d[1:]/d[:-1]))) < 1e-8
        assert not hit[0]
        checked += 1
    assert checked > 200
    _, hit = det_path_increment(np.eye(2)[None]+0j, np.diag([-1., 0.])[None]+0j, np.array([2.]))
    assert hit[0]


def test_gamma_phase_shift_matches_dense_sampling():
    rng = np.random.default_rng(2)
    for _ in range(50):
        Q = rng.normal(size=(2, 2))
        H = rng.normal(size=(2, 2))
        P = (H+H.T)/2@Q
        g1 = rng.uniform(1e-3, 1e6)
        gs = np.geomspace(1., g1, 20001)
        d = np.linalg.det(gs[:, None, None]*Q[None]+1j*P[None])
        assert abs(gamma_phase_shift(Q[None], P[None], 1., g1)[0]-np.sum(np.angle(d[1:]/d[:-1]))) < 1e-8


def test_continuous_phase_matches_maslov_branch_on_curved_walls():
    """Cylinder (up to 3 bounces) and a long torus (up to ~70 bounces and caustics): the
    continuous determinant argument gives the Maslov-branch prefactor for every width."""
    from formula.capsysred.stages.method_safe.field import canonical_prefactor
    source = np.array([.08, -.04, -2.])
    cylinder = SimpleNamespace(z0=0., bores=[{"center": [0., 0.], "radius": 1.}])
    torus = SimpleNamespace(z0=0., bores=[{"center": [0., 0.], "radius": 1., "kind": "torus",
                                           "bend": {"radius": 2000., "toward": [1., 0.]}}])
    cases = [(cylinder, [_cylinder_trace(np.array(q), source, 1., 10., 12.)
                         for q in ([.1, .1], [.5, .15], [.6, -.4], [-.8, .1], [0., .95])], 12.),
             (torus, [_torus_trace(np.array(q), source, 1., 2000., 300., 320.)
                      for q in ([.1, .1], [.5, .15], [-.8, .1])], 320.)]
    for cap, paths, target in cases:
        mode = {"origin": source, "points": np.array([p[0] for p in paths]),
                "directions": np.array([p[1] for p in paths]), "refl": [p[2] for p in paths]}
        result = transport_mode(mode, cap, target, gamma_ref=2.)
        assert result["valid"].all() and result["continuous_valid"].all(), result["diagnostics"]
        assert result["maslov"].max() > 1
        assert result["diagnostics"]["phase_root_on_path_count"] == 0
        for gamma in (.3, 1., 50., 1e5):
            old, _ = canonical_prefactor(result["Q"], result["P"], result["maslov"], gamma)
            new, valid = canonical_prefactor(result["Q"], result["P"], None, gamma,
                                             phase=result["phase"], gamma_ref=result["gamma_ref"])
            assert valid.all()
            np.testing.assert_allclose(new, old, rtol=1e-12)

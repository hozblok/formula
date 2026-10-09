"""Checks against separable Fresnel integrals, not complex-P1 interpolation."""

import numpy as np
import pytest
from scipy.integrate import quad

from formula.capsysred.stages import _b9_phase
from formula.capsysred.stages._b9_phase import phase_field


SQUARE = np.array([[[-.7, -.4], [.8, -.4], [.8, .6]],
                   [[-.7, -.4], [.8, .6], [-.7, .6]]])
AMPLITUDE = np.array([.9+.3j, .27-.11j, -.23+.17j])


def _mesh(k=7., quadratic=(1.4, -.9), linear=(.8, -.35), constant=.37):
    t = SQUARE.copy()
    h, b = np.asarray(quadratic), np.asarray(linear)
    phase = constant+np.sum(.5*h*t*t+b*t, axis=-1)
    gradient = h*t+b
    amplitude = AMPLITUDE[0]+AMPLITUDE[1]*t[..., 0]+AMPLITUDE[2]*t[..., 1]
    return dict(triangles=t, vertex_phase=phase, vertex_amplitude=amplitude,
                vertex_directions=gradient/k)


def _moment(lower, upper, a, b, degree=0):
    return quad(lambda q: q**degree*np.exp(1j*(.5*a*q*q+b*q)),
                lower, upper, epsabs=2e-13, epsrel=2e-13, complex_func=True)[0]


def _rectangle_reference(x, y, kappa, quadratic=(1.4, -.9), linear=(.8, -.35), constant=.37):
    result = np.empty((len(y), len(x)), complex)
    for iy, v in enumerate(y):
        for ix, u in enumerate(x):
            mx = [_moment(-.7, .8, quadratic[0]+kappa, linear[0]-kappa*u, n) for n in (0, 1)]
            my = [_moment(-.4, .6, quadratic[1]+kappa, linear[1]-kappa*v, n) for n in (0, 1)]
            integral = AMPLITUDE[0]*mx[0]*my[0]+AMPLITUDE[1]*mx[1]*my[0]+AMPLITUDE[2]*mx[0]*my[1]
            result[iy, ix] = kappa/(2j*np.pi)*np.exp(1j*(constant+.5*kappa*(u*u+v*v)))*integral
    return result


@pytest.mark.parametrize("backend", ["direct", "finufft"])
def test_p2_quadratic_phase_complex_amplitude_against_separable_integral(backend):
    if backend == "finufft":
        pytest.importorskip("finufft")
    k, distance = 7., 1.3
    x, y = [-.9, 0., .37], [-.2, .5]
    got, stats = phase_field(_mesh(k), k=k, distance=distance, x=x, y=y,
                             quadrature_order=18, backend=backend, eps=1e-13, return_stats=True)
    expected = _rectangle_reference(x, y, k/distance)
    np.testing.assert_allclose(got, expected, atol=4e-13, rtol=3e-12)
    fit = stats["phase_fit"]
    assert fit["edge_trapezoid_residual_rad"]["maximum"] < 2e-15
    assert fit["vertex_gradient_times_diameter_residual_rad"]["maximum"] < 3e-15
    assert fit["midpoint_correction_rad"]["maximum"] > .1
    assert stats["source_nodes"] == 2*18**2


def test_focused_quadratic_is_exact_at_low_order_and_affine_phase_is_different():
    k, distance = 11., 2.
    b = (.6, -.4)
    curvature = (-k/distance,)*2
    mesh = _mesh(k, quadratic=curvature, linear=b)
    x, y = [b[0]*distance/k], [b[1]*distance/k]
    got = phase_field(mesh, k=k, distance=distance, x=x, y=y, quadrature_order=2, backend="direct")
    expected = _rectangle_reference(x, y, k/distance, quadratic=curvature, linear=b)
    np.testing.assert_allclose(got, expected, atol=7e-16, rtol=2e-15)
    affine, stats = phase_field(mesh, k=k, distance=distance, x=x, y=y,
                                quadrature_order=16, phase_degree=1, backend="direct", return_stats=True)
    assert abs(affine-expected).item() > .2*abs(expected).item()
    assert stats["phase_fit"]["vertex_gradient_times_diameter_residual_rad"]["maximum"] > 1


def test_affine_phase_without_directions_and_quadrature_convergence():
    k, distance = 17., 1.1
    mesh = _mesh(k, quadratic=(0., 0.))
    del mesh["vertex_directions"]
    x, y = [-.9, .2], [-.5, .7]
    expected = _rectangle_reference(x, y, k/distance, quadratic=(0., 0.))
    errors = []
    for order in (4, 12, 28):
        got, stats = phase_field(mesh, k=k, distance=distance, x=x, y=y,
                                 quadrature_order=order, phase_degree=1, backend="direct", return_stats=True)
        errors.append(np.linalg.norm(got-expected))
    assert errors[1] < errors[0]/10
    assert errors[2] < 1e-12
    assert not stats["phase_fit"]["available"]


def test_unequal_overlapping_patches_add_coherently_with_finite_batches():
    k = 6.
    mesh = _mesh(k)
    other = {name: value.copy() for name, value in mesh.items()}
    other["vertex_phase"] += .83
    other["vertex_amplitude"] *= -.4+.7j
    both = {name: np.concatenate((value, other[name])) for name, value in mesh.items()}
    common = dict(k=k, distance=1.2, x=[-.3, .2], y=[.07], quadrature_order=12, backend="direct")
    expected = phase_field(mesh, **common)+phase_field(other, **common)
    got = phase_field(both, max_nodes_per_batch=17, **common)
    np.testing.assert_allclose(got, expected, atol=4e-16, rtol=2e-14)
    assert np.linalg.norm(got) < np.linalg.norm(phase_field(mesh, **common))


def test_common_translation_reversed_orientation_and_local_origins():
    mesh = _mesh()
    common = dict(k=7., distance=1.3, x=np.array([-.1, .4]), y=np.array([.2]),
                  quadrature_order=18, backend="direct", cell_width=.12, pixel_order=4)
    expected = phase_field(mesh, **common)
    delta = np.array([300., -70.])
    translated = {name: value.copy() for name, value in mesh.items()}
    translated["triangles"] += delta
    translated = {name: value[:, ::-1].copy() for name, value in translated.items()}
    got = phase_field(translated, **dict(common, x=common["x"]+delta[0], y=common["y"]+delta[1]),
                      max_nodes_per_batch=53)
    np.testing.assert_allclose(got, expected, atol=2e-13, rtol=5e-12)


def test_native_pixel_mean_includes_chirp_at_each_receiver_node():
    k, distance, width = 7., 1.3, .6
    x, y = [.8], [-.3]
    nodes, weights = np.polynomial.legendre.leggauss(20)
    xx, yy = x[0]+width*nodes/2, y[0]+width*nodes/2
    expected = weights@_rectangle_reference(xx, yy, k/distance)@weights/4
    common = dict(k=k, distance=distance, x=x, y=y, quadrature_order=18, backend="direct", cell_width=width)
    coarse = phase_field(_mesh(k), pixel_order=2, **common).item()
    got = phase_field(_mesh(k), pixel_order=10, **common).item()
    assert abs(got-expected) < 3e-13
    assert abs(coarse-expected) > 1e-5
    center = phase_field(_mesh(k), **dict(common, cell_width=0.)).item()
    assert abs(center-expected) > .01


def test_strict_source_and_target_batch_limits_even_when_one_triangle_exceeds_limit(monkeypatch):
    common = dict(k=7., distance=1.3, x=[-.3, .1], y=[.2, .6], quadrature_order=3,
                  cell_width=.1, pixel_order=3, backend="direct")
    mesh = _mesh()
    originals = {name: value.copy() for name, value in mesh.items()}
    expected = phase_field(mesh, **common)
    transform = _b9_phase._point_transform
    calls = []

    def checked(points, coefficients, frequencies, **kwargs):
        calls.append((len(points), len(frequencies)))
        assert len(points) <= 1
        assert len(frequencies) <= 5
        return transform(points, coefficients, frequencies, **kwargs)

    monkeypatch.setattr(_b9_phase, "_point_transform", checked)
    monkeypatch.setattr(_b9_phase, "_MAX_TARGET_NODES", 5)
    got, stats = phase_field(mesh, max_nodes_per_batch=1, return_stats=True, **common)
    np.testing.assert_allclose(got, expected, atol=5e-16, rtol=2e-14)
    assert stats["source_batches"] == 18
    assert stats["transform_calls"] == len(calls) == 18*8
    assert stats["maximum_source_batch"] == 1 and stats["maximum_target_batch"] == 5
    for name in mesh:
        np.testing.assert_array_equal(mesh[name], originals[name])


def test_p2_shared_edge_phase_is_continuous_and_incompatible_directions_are_visible():
    mesh = _mesh()
    args = _b9_phase._mesh(mesh, 2, 7.)
    correction = args[-1]
    t = np.linspace(0, 1, 31)
    # First triangle edge 2->0 is the same geometric edge as second 0->1.
    p = mesh["vertex_phase"]
    first = (1-t)*p[0, 0]+t*p[0, 2]+4*t*(1-t)*correction[0, 2]
    second = (1-t)*p[1, 0]+t*p[1, 1]+4*t*(1-t)*correction[1, 0]
    np.testing.assert_allclose(first, second, atol=2e-16)
    mesh["vertex_directions"][0, 0, 0] += .3
    _, stats = phase_field(mesh, k=7., distance=1., x=[0.], y=[0.], backend="direct", return_stats=True)
    fit = stats["phase_fit"]
    assert fit["edge_trapezoid_residual_rad"]["maximum"] > 1
    assert fit["vertex_gradient_times_diameter_residual_rad"]["maximum"] > 1


def test_empty_mesh_needs_no_directions_or_transform(monkeypatch):
    mesh = dict(triangles=np.empty((0, 3, 2)), vertex_phase=np.empty((0, 3)),
                vertex_amplitude=np.empty((0, 3), complex))

    def fail(*args, **kwargs):
        pytest.fail("empty mesh must not call the transform")

    monkeypatch.setattr(_b9_phase, "_point_transform", fail)
    got, stats = phase_field(mesh, k=1., distance=1., x=[1., 2.], y=[3.], return_stats=True)
    np.testing.assert_array_equal(got, np.zeros((1, 2), complex))
    assert stats["source_nodes"] == stats["transform_calls"] == 0


def test_mixed_xy_curvature_against_independent_cartesian_integral():
    k, distance = 7., 1.3
    hessian = np.array([[1.4, 1.1], [1.1, -.9]])
    linear, target = np.array([.8, -.35]), np.array([.3, -.2])
    mesh = _mesh(k)
    t = mesh["triangles"]
    mesh["vertex_phase"] = .37+.5*np.einsum("...i,ij,...j->...", t, hessian, t)+t@linear
    mesh["vertex_directions"] = (t@hessian+linear)/k

    def integrand(x, y):
        q = np.array([x, y])
        a = AMPLITUDE[0]+AMPLITUDE[1]*x+AMPLITUDE[2]*y
        p = .37+.5*q@hessian@q+linear@q+.5*k/distance*np.sum((q-target)**2)
        return a*np.exp(1j*p)

    integral = quad(lambda y: quad(lambda x: integrand(x, y), -.7, .8,
                                   complex_func=True, epsabs=2e-13, epsrel=2e-13)[0],
                    -.4, .6, complex_func=True, epsabs=2e-13, epsrel=2e-13)[0]
    got, stats = phase_field(mesh, k=k, distance=distance, x=[target[0]], y=[target[1]],
                             backend="direct", quadrature_order=20, return_stats=True)
    np.testing.assert_allclose(got.item(), k/(2j*np.pi*distance)*integral, atol=2e-14, rtol=3e-14)
    assert stats["phase_fit"]["vertex_gradient_times_diameter_residual_rad"]["maximum"] < 3e-15


def test_large_common_optical_phase_preserves_local_chirp_precision():
    mesh = _mesh(quadratic=(0., 0.), linear=(0., 0.), constant=0.)
    common = dict(k=7., distance=1.3, x=[-.7, .4], y=[-.2, .6], backend="direct", quadrature_order=18)
    expected = phase_field(mesh, **common)*np.exp(1j*1e12)
    mesh["vertex_phase"][:] = 1e12
    got = phase_field(mesh, **common)
    np.testing.assert_allclose(got, expected, atol=3e-16, rtol=4e-15)


def test_finufft_with_multiple_source_and_receiver_batches(monkeypatch):
    pytest.importorskip("finufft")
    common = dict(k=7., distance=1.3, x=[-.1, .4], y=[.2], quadrature_order=8,
                  cell_width=.1, pixel_order=2, eps=1e-13)
    expected = phase_field(_mesh(), backend="direct", **common)
    monkeypatch.setattr(_b9_phase, "_MAX_TARGET_NODES", 5)
    got, stats = phase_field(_mesh(), backend="finufft", max_nodes_per_batch=50,
                             return_stats=True, **common)
    np.testing.assert_allclose(got, expected, atol=2e-13, rtol=5e-12)
    assert stats["source_batches"] == 3 and stats["transform_calls"] == 6


def test_micrometre_coordinates_and_xray_wavenumber_preserve_fresnel_scaling():
    mesh = _mesh()
    common = dict(k=7., distance=1.3, x=np.array([-.1, .4]), y=np.array([.2]),
                  cell_width=.12, pixel_order=4, quadrature_order=18, backend="direct")
    expected = phase_field(mesh, **common)
    scale = 1e-5
    scaled = {name: value.copy() for name, value in mesh.items()}
    scaled["triangles"] *= scale
    scaled["vertex_directions"] *= scale
    got = phase_field(scaled, **dict(common, k=common["k"]/scale**2, x=common["x"]*scale,
                                    y=common["y"]*scale, cell_width=common["cell_width"]*scale))
    np.testing.assert_allclose(got, expected, atol=4e-16, rtol=4e-15)


@pytest.mark.parametrize("name,value", [("k", 0.), ("distance", np.inf), ("cell_width", -.1),
    ("eps", 1.), ("eps", 0.), ("pixel_order", 1.2), ("quadrature_order", True),
    ("phase_degree", 3), ("nthreads", 0), ("max_nodes_per_batch", 0),
    ("backend", "bad"), ("x", []), ("y", [[0.]]), ("x", [np.nan]), ("k", 1j)])
def test_invalid_arguments(name, value):
    args = dict(k=7., distance=1., x=[0.], y=[0.], backend="direct")
    args[name] = value
    with pytest.raises(ValueError):
        phase_field(_mesh(), **args)


@pytest.mark.parametrize("kind", ["degenerate", "nan_phase", "nan_amplitude", "nan_direction",
                                 "missing_direction", "wrong_direction_shape", "complex_phase"])
def test_invalid_mesh(kind):
    mesh = _mesh()
    if kind == "degenerate":
        mesh["triangles"][0, 1] = mesh["triangles"][0, 0]
    elif kind == "nan_phase":
        mesh["vertex_phase"][0, 1] = np.nan
    elif kind == "nan_amplitude":
        mesh["vertex_amplitude"][0, 1] = complex(0., np.inf)
    elif kind == "nan_direction":
        mesh["vertex_directions"][0, 1, 0] = np.inf
    elif kind == "missing_direction":
        del mesh["vertex_directions"]
    elif kind == "wrong_direction_shape":
        mesh["vertex_directions"] = mesh["vertex_directions"][:, :2]
    else:
        mesh["vertex_phase"] = mesh["vertex_phase"].astype(complex)+1j
    with pytest.raises(ValueError):
        phase_field(mesh, k=7., distance=1., x=[0.], y=[0.], backend="direct")

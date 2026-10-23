"""Carrier shifts, complex branch sums and coherent receiver integration."""

import numpy as np
import pytest

from formula.capsysred.stages._b9_carrier import carrier_field


SQUARE = np.array([[[-1., -1.], [1., -1.], [1., 1.]],
                   [[-1., -1.], [1., 1.], [-1., 1.]]])
AMPLITUDE = np.array([[1+.2j, .5-.1j, .3+.7j], [.7-.3j, -.2+.4j, .8+.1j]])


def _mesh(scale=1.):
    triangles = SQUARE*scale
    carriers = np.array([[7., -3.], [-5., 4.]])/scale
    return dict(triangles=triangles.copy(), vertex_amplitude=AMPLITUDE.copy(),
                vertex_phase=np.einsum("tvi,ti->tv", triangles, carriers))


def _reference(mesh, kappa, points, order=48):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    bary = np.stack(((1-u)*(1-v), u, (1-u)*v), axis=-1).reshape(-1, 3)
    w = (weights[:, None]*weights[None, :]*(1-u)).ravel()
    spectrum = np.zeros(len(points), complex)
    for t, a, p in zip(mesh["triangles"], mesh["vertex_amplitude"], mesh["vertex_phase"]):
        e1, e2 = t[1]-t[0], t[2]-t[0]
        determinant = abs(e1[0]*e2[1]-e1[1]*e2[0])
        q = bary@t
        source = determinant*w*(bary@a)*np.exp(1j*(bary@p+.5*kappa*np.sum(q*q, axis=-1)))
        spectrum += source@np.exp(-1j*kappa*(q@points.T))
    return kappa/(2j*np.pi)*np.exp(.5j*kappa*np.sum(points*points, axis=-1))*spectrum


@pytest.mark.parametrize("backend", ["direct", "finufft"])
def test_real_frequency_shift_preserves_unequal_complex_fields(backend):
    if backend == "finufft":
        pytest.importorskip("finufft")
    mesh = _mesh()
    kappa = 1e-6
    x, y = np.array([-2., .1, 3.7])/kappa, np.array([-1.3, .5])/kappa
    got, stats = carrier_field(mesh, 1, 2, k=kappa, distance=1., x=x, y=y,
                               backend=backend, edge_order=12, eps=1e-13)
    points = np.array([(a, b) for b in y for a in x])
    expected = _reference(mesh, kappa, points).reshape(len(y), len(x))
    np.testing.assert_allclose(got, expected, rtol=3e-6, atol=3e-14)
    assert stats["carrier_groups_effective"] == 2
    assert np.max(np.abs(np.asarray(stats["carriers_per_m"]))) > 5
    wrong = dict(mesh, vertex_phase=-mesh["vertex_phase"])
    reverse, _ = carrier_field(wrong, 1, 2, k=kappa, distance=1., x=x, y=y, backend="direct")
    assert np.linalg.norm(reverse-expected) > .2*np.linalg.norm(expected)


def test_batches_add_complex_fields_and_do_not_mutate_exit_data(monkeypatch):
    mesh = _mesh(.07)
    original = {key: value.copy() for key, value in mesh.items()}
    common = dict(k=23., distance=1.3, x=[-.2, .1], y=[-.07, .05], backend="direct")
    full, stats = carrier_field(mesh, 2, 2, max_triangles_per_batch=100, **common)
    from formula.capsysred.stages import _b9_carrier

    transform, call_sizes = _b9_carrier.triangle_fourier, []

    def checked_transform(triangles, *args, **kwargs):
        call_sizes.append(len(triangles))
        assert len(triangles) <= 1
        return transform(triangles, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_b9_carrier, "triangle_fourier", checked_transform)
        batched, split = carrier_field(mesh, 2, 2, max_triangles_per_batch=1, **common)
    assert call_sizes == [1]*8
    np.testing.assert_allclose(batched, full, rtol=4e-11, atol=1e-13)
    assert sum(s["contour_batches"] for s in split["group_runs"]) == 8
    for key in original:
        np.testing.assert_array_equal(mesh[key], original[key])
    repeats, repeat_stats = carrier_field(mesh, 2, 2, max_triangles_per_batch=100, **common)
    np.testing.assert_array_equal(repeats, full)
    assert repeat_stats["carriers_per_m"] == stats["carriers_per_m"]
    changed = dict(mesh, vertex_amplitude=mesh["vertex_amplitude"]*(.2+.8j))
    changed_field, changed_stats = carrier_field(changed, 2, 2, **common)
    np.testing.assert_allclose(changed_field, full*(.2+.8j), rtol=3e-12, atol=1e-13)
    assert changed_stats["carriers_per_m"] == stats["carriers_per_m"]


def test_group_count_is_bounded_by_distinct_geometry_phase_gradients():
    mesh = _mesh(.1)
    common = dict(k=4., distance=.7, x=[0., .03], y=[0.], backend="direct")
    two, _ = carrier_field(mesh, 1, 2, **common)
    many, stats = carrier_field(mesh, 1, 100, **common)
    np.testing.assert_array_equal(many, two)
    assert stats["carrier_groups_effective"] == 2
    duplicate = {key: np.repeat(value[:1], 3, axis=0) for key, value in mesh.items()}
    duplicate["vertex_amplitude"] *= np.array([1., .2+.5j, -.3j])[:, None]
    got, stats = carrier_field(duplicate, 1, 100, **common)
    single = {key: value[:1] for key, value in mesh.items()}
    expected, _ = carrier_field(single, 1, 1, **common)
    np.testing.assert_allclose(got, expected*(1.2+.2j), rtol=2e-12, atol=1e-14)
    assert stats["carrier_groups_effective"] == 1


def test_output_chirp_precedes_coherent_pixel_average():
    mesh = _mesh(.02)
    kappa, width = .15, .2
    x, y = np.array([80.]), np.array([-30.])
    got, _ = carrier_field(mesh, 8, 2, k=kappa, distance=1., x=x, y=y,
                           cell_width=width, pixel_order=8, backend="direct", edge_order=12)
    nodes, weights = np.polynomial.legendre.leggauss(12)
    nodes, weights = nodes*width/2, weights/2
    points = np.array([(x[0]+a, y[0]+b) for b in nodes for a in nodes])
    w = np.array([a*b for b in weights for a in weights])
    values = _reference(mesh, kappa, points)
    expected = values@w
    assert got[0, 0] == pytest.approx(expected, rel=3e-6, abs=1e-13)
    wrong = (values*np.exp(-.5j*kappa*np.sum(points*points, axis=-1)))@w
    wrong *= np.exp(.5j*kappa*(x[0]**2+y[0]**2))
    assert abs(wrong-expected) > .01*abs(expected)


def test_refining_demodulated_representation_converges_to_fixed_phase_integral():
    mesh = _mesh(.25)
    x, y = np.array([-.4, .2]), np.array([-.1, .3])
    kappa = 7.
    expected = _reference(mesh, kappa, np.array([(a, b) for b in y for a in x])).reshape(2, 2)
    errors = []
    for subdivisions in (1, 2, 4):
        field, _ = carrier_field(mesh, subdivisions, 2, k=kappa, distance=1., x=x, y=y, backend="direct", edge_order=12)
        errors.append(np.linalg.norm(field-expected))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .12*errors[0]


def test_multigroup_clustering_is_reproducible_and_independent_of_amplitude():
    rng = np.random.default_rng(27)
    centers = rng.uniform(-.1, .1, size=(160, 2))
    triangles = centers[:, None, :]+SQUARE[0][None, :, :]*.001
    slopes = rng.uniform(-20., 20., size=(160, 2))
    mesh = dict(triangles=triangles, vertex_phase=np.einsum("tvi,ti->tv", triangles, slopes),
                vertex_amplitude=np.ones((160, 3), complex))
    kwargs = dict(subdivisions=1, groups=8, k=3., distance=1., x=[0.], y=[0.], backend="direct")
    first, s1 = carrier_field(mesh, **kwargs)
    second, s2 = carrier_field(dict(mesh, vertex_amplitude=mesh["vertex_amplitude"]*(.3-.7j)), **kwargs)
    assert s1["carrier_groups_effective"] == 8
    assert s1["carriers_per_m"] == s2["carriers_per_m"]
    assert s1["triangle_counts"] == s2["triangle_counts"]
    np.testing.assert_allclose(second, first*(.3-.7j), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("key,value", [("groups", 0), ("subdivisions", 0), ("pixel_order", 0),
                                      ("max_triangles_per_batch", 0), ("distance", 0)])
def test_invalid_parameters_are_rejected(key, value):
    arguments = dict(subdivisions=1, groups=2, k=1., distance=1., x=[0.], y=[0.], backend="direct")
    arguments[key] = value
    with pytest.raises(ValueError):
        carrier_field(_mesh(), **arguments)

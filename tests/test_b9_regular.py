"""Shifted type1 receiver lattices against direct/type3 Fresnel quadrature."""

import numpy as np
import pytest

from formula.capsysred.stages._b9_phase import phase_field
from formula.capsysred.stages._b9_regular import regular_phase_field, regular_phase_order_pair


pytest.importorskip("finufft")


def _mesh(k=7.):
    triangles = np.array([[[-.7, -.4], [.8, -.4], [.8, .6]],
                          [[-.7, -.4], [.8, .6], [-.7, .6]]])
    h = np.array([[1.4, .8], [.8, -.9]])
    b = np.array([.8, -.35])
    phase = .37+.5*np.einsum("...i,ij,...j->...", triangles, h, triangles)+triangles@b
    amplitude = np.array([[1+.2j, .3-.7j, -.1+.3j], [.4-.2j, -.5+.1j, 1.2+.3j]])
    return dict(triangles=triangles, vertex_phase=phase, vertex_amplitude=amplitude,
                vertex_directions=(triangles@h+b)/k)


@pytest.mark.parametrize("shape", [(4, 5), (5, 4), (6, 6), (1, 7), (7, 1), (1, 1)])
def test_even_odd_singleton_and_noncentered_grids_against_direct(shape):
    nx, ny = shape
    common = dict(k=7., distance=1.3, x=.27+.17*np.arange(nx), y=-.36+.13*np.arange(ny),
                  cell_width=.11, pixel_order=3, quadrature_order=12, eps=1e-13)
    expected = phase_field(_mesh(), backend="direct", **common)
    got = regular_phase_field(_mesh(), receiver_channels_per_batch=4, **common)
    np.testing.assert_allclose(got, expected, rtol=2e-11, atol=3e-13)


def test_wrapped_angles_retain_unwrapped_physical_carrier():
    k = 83.
    common = dict(k=k, distance=.9, x=1.217+1.37*np.arange(5), y=-2.413+.93*np.arange(4),
                  cell_width=.23, pixel_order=4, quadrature_order=9, eps=1e-13)
    mesh = _mesh(k)
    # Type1 source angles span many periods; the receiver carrier is fractional.
    assert np.max(abs(mesh["triangles"]*k/.9*np.array([1.37, .93]))) > 20*np.pi
    expected = phase_field(mesh, backend="direct", **common)
    got = regular_phase_field(mesh, receiver_channels_per_batch=3, **common)
    np.testing.assert_allclose(got, expected, rtol=3e-10, atol=3e-12)


@pytest.mark.parametrize("backend", ["direct", "finufft"])
def test_xray_scales_and_native_pixel_mean_match_existing_backend(backend):
    scale = 1e-5
    mesh = _mesh()
    mesh["triangles"] *= scale
    mesh["vertex_directions"] *= scale
    common = dict(k=7./scale**2, distance=1.3, x=np.linspace(-160e-6, 160e-6, 8),
                  y=np.linspace(-130e-6, 130e-6, 7), cell_width=.3e-6, pixel_order=4,
                  quadrature_order=16, eps=1e-13)
    expected = phase_field(mesh, backend=backend, **common)
    got, stats = regular_phase_field(mesh, return_stats=True, **common)
    np.testing.assert_allclose(got, expected, atol=2e-12, rtol=3e-10)
    assert stats["maximum_lattice_fourier_phase_defect_bound_rad"] < 1e-12


def test_exact_quadratic_focus_and_unequal_complex_affine_amplitude():
    k, distance = 11., 2.
    mesh = _mesh(k)
    t = mesh["triangles"]
    b = np.array([.6, -.4])
    mesh["vertex_phase"] = -.5*k/distance*np.sum(t*t, axis=-1)+t@b+.37
    mesh["vertex_directions"] = (-k/distance*t+b)/k
    a = np.array([.9+.3j, .27-.11j, -.23+.17j])
    mesh["vertex_amplitude"] = a[0]+a[1]*t[..., 0]+a[2]*t[..., 1]
    target = b*distance/k
    got = regular_phase_field(mesh, k=k, distance=distance, x=[target[0]], y=[target[1]],
                              quadrature_order=2, eps=1e-13).item()
    area = 1.5
    integral = area*(a[0]+a[1]*.05+a[2]*.1)
    expected = k/(2j*np.pi*distance)*np.exp(1j*(.37+.5*k/distance*np.sum(target*target)))*integral
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-13)


def test_common_translation_and_reversed_axes_preserve_field():
    scale = 1e-5
    mesh = _mesh()
    mesh["triangles"] *= scale
    mesh["vertex_directions"] *= scale
    common = dict(k=7./scale**2, distance=1.3, x=np.linspace(-30e-6, 40e-6, 6),
                  y=np.linspace(-50e-6, 20e-6, 5), cell_width=.3e-6, pixel_order=4,
                  quadrature_order=14, eps=1e-13)
    expected = regular_phase_field(mesh, **common)
    shift = np.array([.002, -.003])
    moved = {name: value.copy() for name, value in mesh.items()}
    moved["triangles"] += shift
    got = regular_phase_field(moved, **dict(common, x=common["x"][::-1]+shift[0],
                                          y=common["y"][::-1]+shift[1]))
    np.testing.assert_allclose(got[::-1, ::-1], expected, atol=5e-12, rtol=3e-10)


def test_source_and_receiver_channel_batches_add_signed_complex_fields(monkeypatch):
    import finufft
    mesh = _mesh()
    second = {name: value.copy() for name, value in mesh.items()}
    second["vertex_amplitude"] *= -.93+.07j
    second["vertex_phase"] += .031
    both = {name: np.concatenate((value, second[name])) for name, value in mesh.items()}
    before = {name: value.copy() for name, value in both.items()}
    common = dict(k=7., distance=1.3, x=np.linspace(-.1, .4, 4), y=np.linspace(-.3, .1, 3),
                  cell_width=.15, pixel_order=3, quadrature_order=3, eps=1e-13)
    expected = phase_field(both, backend="direct", **common)
    real_plan, seen = finufft.Plan, []

    class Plan:
        def __init__(self, *args, **kwargs):
            self.plan = real_plan(*args, **kwargs)
        def setpts(self, x, y):
            assert len(x) <= 7 and len(y) == len(x)
            seen.append(len(x))
            self.plan.setpts(x, y)
        def execute(self, values):
            assert values.shape[0] == 4 and values.shape[1] <= 7
            return self.plan.execute(values)

    monkeypatch.setattr(finufft, "Plan", Plan)
    got, stats = regular_phase_field(both, max_nodes_per_batch=7, receiver_channels_per_batch=4,
                                     return_stats=True, **common)
    np.testing.assert_allclose(got, expected, atol=8e-14, rtol=2e-10)
    assert seen == [7, 7, 7, 7, 7, 1]
    assert stats["source_batches"] == 6 and stats["transform_calls"] == 18
    assert stats["padded_channel_transforms"] == 18 and stats["plans"] == 1
    for name in both:
        np.testing.assert_array_equal(both[name], before[name])


def test_affine_phase_without_directions_and_single_receiver_channel():
    mesh = _mesh()
    del mesh["vertex_directions"]
    common = dict(k=7., distance=1.3, x=np.linspace(-.2, .3, 4), y=[.17],
                  phase_degree=1, pixel_order=1, cell_width=.1, quadrature_order=12, eps=1e-13)
    expected = phase_field(mesh, backend="direct", **common)
    got = regular_phase_field(mesh, **common)
    np.testing.assert_allclose(got, expected, atol=2e-13, rtol=2e-11)


def test_order_pair_records_sensitivity_and_returns_finer_field():
    common = dict(k=7., distance=1.3, x=np.linspace(-.3, .5, 4), y=np.linspace(-.4, .3, 3),
                  cell_width=.12, pixel_order=3, eps=1e-13)
    got, stats = regular_phase_order_pair(_mesh(), quadrature_orders=(4, 12), **common)
    expected = phase_field(_mesh(), backend="direct", quadrature_order=12, **common)
    coarse = phase_field(_mesh(), backend="direct", quadrature_order=4, **common)
    np.testing.assert_allclose(got, expected, atol=2e-13, rtol=2e-11)
    assert stats["comparison"]["relative_complex_l2"] == pytest.approx(
        np.linalg.norm(coarse-expected)/np.linalg.norm(expected), rel=1e-9)
    assert stats["comparison"]["relative_complex_l2"] > 1e-4


def test_empty_mesh_returns_zero_without_plan(monkeypatch):
    import finufft
    mesh = dict(triangles=np.empty((0, 3, 2)), vertex_phase=np.empty((0, 3)),
                vertex_amplitude=np.empty((0, 3), complex))
    def fail(*args, **kwargs):
        pytest.fail("empty field must not allocate a plan")
    monkeypatch.setattr(finufft, "Plan", fail)
    got, stats = regular_phase_field(mesh, k=1., distance=1., x=[1., 2.], y=[3.], return_stats=True)
    np.testing.assert_array_equal(got, np.zeros((1, 2), complex))
    assert stats["source_nodes"] == stats["plans"] == 0


@pytest.mark.parametrize("override", [dict(x=[0., .1, .21]), dict(x=[0., 0.]), dict(y=[]),
    dict(x=[np.nan]), dict(x=[[0.]]), dict(receiver_channels_per_batch=0),
    dict(max_nodes_per_batch=0), dict(quadrature_order=1.5), dict(phase_degree=3),
    dict(eps=1.), dict(k=0.), dict(distance=np.inf), dict(cell_width=-.1)])
def test_invalid_arguments(override):
    common = dict(k=7., distance=1.3, x=[0.], y=[0.])
    common.update(override)
    with pytest.raises(ValueError):
        regular_phase_field(_mesh(), **common)

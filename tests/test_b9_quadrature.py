"""Independent checks of real-phase Duffy bounds and mixed-order assembly."""
import numpy as np
import pytest

from formula.capsysred.stages._b9_phase import phase_field
from formula.capsysred.stages._b9_quadrature import (
    _quadratic_abs_max, mixed_phase_field, phase_quadrature_orders,
)


def model():
    t = np.array([[[0., 0.], [1., .2], [-.1, .8]],
                  [[1.3, -.2], [1.5, .15], [1.15, .2]]])
    H = np.array([[7., -4.], [-4., 5.]])
    gradient = np.array([2., -1.])
    p = .31+np.einsum('...i,ij,...j->...', t, H, t)/2+t@gradient
    a = np.array([[.4+.2j, -.3+.4j, .7-.2j], [1.2-.4j, -.1+.7j, .6+.3j]])
    return dict(triangles=t, vertex_phase=p, vertex_directions=(t@H+gradient)/11,
                vertex_amplitude=a)


def test_quadratic_extrema_include_interior_stationary_point():
    np.testing.assert_allclose(_quadratic_abs_max([0., 0., 2., 1.], [4., -4., -1., 0.],
                                               [-4., 4., 0., 0.]), [1., 1., 2., 1.])


@pytest.mark.parametrize('degree', [1, 2])
def test_bounds_dominate_independent_dense_physical_gradient_samples(degree):
    mesh = model()
    common = dict(k=11., distance=1.7, x=[-.5, .8], y=[-.7, .3], cell_width=.2,
                  phase_degree=degree, max_order=512)
    selected = phase_quadrature_orders(mesh, **common)
    gauss = np.linspace(0., 1., 111)
    u, v = np.meshgrid(gauss, gauss)
    s, t = u.ravel(), ((1-u)*v).ravel()
    for index, triangle in enumerate(mesh['triangles']):
        e1, e2 = triangle[1]-triangle[0], triangle[2]-triangle[0]
        points = triangle[0]+s[:, None]*e1+t[:, None]*e2
        if degree == 2:
            gradient = points@np.array([[7., -4.], [-4., 5.]])+[2., -1.]
        else:
            diffs = mesh['vertex_phase'][index, 1:]-mesh['vertex_phase'][index, 0]
            gradient = np.broadcast_to(np.linalg.solve(np.array([e1, e2]), diffs), points.shape)
        maximum = np.zeros(2)
        for X in [[-.6, -.8], [-.6, .4], [.9, -.8], [.9, .4], [.2, .1]]:
            total = gradient+11./1.7*(points-X)
            du = np.sum(total*(e1-v.ravel()[:, None]*e2), axis=1)
            dv = (1-s)*np.sum(total*e2, axis=1)
            maximum = np.maximum(maximum, [abs(du).max(), abs(dv).max()])
        assert np.all(selected['phase_derivative_bounds'][index] >= maximum-1e-12)
        np.testing.assert_allclose(selected['phase_derivative_bounds'][index], maximum, rtol=3e-4, atol=1e-10)


def test_bound_and_orders_invariant_under_common_coordinate_and_phase_translation():
    mesh = model()
    args = dict(k=11., distance=1.7, x=np.linspace(-.5, .8, 3), y=[-.4, .5], cell_width=.2)
    before = phase_quadrature_orders(mesh, **args)
    shift = np.array([2.1, -3.4])
    mesh['triangles'] += shift
    mesh['vertex_phase'] += 172.
    after = phase_quadrature_orders(mesh, **dict(args, x=args['x']+shift[0], y=np.array(args['y'])+shift[1]))
    np.testing.assert_allclose(before['phase_derivative_bounds'], after['phase_derivative_bounds'], rtol=1e-13)
    np.testing.assert_array_equal(before['orders'], after['orders'])


def test_order_selection_monotone_in_safety_and_receiver_box():
    args = dict(k=11., distance=1.7, x=[-.5, .8], y=[-.4, .5], cell_width=.2)
    low = phase_quadrature_orders(model(), safety_factor=1., **args)
    high = phase_quadrature_orders(model(), safety_factor=2., **args)
    big = phase_quadrature_orders(model(), safety_factor=2., **dict(args, x=[-2., 3.], y=[-2., 3.]))
    assert np.all(high['orders'] >= low['orders'])
    assert np.all(big['orders'] >= high['orders'])


def test_order_cap_is_an_error_not_silent_clipping():
    with pytest.raises(ValueError, match='do not silently truncate'):
        phase_quadrature_orders(model(), k=11., distance=.001, x=[2.], y=[3.], max_order=16)


def test_grouped_complex_sum_matches_selected_direct_rules_with_different_origins():
    pytest.importorskip('finufft')
    mesh = model()
    before = {name: value.copy() for name, value in mesh.items()}
    common = dict(k=11., distance=1.7, x=np.linspace(-.2, .2, 4), y=np.linspace(-.1, .3, 3),
                  cell_width=.12, pixel_order=3, eps=1e-13)
    orders = phase_quadrature_orders(mesh, safety_factor=1.5,
                                    **{key: common[key] for key in ('k', 'distance', 'x', 'y', 'cell_width')})['orders'].max(axis=1)
    assert len(np.unique(orders)) == 2
    expected = np.zeros((3, 4), complex)
    for i, q in enumerate(orders):
        sub = {name: value[i:i+1] for name, value in mesh.items()}
        expected += phase_field(sub, quadrature_order=int(q), backend='direct', **common)
    got, stats = mixed_phase_field(mesh, max_nodes_per_batch=37, receiver_channels_per_batch=3,
                                  return_stats=True, **common)
    np.testing.assert_allclose(got, expected, rtol=3e-10, atol=3e-13)
    assert stats['source_nodes'] == sum(int(q)**2 for q in orders)
    assert all(run['maximum_source_batch'] <= 37 for run in stats['runs'])
    for name in before:
        np.testing.assert_array_equal(mesh[name], before[name])


def test_highly_oscillatory_triangle_improves_over_uniform_low_order():
    pytest.importorskip('finufft')
    mesh = model()
    mesh = {name: value[:1].copy() for name, value in mesh.items()}
    t = mesh['triangles']
    b = np.array([91., -43.])
    mesh['vertex_phase'] += t@b
    mesh['vertex_directions'] += b/11
    common = dict(k=11., distance=1.7, x=np.linspace(-.2, .2, 3), y=[.1],
                  cell_width=.12, pixel_order=3, eps=1e-13)
    expected = phase_field(mesh, quadrature_order=96, backend='direct', **common)
    low = phase_field(mesh, quadrature_order=8, backend='direct', **common)
    got = mixed_phase_field(mesh, safety=1.5, **common)
    assert np.linalg.norm(low-expected) > .05*np.linalg.norm(expected)
    np.testing.assert_allclose(got, expected, rtol=2e-8, atol=1e-12)


def test_oscillatory_rectangle_against_analytic_fourier_integral_and_pixel_mean():
    pytest.importorskip('finufft')
    k, distance = 11., 1.7
    kappa = k/distance
    t = np.array([[[0., 0.], [1., 0.], [1., 1.]],
                  [[0., 0.], [1., 1.], [0., 1.]]])
    b = np.array([80., -37.])
    coefficients = np.array([1.2+.1j, .3-.2j, -.1+.4j])
    mesh = dict(triangles=t, vertex_phase=.37+t@b-.5*kappa*np.sum(t*t, axis=-1),
        vertex_directions=(b-kappa*t)/k,
        vertex_amplitude=coefficients[0]+coefficients[1]*t[..., 0]+coefficients[2]*t[..., 1])
    x, y, width, pixel_order = np.linspace(-.2, .2, 3), np.array([-.1, .2]), .025, 6
    def integrals(w):
        h = w/2
        zero = np.exp(1j*h)*np.sinc(h/np.pi)
        first = np.exp(1j*h)*(.5*np.sinc(h/np.pi)+.5j*(np.sin(h)-h*np.cos(h))/h**2)
        return zero, first
    nodes, weights = np.polynomial.legendre.leggauss(pixel_order)
    reference = np.zeros((len(y), len(x)), complex)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            xx, yy = x+width*u/2, y+width*v/2
            ix, jx = integrals(b[0]-kappa*xx)
            iy, jy = integrals(b[1]-kappa*yy)
            integral = coefficients[0]*iy[:, None]*ix+coefficients[1]*iy[:, None]*jx+coefficients[2]*jy[:, None]*ix
            reference += weights[i]*weights[j]/4*kappa/(2j*np.pi)*np.exp(1j*(.37+.5*kappa*(xx[None, :]**2+yy[:, None]**2)))*integral
    got = mixed_phase_field(mesh, k=k, distance=distance, x=x, y=y, cell_width=width,
                            pixel_order=pixel_order, safety=1.5, eps=1e-13)
    np.testing.assert_allclose(got, reference, rtol=3e-9, atol=2e-13)


def test_actual_xray_scale_and_unequal_overlapping_branch_coefficients():
    pytest.importorskip('finufft')
    mesh = model()
    scale = 1e-5
    mesh['triangles'] *= scale
    mesh['vertex_directions'] *= scale
    duplicate = {name: value.copy() for name, value in mesh.items()}
    duplicate['vertex_amplitude'] *= -.93+.09j
    duplicate['vertex_phase'] += .023
    both = {name: np.concatenate((value, duplicate[name])) for name, value in mesh.items()}
    common = dict(k=11/scale**2, distance=1.7, x=np.linspace(-30e-6, 30e-6, 4),
                  y=np.linspace(-25e-6, 35e-6, 3), cell_width=.3e-6, pixel_order=4, eps=1e-13)
    got, stats = mixed_phase_field(both, safety=1.5, return_stats=True, **common)
    q = stats['selection']['maximum_executed_order']+16
    expected = phase_field(both, quadrature_order=q, backend='direct', **common)
    np.testing.assert_allclose(got, expected, rtol=2e-8, atol=8e-13)


def test_empty_mesh():
    empty = dict(triangles=np.empty((0, 3, 2)), vertex_phase=np.empty((0, 3)),
                 vertex_amplitude=np.empty((0, 3), complex))
    got, stats = mixed_phase_field(empty, k=2., distance=1., x=[1., 2.], y=[3.], return_stats=True)
    np.testing.assert_array_equal(got, np.zeros((1, 2), complex))
    assert stats['source_nodes'] == 0
    assert stats['selection']['groups'] == []


@pytest.mark.parametrize('override', [dict(k=0.), dict(distance=np.inf), dict(x=[]), dict(y=[[0.]]),
    dict(cell_width=-1.), dict(safety_factor=0.), dict(safety_factor=np.nan), dict(min_order=1),
    dict(order_multiple=0), dict(order_multiple=1.5), dict(max_order=4), dict(phase_degree=3)])
def test_invalid_selection_args(override):
    common = dict(k=11., distance=1., x=[0.], y=[0.])
    common.update(override)
    with pytest.raises(ValueError):
        phase_quadrature_orders(model(), **common)

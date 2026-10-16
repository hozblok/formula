"""Curved pullback integration: independent analytic/Cartesian/affine controls."""
import numpy as np
import pytest

from formula.capsysred.stages._b9_curved import (
    _derivative_bounds, _duffy_power, curved_phase_field, curved_quadrature_orders,
)
from formula.capsysred.stages._b9_phase import phase_field


BARY = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                 [.5, .5, 0.], [0., .5, .5], [.5, 0., .5]])


def rectangle():
    return np.array([[[0., 0.], [1., 0.], [1., 1.]], [[0., 0.], [1., 1.], [0., 1.]]])


def nonlinear_mesh():
    triangles = rectangle()
    q = np.einsum('ni,tic->tnc', BARY, triangles)
    a, b = q[..., 0], q[..., 1]
    positions = np.stack((a+.27*a*a-.13*a*b, .7*b+.21*a*b+.08*b*b), axis=-1)
    phase = .37+1.2*a-.8*b+.4*a*a+.31*a*b-.2*b*b
    rho = 1+.2j+(.3-.1j)*a+(-.2+.15j)*b+(.11+.07j)*a*b+(.05-.09j)*a*a
    return dict(entrance_triangles=triangles, position_nodes=positions, phase_nodes=phase, weight_nodes=rho)


def test_duffy_power_matches_independent_p2_basis():
    rng = np.random.default_rng(17)
    values = rng.normal(size=(3, 6, 2))
    coefficients = _duffy_power(values)
    u, v = rng.uniform(size=(2, 31))
    l0, l1, l2 = (1-u)*(1-v), u, (1-u)*v
    basis = np.array([l0*(2*l0-1), l1*(2*l1-1), l2*(2*l2-1), 4*l0*l1, 4*l1*l2, 4*l2*l0])
    expected = np.einsum('tac,an->tnc', values, basis)
    result = sum(coefficients[:, i, j, None]*u[None, :, None]**i*v[None, :, None]**j for i in range(3) for j in range(3))
    np.testing.assert_allclose(result, expected, atol=3e-15)


def test_bernstein_bound_dominates_actual_nonlinear_derivatives_everywhere_sampled():
    mesh = nonlinear_mesh()
    receivers = np.array([[-.4, -.7], [-.4, .8], [.9, -.7], [.9, .8]])
    kappa = 11/1.7
    bounds = _derivative_bounds(mesh['position_nodes'], mesh['phase_nodes'], kappa, receivers)
    u, v = np.meshgrid(np.linspace(0, 1, 61), np.linspace(0, 1, 61))
    u, v = u.ravel(), v.ravel()
    for index, triangle in enumerate(mesh['entrance_triangles']):
        e1, e2 = triangle[1]-triangle[0], triangle[2]-triangle[0]
        q = triangle[0]+u[:, None]*e1+((1-u)*v)[:, None]*e2
        a, b = q[:, 0], q[:, 1]
        X = np.column_stack((a+.27*a*a-.13*a*b, .7*b+.21*a*b+.08*b*b))
        derivative_x = np.column_stack((1+.54*a-.13*b, .21*b))
        derivative_y = np.column_stack((-.13*a, .7+.21*a+.16*b))
        phase_x, phase_y = 1.2+.8*a+.31*b, -.8+.31*a-.4*b
        for target in np.r_[receivers, [[.2, .1]]]:
            gx = phase_x+kappa*np.sum((X-target)*derivative_x, axis=1)
            gy = phase_y+kappa*np.sum((X-target)*derivative_y, axis=1)
            gradient = np.column_stack((gx, gy))
            du = np.sum(gradient*(e1-v[:, None]*e2), axis=1)
            dv = (1-u)*np.sum(gradient*e2, axis=1)
            assert np.max(abs(du)) <= bounds[index, 0]+1e-12
            assert np.max(abs(dv)) <= bounds[index, 1]+1e-12


@pytest.mark.parametrize('negative_orientation', [False, True])
def test_affine_map_matches_existing_exit_integral_with_one_measure_factor(negative_orientation):
    pytest.importorskip('finufft')
    entrance = rectangle()
    q = np.einsum('ni,tic->tnc', BARY, entrance)
    matrix = np.array([[1.4, .3], [-.2, .8]])
    if negative_orientation:
        matrix[0] *= -1
    positions = q@matrix.T+[.2, -.3]
    H, b = np.array([[1.2, -.4], [-.4, .7]]), np.array([.6, -.8])
    phase = .31+.5*np.einsum('...i,ij,...j->...', positions, H, positions)+positions@b
    amplitude = .9+.2j+(.3-.1j)*positions[..., 0]+(-.2+.17j)*positions[..., 1]
    curved = dict(entrance_triangles=entrance, position_nodes=positions, phase_nodes=phase,
                  weight_nodes=amplitude*abs(np.linalg.det(matrix)))
    k = 11.
    affine = dict(triangles=positions[:, :3], vertex_phase=phase[:, :3],
                  vertex_amplitude=amplitude[:, :3], vertex_directions=(positions[:, :3]@H+b)/k)
    common = dict(k=k, distance=1.7, x=np.linspace(-.3, .4, 4), y=np.linspace(-.2, .5, 3),
                  cell_width=.08, pixel_order=4, eps=1e-13)
    expected = phase_field(affine, quadrature_order=40, backend='direct', **common)
    got = curved_phase_field(curved, safety=1.5, **common)
    np.testing.assert_allclose(got, expected, atol=5e-13, rtol=2e-10)


def test_nonlinear_map_against_independent_cartesian_integral_and_native_cell_mean():
    pytest.importorskip('finufft')
    common = dict(k=11., distance=1.7, x=np.linspace(-.2, .2, 3), y=np.array([-.1, .15]),
                  cell_width=.12, pixel_order=4, eps=1e-13)
    nodes, weights = np.polynomial.legendre.leggauss(42)
    a, b = np.meshgrid((nodes+1)/2, (nodes+1)/2)
    weight = np.outer(weights/2, weights/2)
    X = np.stack((a+.27*a*a-.13*a*b, .7*b+.21*a*b+.08*b*b), axis=-1)
    phi = .37+1.2*a-.8*b+.4*a*a+.31*a*b-.2*b*b
    rho = 1+.2j+(.3-.1j)*a+(-.2+.15j)*b+(.11+.07j)*a*b+(.05-.09j)*a*a
    expected = np.zeros((2, 3), complex)
    gn, gw = np.polynomial.legendre.leggauss(4)
    kappa = common['k']/common['distance']
    for iy, y in enumerate(common['y']):
        for ix, x in enumerate(common['x']):
            for i, u in enumerate(gn):
                for j, v in enumerate(gn):
                    target = np.array([x, y])+.5*common['cell_width']*np.array([u, v])
                    expected[iy, ix] += gw[i]*gw[j]/4*kappa/(2j*np.pi)*np.sum(weight*rho*np.exp(1j*(phi+.5*kappa*np.sum((X-target)**2, axis=-1))))
    got = curved_phase_field(nonlinear_mesh(), **common)
    np.testing.assert_allclose(got, expected, atol=5e-13, rtol=3e-10)


def test_degenerate_exit_position_zero_density_and_analytic_p2_mass():
    pytest.importorskip('finufft')
    mesh = nonlinear_mesh()
    mesh['position_nodes'][:] = [.27, -.31]
    mesh['phase_nodes'][:] = .47
    mesh['weight_nodes'][0] = 0
    k, distance, target = 11., 1.7, np.array([-.2, .13])
    # Integrals of vertex P2 basis are zero; edge basis integrates to area/3.
    mass = .5*np.sum(mesh['weight_nodes'][1, 3:])/3
    expected = k/(2j*np.pi*distance)*mass*np.exp(1j*(.47+.5*k/distance*np.sum((target-[.27, -.31])**2)))
    got = curved_phase_field(mesh, k=k, distance=distance, x=[target[0]], y=[target[1]], eps=1e-13)
    np.testing.assert_allclose(got.item(), expected, atol=2e-13, rtol=1e-11)


def test_complex_overlapping_branches_batched_and_translated_xray_coordinates(monkeypatch):
    pytest.importorskip('finufft')
    import finufft
    base = nonlinear_mesh()
    scale = 1e-5
    base['entrance_triangles'] *= scale
    base['position_nodes'] *= scale
    second = {name: value.copy() for name, value in base.items()}
    second['weight_nodes'] *= -.91+.13j
    second['phase_nodes'] += .071
    both = {name: np.concatenate((value, second[name])) for name, value in base.items()}
    original = {name: value.copy() for name, value in both.items()}
    common = dict(k=11/scale**2, distance=1.7, x=np.linspace(-2e-6, 2e-6, 4), y=np.array([-.1e-5, .15e-5]),
                  cell_width=.12e-5, pixel_order=3, eps=1e-13)
    expected = curved_phase_field(base, **common)+curved_phase_field(second, **common)
    plan_class, seen = finufft.Plan, []
    class Plan:
        def __init__(self, *args, **kwargs):
            self.plan = plan_class(*args, **kwargs)
        def setpts(self, x, y):
            assert len(x) <= 73
            seen.append(len(x))
            self.plan.setpts(x, y)
        def execute(self, strengths):
            assert strengths.shape[0] == 2
            return self.plan.execute(strengths)
    monkeypatch.setattr(finufft, 'Plan', Plan)
    got, stats = curved_phase_field(both, max_nodes_per_batch=73, receiver_channels_per_batch=2,
                                   return_stats=True, **common)
    np.testing.assert_allclose(got, expected, atol=5e-13, rtol=3e-10)
    assert seen and stats['maximum_source_batch'] <= 73 and stats['plans'] == 1
    for name in both:
        np.testing.assert_array_equal(original[name], both[name])
    monkeypatch.setattr(finufft, 'Plan', plan_class)
    shift = np.array([.002, -.003])
    both['position_nodes'] += shift
    both['entrance_triangles'] += [.005, -.004]  # Only integration measure matters.
    translated = curved_phase_field(both, **dict(common, x=common['x']+shift[0], y=common['y']+shift[1]))
    np.testing.assert_allclose(translated, expected, atol=4e-11, rtol=3e-9)


def test_order_bounds_invariant_to_global_phase_and_position_translation():
    mesh = nonlinear_mesh()
    common = dict(k=11., distance=1.7, x=[-.4, .3], y=[-.2, .5], cell_width=.2)
    before = curved_quadrature_orders(mesh, **common)
    mesh['phase_nodes'] += 120.
    mesh['position_nodes'] += [2.3, -4.1]
    after = curved_quadrature_orders(mesh, **dict(common, x=np.array(common['x'])+2.3, y=np.array(common['y'])-4.1))
    np.testing.assert_allclose(after['phase_derivative_bounds'], before['phase_derivative_bounds'], rtol=2e-13)
    np.testing.assert_array_equal(after['orders'], before['orders'])


def test_max_order_raises_and_safety_orders_increase():
    common = dict(k=110., distance=1.7, x=[-.4, .3], y=[-.2, .5])
    low = curved_quadrature_orders(nonlinear_mesh(), safety=1., **common)
    high = curved_quadrature_orders(nonlinear_mesh(), safety=2., **common)
    assert np.all(high['orders'] >= low['orders'])
    with pytest.raises(ValueError, match='do not silently truncate'):
        curved_quadrature_orders(nonlinear_mesh(), max_order=12, **common)


def test_empty_mesh_is_zero_without_plan():
    empty = dict(entrance_triangles=np.empty((0, 3, 2)), position_nodes=np.empty((0, 6, 2)),
                 phase_nodes=np.empty((0, 6)), weight_nodes=np.empty((0, 6), complex))
    field, stats = curved_phase_field(empty, k=1., distance=1., x=[1., 2.], y=[3.], return_stats=True)
    np.testing.assert_array_equal(field, np.zeros((1, 2)))
    assert stats['source_nodes'] == stats['plans'] == 0


@pytest.mark.parametrize('override', [dict(k=0), dict(distance=np.nan), dict(cell_width=-1), dict(pixel_order=0),
    dict(x=[0, 1, 2.1]), dict(y=[]), dict(safety=0), dict(min_order=1), dict(max_order=4), dict(eps=1),
    dict(max_nodes_per_batch=0), dict(receiver_channels_per_batch=0)])
def test_invalid_scalar_or_receiver_arguments(override):
    common = dict(k=11., distance=1.7, x=[0], y=[0])
    common.update(override)
    with pytest.raises(ValueError):
        curved_phase_field(nonlinear_mesh(), **common)


@pytest.mark.parametrize('fault', ['position_shape', 'phase_nan', 'rho_inf', 'entrance_degenerate'])
def test_invalid_mesh_raises(fault):
    mesh = nonlinear_mesh()
    if fault == 'position_shape':
        mesh['position_nodes'] = mesh['position_nodes'][:, :3]
    elif fault == 'phase_nan':
        mesh['phase_nodes'][0, 0] = np.nan
    elif fault == 'rho_inf':
        mesh['weight_nodes'][0, 0] = np.inf
    else:
        mesh['entrance_triangles'][0, 1] = mesh['entrance_triangles'][0, 0]
    with pytest.raises(ValueError):
        curved_phase_field(mesh, k=11., distance=1.7, x=[0], y=[0])

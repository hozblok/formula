"""Absolute normalization, receiver quadrature and finite-width limitations."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

from formula.capsysred.stages.method_safe._b5_field import (
    canonical_prefactor,
    gaussian_cell_average,
    reconstruct_field,
)


def test_stable_prefactor_matches_curvature_eigenvalues_and_focus_limit():
    rng = np.random.default_rng(38)
    Q = rng.normal(size=(25, 2, 2))
    H = rng.normal(size=Q.shape)
    H = (H + H.transpose(0, 2, 1))/2
    P = H @ Q
    maslov = rng.integers(0, 8, len(Q))
    got, valid = canonical_prefactor(Q, P, maslov, 0.7)
    expected = (np.sqrt(abs(np.linalg.det(Q)))
                * np.prod(np.sqrt(0.7+1j*np.linalg.eigvalsh(H)), axis=1)
                * np.exp(-0.5j*np.pi*maslov))
    assert valid.all()
    np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14)
    tiny = 1e-100
    Q = np.array([np.diag([tiny, 1]), np.diag([-tiny, 1]), np.diag([0, 1])])
    P = np.array([np.diag([-1, 0.2])]*3)
    got, valid = canonical_prefactor(Q, P, [0, 1, 1], 0.7)
    assert valid.tolist() == [True, True, False]
    np.testing.assert_allclose(got[0], got[1], rtol=1e-14)
    assert abs(got[0]) > 0.5


@pytest.mark.parametrize("offset,direction", [(0.13, 0.007), (-0.3, -0.017), (0.0, 0.15)])
def test_coherent_cell_is_complex_quadrature_not_center_or_intensity(offset, direction):
    k, width, cell = 500, 0.2, 0.11
    got = gaussian_cell_average(offset, direction, k, width, cell, order=32)
    def f(t):
        return np.exp(1j*k*direction*t-t*t/(2*width**2))/cell
    lo, hi = offset-cell/2, offset+cell/2
    expected = quad(lambda t: f(t).real, lo, hi, epsabs=1e-12)[0]
    expected += 1j*quad(lambda t: f(t).imag, lo, hi, epsabs=1e-12)[0]
    assert got == pytest.approx(expected, abs=2e-14)
    center = np.exp(1j*k*direction*offset-offset**2/(2*width**2))
    assert abs(got-center) > 1e-3


def _quadratic(width, magnification=1.3, maslov=0, constant=0.73, amplitude=0.8):
    q = (np.arange(141)+0.5)*(2/141)-1
    qx, qy = np.meshgrid(q, q)
    entrance = np.column_stack([qx.ravel(), qy.ravel()])
    points = magnification*entrance
    k, curvature = 50.0, 0.2
    Q = np.broadcast_to(np.eye(2)*magnification, (len(points), 2, 2))
    P = curvature*Q
    axis = np.array([-0.3, 0.0, 0.3])
    result = reconstruct_field(
        points, curvature*points, constant+0.5*k*curvature*np.sum(points**2, axis=1),
        1, Q, P, maslov, area_weights=(2/141)**2, source_amplitude=amplitude,
        k=k, width=width, x=axis, y=axis, cell_width=0, cutoff=7)
    xx, yy = np.meshgrid(axis, axis)
    expected = amplitude/abs(magnification)*np.exp(
        1j*constant+0.5j*k*curvature*(xx**2+yy**2)-0.5j*np.pi*maslov)
    return result, expected


@pytest.mark.parametrize("magnification,maslov", [(1.3, 0), (-1.3, 2)])
def test_quadratic_free_and_post_focus_absolute_field(magnification, maslov):
    coarse, expected = _quadratic(0.12, magnification, maslov)
    fine, _ = _quadratic(0.08, magnification, maslov)
    np.testing.assert_allclose(coarse["field"], expected, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(fine["field"], expected, atol=2e-10, rtol=2e-10)
    ref = coarse["ref_index"]
    assert coarse["self_cross"][ref] == pytest.approx(coarse["self_intensity"][ref])
    assert coarse["self_intensity"][ref] > 0


def test_unequal_channel_phases_amplitudes_do_not_cancel():
    first, e1 = _quadratic(0.1, amplitude=0.8, constant=0.15)
    second, e2 = _quadratic(0.1, amplitude=0.23, constant=1.7)
    expected = e1+e2
    actual = first["field"]+second["field"]
    np.testing.assert_allclose(actual, expected, atol=2e-10)
    incorrect = e1+np.abs(e2)*np.exp(0.15j)*np.exp(1j*np.angle(e1*np.exp(-0.15j)))
    assert np.max(abs(actual-incorrect)) > 0.2


def test_finite_aperture_has_width_dependent_smoothing_not_exact_identity():
    q = (np.arange(180)+0.5)/180-0.5
    qx, qy = np.meshgrid(q, q)
    points = np.column_stack([qx.ravel(), qy.ravel()])
    Q = np.broadcast_to(np.eye(2), (len(points), 2, 2))
    P = np.zeros_like(Q)
    axis = np.array([0.45, 0.5, 0.55])
    fields = []
    for width in [0.1, 0.05]:
        result = reconstruct_field(points, np.zeros_like(points), 0, 1, Q, P, 0,
            area_weights=1/180**2, source_amplitude=1, k=100, width=width,
            x=axis, y=np.array([-0.05, 0, 0.05]), cell_width=0, cutoff=7)
        expected_x = (erf((0.5-axis)/(np.sqrt(2)*width))-erf((-0.5-axis)/(np.sqrt(2)*width)))/2
        expected_y = erf(0.5/(np.sqrt(2)*width))
        np.testing.assert_allclose(result["field"][1], expected_x*expected_y, atol=1.3e-4)
        fields.append(result["field"][1])
    assert 0.3 < fields[0][2].real < 0.32
    assert 0.15 < fields[1][2].real < 0.17
    assert abs(fields[0][0]-1) > 0.3


def test_self_terms_equal_explicit_individual_ray_sum():
    points = np.array([[0.0, 0.0], [0.03, -0.05], [-0.08, 0.01]])
    directions = np.array([[0.01, 0.02], [0.005, -0.01], [0.01, -0.005]])
    phase = np.array([0.3, 1.7, -0.8])
    fresnel = np.array([1.0, 0.8j, 0.17*np.exp(0.4j)])
    Q = np.broadcast_to(np.eye(2), (3, 2, 2))
    P = Q*0.2
    common = dict(area_weights=0.003, source_amplitude=0.8, k=100, width=0.08,
                  x=np.linspace(-0.2, 0.2, 5), y=np.linspace(-0.2, 0.2, 5), cell_width=0.04)
    combined = reconstruct_field(points, directions, phase, fresnel, Q, P, 0, **common)
    individual = [reconstruct_field(points[i:i+1], directions[i:i+1], phase[i:i+1],
        fresnel[i:i+1], Q[i:i+1], P[i:i+1], 0, **common)["field"] for i in range(3)]
    ref = combined["ref_index"]
    np.testing.assert_allclose(combined["field"], np.sum(individual, axis=0), atol=1e-16)
    np.testing.assert_allclose(combined["self_intensity"], np.sum(np.abs(individual)**2, axis=0), atol=1e-16)
    np.testing.assert_allclose(combined["self_cross"], sum(v*v[ref].conjugate() for v in individual), atol=1e-16)


def test_automatic_cell_rule_resolves_rapid_phase():
    points = np.array([[0.07e-6, -0.05e-6]])
    directions = np.array([[0.01, -0.008]])
    Q = np.eye(2)[None]
    k, width, cell = 4e10, 1e-6, 0.3e-6
    result = reconstruct_field(points, directions, 0.7, 0.6j, Q, Q*0, 0,
        area_weights=1e-12, source_amplitude=0.8, k=k, width=width,
        x=np.array([-1e-6, 0, 1e-6]), y=np.array([-1e-6, 0, 1e-6]), cell_width=cell)
    rx = np.array([-1e-6, 0, 1e-6])-points[0, 0]
    ry = np.array([-1e-6, 0, 1e-6])-points[0, 1]
    gx = gaussian_cell_average(rx, directions[0, 0], k, width, cell, order=128)
    gy = gaussian_cell_average(ry, directions[0, 1], k, width, cell, order=128)
    expected = 0.8*0.6j*np.exp(0.7j)/(2*np.pi)*gy[:, None]*gx[None, :]
    np.testing.assert_allclose(result["field"], expected, atol=3e-16, rtol=3e-10)
    assert result["metadata"]["cell_quadrature_order"] >= 60


def test_rejects_invalid_grid_and_ray_parameters():
    p = np.zeros((1, 2))
    q = np.eye(2)[None]
    args = dict(area_weights=1, source_amplitude=1, k=100, width=0.1,
                x=[0, 1, 3], y=[0, 1], cell_width=0)
    with pytest.raises(ValueError, match="uniformly"):
        reconstruct_field(p, p, 0, 1, q, q, 0, **args)
    args.update(x=[0, 1], area_weights=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        reconstruct_field(p, p, 0, 1, q, q, 0, **args)


def test_continuous_branch_is_defined_on_an_exact_caustic():
    """Q = diag(1 - z, 1), P = V = diag(-1, 0): det Q vanishes at z = 1.  The continued root
    equals the Maslov branch on both sides and stays finite and continuous on the caustic."""
    from formula.capsysred.stages.variational_transport import det_phase_increment
    gamma = 0.7
    zs = np.array([0.5, 1.0-1e-9, 1.0, 1.0+1e-9, 1.5])
    P = np.repeat(np.diag([-1.0, 0.0])[None], len(zs), 0)
    Q0 = np.repeat(np.eye(2)[None], len(zs), 0)
    Q = Q0+zs[:, None, None]*P
    phase = np.angle(np.linalg.det(gamma*Q0+1j*P))+det_phase_increment(Q0, P, P, zs, gamma)[0]
    continued, valid = canonical_prefactor(Q, P, None, gamma, phase=phase, gamma_ref=gamma)
    maslov, valid_m = canonical_prefactor(Q, P, (zs > 1).astype(int), gamma)
    assert valid.all() and valid_m.tolist() == [True, True, False, True, True]
    np.testing.assert_allclose(continued[valid_m], maslov[valid_m], rtol=1e-12)
    assert abs(continued[2]-continued[1]) < 1e-8 and abs(continued[3]-continued[2]) < 1e-8


def test_complex_window_cell_average_is_complex_quadrature():
    k, width, cell, offset, direction = 500, 0.2, 0.11, 0.13, 0.007
    m = 1/(k*width**2)/(1+0.8j)
    got = gaussian_cell_average(offset, direction, k, width, cell, order=32, m=m)
    def f(t):
        return np.exp(1j*k*direction*t-0.5*k*m*t*t)/cell
    lo, hi = offset-cell/2, offset+cell/2
    expected = quad(lambda t: f(t).real, lo, hi, epsabs=1e-12)[0]+1j*quad(lambda t: f(t).imag, lo, hi, epsabs=1e-12)[0]
    assert got == pytest.approx(expected, abs=2e-14)


def test_exit_propagation_reproduces_the_quadratic_field():
    """Windows formed on a plane L before the receiver (Q there, centres and phases on the
    receiver) and propagated exactly give the exact field of a quadratic family."""
    k, c, M, width = 50.0, 0.2, 1.3, 0.1
    L = k*width**2                                     # the element widens by sqrt(2)
    q = (np.arange(141)+0.5)*(2/141)-1
    qx, qy = np.meshgrid(q, q)
    exit_points = M*np.column_stack([qx.ravel(), qy.ravel()])
    slopes = c*exit_points
    points = exit_points+L*slopes
    phase = 0.73+0.5*k*c*np.sum(exit_points**2, axis=1)+0.5*k*L*np.sum(slopes**2, axis=1)
    Q = np.broadcast_to(np.eye(2)*M, (len(points), 2, 2))
    axis = np.array([-0.3, 0.0, 0.3])
    result = reconstruct_field(points, slopes, phase, 1, Q, c*Q, 0, area_weights=(2/141)**2,
                               source_amplitude=0.8, k=k, width=width, x=axis, y=axis, cell_width=0,
                               cutoff=7, spread=L)
    xx, yy = np.meshgrid(axis, axis)
    expected = 0.8/(M*(1+c*L))*np.exp(0.73j+0.5j*k*c/(1+c*L)*(xx**2+yy**2))
    np.testing.assert_allclose(result["field"], expected, atol=1e-9, rtol=1e-9)
    assert result["metadata"]["effective_width_m"] == pytest.approx(width*np.sqrt(2))



def test_cell_order_resolves_the_chirp_of_a_spread_window():
    """A window propagated far develops the chirp k Im(m) t; the cell rule must resolve its
    slope at the farthest kept cell, not only the ray tilt and the curvature at the centre
    (16 nodes gave a relative error of 40 at one effective width)."""
    from formula.capsysred.stages.method_safe._b5_field import (
        cell_order,
        gaussian_cell_average,
        spread_window,
    )
    k, width, L, cell = 4e10, 1e-8, 0.1, 1e-6
    m, _ = spread_window(1/(k*width**2), L)
    weff = 1/np.sqrt(k*m.real)
    reach = 5*weff+cell/2
    order = cell_order(k, cell, 0.0, m, weff, reach)
    assert 16 < order <= 256
    f = lambda t: np.exp(-0.5*k*m*t*t)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    def composite(offset, parts=64):
        edges = np.linspace(offset-cell/2, offset+cell/2, parts+1)
        total = sum(0.5*(hi-lo)*np.sum(weights*f(0.5*(hi-lo)*nodes+0.5*(hi+lo)))
                    for lo, hi in zip(edges[:-1], edges[1:]))
        return total/cell
    for offset in (weff, 3*weff, 5*weff):
        got = gaussian_cell_average(offset, 0.0, k, width, cell, order, m)
        assert abs(got-composite(offset)) < 1e-12
    assert abs(gaussian_cell_average(weff, 0.0, k, width, cell, 16, m)-composite(weff)) > 0.1
    # the same rule inside reconstruct_field: one ray at the origin, cells at +-weff
    axis = np.array([-weff, 0.0, weff])
    result = reconstruct_field(np.zeros((1, 2)), np.zeros((1, 2)), 0.0, 1.0, np.eye(2)[None], np.zeros((1, 2, 2)),
                               0, area_weights=1.0, source_amplitude=1.0, k=k, width=width, x=axis, y=axis,
                               cell_width=cell, cutoff=5.0, spread=L)
    assert result["metadata"]["cell_quadrature_order"] == order
    field = result["field"]
    assert field[1, 2]/field[1, 1] == pytest.approx(composite(weff)/composite(0.0), rel=1e-9)

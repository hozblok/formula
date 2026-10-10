"""Independent normalization, edge jumps, affine terms and receiver checks."""

import numpy as np
import pytest
from scipy.integrate import quad

from formula.capsysred.stages._b9_contour import (
    contour_fourier, fresnel_from_chirped, prepare_contour, triangle_fourier,
)


SQUARE = np.array([[[-1., -1.], [1., -1.], [1., 1.]],
                   [[-1., -1.], [1., 1.], [-1., 1.]]])


def _quad_triangle(triangle, values, frequency, order=40):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    t, s = np.meshgrid(nodes, nodes, indexing="ij")
    bary = np.stack(((1-t)*(1-s), t, (1-t)*s), axis=-1)
    q = bary @ triangle
    u = bary @ values
    e1, e2 = triangle[1]-triangle[0], triangle[2]-triangle[0]
    determinant = abs(e1[0]*e2[1]-e1[1]*e2[0])
    return np.sum(weights[:, None]*weights[None, :]*(1-t)*determinant*u*np.exp(-1j*(q @ frequency)))


@pytest.mark.parametrize("backend", ["direct", "finufft"])
def test_rectangle_transform_exact_dc_near_dc_and_oscillations(backend):
    if backend == "finufft":
        pytest.importorskip("finufft")
    kx = np.linspace(-17, 17, 35)
    ky = np.linspace(-13, 13, 27)
    got = triangle_fourier(SQUARE, np.ones((2, 3)), kx, ky, backend=backend, eps=1e-13)
    exact = 4*np.sinc(kx/np.pi)[None, :]*np.sinc(ky/np.pi)[:, None]
    np.testing.assert_allclose(got, exact, rtol=0, atol=2e-12)
    small = triangle_fourier(SQUARE, np.ones((2, 3)), [0, 1e-15, .01], [0, -.001], backend="direct")
    exact = 4*np.sinc(np.array([0, 1e-15, .01])/np.pi)[None, :]*np.sinc(np.array([0, -.001])/np.pi)[:, None]
    np.testing.assert_allclose(small, exact, rtol=0, atol=4e-15)


@pytest.mark.parametrize("reverse", [False, True])
def test_affine_gradient_and_unequal_internal_edge_values(reverse):
    mesh = SQUARE[:, ::-1] if reverse else SQUARE.copy()
    values = (1 + (.2+.7j)*mesh[:, :, 0] + (.3-.5j)*mesh[:, :, 1])
    values[1] *= .7*np.exp(.4j)
    kx, ky = np.array([-.1, 1., 11.]), np.array([0., 1.3, -7.])
    got = triangle_fourier(mesh, values, kx, ky, backend="direct", edge_order=10)
    expected = np.array([[sum(_quad_triangle(t, u, [fx, fy]) for t, u in zip(mesh, values))
                          for fx in kx] for fy in ky])
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-13)
    constant_approx = triangle_fourier(mesh, np.repeat(values.mean(axis=1)[:, None], 3, axis=1), kx, ky, backend="direct")
    assert np.max(abs(got-constant_approx)) > .1
    averaged_wrong = triangle_fourier(mesh, np.ones((2, 3))*values.mean(), kx, ky, backend="direct")
    assert np.max(abs(got-averaged_wrong)) > .1


def test_translation_physical_scaling_and_noncentered_even_frequency_grid():
    pytest.importorskip("finufft")
    mesh = SQUARE*2.7e-6 + [71e-6, -19e-6]
    values = np.array([[1, 2+.3j, -.2j], [1+.2j, .3-.5j, 2.]])
    kx, ky = np.linspace(-1.7e6, 2.3e6, 14), np.linspace(2.3e5, 1.9e6, 12)
    direct = triangle_fourier(mesh, values, kx, ky, backend="direct", edge_order=10)
    nufft = triangle_fourier(mesh, values, kx, ky, backend="finufft", eps=1e-13, edge_order=10)
    np.testing.assert_allclose(nufft, direct, rtol=1e-10, atol=2e-23)
    expected = _quad_triangle(mesh[0], values[0], [kx[3], ky[4]]) + _quad_triangle(mesh[1], values[1], [kx[3], ky[4]])
    assert direct[4, 3] == pytest.approx(expected, rel=1e-12, abs=1e-24)


def test_mesh_diagonal_and_uniform_subdivision_preserve_affine_field():
    affine = lambda q: .7 + .2j + (.3+.4j)*q[..., 0] - .2*q[..., 1]
    opposite = np.array([[[-1., -1.], [1., -1.], [-1., 1.]], [[1., -1.], [1., 1.], [-1., 1.]]])
    refined = np.concatenate([(SQUARE+shift)/2 for shift in [[-1,-1], [1,-1], [-1,1], [1,1]]])
    kx, ky = np.arange(-4., 5.), np.arange(-3., 4.)
    reference = triangle_fourier(SQUARE, affine(SQUARE), kx, ky, backend="direct")
    for mesh in [opposite, refined, SQUARE[:, ::-1]]:
        got = triangle_fourier(mesh, affine(mesh), kx, ky, backend="direct")
        np.testing.assert_allclose(got, reference, rtol=3e-13, atol=2e-13)


@pytest.mark.parametrize("backend", ["direct", "finufft"])
def test_focus_normalization_and_coherent_pixel_average(backend):
    if backend == "finufft":
        pytest.importorskip("finufft")
    k, distance, halfwidth = 270., 1.3, .17
    mesh = SQUARE*halfwidth
    chirped_values = np.ones((2, 3), complex)
    x, y = np.linspace(-.31, .32, 7), np.linspace(-.27, .28, 5)
    got = fresnel_from_chirped(mesh, chirped_values, k=k, distance=distance, x=x, y=y, backend=backend, eps=1e-13)
    def axis(t):
        return np.exp(.5j*k/distance*t*t)*2*halfwidth*np.sinc(k*halfwidth*t/(distance*np.pi))
    exact = k/(2j*np.pi*distance)*axis(x)[None, :]*axis(y)[:, None]
    np.testing.assert_allclose(got, exact, rtol=2e-13, atol=2e-13)
    cell = .023
    averaged = fresnel_from_chirped(mesh, chirped_values, k=k, distance=distance, x=x, y=y,
                                    cell_width=cell, pixel_order=8, backend=backend, eps=1e-13)
    def average(t):
        real = quad(lambda a: axis(a).real, t-cell/2, t+cell/2, epsabs=1e-13)[0]
        imag = quad(lambda a: axis(a).imag, t-cell/2, t+cell/2, epsabs=1e-13)[0]
        return (real+1j*imag)/cell
    exact_average = k/(2j*np.pi*distance)*np.array([average(t) for t in x])[None, :]*np.array([average(t) for t in y])[:, None]
    np.testing.assert_allclose(averaged, exact_average, rtol=2e-12, atol=3e-13)
    assert np.max(abs(averaged-got)) > .01


@pytest.mark.parametrize("alpha", [-4., 0., 4.])
def test_quadratic_input_mesh_convergence_through_focus(alpha):
    kx, ky = np.array([-1.7, 0., 2.1]), np.array([-.9, 0., 1.3])
    def exact_axis(omega):
        f = lambda x: np.exp(1j*alpha*x*x-1j*omega*x)
        return quad(lambda x: f(x).real, -1, 1, epsabs=1e-13)[0] + 1j*quad(lambda x: f(x).imag, -1, 1, epsabs=1e-13)[0]
    exact = np.array([exact_axis(w) for w in ky])[:, None]*np.array([exact_axis(w) for w in kx])[None, :]
    errors = []
    for cells in [8, 16, 32]:
        edges = np.linspace(-1., 1., cells+1)
        origins = np.array([(x, y) for x in edges[:-1] for y in edges[:-1]])
        triangles = (SQUARE[None, :, :, :]+1)/cells + origins[:, None, None, :]
        triangles = triangles.reshape(-1, 3, 2)
        values = np.exp(1j*alpha*np.sum(triangles**2, axis=2))
        got = triangle_fourier(triangles, values, kx, ky, backend="direct", edge_order=8)
        errors.append(np.linalg.norm(got-exact))
    if alpha == 0:
        assert max(errors) < 1e-12
    else:
        assert errors[2] < .28*errors[1] < .08*errors[0]


def test_edge_quadrature_convergence_for_oscillatory_affine_patch():
    kx, ky = [48.7], [-37.1]
    values = np.array([[1., 2+.4j, -.7j], [.3j, -.5, 1.2]])
    exact = sum(_quad_triangle(t, u, [kx[0], ky[0]], order=100) for t, u in zip(SQUARE, values))
    errors = []
    for order in [2, 4, 8]:
        got = triangle_fourier(SQUARE, values, kx, ky, edge_order=order, phase_step=4, backend="direct")[0, 0]
        errors.append(abs(got-exact))
    assert errors[2] < 1e-13
    assert errors[2] < errors[1] < errors[0]


def test_prepared_bounds_and_invalid_inputs():
    prepared = prepare_contour(SQUARE, np.ones((2, 3)), max_frequency=[1, 2])
    assert prepared.stats["area"] == pytest.approx(4)
    with pytest.raises(ValueError, match="exceed"):
        contour_fourier(prepared, [1.1], [0])
    with pytest.raises(ValueError, match="zero extent"):
        triangle_fourier(np.zeros((1, 3, 2)), np.ones((1, 3)), [0], [0])
    with pytest.raises(ValueError):
        fresnel_from_chirped(SQUARE, np.ones((2, 3)), k=1, distance=0, x=[0], y=[0])

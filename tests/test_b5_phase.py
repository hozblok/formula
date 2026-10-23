"""B5 scalar action reconstruction and physically consistent pair phases."""

import numpy as np
import pytest

from formula.capsysred.stages._b5_phase import fit_phase_gradient


def _grid(n=7):
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    return np.stack((x, y), axis=-1).reshape(-1, 2)


@pytest.mark.parametrize("degree", [2, 3, 4, 5])
def test_exact_polynomial_gradients_and_anchored_phase(degree):
    center, scale = np.array([7e-5, -2e-5]), np.array([2.4e-5, 1.1e-5])
    points = center + _grid() * scale
    anchor = center + np.array([0.17, -0.21]) * scale

    def phase(p):
        x, y = ((p - center) / scale).T
        return 0.7 * x + 0.2 * y + x * y + 0.5 * x ** 2 + 0.3 * y ** degree + 0.6 * x ** degree

    def grad(p):
        x, y = ((p - center) / scale).T
        return np.stack((0.7 + y + x + 0.6 * degree * x ** (degree - 1),
                         0.2 + x + 0.3 * degree * y ** (degree - 1)), axis=-1) / scale

    model = fit_phase_gradient(points, grad(points), degree=degree,
                               anchor_point=anchor, anchor_phase=2.3)
    queries = center + np.random.default_rng(1234).uniform(-0.9, 0.9, (31, 2)) * scale
    np.testing.assert_allclose(model.phase(queries), phase(queries) - phase(anchor[None, :])[0] + 2.3,
                               rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(model.gradient(queries), grad(queries), rtol=2e-12, atol=1e-8)
    assert model.diagnostics["full_rank"]
    assert model.diagnostics["gradient_relative_l2"] < 2e-14
    assert np.isfinite(model.diagnostics["condition"])
    assert model.phase(anchor) == pytest.approx(2.3)


def test_nonintegrable_observations_leave_a_measurable_residual():
    points = _grid(9)
    # Curl of (-y, x) is 2; no scalar action has this gradient.
    gradients = np.stack((-points[:, 1], points[:, 0]), axis=-1)
    model = fit_phase_gradient(points, gradients, degree=5)
    assert model.diagnostics["full_rank"]
    assert model.diagnostics["gradient_relative_l2"] > 0.65
    assert model.diagnostics["gradient_rms"] > 0.4


def test_pair_multiplier_preserves_psd_and_exact_cubic_residual():
    points = _grid(5)
    x, y = points.T
    gradients = np.stack((-x + x ** 2 + 0.6 * x * y, 0.3 * x ** 2 + 0.8 * y ** 3), axis=-1)
    model = fit_phase_gradient(points, gradients, degree=4)
    rng = np.random.default_rng(311)
    modes = rng.normal(size=(len(points), 4)) + 1j * rng.normal(size=(len(points), 4))
    rho = modes @ modes.conj().T
    multiplier = model.pair_multiplier(points)
    output = multiplier * rho
    diagonal = np.exp(1j * model.phase(points))
    np.testing.assert_allclose(output, diagonal[:, None] * rho * diagonal.conj()[None, :], atol=1e-13)
    np.testing.assert_allclose(output, output.conj().T, atol=1e-13)
    assert np.linalg.eigvalsh(output).min() > -1e-12
    np.testing.assert_allclose(np.diag(output), np.diag(rho))

    chord = points[:, None, :] - points[None, :, :]
    midpoint = (points[:, None, :] + points[None, :, :]) / 2
    corrected = model.residual_multiplier(points, baseline_gradient=model.gradient)
    dx, dy = chord[..., 0], chord[..., 1]
    expected_delta = dx ** 3 / 12 + 0.3 * dx ** 2 * dy / 4 + 0.2 * midpoint[..., 1] * dy ** 3
    np.testing.assert_allclose(corrected, np.exp(1j * expected_delta), atol=2e-14)
    classical = np.exp(1j * np.sum(chord * model.gradient(midpoint), axis=-1))
    np.testing.assert_allclose(classical * corrected, multiplier, atol=2e-14)


def test_residual_uses_actual_baseline_not_the_fitted_kick():
    points = _grid(4)
    model = fit_phase_gradient(points, np.stack((points[:, 0] ** 2, points[:, 1]), axis=-1))
    chord = points[:, None, :] - points[None, :, :]
    midpoint = (points[:, None, :] + points[None, :, :]) / 2
    actual = model.gradient(midpoint) + np.array([0.73, -0.41])
    baseline = np.exp(1j * np.sum(chord * actual, axis=-1))
    residual = model.residual_multiplier(points, baseline_gradient=actual)
    np.testing.assert_allclose(baseline * residual, model.pair_multiplier(points), atol=1e-14)
    implicit = model.residual_multiplier(points, baseline_gradient=model.gradient)
    assert np.max(np.abs(baseline * implicit - model.pair_multiplier(points))) > 1.5


def test_unequal_channel_anchors_change_interference():
    points = _grid()
    gradients = 0.2 * points
    first = fit_phase_gradient(points, gradients, degree=2, anchor_phase=0)
    second = fit_phase_gradient(points, gradients, degree=2, anchor_phase=np.pi)
    e1, e2 = np.exp(1j * first.phase(points)), np.exp(1j * second.phase(points))
    np.testing.assert_allclose(np.abs(e1 + e2) ** 2, 0.0, atol=1e-28)
    np.testing.assert_allclose(np.abs(e1 + e1) ** 2, 4.0, atol=2e-15)
    np.testing.assert_allclose(first.pair_multiplier(points), second.pair_multiplier(points))


def test_rank_deficiency_is_reported():
    points = np.array([[-1, 0], [0, 0], [1, 0]], dtype=float)
    model = fit_phase_gradient(points, points, degree=5)
    assert not model.diagnostics["full_rank"]
    assert model.diagnostics["rank"] < model.diagnostics["n_coefficients"]
    assert model.diagnostics["condition"] == float("inf")
    np.testing.assert_allclose(model.gradient(points), points, atol=1e-14)


@pytest.mark.parametrize("kwargs", [
    {"degree": 1}, {"degree": 6}, {"degree": 3.0}, {"degree": True},
    {"anchor_phase": np.nan}, {"anchor_phase": [0]},
    {"anchor_point": [0, np.inf]}, {"anchor_point": [[0, 0]]},
])
def test_invalid_fit_options(kwargs):
    points = _grid()
    with pytest.raises(ValueError):
        fit_phase_gradient(points, points, **kwargs)


@pytest.mark.parametrize("points, gradients", [
    ([], []), ([1, 2], [1, 2]), ([[0, 1, 2]], [[0, 1, 2]]),
    ([[0, np.nan]], [[1, 2]]), ([[0, 1]], [[1, np.inf]]),
    ([[0, 1]], [[1, 2], [3, 4]]), ([[0, 1j]], [[1, 2]]),
])
def test_invalid_observations(points, gradients):
    with pytest.raises(ValueError):
        fit_phase_gradient(points, gradients)


def test_pair_shapes_and_invalid_baseline():
    points = _grid(4)
    model = fit_phase_gradient(points, points)
    assert model.pair_multiplier(points[:3], points[:5]).shape == (3, 5)
    assert model.gradient(points.reshape(4, 4, 2)).shape == (4, 4, 2)
    with pytest.raises(ValueError, match="left"):
        model.pair_multiplier(points[0])
    with pytest.raises(ValueError, match="baseline_gradient"):
        model.residual_multiplier(points, baseline_gradient=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        model.residual_multiplier(points, baseline_gradient=[np.nan, 0])
    with pytest.raises(TypeError, match="baseline_gradient"):
        model.residual_multiplier(points)

"""Independent fold action test; no field amplitudes or caustic wave integral."""

import numpy as np
import pytest

from formula.capsysred.stages._b5_phase import fit_phase_gradient
from formula.capsysred.stages.stage17 import ray_chart


@pytest.mark.parametrize("chart", ["xx", "px", "xp", "pp"])
def test_fold_needs_first_momentum_chart_and_keeps_absolute_action(chart):
    a, b, k, constant = 1e-5, 1e-4, 4e10, 0.73
    distance = a / b
    parameters = np.random.default_rng(17).uniform(-1, 1, (180, 2))
    t, s = parameters.T
    points = np.column_stack((a * t**2, a * s))
    directions = b * parameters
    phases = constant + k * a * b * (2 * t**3 / 3 + s**2 / 2)
    train, test = np.arange(120), np.arange(120, 180)
    assert not np.intersect1d(train, test).size

    coordinates, gradients, action = ray_chart(
        points, directions, phases, k, distance, chart)
    expected = constant + k * a * b * (
        (-1 / 3 if chart[0] == "p" else 2 / 3) * t**3
        + (-1 / 2 if chart[1] == "p" else 1 / 2) * s**2)
    np.testing.assert_allclose(action, expected, atol=2e-14, rtol=2e-14)
    model = fit_phase_gradient(
        coordinates[train], gradients[train], degree=3,
        anchor_point=coordinates[train[0]], anchor_phase=action[train[0]])
    assert model.diagnostics["full_rank"]
    assert model.phase(coordinates[train[0]]) == pytest.approx(action[train[0]])

    fitted_action = model.phase(coordinates[test])
    physical_phase = fitted_action.copy()
    for axis, variable in enumerate(chart):
        if variable == "p":
            physical_phase += k * directions[test, axis] * points[test, axis]
    np.testing.assert_allclose(
        physical_phase - phases[test], fitted_action - action[test], atol=2e-14)

    error = np.max(np.abs(physical_phase - phases[test]))
    if chart[0] == "p":
        assert error < 1e-11
        assert model.diagnostics["gradient_relative_l2"] < 1e-12
        assert model.phase(np.zeros(2)) == pytest.approx(constant, abs=1e-11)
    else:
        assert error > 20
        assert model.diagnostics["gradient_relative_l2"] > 0.5

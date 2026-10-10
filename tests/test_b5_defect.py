"""Stage 17 defect indicator: entrance kernel, analytic residual of the frozen windows
against finite differences, and the exit tails of a straight ray."""
import math

import numpy as np
import pytest

from formula.capsysred.stages import _b5_defect as D

LAM = 12.398419843320026e-10 / 8.048
K = 2.0 * math.pi / LAM


def test_entrance_defect_scales_with_the_window():
    a_w = 24e-6
    d1, d2 = D.entrance_defect(a_w, 0.5e-6), D.entrance_defect(a_w, 2e-6)
    assert 0.0 < d1 < d2 < 0.5
    assert d2 / d1 == pytest.approx(2.0, rel=0.3)          # ~ sqrt(w / a)


def _one_ray_scene():
    from formula.capsysred.simulation import Simulation
    raw = {"precision": 32, "energy_kev": 8.048, "seed": 1,
           "capillary": {"bores": [{"center": [0.0, 0.0], "radius": 5e-6}], "z0": 0.0, "z1": 0.2,
                         "source": {"shape": "gaussian", "size": 1e-7, "position": [0.0, 0.0, -0.5],
                                    "n_modes": 1, "n_rays": 1},
                         "screen": {"z": 0.3, "center": [0.0, 0.0], "edge_x": 1e-5, "edge_y": 1e-5, "nx": 10, "ny": 10}},
           "b5_estimator": {"provider": "archive_canonical", "screen_index": 0}}
    sim = Simulation.from_dict(raw)
    return sim


def test_residual_matches_finite_differences():
    """One straight ray from the source: i d/dz psi + Lap psi / 2k of the frozen window
    (analytic prefactor derivative) against central differences of the window itself."""
    sim = _one_ray_scene()
    bores = sim.cfg.raw["capillary"]["bores"]
    origin = [0.0, 0.0, -0.5]
    slope = 1.5e-6 / 0.7
    rows = [(1.5e-6, 0.0, slope / math.sqrt(1 + slope * slope), 0.0, 0.7 * math.sqrt(1 + slope * slope), [], [])]
    seg = D.build_segments(rows, origin, 0.2, bores, 0.0, K, (2 * sim.delta_f, 2 * sim.beta_f))
    assert seg.p0.shape == (1, 3) and seg.final[0] and seg.valid[0]
    width = 1e-6
    z = 0.1
    x = np.linspace(-2e-6, 2e-6, 9) + slope * (z + 0.5) * 0.0 + 1.5e-6 * (z + 0.5) / 0.7
    y = np.linspace(-2e-6, 2e-6, 9)
    px, py = np.meshgrid(x, y)
    hx, hz = 2e-9, 2e-3                        # hz >> roundoff of the k * 0.6 m phase
    idx = np.array([0])
    def psi(zz, dx=0.0, dy=0.0):
        return D.window_values(seg, idx, zz, px.reshape(1, -1) + dx, py.reshape(1, -1) + dy, K, width, origin[2])[0].reshape(px.shape)
    base = psi(z)
    dz = (psi(z + hz) - psi(z - hz)) / (2 * hz)
    lap = (psi(z, dx=hx) + psi(z, dx=-hx) + psi(z, dy=hx) + psi(z, dy=-hx) - 4 * base) / (hx * hx)
    fd = 1j * dz + lap / (2 * K)
    analytic = D.window_values(seg, idx, z, px.reshape(1, -1), py.reshape(1, -1), K, width, origin[2], residual=True)[0].reshape(px.shape)
    scale = np.max(np.abs(base)) / (K * width * width)
    assert np.max(np.abs(fd - analytic)) < 5e-4 * scale


def test_exit_tails_and_trace_of_a_straight_ray():
    sim = _one_ray_scene()
    bores = sim.cfg.raw["capillary"]["bores"]
    origin = [0.0, 0.0, -0.5]
    rows = [(1e-6, 0.0, 1e-6 / 0.7, 0.0, 0.7, [], [])]
    seg = D.build_segments(rows, origin, 0.2, bores, 0.0, K, (2 * sim.delta_f, 2 * sim.beta_f))
    assert D.exit_tails(seg, bores, 5e-6, 0.2, K, 1e-6, origin[2], 0.5e-6, 4.0) == (0.0, 0.0)
    g, g_noise = D.wall_trace(seg, bores, 5e-6, np.array([0.05, 0.15]), K, 1e-6, origin[2], 0.5e-6, 4.0)
    assert g.shape == (2,) and np.all(g >= 0.0) and np.allclose(g_noise, g)   # one ray: all noise
    r, r_noise = D.volume_residual(seg, bores, 5e-6, np.array([0.1]), K, 1e-6, origin[2], 300, 4.0)
    assert r.shape == (1,) and r[0] > 0.0 and r_noise[0] == r[0]

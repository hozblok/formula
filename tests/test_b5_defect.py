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


def test_glass_form_cancels_a_window_pair_with_its_fresnel_coefficient():
    """An incident window and its mirror image times r on the wall x = a: the glass form
    vanishes up to the envelope gradient, the Dirichlet trace is |1 + r|."""
    sim = _one_ray_scene()
    fresnel = (2 * sim.delta_f, 2 * sim.beta_f)
    a, s, width, x_c = 5e-6, 4e-5, 50e-6, 4e-6
    root = np.sqrt(complex(s * s - fresnel[0], fresnel[1]))
    r = (s - root) / (s + root)
    pre = (np.array([[x_c, 0.0], [2 * a - x_c, 0.0]]), np.array([[s, 0.0], [-s, 0.0]]), np.array([1.0, r]), np.zeros(2))
    y = np.linspace(-20e-6, 20e-6, 9)
    px, py = np.full((2, y.size), a), np.broadcast_to(y, (2, y.size))
    normal = (np.ones(px.shape), np.zeros(px.shape))
    plain = D.window_values(None, np.arange(2), 0.0, px, py, K, width, 0.0, pre=pre)
    glass = D.window_values(None, np.arange(2), 0.0, px, py, K, width, 0.0, pre=pre, normal=normal, fresnel=fresnel)
    scale = np.max(np.abs(plain[0]))
    assert np.max(np.abs(plain.sum(axis=0))) == pytest.approx(abs(1 + r) * scale, rel=1e-6)
    assert np.max(np.abs(glass.sum(axis=0))) < 0.05 * abs(1 + r) * scale
    # the normal derivative against central differences
    h = 1e-9
    fd = (D.window_values(None, np.arange(2), 0.0, px + h, py, K, width, 0.0, pre=pre)
          - D.window_values(None, np.arange(2), 0.0, px - h, py, K, width, 0.0, pre=pre)) / (2 * h)
    k_g = K * np.sqrt(s * s - fresnel[0] + 1j * fresnel[1])
    np.testing.assert_allclose(glass, plain - fd / (1j * k_g), rtol=1e-5, atol=1e-5 * scale)


def test_outside_tails_of_a_final_window_match_direct_integration():
    sim = _one_ray_scene()
    bores = sim.cfg.raw["capillary"]["bores"]
    origin = [0.0, 0.0, -0.5]
    rows = [(4.2e-6 * 0.7 / 0.5, 0.0, 4.2e-6 / 0.5, 0.0, 0.7, [], [])]     # 4.2 um off axis at the exit z = 0.2
    seg = D.build_segments(rows, origin, 0.2, bores, 0.0, K, (2 * sim.delta_f, 2 * sim.beta_f))
    full, shot = D.exit_tails(seg, bores, 5e-6, 0.2, K, 1e-6, origin[2], 0.1e-6, 5.0, outside=True)
    assert full == shot > 0.0
    h = 0.02e-6
    x = np.arange(-12e-6, 12e-6, h) + 0.5 * h
    px, py = np.meshgrid(x, x)
    vals = D.window_values(seg, np.array([0]), 0.2, px.reshape(1, -1), py.reshape(1, -1), K, 1e-6, origin[2])[0]
    direct = float(np.sum(np.abs(vals) ** 2 * (np.hypot(px, py).ravel() > 5e-6)) * h * h)
    assert full == pytest.approx(direct, rel=1e-2)


def test_mask_of_a_stage17_map(tmp_path):
    I = np.array([[4.0, 1.0], [0.25, -1.0]])
    rays = np.array([[3, 1], [0, 0]])
    path = tmp_path / "map.npz"
    np.savez(path, I=I, matched_stage14_rays=rays, ref_index=np.array([0, 0]), x=np.arange(2.0), y=np.arange(2.0))
    summary = D.write_map_mask(path, 0.1, 0.35, tmp_path / "mask.npz")
    assert summary == {"cells": 4, "lit": 2, "masked": 3, "masked_lit": 1}
    data = np.load(tmp_path / "mask.npz")
    eta = 0.1 * np.sqrt(5.25 / 2)
    assert data["dmu"][0, 1] == pytest.approx(2 * eta * (1.0 + 0.5))
    assert data["masked"].tolist() == [[False, True], [True, True]]


def test_indicator_noise_rescales_to_the_masked_maps_budget():
    # a 100-ray measurement with R_shot = 1: eps 1 at a 100-ray map, 0.1 at 10000 rays
    terms = {"R": 0.0, "R_shot": 1.0, "R_inside": 0.0, "R_inside_shot": 1.0,
             "G": 0.0, "G_shot": 0.0, "T": 0.0, "T_shot": 0.0}
    assert D.indicator_at_budget(terms, 0.0, 100, 100, False) == pytest.approx(1.0)
    assert D.indicator_at_budget(terms, 0.0, 100, 10000, False) == pytest.approx(0.1)
    assert D.indicator_at_budget(terms, 0.2, 100, 10000, True) == pytest.approx(0.3)


def test_defect_diagnostics_refuse_a_continuous_branch_config(tmp_path):
    # the diagnostics drop the prefactor on an exact caustic (Maslov branch); a
    # continuous-branch map is another field there, so the scan refuses up front
    from formula.capsysred.simulation import Simulation
    raw = dict(_one_ray_scene().cfg.raw)
    raw["b5_estimator"] = dict(raw["b5_estimator"], branch="continuous")
    sim = Simulation.from_dict(raw)
    with pytest.raises(ValueError, match="continuous"):
        D.run_defect(sim, str(tmp_path / "no-archive"), str(tmp_path), D.options(sim.cfg), [1e-6])


def test_map_eps_uses_the_snapshots_mode_prefix():
    # a snapshot map of the first mode: its eps is that mode's indicator, not the
    # scan-wide rms; more modes than diagnosed is an error
    results = [{"rays": 100, "widths": {"1": {"R": 0.0, "R_shot": 0.0, "R_inside": 0.0, "R_inside_shot": 0.0,
                                               "G": g, "G_shot": 0.0, "T": 0.0, "T_shot": 0.0}}}
               for g in (0.02, 0.0, 0.0, 0.0)]
    assert D.map_eps(results, "1", 0.0, 100, 1, False) == pytest.approx(0.02)
    assert D.map_eps(results, "1", 0.0, 100, 4, False) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="5 modes"):
        D.map_eps(results, "1", 0.0, 100, 5, False)

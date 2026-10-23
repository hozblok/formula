"""Stage 16 provider `march`: config contract, agreement with the exact image sum on an ideal
channel, a roughness series that is reproducible and changes the field, a width step, a circular
bore against `fb`-free expectations (power stays inside), and the far-field files."""

import json
import math

import numpy as np
import pytest

from formula.capsysred import Simulation
from formula.capsysred.screen import ScreenGrid
from formula.capsysred.stages import wave as W
from formula.capsysred.stages import wave_images as WI
from formula.capsysred.stages import wave_march as WM

LAM = 12.398419843320026e-10 / 8.048
K = 2.0 * math.pi / LAM


def _tiny(provider="march", bores=None, length=1.0e-3, **wave):
    """Si channel 2 x 2 um x 1 mm, point source 0.5 mm before the entrance, screen on the exit face."""
    m = wave.pop("march", {})
    march = None if m is None else {"dz": 1.0e-6, "box": 1.0e-6, "sponge": 0.5e-6, "dx": 5.0e-9, "defects": [], **m}
    return {"precision": 32, "energy_kev": 8.048, "material": "silicon",
            "capillary": {"bores": bores or [{"center": [0.0, 0.0], "radius": 1.0e-6, "sides": 4, "rotation_deg": 0}],
                          "z0": 0.0, "z1": length,
                          "source": {"shape": "point", "size": 0.0, "position": [-1.0e-6, 0.0, -5.0e-4],
                                     "n_modes": 1, "n_rays": 1},
                          "screen": {"z": length, "center": [0.0, 0.0], "edge_x": 2.0e-6, "edge_y": 2.0e-6,
                                     "nx": 11, "ny": 11, "reference": [0.0, 0.0]}},
            "wave_estimator": {"provider": provider, "observable": "point", "grid_dx": 5.0e-9, "march": march, **wave}}


def _solver(raw, cls, **extra):
    sim = Simulation.from_dict(raw)
    wave = dict(sim.cfg.validate_wave_estimator(), **extra)
    cap = sim.cfg.capillary
    theta = W._theta_max(cap, np.array([[-1.0e-6, 0.0]]), [ScreenGrid(cap.screen)])
    z_in = float(cap.z0) - float(cap.source.position[2])
    length = float(cap.z1) - float(cap.z0)
    thc = math.sqrt(2.0 * sim.delta_f)
    if cls is WI.ImageSolver:
        return sim, WI.ImageSolver(cap.bores[0], z_in, length, theta, LAM, wave, theta_c_max=thc)
    return sim, WM.MarchSolver(cap.bores[0], z_in, length, theta, LAM, wave)


def _centroid(u, xs):
    p = np.abs(u) ** 2
    return float(np.sum(p * xs[:, None]) / np.sum(p))


def test_config_contract():
    w = Simulation.from_dict(_tiny()).cfg.validate_wave_estimator()
    assert w["provider"] == "march" and w["march"]["dz"] == 1e-6 and w["march"]["defects"] == []
    with pytest.raises(ValueError, match="march"):
        Simulation.from_dict(_tiny(march=None)).cfg.validate_wave_estimator()
    for bad in ({"dz": 0}, {"dz": 1e-6, "box": -1}, {"dz": 1e-6, "steps": 3}, {"dz": 1e-6, "defects": 5},
                {"dz": 1e-6, "defects": [{"type": "hole"}]}, {"dz": 1e-6, "defects": [{"type": "roughness", "sigma": 1e-9}]},
                {"dz": 1e-6, "defects": [{"type": "roughness", "sigma": 1e-9, "lc": 1e-7, "seed": 1, "walls": ["z+"]}]},
                {"dz": 1e-6, "defects": [{"type": "step", "z": 1e-4, "dw": [1e-9]}]}):
        raw = _tiny()
        raw["wave_estimator"]["march"] = bad
        with pytest.raises(ValueError, match="march"):
            Simulation.from_dict(raw).cfg.validate_wave_estimator()
    raw = _tiny(march={"defects": [{"type": "roughness", "sigma": 1e-9, "lc": 1e-7, "seed": 3, "walls": ["x-", "x+"]},
                                   {"type": "step", "z": 5e-4, "dw": [5e-9, 0.0]}, {"type": "slope", "deg": 3},
                                   {"type": "rounded", "rc": 4e-8}, {"type": "taper", "dw": [1e-8, 1e-8]},
                                   {"type": "drift", "dxy": [1e-8, 0.0]}]})
    assert len(Simulation.from_dict(raw).cfg.validate_wave_estimator()["march"]["defects"]) == 6
    bent = _tiny(bores=[{"center": [0.0, 0.0], "radius": 1e-6, "bend": {"radius": 1.0, "toward": [1.0, 0.0]}}])
    sim = Simulation.from_dict(bent)
    with pytest.raises(ValueError, match="provider march"):
        W.capillary_provider(sim.cfg.capillary, sim.cfg.validate_wave_estimator())


def test_ideal_channel_converges_to_the_image_sum():
    """First order in dz: the wall power loss of the split step approaches the exact image sum as dz halves."""
    sim, img = _solver(_tiny(provider="images", length=5.0e-4), WI.ImageSolver)
    u_i = img.exit_field((-1.0e-6, 0.0), K, sim.delta_f, sim.beta_f, -1)
    power = lambda u: float(np.sum(np.abs(u) ** 2))
    ratios = []
    for dz in (5.0e-7, 2.5e-7):
        _, mar = _solver(_tiny(length=5.0e-4, march={"dz": dz, "box": 0.6e-6, "sponge": 0.3e-6}), WM.MarchSolver)
        assert mar.separable and not mar.z_dependent and mar.n_steps == round(5.0e-4 / dz)
        assert np.allclose(mar.xs, img.xs) and np.allclose(mar.ys, img.ys)
        u_m = mar.exit_field((-1.0e-6, 0.0), K, sim.delta_f, sim.beta_f, -1)
        ratios.append(power(u_m) / power(u_i))
        assert abs(_centroid(u_m, mar.xs) - _centroid(u_i, img.xs)) < 0.03e-6
        assert abs(np.vdot(u_i, u_m)) / math.sqrt(power(u_i) * power(u_m)) > 0.99
    assert abs(ratios[1] - 1.0) < 0.6 * abs(ratios[0] - 1.0) and abs(ratios[1] - 1.0) < 0.03


def test_roughness_is_reproducible_and_a_step_moves_the_field():
    rough = {"defects": [{"type": "roughness", "sigma": 3e-9, "lc": 1e-7, "seed": 7}]}
    sim, a = _solver(_tiny(march=rough), WM.MarchSolver)
    _, b = _solver(_tiny(march=rough), WM.MarchSolver)
    assert a.z_dependent and np.array_equal(a.offsets, b.offsets) and a.offsets.std() == pytest.approx(3e-9, rel=0.05)
    _, c = _solver(_tiny(march={"defects": [{"type": "roughness", "sigma": 3e-9, "lc": 1e-7, "seed": 8}]}), WM.MarchSolver)
    assert not np.array_equal(a.offsets, c.offsets)
    _, ideal = _solver(_tiny(), WM.MarchSolver)
    u0 = ideal.exit_field((-1.0e-6, 0.0), K, sim.delta_f, sim.beta_f, -1)
    ua = a.exit_field((-1.0e-6, 0.0), K, sim.delta_f, sim.beta_f, -1)
    assert np.abs(ua - u0).max() > 1e-3 * np.abs(u0).max()
    _, st = _solver(_tiny(march={"defects": [{"type": "step", "z": 5e-4, "dw": [2e-7, 0.0]}]}), WM.MarchSolver)
    xl, xr, yl, yr = st._walls(st.n_steps)
    assert xr - xl == pytest.approx(2.2e-6) and yr - yl == pytest.approx(2.0e-6)
    assert st._walls(0)[1] - st._walls(0)[0] == pytest.approx(2.0e-6)
    us = st.exit_field((-1.0e-6, 0.0), K, sim.delta_f, sim.beta_f, -1)
    assert float(np.sum(np.abs(us) ** 2)) > 0.5 * float(np.sum(np.abs(u0) ** 2))


def test_shapes_fill_fractions():
    _, sl = _solver(_tiny(march={"defects": [{"type": "slope", "deg": 3}, {"type": "rounded", "rc": 1e-7}]}), WM.MarchSolver)
    assert not sl.separable and sl._static is not None
    f = sl.fill(0)
    assert f.min() == 0.0 and f.max() == 1.0 and 0.0 < f.mean() < 1.0
    area = float(f.sum()) * sl.h * sl.h
    assert 0.9 * 4e-12 < area < 1.05 * 4e-12                   # near the 2 x 2 um square, corners rounded
    _, circ = _solver(_tiny(bores=[{"center": [0.0, 0.0], "radius": 1.0e-6}]), WM.MarchSolver)
    assert circ.circular
    fc = circ.fill(0)
    assert float(fc.sum()) * circ.h * circ.h == pytest.approx(math.pi * 1e-12, rel=2e-3)
    _, ell = _solver(_tiny(bores=[{"center": [0.0, 0.0], "radius": 1.0e-6}],
                           march={"defects": [{"type": "ellipse", "ratio": 0.5}]}), WM.MarchSolver)
    assert float(ell.fill(0).sum()) * ell.h * ell.h == pytest.approx(0.5 * math.pi * 1e-12, rel=2e-3)


def test_run_writes_far_field_and_dump(tmp_path):
    raw = _tiny(far_field={"theta_max": 2e-3, "pad": 1024}, dump_field=True)
    sim = Simulation.from_dict(raw)
    sim.run(str(tmp_path), stages=[16])
    out = tmp_path / "stage16"
    assert (out / "far-field.json").exists() and (out / "field-capillary.npz").exists()
    meta = json.load(open(out / "meta.json", encoding="utf-8"))
    assert meta["provider"] == "march" and meta["model_class"] == "paraxial-medium" and meta["march"]["n_steps"] == 1000
    ff = json.load(open(out / "far-field.json", encoding="utf-8"))
    assert ff["weighted"]["x"]["centroid_rad"] is not None and 0.5 < ff["weighted"]["power_in_window"] <= 1.0

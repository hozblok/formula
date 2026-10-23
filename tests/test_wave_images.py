"""Stage 16 provider `images`: config contract, agreement with `uisk` on the same tiles,
separability, convergence in the image count, the coherent entrance spot, the far-field
observable, and the certified parity state of the 250 nm silicon channel."""

import json
import math

import numpy as np
import pytest

from formula.capsysred import Simulation
from formula.capsysred.screen import ScreenGrid
from formula.capsysred.stages import wave as W
from formula.capsysred.stages import wave_images as WI

LAM = 12.398419843320026e-10 / 8.048
K = 2.0 * math.pi / LAM


def _square(provider="images", nx=41, rotation_deg=0, sides=4, **wave):
    return {"precision": 32, "energy_kev": 8.048,
            "capillary": {"bores": [{"center": [0.0, 0.0], "radius": 24e-6, "sides": sides,
                                     "rotation_deg": rotation_deg}],
                          "z0": 0.0, "z1": 0.0525,
                          "source": {"shape": "point", "size": 0.0,
                                     "position": [0.0, 0.0, -0.117], "n_modes": 1, "n_rays": 1},
                          "screen": {"z": 0.183, "center": [0.0, 0.0], "edge_x": 6e-6,
                                     "edge_y": 6e-6, "nx": nx, "ny": nx, "reference": [0.0, 0.0]}},
            "wave_estimator": {"provider": provider, "observable": "point", **wave}}


def _channel(e_kev=11.2, offset=-0.6e-6, **wave):
    """Program-A scene: Si channel 250 nm x 5.0026 mm, point source 1 mm before the entrance."""
    return {"precision": 32, "energy_kev": e_kev, "material": "silicon",
            "capillary": {"bores": [{"center": [0.0, 0.0], "radius": 1.25e-7, "sides": 4, "rotation_deg": 0}],
                          "z0": 0.0, "z1": 5.0026e-3,
                          "source": {"shape": "point", "size": 0.0, "position": [offset, 0.0, -1.0e-3],
                                     "n_modes": 1, "n_rays": 1},
                          "screen": {"z": 5.0026e-3, "center": [0.0, 0.0], "edge_x": 2.5e-7, "edge_y": 2.5e-7,
                                     "nx": 11, "ny": 11, "reference": [0.0, 0.0]}},
            "wave_estimator": {"provider": "images", "observable": "point", "grid_dx": 2.0e-8, **wave}}


def _solvers(raw, **extra):
    sim = Simulation.from_dict(raw)
    wave = dict(sim.cfg.validate_wave_estimator(), **extra)
    cap = sim.cfg.capillary
    theta = W._theta_max(cap, np.array([[0.0, 0.0]]), [ScreenGrid(cap.screen)])
    z_in = float(cap.z0) - float(cap.source.position[2])
    length = float(cap.z1) - float(cap.z0)
    return sim, wave, cap, theta, z_in, length


def test_config_and_provider_contract():
    raw = _square()
    wave = Simulation.from_dict(raw).cfg.validate_wave_estimator()
    assert wave["provider"] == "images" and wave["images"] is None and wave["far_field"] is None
    cap = Simulation.from_dict(raw).cfg.capillary
    assert W.capillary_provider(cap, wave) == "images"
    assert WI.axis_rectangle(cap.bores[0]) == pytest.approx((0.0, 24e-6, 0.0, 24e-6))
    for bad in (_square(rotation_deg=45), _square(sides=6)):
        sim = Simulation.from_dict(bad)
        with pytest.raises(ValueError, match="provider images"):
            W.capillary_provider(sim.cfg.capillary, sim.cfg.validate_wave_estimator())
    assert W.capillary_provider(Simulation.from_dict(_square(rotation_deg=90)).cfg.capillary, wave) == "images"
    for bad in ({"images": -1}, {"images": 2.5}, {"far_field": {"pad": 256}}, {"far_field": {"theta_max": 0}},
                {"far_field": {"theta_max": 1e-3, "pad": 8}}, {"far_field": {"theta_max": 1e-3, "window": 1}}):
        with pytest.raises(ValueError, match="wave_estimator"):
            Simulation.from_dict(_square(**bad)).cfg.validate_wave_estimator()
    w = Simulation.from_dict(_square(images=3, far_field={"theta_max": 2.5e-3})).cfg.validate_wave_estimator()
    assert w["images"] == 3 and w["far_field"] == {"theta_max": 2.5e-3, "pad": 2048}
    rect = _square()
    rect["capillary"]["bores"][0]["apothems"] = [20e-6, 24e-6, 28e-6, 24e-6]
    assert WI.axis_rectangle(Simulation.from_dict(rect).cfg.capillary.bores[0]) == pytest.approx((-4e-6, 24e-6, 0.0, 24e-6))


def test_images_match_uisk_on_the_same_tiles():
    """uisk at max_bounces 2 keeps the 3 x 3 box of tiles; `images: 1` is the same sum without the band window."""
    sim, wave, cap, theta, z_in, length = _solvers(_square())
    uisk = W.BoreSolver(cap.bores[0], z_in, length, theta, LAM, wave)
    assert len(uisk.families) == 9
    img = WI.ImageSolver(cap.bores[0], z_in, length, theta, LAM, dict(wave, images=1, grid_dx=uisk.h), theta_c_max=None)
    assert img.n_images == [1, 1] and img.win == uisk.win and np.allclose(img.xs, uisk.xs) and np.allclose(img.ys, uisk.ys)
    u_u = uisk.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    u_i = img.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    assert u_i.shape == u_u.shape
    # on the exit lattice the two differ by uisk's raised-cosine band window (edge diffraction
    # above 0.4 mrad); on the detector 13 cm behind the exit those components are outside the pixels
    rms = lambda u: math.sqrt(float(np.mean(np.abs(u) ** 2)))
    assert rms(u_i - u_u) < 0.05 * rms(u_u)
    sv = np.linalg.svd(u_i, compute_uv=False)
    assert sv[1] / sv[0] < 1e-9                                            # X(x) Y(y)
    grid = ScreenGrid(cap.screen)
    samp = W.ScreenSampler(grid, (0.0, 0.0), "point", 1)
    d = 0.183 - 0.0525
    e_i, _, _ = samp.fields(u_i, samp.kernels(img.xs, img.ys, K, d, img.h), K, d, img.h)
    e_u, _, _ = samp.fields(u_u, samp.kernels(uisk.xs, uisk.ys, K, d, uisk.h), K, d, uisk.h)
    assert np.abs(e_i - e_u).max() < 1e-4 * np.abs(e_u).max()
    assert np.abs(e_i - e_i[::-1, :]).max() < 1e-10 * np.abs(e_i).max()
    assert np.abs(e_i - e_i.T).max() < 1e-10 * np.abs(e_i).max()


def test_image_count_converges_and_defaults_to_the_guided_horizon():
    sim, wave, cap, theta, z_in, length = _solvers(_square())
    thc = math.sqrt(2.0 * sim.delta_f)
    auto = WI.ImageSolver(cap.bores[0], z_in, length, theta, LAM, wave, theta_c_max=thc)
    expect = int(math.ceil(1.5 * thc * length / 48e-6)) + 4
    assert auto.n_images == [expect, expect] and auto.theta_horizon == pytest.approx(1.5 * thc)
    more = WI.ImageSolver(cap.bores[0], z_in, length, theta, LAM, dict(wave, images=2 * expect, grid_dx=auto.h), theta_c_max=thc)
    u_a = auto.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    u_m = more.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    assert np.abs(u_a - u_m).max() < 2e-5 * np.abs(u_m).max()
    power = lambda u, h: float(np.sum(np.abs(u) ** 2) * h * h)
    assert 0.5 < power(u_a, auto.h) / power(more.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1), more.h) < 2.0


def test_entrance_spot_is_a_unit_power_tilted_gaussian():
    sim, wave, cap, theta, z_in, length = _solvers(_square())
    spot = {"center": [0.0, 0.0], "fwhm": 8e-6, "tilt": [0.0, 0.0]}
    sol = WI.ImageSolver(cap.bores[0], z_in, length, theta, LAM, dict(wave, images=0), theta_c_max=None, spot=spot)
    fx, fy = sol.entrance_factors((0.0, 0.0), K)[0]
    x = np.linspace(-24e-6, 24e-6, 20001)
    assert float(np.trapezoid(np.abs(fx(x)) ** 2, x) * np.trapezoid(np.abs(fy(x)) ** 2, x)) == pytest.approx(1.0, rel=1e-6)
    u0 = sol.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    samp = W.ScreenSampler(ScreenGrid(cap.screen), (0.0, 0.0), "point", 1)
    d = 0.183 - 0.0525
    e0, _, _ = samp.fields(u0, samp.kernels(sol.xs, sol.ys, K, d, sol.h), K, d, sol.h)
    assert np.abs(e0 - e0[::-1, :]).max() < 1e-10 * np.abs(e0).max()
    tilted = WI.ImageSolver(cap.bores[0], z_in, length, theta, LAM, dict(wave, images=0, grid_dx=sol.h), theta_c_max=None,
                            spot=dict(spot, tilt=[2e-4, 0.0]))
    u1 = tilted.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    cx = lambda u, xs: float(np.sum(np.abs(u) ** 2 * xs[:, None]) / np.sum(np.abs(u) ** 2))
    assert cx(u0, sol.xs) == pytest.approx(0.0, abs=1e-9)
    assert cx(u1, tilted.xs) == pytest.approx(2e-4 * length, rel=0.05)      # the spot drifts by tilt x L


def test_channel_parity_state_and_far_field(tmp_path):
    """Certified program-A state at 11.2 keV: far-field centroid +0.94 theta_0 (exact slab +0.941,
    xwglib +0.943, uisk +0.940); the mirrored state at 7.48 keV; files of the run."""
    raw = _channel(far_field={"theta_max": 2.5e-3, "pad": 2048}, dump_field=True)
    sim = Simulation.from_dict(raw)
    sim.run(str(tmp_path / "plus"), stages=[16])
    out = tmp_path / "plus" / "stage16"
    assert (out / "far-field.json").exists() and (out / "16-far-field.svg").exists() and (out / "field-capillary.npz").exists()
    ff = json.load(open(out / "far-field.json", encoding="utf-8"))
    assert ff["pad"] == 2048 and len(ff["lines"]) == 1 and ff["lines"][0]["e_kev"] == pytest.approx(11.2, rel=1e-9)
    assert ff["weighted"]["x"]["centroid_rad"] / 0.6e-3 == pytest.approx(0.943, abs=0.012)
    assert abs(ff["weighted"]["y"]["centroid_rad"]) < 1e-5 * 0.6e-3
    assert ff["weighted"]["x"]["P_plus_over_minus"] > 10.0 and 0.9 < ff["weighted"]["power_in_window"] <= 1.0
    meta = json.load(open(out / "meta.json", encoding="utf-8"))
    assert meta["provider"] == "images" and meta["model_class"] == "kirchhoff-rims"
    assert meta["images_per_axis"][0] == [meta["images_per_axis"][0][0]] * 2 and meta["images_per_axis"][0][0] >= 60
    raw_m = _channel(e_kev=7.48, far_field={"theta_max": 2.5e-3, "pad": 2048})
    Simulation.from_dict(raw_m).run(str(tmp_path / "minus"), stages=[16])
    ffm = json.load(open(tmp_path / "minus" / "stage16" / "far-field.json", encoding="utf-8"))
    assert ffm["weighted"]["x"]["centroid_rad"] / 0.6e-3 == pytest.approx(-0.925, abs=0.012)


def test_far_field_needs_one_rectangular_bore_and_rejects_fb(tmp_path):
    raw = _square(far_field={"theta_max": 1e-3})
    raw["capillary"]["bores"].append({"center": [60e-6, 0.0], "radius": 24e-6, "sides": 4, "rotation_deg": 0})
    with pytest.raises(ValueError, match="far_field"):
        Simulation.from_dict(raw).run(str(tmp_path / "two"), stages=[16])
    raw = _square(provider="fb", far_field={"theta_max": 1e-3})
    raw["capillary"]["bores"] = [{"center": [0.0, 0.0], "radius": 24e-6}]
    with pytest.raises(ValueError, match="far_field"):
        Simulation.from_dict(raw).run(str(tmp_path / "fb"), stages=[16])

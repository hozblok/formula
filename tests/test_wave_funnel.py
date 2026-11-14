"""Stage 16 provider fb on conformal funnel bores (wave_funnel): the straight limit is the fb bore with exact roots,
the basis dilation follows the effective radius, power is kept without absorption, the result agrees with the march
funnel, and a run writes the screen and the matrix of mu between the bore centres."""

import json
import math

import numpy as np
import pytest

from formula.capsysred import Simulation
from formula.capsysred.stages import wave_funnel as WF
from formula.xray import OE2012_GLASS

E_KEV = 8.0
LAM = 12.398419843320026e-10 / E_KEV
K = 2.0 * math.pi / LAM
DELTA, BETA = OE2012_GLASS.optical(E_KEV)
Z26_G = (1.0 / 0.03, -1.0 / (0.03 * 0.46))


def test_straight_limit_is_the_fb_bore():
    """g = 0: G = s = 1 and no lens, so the propagator is the diagonal exp(-i h L) of the exact roots."""
    basis = WF.FunnelBasis(0.5e-6, (0.0, 0.0), K, DELTA, BETA, 40)
    ch = WF.FunnelChannel(basis, (0.3e-6, 0.0))
    c0 = ch.entrance_modes([(0.0, 0.0), (-0.1e-6, 0.2e-6)], 1.5e-3)
    cL, _, _ = ch.propagate(c0, 2.0e-3, 1.0e-3)
    hp = np.zeros(basis.disc.zp.shape, complex)
    hp[basis.mr, basis.nr_] = basis.h(0.0)
    ref = c0 * np.exp(-1j * hp[basis.disc.am] * 2.0e-3)[:, :, None]
    assert np.max(np.abs(cL - ref)) < 1e-9 * np.max(np.abs(c0))


def test_dilation_tracks_the_effective_radius():
    """s A0 = a + Re ell / G, s(0) = 1; the derivatives against finite differences."""
    basis = WF.FunnelBasis(6.0e-6, Z26_G, K, DELTA, BETA, 20)
    for z in (0.0, 0.1, 0.23):
        assert basis.s(z)[0] * basis.A0 == pytest.approx(basis.a + basis.ell.real / basis.G(z), rel=1e-14)
    assert basis.s(0.0)[0] == 1.0
    z, h = 0.1, 1.0e-5
    assert basis.s(z)[1] == pytest.approx((basis.s(z + h)[0] - basis.s(z - h)[0]) / (2 * h), rel=1e-6)
    assert basis.s(z)[2] == pytest.approx((basis.s(z + h)[1] - basis.s(z - h)[1]) / (2 * h), rel=1e-5)


def test_power_is_kept_without_absorption():
    """beta = 0: real exact roots, real lens / axis / dilation terms, so the norm of w (= of E) is kept."""
    basis = WF.FunnelBasis(0.5e-6, (3.0e3, -1.0e6), K, DELTA, 0.0, 40)
    ch = WF.FunnelChannel(basis, (0.3e-6, 0.1e-6))
    c0 = ch.entrance_modes([(0.0, 0.0), (-0.2e-6, 0.15e-6)], 1.0e-3 / 3.0)
    cL, _, _ = ch.propagate(c0, 1.0e-3, 2.0e-5)
    for i in range(2):
        p0, p1 = basis.disc.norm(c0[:, :, i]), basis.disc.norm(cL[:, :, i])
        assert p1 == pytest.approx(p0, rel=1e-6)


def _scene(nx=45, ref=(0.0, 0.0)):
    L = 1.0e-3
    bores = [{"center": [c, 0.0], "radius": 0.5e-6, "funnel": {"g": [3.0 / L, -1.0 / L ** 2]}} for c in (0.0, 1.2e-6)]
    return {"precision": 32, "energy_kev": E_KEV, "material": "glass_oe2012",
            "capillary": {"bores": bores, "z0": 0.0, "z1": L,
                          "source": {"shape": "grid", "grid_n": 3, "grid_step": 0.1e-6, "grid_r_max": 0.15e-6, "size": 0.1e-6,
                                     "position": [0.0, 0.0, -L / 3.0], "n_modes": 5, "n_rays": 1},
                          "screen": {"z": L, "center": [0.0, 0.0], "edge_x": 10.8e-6, "edge_y": 3.6e-6, "nx": nx, "ny": 15,
                                     "reference": list(ref)}},
            "wave_estimator": {"provider": "fb", "observable": "point", "fb_wall": "exact", "fb_jmax": 30, "fb_dz": 2.0e-5}}


def test_run_two_bores(tmp_path):
    sim = Simulation.from_dict(_scene())
    sim.run(str(tmp_path), stages=[16])
    out = tmp_path / "stage16"
    meta = json.load(open(out / "meta.json", encoding="utf-8"))
    assert meta["provider"] == "fb" and meta["model_class"] == "exact-jk,funnel-xi" and meta["funnel"]["G_exit"] == pytest.approx(3.0)
    cen = json.load(open(out / "funnel-centres.json", encoding="utf-8"))
    mu = np.array(cen["mu_re"]) + 1j * np.array(cen["mu_im"])
    assert np.allclose(np.diag(mu), 1.0) and np.allclose(mu, mu.conj().T) and np.all(np.abs(mu) <= 1.0 + 1e-12)
    rows = [json.loads(l) for l in open(out / "mu-wave.jsonl", encoding="utf-8")]
    lit = [r for r in rows if r["I"] > 0.0]
    assert lit and all(r["mu_abs"] <= 1.0 + 1e-9 for r in lit)
    ref = [r for r in lit if abs(r["x_um"]) < 1e-9 and abs(r["y_um"]) < 1e-9]
    assert ref and ref[0]["mu_abs"] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="fb_wall: exact"):
        raw = _scene()
        raw["wave_estimator"]["fb_wall"] = "dir-ell"
        Simulation.from_dict(raw).run(str(tmp_path / "x"), stages=[16])


def test_agrees_with_the_march_funnel():
    """A conformal funnel at grazing angles below 0.2 theta_c (a = 0.5 um, L = 3 mm, D = 1.5 mm, G(L) = 2.5, centre
    0.6 um): the exit fields of the modes and of the split step (dx 5 nm) overlap to better than 0.999."""
    from formula.capsysred.stages import wave_march as WM
    L, D, a, c0, g = 3.0e-3, 1.5e-3, 0.5e-6, (0.6e-6, 0.0), (1.0 / 1.5e-3, -0.5 / 9.0e-6)
    raw = {"precision": 32, "energy_kev": E_KEV, "material": "glass_oe2012",
           "capillary": {"bores": [{"center": list(c0), "radius": a, "funnel": {"g": list(g)}}], "z0": 0.0, "z1": L,
                         "source": {"shape": "point", "size": 0.0, "position": [0.0, 0.0, -D], "n_modes": 1, "n_rays": 1},
                         "screen": {"z": L, "center": [0.0, 0.0], "edge_x": 4e-6, "edge_y": 4e-6, "nx": 11, "ny": 11,
                                    "reference": [0.0, 0.0]}},
           "wave_estimator": {"provider": "march", "observable": "point", "grid_dx": 5e-9,
                              "march": {"dz": 5e-7, "box": 1e-6, "sponge": 5e-7, "dx": 5e-9, "defects": []}}}
    sim = Simulation.from_dict(raw)
    s = WM.MarchSolver(sim.cfg.capillary.bores[0], D, L, 5e-3, LAM, sim.cfg.validate_wave_estimator())
    src = (-0.2e-6, 0.15e-6)
    um = s.exit_field(np.array(src), K, DELTA, BETA, 4)
    basis = WF.FunnelBasis(a, g, K, DELTA, BETA, 60)
    ch = WF.FunnelChannel(basis, c0)
    cL, _, _ = ch.propagate(ch.entrance_modes([src], D), L, 5e-5)
    X, Y = np.meshgrid(s.xs, s.ys, indexing="ij")
    inside = s.exit_mask > 0.999
    e, ins = ch.points(cL, L, X[inside], Y[inside], basis.disc.aw / 1500.0)
    ef = np.zeros(int(inside.sum()), complex)
    ef[ins] = e[:, 0]
    a0 = um[inside]
    ov = abs(np.vdot(a0, ef)) ** 2 / (np.vdot(a0, a0).real * np.vdot(ef, ef).real)
    assert ov > 0.999 and np.vdot(ef, ef).real / np.vdot(a0, a0).real == pytest.approx(1.0, abs=0.005)

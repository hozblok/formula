"""Stage 20 (wave packet along one circular bore): config contract, unitarity and the return at z = 0 in a straight
lossless bore, the pendulum of a packet along the wall of a bent bore, the consistency of the carpets, the slices
and the far field with the mode coefficients, and the end-to-end run with its files."""

import json
import math
import os

import numpy as np
import pytest

from formula.capsysred import Simulation
from formula.capsysred.stages.wave.providers.fb import basis as FBB
from formula.capsysred.stages.wave.providers.fb import channel as FBC
from formula.capsysred.stages import wave_packet as WP

E_KEV = 8.0
K = 2.0 * math.pi * E_KEV / 12.398419843320026e-10


def _scene(a=2.0e-6, length=4.0e-3, bend=None, spot=None, packet=None, wave=None, material="glass_oe2012"):
    bore = {"center": [0.0, 0.0], "radius": a}
    if bend:
        bore["bend"] = bend
    spot = spot or {"center": [-(a - 2.5e-7), 0.0], "fwhm": 1.5e-7, "tilt": [0.0, 0.0]}
    w = {"provider": "fb", "fb_jmax": 60, "fb_wall": "dir-ell-real", "entrance_spot": spot, "observable": "point"}
    w.update(wave or {})
    raw = {"precision": 32, "energy_kev": E_KEV, "seed": 1, "material": material,
           "capillary": {"bores": [bore], "z0": 0.0, "z1": length,
                         "source": {"shape": "point", "size": 0.0, "position": [0.0, 0.0, -1.0e-3], "n_modes": 1, "n_rays": 1},
                         "screen": {"z": length, "center": [0.0, 0.0], "edge_x": 2.2 * a, "edge_y": 2.2 * a, "nx": 11, "ny": 11,
                                    "reference": [0.0, 0.0]}},
           "wave_estimator": w, "wave_packet": packet or {}}
    return raw


# ---------------------------------------------------------------- contract

def test_packet_config_defaults_and_errors():
    sim = Simulation.from_dict(_scene())
    p = sim.cfg.validate_wave_packet()
    assert p["z_step"] is None and p["z_max"] is None and p["slices"] == [] and p["dephasing_from"] == 0.5
    for bad in ({"z_step": 0.0}, {"z_max": -1.0}, {"slice_pixel": 0}, {"slices": 3.0}, {"slices": [-1e-3]},
                {"dephasing_from": 1.0}, {"far_field_at": "end"}, {"zstep": 1.0}):
        with pytest.raises(ValueError, match="wave_packet"):
            Simulation.from_dict(_scene(packet=bad)).cfg.validate_wave_packet()
    sim = Simulation.from_dict(_scene(packet={"slices": [2.0e-3, 1.0e-3], "z_max": 3.0e-3}))
    assert sim.cfg.validate_wave_packet()["slices"] == [1.0e-3, 2.0e-3]


def test_packet_preflight_contract():
    def check(raw, match):
        sim = Simulation.from_dict(raw)
        with pytest.raises(ValueError, match=match):
            WP.preflight_packet_inputs(sim, sim.cfg.validate_wave_estimator(), sim.cfg.validate_wave_packet())

    raw = _scene()
    raw["wave_estimator"].pop("entrance_spot")
    check(raw, "entrance_spot")
    raw = _scene(wave={"provider": "uisk"})
    check(raw, "provider fb")
    raw = _scene()
    raw["capillary"]["bores"].append({"center": [5.0e-6, 0.0], "radius": 2.0e-6})
    check(raw, "one bore")
    check(_scene(packet={"z_max": 5.0e-3}), "beyond the bore length")
    check(_scene(packet={"slices": [4.5e-3]}), "outside")
    check(_scene(packet={"far_field_at": [1.0e-3]}), "far_field")
    sq = _scene()
    sq["capillary"]["bores"][0]["sides"] = 4
    check(sq, "circular bore")
    sim = Simulation.from_dict(_scene(packet={"slices": [1.0e-3]}))
    WP.preflight_packet_inputs(sim, sim.cfg.validate_wave_estimator(), sim.cfg.validate_wave_packet())
    assert 20 in Simulation.from_dict(_scene()).run.__globals__["KNOWN_STAGES"]


# ---------------------------------------------------------------- physics oracles

def test_straight_lossless_bore_is_unitary_and_returns_at_zero(tmp_path):
    raw = _scene(packet={"z_step": 5.0e-4, "slices": [2.0e-3]})
    sim = Simulation.from_dict(raw)
    sim.run(str(tmp_path), stages=[20])
    d = np.load(tmp_path / "stage20" / "packet.npz")
    z, T, F = d["z"], d["T"][0], d["F"][0]
    assert z[0] == 0.0 and z[-1] == pytest.approx(4.0e-3) and z.size == 9
    assert F[0] == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(T - 1.0)) < 1e-10                       # no absorption, no leak: the power is conserved
    assert F[-1] < 0.9                                            # the packet spreads and the overlap falls
    # the slice coefficients equal the diagonal propagation of the entrance coefficients
    disc = FBB.Disc(aw=float(d["aw"]), jmax=60, wall_cut=None)
    c0, c = d["c_in"][0], d["coef"][0, 0]
    h0 = disc.kap2 / (2.0 * K)
    assert np.max(np.abs(c - c0 * np.exp(-1j * h0 * 2.0e-3))) < 1e-9 * np.max(np.abs(c0))
    # the azimuthal carpet integrates to the power of the line, the cross-section field as well (1 % quadrature)
    P = d["P"][0]
    assert np.sum(d["azimuth_I"][-1]) * (2 * math.pi / d["phi"].size) == pytest.approx(P[-1], rel=1e-6)
    dx = float(d["slice_x"][1] - d["slice_x"][0])
    assert np.sum(np.abs(d["slice_E"][0, 0]) ** 2) * dx * dx == pytest.approx(P[4], rel=1e-2)
    # the entrance centroid sits at the spot centre, the straight bore keeps the mean xi within the spread
    assert d["xi_mean"][0, 0] == pytest.approx(-(2.0e-6 - 2.5e-7), abs=5e-9)
    assert abs(d["eta_mean"][0, 0]) < 1e-9


def test_pendulum_of_a_packet_along_the_wall_of_a_bent_bore(tmp_path):
    """A spot at the outer wall displaced along the wall swings about the lowest point of the potential k xi / R with the
    small-amplitude period 2 pi sqrt(R a) of the paraxial pendulum (a / R corrections of a few per cent)."""
    a, R = 3.0e-6, 1.0
    eta0 = 2.5e-7
    spot = {"center": [-(a - 2.0e-7), eta0], "fwhm": 1.4e-7, "tilt": [0.0, 0.0]}
    period = 2.0 * math.pi * math.sqrt(R * a)
    raw = _scene(a=a, length=period, bend={"radius": R, "toward": [1.0, 0.0]}, spot=spot,
                 packet={"z_step": period / 40}, wave={"fb_jmax": 130, "fb_theta_cut": 1.5e-3})
    sim = Simulation.from_dict(raw)
    sim.run(str(tmp_path), stages=[20])
    d = np.load(tmp_path / "stage20" / "packet.npz")
    z, eta, T = d["z"], d["eta_mean"][0], d["T"][0]
    assert np.max(np.abs(T - 1.0)) < 1e-8                         # lossless wall: the Chebyshev propagator is unitary
    assert eta[0] == pytest.approx(eta0, rel=0.02)
    # fit eta0 cos(w z): the frequency from the zero crossings of the centroid
    sign = np.sign(eta)
    crossings = [z[i] - eta[i] * (z[i + 1] - z[i]) / (eta[i + 1] - eta[i]) for i in range(z.size - 1) if sign[i] != sign[i + 1]]
    assert len(crossings) >= 2
    w_fit = math.pi / (crossings[1] - crossings[0])
    assert w_fit == pytest.approx(1.0 / math.sqrt(R * a), rel=0.08)
    assert eta[np.argmin(np.abs(z - period / 2))] == pytest.approx(-eta0, rel=0.15)   # the swing reaches the far side
    meta = json.load(open(tmp_path / "stage20" / "meta.json", encoding="utf-8"))
    assert meta["propagator"] == "chebyshev" and meta["chebyshev_degree_median"] > 0
    assert meta["model_class"] == "dir-ell,no-absorption,parabolic-axis"


def test_far_field_and_return_peaks(tmp_path):
    a = 2.0e-6
    raw = _scene(a=a, length=3.0e-3, bend={"radius": 2.0, "toward": [1.0, 0.0]},
                 packet={"z_step": 1.0e-4, "far_field_at": [3.0e-3], "slices": [3.0e-3]},
                 wave={"fb_jmax": 80, "fb_wall": "dir-ell", "far_field": {"theta_max": 2.0e-3, "n_theta": 41, "n_phi": 256, "ring": [0.0, 2.0e-3]}})
    sim = Simulation.from_dict(raw)
    sim.run(str(tmp_path), stages=[20])
    d = np.load(tmp_path / "stage20" / "packet.npz")
    assert d["ff_map"].shape == (1, 41, 256) and d["ff_z"][0] == pytest.approx(3.0e-3)
    # Parseval: the power in the whole far field equals the exit power in the basis (theta_max holds the packet)
    k = float(d["k"][0])
    th = d["ff_theta"]
    w = np.gradient(th) * th * (k / (2 * math.pi)) ** 2 * (2 * math.pi / 256)
    assert np.sum(w[:, None] * d["ff_map"][0]) == pytest.approx(d["P"][0, -1], rel=0.03)
    # the rms far-field angle from the coefficients equals the second moment of the far-field map
    th2 = np.sum(w[:, None] * th[:, None] ** 2 * d["ff_map"][0]) / np.sum(w[:, None] * d["ff_map"][0])
    assert d["theta_rms"][0, -1] == pytest.approx(math.sqrt(th2), rel=0.03)
    T = d["T"][0]
    assert T[-1] < 1.0 and np.all(np.diff(T) <= 1e-12)            # absorbing wall: the power only falls
    meta = json.load(open(tmp_path / "stage20" / "meta.json", encoding="utf-8"))
    assert meta["far_field_rings"][0]["z_m"] == pytest.approx(3.0e-3)
    s = meta["lines"][0]
    assert s["dephased_from_m"] == pytest.approx(1.5e-3) and 0.0 < s["dephased_mean_F"] < 1.0
    for p in s["return_peaks"]:
        i = int(np.argmin(np.abs(d["z"] - p["z_m"])))
        assert d["F"][0, i] == pytest.approx(p["F"]) and d["T"][0, i] == pytest.approx(p["T"])


def test_boundary_loss_operator_matches_the_diagonal_damping_and_the_bouncer_law(tmp_path):
    """Straight bore, one disc mode: the boundary form of i Im ell damps it as mode_damping without its Fresnel factor.
    Bent bore, a packet at the outer wall: every Airy mode loses 2 k Im ell / R per unit length, so T(z) = exp(-2 k Im ell z / R)."""
    a, delta, beta = 2.0e-6, 4.5575e-6, 5.725e-8
    ell = FBB.wall_offset(K, delta, beta)
    d = FBB.Disc(aw=a + ell.real, jmax=60)
    ch = FBC.FBChannel(c0=(0.0, 0.0), t=None, rbend=None, disc=d, k=K, wall_loss=1j * ell.imag)
    c = d.empty()
    c[d.M + 7, 2] = 1.0                                           # m = 7, n = 3
    out, _ = ch.propagate(c, 2.0e-3)
    gamma = d.kap2[d.M + 7, 2] * ell.imag / (d.aw * K)
    assert abs(out[d.M + 7, 2]) == pytest.approx(math.exp(-gamma * 2.0e-3), rel=1e-6)
    assert d.norm(out) == pytest.approx(d.norm(c) * math.exp(-2 * gamma * 2.0e-3), rel=1e-5)   # no leak into other modes
    R = 1.0
    raw = _scene(a=a, length=6.0e-3, bend={"radius": R, "toward": [1.0, 0.0]},
                 spot={"center": [-(a - 2.5e-7), 0.0], "fwhm": 1.2e-7, "tilt": [0.0, 0.0]},
                 packet={"z_step": 5.0e-4}, wave={"fb_jmax": 120, "fb_theta_cut": 1.5e-3, "fb_wall": "dir-ell"})
    sim = Simulation.from_dict(raw)
    sim.run(str(tmp_path), stages=[20])
    dd = np.load(tmp_path / "stage20" / "packet.npz")
    z, T = dd["z"], dd["T"][0]
    law = np.exp(-2.0 * K * ell.imag * z / R)
    assert np.max(np.abs(T - law)) < 0.05 * (1.0 - law[-1])        # within 5 % of the loss
    meta = json.load(open(tmp_path / "stage20" / "meta.json", encoding="utf-8"))
    assert meta["loss"] == "boundary" and meta["model_class"].startswith("dir-ell,boundary-loss")
    # the diagonal damping of stage 16 misses the coherent wall concentration of the packet: far less loss
    raw["wave_packet"]["loss"] = "diagonal"
    Simulation.from_dict(raw).run(str(tmp_path / "diag"), stages=[20])
    Td = np.load(tmp_path / "diag" / "stage20" / "packet.npz")["T"][0]
    assert 1.0 - Td[-1] < 0.3 * (1.0 - T[-1])
    with pytest.raises(ValueError, match="wave_packet.loss"):
        Simulation.from_dict(_scene(packet={"loss": "none"})).cfg.validate_wave_packet()


def test_return_peaks_helper():
    z = np.linspace(0.0, 1.0, 101)
    F = 0.5 + 0.5 * np.cos(2 * math.pi * z * 2)                   # minima at 0.25, 0.75; maxima at 0.5, 1.0
    peaks = WP.return_peaks(z, F)
    assert [round(p[0], 2) for p in peaks] == [0.5, 1.0]
    assert WP.return_peaks(z, np.exp(-z)) == []                   # monotone decay: no return


# ---------------------------------------------------------------- end to end

def test_stage20_files_report_and_no_clobber(tmp_path):
    raw = _scene(length=2.0e-3, packet={"z_step": 1.0e-3})
    sim = Simulation.from_dict(raw)
    out = sim.run(str(tmp_path), stages=[20])
    names = set(out["files"])
    assert {"stage20/packet.npz", "stage20/meta.json", "stage20/20-return.svg"} <= names
    report = open(tmp_path / [n for n in os.listdir(tmp_path) if n.startswith("report-")][0], encoding="utf-8").read()
    assert "## Stage 20 — wave packet along the bore [capillary]" in report and "return peaks" in report
    meta = json.load(open(tmp_path / "stage20" / "meta.json", encoding="utf-8"))
    assert meta["stage"] == 20 and meta["basis"]["n_modes"] > 0 and meta["wave_packet"]["z_step"] == 1.0e-3
    with pytest.raises(ValueError, match="publication conflict"):
        Simulation.from_dict(raw).run(str(tmp_path), stages=[20])
    with pytest.raises(ValueError, match="unknown stages"):
        Simulation.from_dict(raw).run(str(tmp_path / "x"), stages=[19])

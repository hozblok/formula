"""Stage 16 fb: exact circular-bore dispersion, Hadamard wall figure, co-moving axis path, far field from the
modes, coefficient dump, entrance-spot focus and the images hard wall."""

import json
import math
import os

import numpy as np
import pytest
from scipy import fft as sfft
from scipy.integrate import quad
from scipy.special import jn_zeros, jv, jvp, kv, kvp

from formula.capsysred import Simulation
from formula.capsysred.stages import wave as W
from formula.capsysred.stages import wave_fb as FB
from formula.capsysred.stages import wave_images as WI
from formula.xray import FUSED_SILICA

E_KEV = 8.048
LAM = 12.398419843320026e-10 / E_KEV
K = 2.0 * math.pi / LAM
DELTA, BETA = FUSED_SILICA.optical(E_KEV)


def _disc(a, theta_cut=4.5e-3, cut=True):
    aw = a + FB.wall_offset(K, DELTA, BETA).real
    J = K * aw * math.sqrt(2.0 * DELTA)
    return FB.Disc(aw=aw, jmax=int(math.ceil(K * aw * theta_cut)), wall_cut=J if cut else None)


# ---------------------------------------------------------------- exact dispersion

def test_exact_kappa_roots_and_the_angular_talbot_length():
    a = 1.0e-6
    d = _disc(a)
    kap, ok = FB.exact_kappa(d, a, K, DELTA, BETA)
    assert ok[d.validp].all()
    # independent script values (J_m / K_m continuity, numeric Newton), kappa * a
    for (m, n), (re, im) in {(0, 1): (2.3893, -9.962e-05), (90, 1): (97.9037, -4.225e-03),
                             (90, 2): (104.5499, -4.655e-03), (145, 1): (153.9244, -6.856e-03)}.items():
        v = kap[m, n - 1] * a
        assert abs(v.real - re) < 1e-4 and abs(v.imag - im) < 2e-3 * abs(im)
    # the root satisfies the unscaled product form of the characteristic equation
    k2 = K * K * (1.0 - (1.0 - DELTA + 1j * BETA) ** 2)
    for m, n in ((30, 1), (90, 1), (12, 7)):
        x = kap[m, n - 1]
        Q = np.sqrt(k2 - x * x)
        Q = -Q if Q.real < 0 else Q
        t1, t2 = x * jvp(m, x * a) * kv(m, Q * a), Q * kvp(m, Q * a) * jv(m, x * a)
        assert abs(t1 - t2) < 1e-9 * abs(t1)
    ms = np.arange(84, 97)
    f = np.array([(kap[m, 0].real * a) ** 2 for m in ms])
    f2 = np.gradient(np.gradient(f, ms), ms)[len(ms) // 2]
    assert 8 * math.pi * K * a * a / f2 == pytest.approx(0.49975, abs=5e-5)    # exact L_A at m0 = 90


# ---------------------------------------------------------------- Hadamard figure and axis path

def test_hadamard_term_is_the_translation_generator():
    """A rigid shift dr = d cos phi: V_ij = -d (h_j - h_i) <psi_i | d_x psi_j> / <psi_i|psi_i> for the code's V."""
    d = FB.Disc(aw=1.0e-6, jmax=60)
    ch = FB.FBChannel((0.0, 0.0), None, None, d, K)
    shift = 1e-10
    ch._dr = FB.WallFigure([{"q": 1, "amp": shift}], None, 1.0, axis=False).coeffs(0.0)
    mj, nj, M = 20, 1, d.M
    e = d.empty()
    e[mj + M, nj - 1] = 1.0
    out = ch.apply_v(e)
    jj = d.zp[mj, nj - 1]
    for ni in range(1, 5):
        mi = mj + 1
        ji = d.zp[mi, ni - 1]
        ov = quad(lambda r: jv(mi, ji / d.aw * r) * jv(mi, jj / d.aw * r) * r, 0, d.aw, limit=400)[0]
        X = (-jj / d.aw / 2) * ov / (0.5 * d.aw ** 2 * jv(mi + 1, ji) ** 2)
        ident = -shift * (jj * jj - ji * ji) / (2 * K * d.aw ** 2) * X
        assert out[mi + M, ni - 1] == pytest.approx(ident, rel=1e-6)


def test_axis_path_against_hadamard_first_order():
    """Axis c(z) = A (1 - cos 2 pi z / P): the exact co-moving frame and the first-order Hadamard q = 1 agree up to
    a mismatch that falls linearly with A (second order); the co-moving result does not depend on the basis size."""
    a, P = 0.5e-6, 1.0e-3
    L = 3 * P
    spot = {"center": [a - 40e-9, 0.0], "fwhm": 80e-9, "tilt": [0.0, 1.6e-3]}
    res = {}
    for cut in (True, False):
        d = _disc(a, theta_cut=4.5e-3 if cut else 1.5 * math.sqrt(2 * DELTA), cut=cut)
        g = FB.mode_damping(d, K, DELTA, BETA)
        ch0 = FB.FBChannel((0, 0), None, None, d, K, g)
        c0 = ch0.spot_modes(spot)
        cu = ch0.propagate(c0, L)[0]
        for A in (0.1e-9, 0.03e-9):
            terms = [{"q": 1, "amp": A, "angle": 0.3}, {"q": 1, "amp": -A, "angle": 0.3, "period": P}]
            c = {ax: FB.FBChannel((0, 0), None, None, d, K, g, figure=FB.WallFigure(terms, None, L, axis=ax),
                                  dz=P / 64).propagate(c0, L)[0] for ax in (True, False)}
            eff = math.sqrt(d.norm(c[True] - cu) / d.norm(cu))
            res[cut, A] = (eff, math.sqrt(d.norm(c[False] - c[True]) / d.norm(c[True] - cu)))
    assert res[True, 0.1e-9][0] == pytest.approx(res[False, 0.1e-9][0], rel=1e-3)
    mis1, mis2 = res[True, 0.1e-9][1], res[True, 0.03e-9][1]
    assert mis1 < 0.08 and 2.5 < mis1 / mis2 < 4.0


def test_axis_path_lab_frame_and_static_shift():
    """A static q = 1 term moves the bore; the exit field is the nominal one translated, exactly."""
    a = 0.5e-6
    d = _disc(a)
    d.fine_table(2e-9)
    spot = {"center": [0.1e-6, 0.05e-6], "fwhm": 150e-9, "tilt": [0.8e-3, 0.4e-3]}
    s = 3e-9
    fig = FB.WallFigure([{"q": 1, "amp": s, "angle": 0.0}], None, 2e-3)
    ch = FB.FBChannel((0, 0), None, None, d, K, figure=fig, dz=1e-4)
    ch0 = FB.FBChannel((0, 0), None, None, d, K)
    assert np.allclose(ch.axis(1e-3), [s, 0.0]) and np.allclose(ch.slope(1e-3), 0.0) and ch.lab_phase_g(1e-3) == 0.0
    sp = dict(spot, center=[spot["center"][0] - s, spot["center"][1]])
    c1 = ch.propagate(ch.spot_modes(spot), 2e-3)[0]
    c2 = ch0.propagate(ch0.spot_modes(sp), 2e-3)[0] * np.exp(1j * K * spot["tilt"][0] * s)   # the tilt phase moves too
    assert np.max(np.abs(c1 - c2)) < 1e-10 * np.max(np.abs(c2))


# ---------------------------------------------------------------- far field

def test_far_field_from_modes_parseval_and_fft():
    d = _disc(1.0e-6)
    rng = np.random.default_rng(3)
    c = np.zeros(d.kap2.shape, complex)
    sel = d.valid & (np.abs(np.array(d.ms))[:, None] < 30)
    c[sel] = rng.standard_normal(sel.sum()) + 1j * rng.standard_normal(sel.sum())
    c *= np.exp(-d.kap2 / (2 * (K * 1.5e-3) ** 2))
    th = np.linspace(0, 8e-3, 1601)
    nphi = 1 << int(math.ceil(math.log2(2 * d.M + 1)))
    A = FB.far_field_amplitude(d, c, K, th, nphi)
    ring, _ = FB.far_field_profiles(np.abs(A) ** 2, th, K, (0.0, 1.0))
    assert ring.sum() * 2 * math.pi / nphi == pytest.approx(d.norm(c), rel=1e-4)
    d.fine_table(2e-9)
    ch = FB.FBChannel((0, 0), None, None, d, K)
    h = 4e-9
    xs = (np.arange(-300, 300) + 0.5) * h
    F = sfft.fft2(ch.lab_field(c, 0.0, xs, xs, 2e-9), s=(8192, 8192)) * h * h
    fq = sfft.fftfreq(8192, h) * 2 * math.pi
    for i in (186, 280):                                             # ky = 0, kx = q: phi = 0
        a_mod = FB.far_field_amplitude(d, c, K, [fq[i] / K], nphi)[0, 0]
        assert abs(a_mod) == pytest.approx(abs(F[0, i]), rel=2e-3)


# ---------------------------------------------------------------- stage

def _scene(extra, length=0.005, a=0.5e-6):
    return {"precision": 32, "energy_kev": E_KEV, "seed": 1,
            "capillary": {"bores": [{"center": [0.0, 0.0], "radius": a, "material": "fused_silica"}], "z0": 0.0, "z1": length,
                          "source": {"shape": "gaussian", "size": 1e-7, "position": [0.0, 0.0, -0.05], "n_modes": 1, "n_rays": 1},
                          "screen": {"z": length, "center": [0.0, 0.0], "edge_x": 1.2e-6, "edge_y": 1.2e-6, "nx": 12, "ny": 12,
                                     "reference": [0.0, 0.0]}},
            "wave_estimator": {"provider": "fb", "fb_theta_cut": 4.0e-3, "observable": "point", **extra}}


def test_stage_far_field_dump_exact_and_mirror_symmetry(tmp_path):
    plane = {"center": [0.0, 0.0], "fwhm": 30e-6, "tilt": [1.2e-3, 0.0]}
    raw = _scene({"entrance_spot": plane, "fb_wall": "exact", "dump_field": True,
                  "far_field": {"theta_max": 2.5e-3, "ring": [0.9e-3, 1.5e-3], "n_theta": 101}})
    sim = Simulation.from_dict(raw)
    res = W.run_wave_stage(sim, str(tmp_path), sim.cfg.validate_wave_estimator(), log=lambda *a: None)
    out = tmp_path / "stage16"
    for name in ("far-field.json", "far-field-fb.npz", "16-far-field.svg", "field-capillary.npz"):
        assert (out / name).exists() and "stage16/" + name in res["files"]
    meta = res["results"]["capillary"]["meta"]
    assert meta["model_class"].startswith("exact-jk") and 0.0 < meta["transmission"][0] < 1.0
    assert sum(meta["exact_fallback_modes"].values()) == 0
    ff = json.loads((out / "far-field.json").read_text(encoding="utf-8"))
    assert ff["lines"][0]["transmission"] == pytest.approx(meta["transmission"][0])
    z = np.load(out / "far-field-fb.npz")
    ring = z["ring"][0]
    assert np.allclose(ring, np.roll(ring[::-1], 1), rtol=1e-8, atol=1e-12 * ring.max())      # P(phi) = P(-phi)
    # a large share of the exit power lies on the ring band |theta| ~ theta_in
    assert ring.sum() * 2 * math.pi / ring.size > 0.5 * z["p_out"][0]
    dump = np.load(out / "field-capillary.npz")
    assert dump["c_out_line0_node0_bore0"].shape == (dump["m"].size, dump["zeros"].shape[1])
    # a static q = 2 figure breaks the mirror symmetry only when it is rotated off the plane of incidence
    for ang, sym in ((0.0, True), (0.4, False)):
        raw2 = _scene({"entrance_spot": plane, "fb_figure": [{"q": 2, "amp": 2e-9, "angle": ang}],
                       "far_field": {"theta_max": 2.5e-3, "ring": [0.9e-3, 1.5e-3], "n_theta": 101}})
        sim2 = Simulation.from_dict(raw2)
        W.run_wave_stage(sim2, str(tmp_path / f"f{ang}"), sim2.cfg.validate_wave_estimator(), log=lambda *a: None)
        r2 = np.load(tmp_path / f"f{ang}" / "stage16" / "far-field-fb.npz")["ring"][0]
        dev = np.max(np.abs(r2 - np.roll(r2[::-1], 1))) / r2.max()
        assert (dev < 1e-8) == sym


def test_config_contract_of_the_new_keys(tmp_path):
    ok = Simulation.from_dict(_scene({"fb_figure": [{"q": 2, "amp": 1e-9, "rate": 10.0}],
                                      "fb_figure_noise": {"sigma": 1e-10, "lc": 1e-3, "q": [0, 2, 3], "seed": 4},
                                      "entrance_spot": {"center": [0, 0], "fwhm": 1e-7, "focus": -2e-4}}))
    w = ok.cfg.validate_wave_estimator()
    assert w["fb_figure"][0]["period"] is None and w["fb_figure_noise"]["q"] == [0, 2, 3]
    assert w["entrance_spot"]["focus"] == -2e-4
    for bad in ({"fb_figure": [{"q": -1, "amp": 1e-9}]}, {"fb_figure": [{"q": 2}]}, {"fb_figure": [{"q": 2, "amp": 1e-9, "period": 0}]},
                {"fb_figure": {"q": 2}}, {"fb_figure_noise": {"sigma": 1e-9, "lc": 0.0, "q": [2], "seed": 1}},
                {"fb_figure_noise": {"sigma": 1e-9, "lc": 1e-3, "q": [2, 2], "seed": 1}}, {"fb_wall": "exact-q"},
                {"hard_wall": "yes"}, {"fb_dz": 0.0}, {"far_field": {"theta_max": 1e-3, "ring": [2e-3, 1e-3]}},
                {"entrance_spot": {"center": [0, 0], "fwhm": 1e-7, "focus": "far"}}):
        with pytest.raises(ValueError, match="wave_estimator"):
            Simulation.from_dict(_scene(bad)).cfg.validate_wave_estimator()
    for extra, match in (({"hard_wall": True}, "hard_wall"),):
        sim = Simulation.from_dict(_scene(extra))
        with pytest.raises(ValueError, match=match):
            W.run_wave_stage(sim, str(tmp_path / "x"), sim.cfg.validate_wave_estimator(), log=lambda *a: None)


# ---------------------------------------------------------------- entrance spot focus and the hard wall

def test_spot_focus_is_the_free_propagation_of_the_waist():
    spot = {"center": [0.2e-6, 0.0], "fwhm": 100e-9, "tilt": [0.5e-3, 0.0]}
    h, n = 2e-9, 1 << 13
    x = (np.arange(n) - n // 2) * h
    waist = W.spot_axis(spot, K, x, 0)
    f = -150e-6                                                      # waist 150 um before the entrance
    kx = 2 * np.pi * sfft.fftfreq(n, h)
    prop = sfft.ifft(sfft.fft(waist) * np.exp(-1j * kx ** 2 * (-f) / (2 * K)))
    # the propagated beam is centred at c + t |f|; the spot is defined by its centre on the entrance plane
    moved = W.spot_axis(dict(spot, focus=f, center=[spot["center"][0] + spot["tilt"][0] * (-f), 0.0]), K, x, 0)
    moved = moved * np.exp(-1j * K * spot["tilt"][0] ** 2 * (-f) / 2)    # plane-wave phase of the tilt over |f|
    m = np.abs(x - 0.275e-6) < 0.6e-6
    assert np.max(np.abs(prop[m] - moved[m])) < 1e-9 * np.max(np.abs(moved))
    assert np.sum(np.abs(moved) ** 2) * h == pytest.approx(1.0, rel=1e-6)


def test_images_hard_wall_is_the_dirichlet_slab():
    a, L, lam = 125e-9, 1.0e-3, 12.398419843320026e-10 / 11.24
    k = 2 * math.pi / lam
    h = 2.5e-9
    ax = WI.AxisSolver(0.0, a, 1e-3, L, h, 4, int(math.ceil(8e-3 * L / (2 * a))) + 4, 8e-3 * L)
    ent = lambda x: np.exp(-((x - 30e-9) ** 2) / (2 * (40e-9) ** 2)) * np.exp(1j * k * 0.6e-3 * x)
    u = ax.exit_line(ent, k, 3.86e-6, 4.6e-8, hard=True)
    xs = ax.exit_xs
    W2 = 2 * a
    ref = np.zeros(xs.size, complex)
    xq = np.linspace(-a, a, 8001)
    for nn in range(1, 160):
        phi = lambda x: np.sin(nn * np.pi * (x + a) / W2)
        cn = np.trapezoid(ent(xq) * phi(xq), xq) * 2 / W2
        ref += cn * np.exp(-1j * (nn * np.pi / W2) ** 2 * L / (2 * k)) * np.where(np.abs(xs) <= a, phi(xs), 0.0)
    inside = np.abs(xs) < a - 2 * h
    assert np.max(np.abs(u[inside] - ref[inside])) < 2e-2 * np.max(np.abs(ref))

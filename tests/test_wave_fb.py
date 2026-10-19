"""Stage 16 provider fb (Fourier-Bessel modes of circular / torus bores): basis, free-space
and Gaussian-beam oracles, wall damping, Chebyshev vs dense exponential,
bend limits and symmetry, bend frame against the tracer's torus and against a free beam,
end-to-end two-bore run, provider resolution and config contract."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from formula.capsysred import Simulation
from formula.formula import Number
from formula.capsysred.stages import wave as W
from formula.capsysred.stages import wave_fb as FB
from formula.capsysred.walls.wall_torus import TorusWall

LAM = 12.398419843320026e-10 / 8.048
K = 2.0 * math.pi / LAM
DELTA, BETA = 7.1e-6, 1.0e-7


def _scene(bores, z1=0.03, screen_z=None, nodes=1, jmax=40, extra=None, screens=(), reference=(0.0, 0.0)):
    raw = {"precision": 32, "energy_kev": 8.048, "seed": 1,
           "capillary": {"bores": bores, "z0": 0.0, "z1": z1,
                         "source": {"shape": "gaussian", "size": 1e-7, "position": [0.0, 0.0, -0.05],
                                    "n_modes": 1, "n_rays": 1},
                         "screen": {"z": z1 if screen_z is None else screen_z, "center": [0.0, 0.0],
                                    "edge_x": 20e-6, "edge_y": 2e-6, "nx": 40, "ny": 4, "reference": list(reference)},
                         "screens": list(screens)},
           "wave_estimator": {"source_nodes": nodes, "fb_jmax": jmax, **(extra or {})}}
    return raw


# ---------------------------------------------------------------- config and provider

def test_fb_config_and_provider_resolution():
    raw = _scene([{"center": [0.0, 0.0], "radius": 3e-6}])
    wave = Simulation.from_dict(raw).cfg.validate_wave_estimator()
    assert wave["fb_wall"] == "dir-ell" and wave["fb_jmax"] == 40
    for bad in ({"fb_wall": "robin"}, {"fb_propagator": "split_step"}, {"fb_jmax": 2}, {"fb_jobs": 0},
                {"fb_grid_dtype": "float32"}, {"fb_per_bore_maps": "yes"}, {"fb_angle_margin": 0.5}):
        r = _scene([{"center": [0.0, 0.0], "radius": 3e-6}], extra=bad)
        with pytest.raises(ValueError, match="wave_estimator"):
            Simulation.from_dict(r).cfg.validate_wave_estimator()
    cap = Simulation.from_dict(raw).cfg.capillary
    assert W.capillary_provider(cap, {"provider": "auto"}) == "fb"
    with pytest.raises(ValueError, match="uisk supports polygon"):
        W.capillary_provider(cap, {"provider": "uisk"})
    sq = Simulation.from_dict(_scene([{"center": [0.0, 0.0], "radius": 3e-6, "sides": 4}])).cfg.capillary
    assert W.capillary_provider(sq, {"provider": "auto"}) == "uisk"
    with pytest.raises(ValueError, match="fb supports circular"):
        W.capillary_provider(sq, {"provider": "fb"})
    mixed = Simulation.from_dict(_scene([{"center": [0.0, 0.0], "radius": 3e-6, "sides": 4},
                                         {"center": [8e-6, 0.0], "radius": 3e-6}])).cfg.capillary
    with pytest.raises(ValueError, match="no provider"):
        W.capillary_provider(mixed, {"provider": "auto"})
    # a screen on the exit plane is accepted by fb and still rejected by uisk
    W.preflight_wave_inputs(Simulation.from_dict(raw), Simulation.from_dict(raw).cfg.validate_wave_estimator())
    sq_exit = _scene([{"center": [0.0, 0.0], "radius": 3e-6, "sides": 4}])
    with pytest.raises(ValueError, match="exit plane"):
        W.preflight_wave_inputs(Simulation.from_dict(sq_exit), Simulation.from_dict(sq_exit).cfg.validate_wave_estimator())


def test_wall_offset_dir_ell():
    ell = FB.wall_offset(K, DELTA, BETA)
    assert ell.real == pytest.approx(1.0 / (K * math.sqrt(2 * DELTA)), rel=2e-4)   # 6.5 nm
    assert 0.0 < ell.imag < 0.02 * ell.real                                       # absorption: small positive


# ---------------------------------------------------------------- basis and oracles

def test_disc_transform_pair():
    d = FB.Disc(3e-6, 40)
    rng = np.random.default_rng(0)
    c = (rng.standard_normal(d.kap2.shape) + 1j * rng.standard_normal(d.kap2.shape)) * d.valid
    back = d.to_modes(d.to_grid(c))
    assert np.max(np.abs(back - c)) < 1e-11 * np.max(np.abs(c))


def test_free_space_point_source_sphere():
    """Wide disc, point source on the axis: the exit field equals the paraxial spherical
    wave at z_in + L up to the rim diffraction (knife-edge ripples of a few % at v ~ 10)."""
    aw, z_in, L = 20e-6, 0.5, 0.02
    d = FB.Disc(aw, 120)
    d.fine_table(20e-9)
    ch = FB.FBChannel((0.0, 0.0), None, None, d, K)
    c0 = ch.entrance_modes((0.0, 0.0), z_in)
    cL, _ = ch.propagate_chebyshev(c0, L)
    x = np.arange(-8e-6, 8e-6, 0.2e-6)
    E = ch.lab_field(cL, L, x, x, 20e-9)
    X, Y = np.meshgrid(x, x)
    z = z_in + L
    Ex = np.exp(1j * K * (X ** 2 + Y ** 2) / (2 * z)) / (1j * LAM * z)
    m = np.hypot(X, Y) < 6e-6
    assert np.sqrt(np.mean(np.abs(E - Ex)[m] ** 2) / np.mean(np.abs(Ex)[m] ** 2)) < 5e-2
    assert np.sqrt(np.mean(np.abs(E[m]) ** 2) / np.mean(np.abs(Ex[m]) ** 2)) == pytest.approx(1.0, abs=3e-2)
    assert abs(np.angle(np.vdot(Ex[m], E[m]))) < 5e-2                    # normalisation 1/(i lambda z) and chirp


def test_gaussian_beam_q_law():
    """Straight wide bore: a Gaussian beam follows q -> q + L with amplitude q0/q (complex,
    no fitted constant)."""
    aw, L, w0 = 15e-6, 0.02, 2.5e-6
    d = FB.Disc(aw, 100)
    d.fine_table(20e-9)
    ch = FB.FBChannel((0.0, 0.0), None, None, d, K)
    xi = d.r[:, None] * np.cos(d.phi)[None, :]
    eta = d.r[:, None] * np.sin(d.phi)[None, :]
    q0 = -1j * math.pi * w0 ** 2 / LAM
    c0 = d.to_modes(np.exp(1j * K * (xi ** 2 + eta ** 2) / (2 * q0)))
    cL, _ = ch.propagate_chebyshev(c0, L)
    x = np.arange(-6e-6, 6e-6, 0.2e-6)
    E = ch.lab_field(cL, L, x, x, 20e-9)
    X, Y = np.meshgrid(x, x)
    qL = q0 + L
    Ex = (q0 / qL) * np.exp(1j * K * (X ** 2 + Y ** 2) / (2 * qL))
    assert np.sqrt(np.mean(np.abs(E - Ex) ** 2) / np.mean(np.abs(Ex) ** 2)) < 1e-4


def test_unitarity_and_wall_damping():
    aw = 3e-6
    d = FB.Disc(aw, 30)
    ell = FB.wall_offset(K, DELTA, BETA)
    gamma = d.kap2 * ell.imag / (aw * K)
    c0 = FB.FBChannel((0.0, 0.0), (1.0, 0.0), 500.0, d, K).entrance_modes((0.2e-6, 0.0), 0.05)
    lossless = FB.FBChannel((0.0, 0.0), (1.0, 0.0), 500.0, d, K)
    assert d.norm(lossless.propagate_chebyshev(c0, 0.02)[0]) / d.norm(c0) == pytest.approx(1.0, abs=1e-12)
    # straight bore: every mode decays as exp(-2 gamma L) in norm
    lossy = FB.FBChannel((0.0, 0.0), None, None, d, K, gamma)
    e = np.zeros(d.kap2.shape, complex)
    e[d.M + 2, 1] = 1.0                       # one valid mode (m = 2, n = 2)
    expected = math.exp(-2.0 * gamma[d.M + 2, 1] * 0.02)
    assert d.norm(lossy.propagate_chebyshev(e, 0.02)[0]) / d.norm(e) == pytest.approx(expected, rel=1e-10)
    assert expected < 1.0


def test_chebyshev_matches_dense_exponential():
    aw = 3e-6
    d = FB.Disc(aw, 20)
    ell = FB.wall_offset(K, DELTA, BETA)
    gamma = d.kap2 * ell.imag / (aw * K)
    ch = FB.FBChannel((0.0, 0.0), (1.0, 0.0), 2000.0, d, K, gamma)
    c0 = ch.entrance_modes((0.3e-6, 0.1e-6), 0.05)
    assert np.linalg.norm(ch.apply_h(c0) - ch.apply_h_grid(c0)) < 1e-13 * np.linalg.norm(ch.apply_h_grid(c0))
    n = c0.size
    H = np.zeros((n, n), complex)
    for j in range(n):
        e = np.zeros(n, complex)
        e[j] = 1.0
        H[:, j] = ch.apply_h(e.reshape(c0.shape)).ravel()
    L = 0.03
    exact = (expm(-1j * H * L) @ c0.ravel()).reshape(c0.shape)
    cheb, deg = ch.propagate_chebyshev(c0, L, 1e-12)
    assert np.linalg.norm(cheb - exact) < 1e-11 * np.linalg.norm(exact)
    assert deg > 10


# ---------------------------------------------------------------- bend

def _lab_exit(ch, cL, L, x):
    cz = ch.axis(L)
    return ch.lab_field(cL, L, cz[0] + x, cz[1] + x, 20e-9)


def test_bend_limit_and_mirror_symmetry():
    aw, L = 3e-6, 0.03
    d = FB.Disc(aw, 40)
    d.fine_table(20e-9)
    x = np.arange(-2.8e-6, 2.81e-6, 0.1e-6)
    straight = FB.FBChannel((0.0, 0.0), None, None, d, K)
    far = FB.FBChannel((0.0, 0.0), (1.0, 0.0), 1e12, d, K)
    c0 = straight.entrance_modes((0.0, 0.0), 0.05)
    Es = _lab_exit(straight, straight.propagate_chebyshev(c0, L)[0], L, x)
    Ef = _lab_exit(far, far.propagate_chebyshev(c0, L)[0], L, x)
    assert np.max(np.abs(Ef - Es)) < 1e-8 * np.max(np.abs(Es))          # (a) R -> infinity
    plus = FB.FBChannel((0.0, 0.0), (1.0, 0.0), 50.0, d, K)
    minus = FB.FBChannel((0.0, 0.0), (-1.0, 0.0), 50.0, d, K)
    Ep = _lab_exit(plus, plus.propagate_chebyshev(c0, L)[0], L, x)
    Em = _lab_exit(minus, minus.propagate_chebyshev(c0, L)[0], L, x)
    assert np.max(np.abs(Ep - Em[:, ::-1])) < 1e-9 * np.max(np.abs(Ep))  # (b) mirror x -> -x
    assert np.max(np.abs(Ep - Es)) > 1e-2 * np.max(np.abs(Es))            # the bend does act


def test_bend_frame_against_free_gaussian_beam():
    """Accelerated frame: a Gaussian beam far from the walls propagates straight in the lab;
    the frame solution transformed back (sag, tilt, carrier g) must equal it, phase included."""
    aw, L, R, w0 = 10e-6, 0.01, 50.0, 3e-6
    d = FB.Disc(aw, 110)
    d.fine_table(20e-9)
    ch = FB.FBChannel((20e-6, 0.0), (-1.0, 0.0), R, d, K)
    xi = d.r[:, None] * np.cos(d.phi)[None, :]
    eta = d.r[:, None] * np.sin(d.phi)[None, :]
    q0 = -1j * math.pi * w0 ** 2 / LAM
    c0 = d.to_modes(np.exp(1j * K * (xi ** 2 + eta ** 2) / (2 * q0)))      # beam on the lab axis x = 20 um
    cL, _ = ch.propagate_chebyshev(c0, L)
    x = np.arange(20e-6 - 5e-6, 20e-6 + 5e-6, 0.2e-6)
    y = np.arange(-5e-6, 5e-6, 0.2e-6)
    E = ch.lab_field(cL, L, x, y, 20e-9)
    X, Y = np.meshgrid(x, y)
    qL = q0 + L
    Ex = (q0 / qL) * np.exp(1j * K * ((X - 20e-6) ** 2 + Y ** 2) / (2 * qL))
    m = np.hypot(X - 20e-6, Y) < 4e-6
    rel = np.sqrt(np.mean(np.abs(E - Ex)[m] ** 2) / np.mean(np.abs(Ex)[m] ** 2))
    assert rel < 2e-3
    assert abs(np.angle(np.vdot(Ex[m], E[m]))) < 1e-2                        # no fitted phase
    assert abs(ch.axis(L)[0] - (20e-6 - L * L / (2 * R))) < 1e-15             # sag toward the bend centre


def test_bend_axis_matches_tracer_torus():
    """Wall hits of the tracer's torus (exact quartic) against the parabolic axis
    c(z) = c0 + toward z^2/(2R) of the wave frame, both walls."""
    p = 32
    a, R, z0 = 3e-6, 50.0, 0.0
    center = (Number("1e-6", p), Number("-2e-6", p))
    toward = (Number("0.6", p), Number("0.8", p))
    wall = TorusWall(center, Number(str(a), p), Number(str(R), p), toward, Number(str(z0), p))
    for sign in (+1.0, -1.0):
        theta = 5e-4
        ux, uy = 0.6, 0.8
        O = (center[0], center[1], Number("0", p))
        dvec = (Number(str(sign * theta * ux), p), Number(str(sign * theta * uy), p), Number("1", p))
        t, P, n = wall.hit(O, dvec, Number("0.1", p))
        z_hit = float(P[2])
        # parabolic model: axis displacement toward +u is z^2/(2R); ray x = c0 +- u theta z
        disc = theta ** 2 - sign * 2 * a / R
        z_model = R * (theta - math.sqrt(disc)) if sign > 0 else R * (-theta + math.sqrt(theta ** 2 + 2 * a / R))
        assert abs(z_hit - z_model) < 2e-9
        r_model = a - 0.0                                                    # hit on the wall
        xw = float(P[0]) - (1e-6 + ux * z_hit ** 2 / (2 * R))
        yw = float(P[1]) - (-2e-6 + uy * z_hit ** 2 / (2 * R))
        assert abs(math.hypot(xw, yw) - r_model) < 2e-12


# ---------------------------------------------------------------- stage end to end

def _two_bores(bend=None, **extra):
    bores = [{"center": [-4e-6, 0.0], "radius": 2.5e-6}, {"center": [4e-6, 0.0], "radius": 2.5e-6}]
    if bend:
        bores[0]["bend"] = {"radius": bend, "toward": [1.0, 0.0]}
        bores[1]["bend"] = {"radius": bend, "toward": [-1.0, 0.0]}
    return _scene(bores, z1=0.02, nodes=1, jmax=36, extra={"pixel_subsamples": 2, **extra}, reference=(4e-6, 0.0),
                  screens=[{"z": 0.03, "nx": 40, "ny": 4, "edge_x": 20e-6, "edge_y": 2e-6, "reference": [4e-6, 0.0]}])


def test_fb_stage_jobs_split_is_exact(tmp_path):
    """The multiprocess node split reproduces the single-process accumulators."""
    scene = _two_bores(bend=2000.0)
    scene["capillary"]["source"]["size"] = 1.5e-6
    scene["wave_estimator"]["source_nodes"] = 4
    outs = []
    for jobs in (1, 2):
        scene["wave_estimator"]["fb_jobs"] = jobs
        sim = Simulation.from_dict(scene)
        outs.append(W.run_wave_stage(sim, str(tmp_path / f"j{jobs}"), sim.cfg.validate_wave_estimator(), log=lambda *a: None))
    for label in outs[0]["results"]:
        r1, r2 = outs[0]["results"][label], outs[1]["results"][label]
        assert r2["meta"]["jobs"] == 2 and r1["meta"]["n_nodes"] == 4
        for key in ("I", "W", "I_ref", "I_pixel"):
            assert np.allclose(r1[key], r2[key], rtol=1e-10, atol=1e-13 * np.abs(r1["I"]).max())
        assert np.allclose(r1["G12"], r2["G12"], rtol=1e-10, atol=1e-13 * np.abs(r1["I"]).max())


def test_fb_stage_two_bores(tmp_path):
    sim = Simulation.from_dict(_two_bores(bend=2000.0))
    wave = sim.cfg.validate_wave_estimator()
    res = W.run_wave_stage(sim, str(tmp_path), wave, log=lambda *a: None)
    assert (tmp_path / "stage16" / "mu-wave.jsonl").exists()
    assert (tmp_path / "stage16" / "screen-1" / "mu-wave.jsonl").exists()
    for label, r in res["results"].items():
        m = r["meta"]
        assert m["provider"] == "fb" and m["model_class"].startswith("dir-ell") and m["propagator"] == "chebyshev"
        assert m["chebyshev_degree_median"] > 0 and m["wall_offset_nm"][0][0] == pytest.approx(6.49, rel=0.02)
        assert r["n_trusted"] > 0 and r["over_unity"] == 0
        ix, iy = r["ref_index"]
        assert abs(r["mu"][ix, iy]) == pytest.approx(1.0, abs=1e-9)
        assert r["I_bore"] is not None and r["G12"] is not None
        assert np.allclose(r["I"], r["I_bore"][0] + r["I_bore"][1] + 2 * r["G12"].real, rtol=1e-9, atol=1e-12 * r["I"].max())
        assert r["I_pixel"] is not None and np.all(r["I_pixel"] >= 0.0)
        assert m["fresnel_numbers"]["exit"] is None if label == "capillary" else m["fresnel_numbers"]["exit"] > 0
    # exit-plane screen is the exit field itself: no light outside the bores (regression: the
    # worker once drifted the exit field by -(z_in) to the source plane)
    r0 = res["results"]["capillary"]
    px = np.array(r0["sampler"].px); py = np.array(r0["sampler"].py)
    PX, PY = np.meshgrid(px, py, indexing="ij")
    inside = (np.hypot(PX + 4e-6, PY) < 2.5e-6 + 0.6e-6) | (np.hypot(PX - 4e-6, PY) < 2.5e-6 + 0.6e-6)
    assert r0["I"][~inside].sum() < 1e-3 * r0["I"].sum()
    # summed-field path (no per-bore maps) gives the same total intensity
    sim3 = Simulation.from_dict(_two_bores(bend=2000.0, fb_per_bore_maps=False))
    res3 = W.run_wave_stage(sim3, str(tmp_path / "c"), sim3.cfg.validate_wave_estimator(), log=lambda *a: None)
    for label in res["results"]:
        assert res3["results"][label]["I_bore"] is None
        assert np.allclose(res3["results"][label]["I"], res["results"][label]["I"], rtol=1e-9, atol=1e-12 * r0["I"].max())
        assert np.allclose(res3["results"][label]["W"], res["results"][label]["W"], rtol=1e-9, atol=1e-12 * r0["I"].max())


# ---------------------------------------------------------------- guided-mode cut and Fresnel damping

def test_disc_wall_cut_keeps_guided_modes_only():
    aw, J = 1e-6, 155.0
    cut, full = FB.Disc(aw, 300, wall_cut=J), FB.Disc(aw, 300)
    assert cut.M == full.M and cut.nmodes < full.nmodes
    for m, z in enumerate(cut.zeros):
        assert np.all(z * z - m * m < J * J) and np.all(z < 300)
        dropped = full.zeros[m][~np.isin(full.zeros[m], z)]
        assert np.all(dropped * dropped - m * m >= J * J)
    assert cut.zeros[0].max() < J < full.zeros[0].max()
    # nothing to cut below the kappa cap: the basis is unchanged
    a, b = FB.Disc(aw, 120, wall_cut=J), FB.Disc(aw, 120)
    assert a.M == b.M and all(np.array_equal(x, y) for x, y in zip(a.zeros, b.zeros))


def test_mode_damping_fresnel_factor():
    aw, thc = 1e-6, math.sqrt(2 * DELTA)
    d = FB.Disc(aw, int(K * aw * 0.9 * thc), wall_cut=K * aw * thc)
    g = FB.mode_damping(d, K, DELTA, BETA)
    base = d.kap2 * FB.wall_offset(K, DELTA, BETA).imag / (aw * K)
    x = np.maximum(d.kap2 - (d.am[:, None] / aw) ** 2, 0.0) / (K * thc) ** 2
    v = d.valid
    assert np.isfinite(g).all() and np.all(g[v] > 0.0) and np.all(x[v] < 0.81)
    # (1 - x)^(-1/2) up to the first-order beta/delta term of the exact Fresnel loss
    tol = 3.0 * np.sqrt(x[v]) * (BETA / DELTA) / np.sqrt(1.0 - x[v]) + 1e-6
    assert np.all(np.abs(g[v] / base[v] * np.sqrt(1.0 - x[v]) - 1.0) < tol)
    assert abs(g[d.M, 0] / base[d.M, 0] - 1.0) < 1e-3                       # m = 0, n = 1: grazing limit
    one = SimpleNamespace(kap2=np.array([[(K * thc) ** 2]]), am=np.array([0]), aw=aw)
    assert np.isfinite(FB.mode_damping(one, K, DELTA, BETA)).all()          # critical angle: finite


def test_fb_wall_cut_config():
    raw = _scene([{"center": [0.0, 0.0], "radius": 3e-6}])
    assert Simulation.from_dict(raw).cfg.validate_wave_estimator()["fb_wall_cut"] is True
    bad = _scene([{"center": [0.0, 0.0], "radius": 3e-6}], extra={"fb_wall_cut": "yes"})
    with pytest.raises(ValueError, match="fb_wall_cut"):
        Simulation.from_dict(bad).cfg.validate_wave_estimator()
    sim = Simulation.from_dict(raw)
    spec = FB.build_spec(sim, sim.cfg.validate_wave_estimator(), [(K, 1.0, DELTA, BETA)])
    assert spec["wall_cut"] == pytest.approx(K * spec["aw"] * math.sqrt(2 * DELTA))
    off = Simulation.from_dict(_scene([{"center": [0.0, 0.0], "radius": 3e-6}], extra={"fb_wall_cut": False}))
    assert FB.build_spec(off, off.cfg.validate_wave_estimator(), [(K, 1.0, DELTA, BETA)])["wall_cut"] is None


# ---------------------------------------------------------------- entrance spot (whispering-gallery launch)

def test_entrance_spot_modes_and_stage(tmp_path):
    aw = 1e-6 + FB.wall_offset(K, DELTA, BETA).real
    thc = math.sqrt(2 * DELTA)
    d = FB.Disc(aw, int(K * aw * 1.15 * thc), wall_cut=K * aw * thc)
    ch = FB.FBChannel((0.0, 0.0), None, None, d, K)
    r0, tilt = aw - 51e-9, 2.31e-3
    spot = {"center": [r0, 0.0], "fwhm": 100e-9, "tilt": [0.0, tilt]}
    c = ch.spot_modes(spot)
    P = d.norm(c)
    assert 0.8 < P < 1.0                                                  # unit-power gaussian; the tail beyond the wall is cut
    m_mean = float(np.sum(np.array(d.ms)[:, None] * np.abs(c) ** 2 * d.mnorm) / P)
    assert abs(m_mean - K * r0 * tilt) < 3.0                               # angular momentum of the tilted spot
    assert np.sum(np.abs(c[:, 0]) ** 2 * d.mnorm[:, 0]) / P > 0.9           # whispering-gallery family n = 1
    raw = _scene([{"center": [0.0, 0.0], "radius": 1e-6}], z1=0.005, jmax=None,
                 extra={"entrance_spot": spot, "fb_theta_cut": 4.5e-3, "observable": "point"})
    raw["capillary"]["screen"].update({"edge_x": 2.2e-6, "edge_y": 2.2e-6, "nx": 11, "ny": 11, "reference": [r0, 0.0]})
    sim = Simulation.from_dict(raw)
    wave = sim.cfg.validate_wave_estimator()
    assert wave["entrance_spot"]["tilt"] == [0.0, tilt]
    res = W.run_wave_stage(sim, str(tmp_path), wave, log=lambda *a: None)
    r = res["results"]["capillary"]
    assert r["meta"]["n_nodes"] == 1 and r["meta"]["source_rule"].startswith("entrance spot")
    assert r["meta"]["fresnel_numbers"]["source"] is None
    assert np.isfinite(r["I"]).all() and r["I"].sum() > 0.0
    for bad in ({"entrance_spot": {"center": [0.0], "fwhm": 1e-7}}, {"entrance_spot": {"center": [0, 0], "fwhm": 0.0}},
                {"entrance_spot": {"fwhm": 1e-7}}, {"entrance_spot": {"center": [0, 0], "fwhm": 1e-7, "waist": 1}}):
        with pytest.raises(ValueError, match="entrance_spot"):
            Simulation.from_dict(_scene([{"center": [0.0, 0.0], "radius": 1e-6}], extra=bad)).cfg.validate_wave_estimator()
    sq = _scene([{"center": [0.0, 0.0], "radius": 1e-6, "sides": 4}], extra={"entrance_spot": spot})
    with pytest.raises(ValueError, match="entrance_spot is implemented for provider fb"):
        W.run_wave_stage(Simulation.from_dict(sq), str(tmp_path / "sq"), Simulation.from_dict(sq).cfg.validate_wave_estimator(), log=lambda *a: None)


def test_propagate_is_exact_for_straight_bores_and_substeps_strong_damping():
    aw, thc = 1e-6 + FB.wall_offset(K, DELTA, BETA).real, math.sqrt(2 * DELTA)
    d = FB.Disc(aw, int(K * aw * 0.8 * thc), wall_cut=K * aw * thc)        # kappa cap below theta_c: a small basis
    g = FB.mode_damping(d, K, DELTA, BETA)
    spot = {"center": [aw - 51e-9, 0.0], "fwhm": 100e-9, "tilt": [0.0, 2.31e-3]}
    L = 0.03
    straight = FB.FBChannel((0.0, 0.0), None, None, d, K, g)
    c0 = straight.spot_modes(spot)
    out, deg = straight.propagate(c0, L)
    assert deg == 0 and np.allclose(out, c0 * np.exp(-1j * straight.h0 * L), rtol=0, atol=0)
    T = d.norm(out) / d.norm(c0)
    assert 0.4 < T < 0.75                                                 # whispering-gallery launch, 30 mm
    bent = FB.FBChannel((0.0, 0.0), (1.0, 0.0), 100.0, d, K, g)
    one, deg1 = bent.propagate(c0, L)
    two_a, _ = bent.propagate(c0, L / 2)
    two, deg2 = bent.propagate(two_a, L / 2)
    assert deg1 > 100 and np.isfinite(one).all()
    assert np.linalg.norm(one - two) < 1e-8 * np.linalg.norm(one)         # sub-step count does not matter
    assert abs(d.norm(one) / d.norm(c0) - T) < 0.05                        # R = 100 m barely changes the loss
    # (the unsplit series on the full length diverges: T_n(x - iy) grows as exp(n y / sqrt(1 - x^2)))

"""Stage 16 (wave estimator): config contract, free-scene oracle, UISK invariants,
two-bore run, source/replay contracts and Stage-14 isolation."""

import json
import math
import os

import numpy as np
import pytest
from scipy.special import fresnel, j1

from formula.capsysred import Simulation, render
from formula.capsysred.config import WAVE_DEFAULTS
from formula.capsysred.screen import ScreenGrid
from formula.capsysred.simulation import KNOWN_STAGES
from formula.capsysred.stages import wave as W

LAM = 12.398419843320026e-10 / 8.048
K = 2.0 * math.pi / LAM


def _free(reference=(2e-6, -4e-6)):
    return {"precision": 32, "energy_kev": 8.048,
            "free": {"source": {"shape": "disk", "size": 2e-6, "position": [0.7e-6, -0.9e-6, -0.3],
                                "n_modes": 1, "n_rays": 1},
                     "screen": {"z": 0.0, "center": [0.0, 0.0], "edge_x": 40e-6, "edge_y": 20e-6,
                                "nx": 41, "ny": 21, "reference": list(reference)}}}


def _square(source_shape="point", size=0.0, nx=41, rotation_deg=0, screen_z=0.183):
    return {"precision": 32, "energy_kev": 8.048,
            "capillary": {"bores": [{"center": [0.0, 0.0], "radius": 24e-6, "sides": 4,
                                     "rotation_deg": rotation_deg}],
                          "z0": 0.0, "z1": 0.0525,
                          "source": {"shape": source_shape, "size": size,
                                     "position": [0.0, 0.0, -0.117], "n_modes": 1, "n_rays": 1},
                          "screen": {"z": screen_z, "center": [0.0, 0.0], "edge_x": 6e-6,
                                     "edge_y": 6e-6, "nx": nx, "ny": nx, "reference": [0.0, 0.0]}}}


def _two_bores(nx=41, ny=5, observable="coherent_cell"):
    return {"precision": 32, "energy_kev": 8.048, "seed": 1,
            "capillary": {"bores": [{"center": [-4.5e-6, 0.0], "radius": 3e-6, "sides": 4},
                                    {"center": [4.5e-6, 0.0], "radius": 3e-6, "sides": 4}],
                          "z0": 0.0, "z1": 0.03,
                          "source": {"shape": "gaussian", "size": 1e-7, "position": [0.0, 0.0, -0.05],
                                     "n_modes": 1, "n_rays": 1},
                          "screen": {"z": 0.13, "center": [0.0, 0.0], "edge_x": 40e-6, "edge_y": 4e-6,
                                     "nx": nx, "ny": ny, "reference": [5e-6, 0.0]},
                          "screens": [{"z": 0.13, "nx": 21, "ny": 1, "edge_x": 40e-6, "edge_y": 1e-6,
                                       "reference": [5e-6, 0.0]}]},
            "wave_estimator": {"source_nodes": 4, "observable": observable, "pixel_subsamples": 2}}


def _lines(sim):
    return [(float(ln.k), float(ln.weight), sim.delta_f, sim.beta_f) for ln in sim.lines]


# ---------------------------------------------------------------- config contract

def test_wave_section_is_lazy_and_validated():
    sim = Simulation.from_dict(_free())
    assert "wave_estimator" not in sim.cfg.raw          # absent section leaves raw untouched
    wave = sim.cfg.validate_wave_estimator()
    assert wave == {**WAVE_DEFAULTS, **{k: float(v) for k, v in WAVE_DEFAULTS.items()
                                         if isinstance(v, float)}}
    assert 16 in KNOWN_STAGES
    for bad in ({"provider": "bpm"}, {"observable": "intensity_pixel"}, {"source_mode": "mc"},
                {"target_error": 0}, {"source_nodes": 0}, {"grid_dx": -1e-9}, {"angle_margin": 0.5},
                {"max_bounces": 1.5}, {"intensity_floor": 2}, {"nodes": 5}, {"workers": True}):
        raw = _free()
        raw["wave_estimator"] = bad
        with pytest.raises(ValueError, match="wave_estimator"):
            Simulation.from_dict(raw).cfg.validate_wave_estimator()
    raw = _free()
    raw["wave_estimator"] = {"grid_dx": 1e-7, "workers": 2, "source_nodes": 12}
    wave = Simulation.from_dict(raw).cfg.validate_wave_estimator()
    assert wave["grid_dx"] == 1e-7 and wave["workers"] == 2 and wave["source_nodes"] == 12


def test_stage14_contract_ignores_wave_section():
    from formula.capsysred.rays import geometry_metadata, sidecar_metadata
    from formula.capsysred.stages.stage14 import _analysis_signature, _screen_contract

    base = Simulation.from_dict(_square("disk", 3e-7))
    raw = _square("disk", 3e-7)
    raw["wave_estimator"] = {"provider": "uisk", "grid_dx": 1e-7, "pad": 3}
    other = Simulation.from_dict(raw)
    assert geometry_metadata(other.cfg) == geometry_metadata(base.cfg)
    assert sidecar_metadata(other.cfg) == sidecar_metadata(base.cfg)
    a, b = ScreenGrid(base.cfg.capillary.screen), ScreenGrid(other.cfg.capillary.screen)
    assert (_analysis_signature(base, _screen_contract(a, a.ref_pixel(base.cfg.capillary.screen.reference)))
            == _analysis_signature(other, _screen_contract(b, b.ref_pixel(other.cfg.capillary.screen.reference))))


def test_recorded_origins_need_an_archive(tmp_path):
    raw = _square("disk", 3e-7)
    raw["wave_estimator"] = {"source_mode": "recorded_origins"}
    sim = Simulation.from_dict(raw)
    with pytest.raises(ValueError, match="recorded_origins needs a rays archive"):
        sim.run(str(tmp_path), stages=[16])
    assert not (tmp_path / "stage16").exists()
    assert not (tmp_path / "stage16.partial").exists()      # failure leaves no partial tree


def test_input_preflight_rejects_bad_spectrum_and_exit_screen(tmp_path):
    raw = _free()
    raw["spectrum"] = {"mode": "lines", "lines": [{"energy_kev": 8.0, "weight": 1.0},
                                                   {"energy_kev": 16.0, "weight": -0.5}]}
    with pytest.raises(ValueError, match="non-negative weights"):
        Simulation.from_dict(raw).run(str(tmp_path / "neg"), stages=[16])
    raw = _square(screen_z=0.0525)                          # screen on the exit plane
    with pytest.raises(ValueError, match="outside the Stage-16 MVP"):
        Simulation.from_dict(raw).run(str(tmp_path / "exit"), stages=[16])
    assert not (tmp_path / "exit" / "stage16").exists()


def test_reference_outside_window():
    raw = _free(reference=(100e-6, 0.0))
    sim = Simulation.from_dict(raw)
    wave = sim.cfg.validate_wave_estimator()
    wave.update(observable="coherent_cell", source_nodes=4)
    with pytest.raises(ValueError, match="outside the screen window"):
        W._free_scene(sim, wave, _lines(sim), None)
    wave.update(observable="point")
    res = W._free_scene(sim, wave, _lines(sim), None)      # point reference: computed exactly
    assert res["ref_index"] is None and res["sampler"].ref_inside is False
    assert np.all(np.abs(res["mu"]) <= 1.0 + 1e-12) and res["I_ref"] > 0


# ---------------------------------------------------------------- sources

def test_source_rules_grid_and_weights():
    class Src:
        shape, size, position = "grid", 0.0, (1e-6, -2e-6, -0.1)
        grid_n, grid_step, grid_rot_deg, grid_r_max = 3, 1e-6, 0.0, None

    nodes, w, label = W.source_rule(Src(), {"source_nodes": 9})
    assert len(nodes) == 9 and np.allclose(w, 1.0 / 9)          # size 0: equal weights, all nodes
    assert np.allclose(nodes.mean(0), [1e-6, -2e-6])
    Src.size = 0.5e-6
    nodes, w, _ = W.source_rule(Src(), {"source_nodes": 9})
    r2 = ((nodes - np.array([1e-6, -2e-6])) ** 2).sum(1)
    assert np.allclose(w, np.exp(-r2 / (2 * 0.25e-12)) / np.exp(-r2 / (2 * 0.25e-12)).sum())
    Src.grid_r_max = 1.1e-6
    nodes, w, _ = W.source_rule(Src(), {"source_nodes": 9})
    assert len(nodes) == 5 and abs(w.sum() - 1.0) < 1e-15

    class Disk:
        shape, size, position = "disk", 2e-6, (0.0, 0.0, -0.3)

    nodes, w, _ = W.source_rule(Disk(), {"source_nodes": 768})
    assert len(nodes) == 4 * 14 * 14 and np.all(w > 0) and abs(w.sum() - 1.0) < 1e-14
    assert abs((nodes ** 2).sum(1) @ w - (2e-6) ** 2 / 2) < 1e-24   # <r^2> of a uniform disk = a^2/2


def _trace_archive(config, archive):
    from formula.capsysred.trace_v3 import trace
    trace(str(config), str(archive), jobs=1, level=1, log=lambda _: None, scenes=("capillary",))


def test_recorded_origins_merge_every_archive(tmp_path):
    import yaml
    from formula.capsysred import rays_v3
    raw = _square("disk", 3e-7)
    raw["seed"] = 271828
    raw["capillary"]["source"].update(n_modes=3, n_rays=40)
    archives = []
    for i, seed in enumerate((271828, 271829)):
        raw["seed"] = seed
        cfg = tmp_path / f"config{i}.yaml"
        cfg.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        archive = tmp_path / f"rays-{i}"
        _trace_archive(cfg, archive)
        archives.append(str(archive))
    src = Simulation.from_dict(raw).cfg.capillary.source
    nodes, w, label = W.recorded_rule(archives, "capillary", src)
    expected = []
    for a in archives:
        expected += [o[:2] for o in rays_v3.origins(a, rays_v3.load_index(a), "capillary")]
    assert len(nodes) == 6 and np.allclose(nodes, np.array(expected, dtype=float))
    assert np.allclose(w, 1.0 / 6) and "2 archive(s)" in label
    raw["wave_estimator"] = {"source_mode": "recorded_origins", "observable": "point",
                             "grid_dx": 4e-7, "pad": 0.5, "max_bounces": 1}
    raw["capillary"]["screen"].update(nx=5, ny=5)
    raw["screen"] = dict(raw["capillary"]["screen"])
    sim = Simulation.from_dict(raw)
    out = tmp_path / "out"
    sim.replay(archives, str(out), stages=[16])
    meta = json.loads((out / "stage16" / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_nodes"] == 6 and "2 archive(s)" in meta["source_rule"]


# ---------------------------------------------------------------- free scene

def test_free_point_matches_complex_vcz_oracle():
    sim = Simulation.from_dict(_free())
    wave = sim.cfg.validate_wave_estimator()
    wave.update(observable="point", source_nodes=768)
    res = W._free_scene(sim, wave, _lines(sim), None)
    s = res["sampler"]
    px, py = np.meshgrid(s.px, s.py, indexing="ij")
    r = np.array(res["ref_xy"])
    xi0 = np.array([0.7e-6, -0.9e-6])
    D, a = 0.3, 2e-6
    u = K * a * np.hypot(px - r[0], py - r[1]) / D
    jinc = np.where(u > 0, 2 * j1(u) / np.where(u > 0, u, 1), 1.0)
    phase = (K * ((px ** 2 + py ** 2) - r @ r) / (2 * D)
             - K * (xi0[0] * (px - r[0]) + xi0[1] * (py - r[1])) / D)
    oracle = np.exp(1j * phase) * jinc
    assert (jinc < 0).sum() > 100                     # negative lobe covered
    assert np.abs(res["mu"] - oracle).max() < 1e-10
    assert res["over_unity"] == 0 and res["n_trusted"] == 41 * 21


def test_free_coherent_cell_is_psd_and_bounded_by_pixel_intensity():
    sim = Simulation.from_dict(_free())
    wave = sim.cfg.validate_wave_estimator()
    wave.update(observable="coherent_cell", source_nodes=64, pixel_subsamples=3)
    res = W._free_scene(sim, wave, _lines(sim), None)
    s = res["sampler"]
    assert abs(abs(res["mu"][res["ref_index"]]) - 1.0) < 1e-12
    assert np.all(np.abs(res["mu"]) <= 1.0 + 1e-12)
    ratio = s.hx * s.hy * res["I_pixel"] / res["I"]     # Cauchy: I_coherent <= area * I_pixel
    assert np.all(ratio >= 1.0 - 1e-6) and ratio.max() > 1.5


def test_broadband_lines_sum_in_w_and_i():
    raw = _free()
    raw["spectrum"] = {"mode": "lines", "lines": [{"energy_kev": 8.048, "weight": 1.0},
                                                   {"energy_kev": 9.0, "weight": 3.0}]}
    sim = Simulation.from_dict(raw)
    wave = sim.cfg.validate_wave_estimator()
    wave.update(observable="point", source_nodes=16)
    res = W._free_scene(sim, wave, _lines(sim), None)
    parts = []
    for e in (8.048, 9.0):
        raw_m = _free()
        raw_m["energy_kev"] = e
        sim_m = Simulation.from_dict(raw_m)
        parts.append(W._free_scene(sim_m, wave, _lines(sim_m), None))
    W_sum = 0.25 * parts[0]["W"] + 0.75 * parts[1]["W"]
    I_sum = 0.25 * parts[0]["I"] + 0.75 * parts[1]["I"]
    assert np.allclose(res["W"], W_sum, rtol=1e-12, atol=0) and np.allclose(res["I"], I_sum, rtol=1e-12)


# ---------------------------------------------------------------- UISK invariants

def test_families_dedup_is_scale_and_rotation_robust():
    for a, rot in ((24e-6, 0.0), (24e-6, math.pi / 4), (3e-6, 0.137), (1e-6, 0.0), (7e-6, 1.1)):
        poly = W.Polygon({"center": [0.0, 0.0], "radius": a, "sides": 4, "rotation": rot})
        assert [len(W.enumerate_families(poly, m)) for m in (2, 3, 4)] == [13, 25, 41], (a, rot)
    square = W.Polygon({"center": [0.0, 0.0], "radius": 24e-6, "sides": 4, "rotation": 0.0})
    fams = W.enumerate_families(square, 2)
    corner = [f for f in fams if len(f.seq) == 2 and abs(f.center[0]) > 1e-9 and abs(f.center[1]) > 1e-9]
    assert len(corner) == 4 and np.allclose(np.abs(corner[0].center), [48e-6, 48e-6])
    assert np.allclose([abs(n @ m) for n, m in zip(corner[0].unfolded, corner[0].unfolded[::-1])], 0.0)
    hexa = W.Polygon({"center": [0.0, 0.0], "radius": 24e-6, "sides": 6, "rotation": 0.3})
    assert len(W.enumerate_families(hexa, 2)) == 1 + 6 + 30      # no commuting pairs at 120 degrees
    one = [f for f in fams if f.seq == (0,)][0]
    assert np.allclose(one.center, [48e-6, 0.0]) and np.allclose(one.unfolded[0], [1.0, 0.0])
    assert np.allclose(one.map_point(np.array([1e-6, 2e-6])), [47e-6, 2e-6])


@pytest.fixture(scope="module")
def square_point_solver():
    sim = Simulation.from_dict(_square())
    wave = sim.cfg.validate_wave_estimator()
    wave.update(observable="point")
    cap = sim.cfg.capillary
    grid = ScreenGrid(cap.screen)
    theta = W._theta_max(cap, np.array([[0.0, 0.0]]), [grid])
    solver = W.BoreSolver(cap.bores[0], 0.117, 0.0525, theta, LAM, wave)
    return sim, wave, solver, grid, theta


def test_square_point_source_is_separable_and_symmetric(square_point_solver):
    sim, wave, solver, grid, theta = square_point_solver
    assert len(solver.families) == 9 and solver.pruned == 4
    u = solver.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    sv = np.linalg.svd(u, compute_uv=False)
    assert sv[1] / sv[0] < 1e-9                        # X(x) Y(y): rank one
    samp = W.ScreenSampler(grid, (0.0, 0.0), "point", 1)
    d = 0.183 - 0.0525
    e, ref, _ = samp.fields(u, samp.kernels(solver.xs, solver.ys, K, d, solver.h), K, d, solver.h)
    assert np.abs(e - e[::-1, :]).max() / np.abs(e).max() < 1e-10
    assert np.abs(e - e.T).max() / np.abs(e).max() < 1e-10
    assert abs(ref - e[20, 20]) / abs(ref) < 1e-10


def test_rotated_square_keeps_symmetries_and_power(square_point_solver):
    sim, wave, solver, grid, theta = square_point_solver
    raw = _square(rotation_deg=45)
    sim_r = Simulation.from_dict(raw)
    cap = sim_r.cfg.capillary
    rot = W.BoreSolver(cap.bores[0], 0.117, 0.0525, theta, LAM, dict(wave, grid_dx=solver.h))
    assert len(rot.families) + rot.pruned == 13         # 13 unique tiles, none duplicated
    u0 = solver.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    ur = rot.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    power = lambda u, h: float(np.sum(np.abs(u) ** 2) * h * h)
    assert abs(power(ur, rot.h) / power(u0, solver.h) - 1.0) < 2e-2
    samp = W.ScreenSampler(grid, (0.0, 0.0), "point", 1)
    d = 0.183 - 0.0525
    e, _, _ = samp.fields(ur, samp.kernels(rot.xs, rot.ys, K, d, rot.h), K, d, rot.h)
    assert np.abs(e - e[::-1, :]).max() / np.abs(e).max() < 1e-9
    assert np.abs(e - e.T).max() / np.abs(e).max() < 1e-9


def test_direct_family_matches_closed_form_fresnel():
    """max_bounces 0: the square aperture lit by a point source is a product of
    Fresnel integrals; the FFT step at h = 66 nm, pad 2.25 reproduces it."""
    sim = Simulation.from_dict(_square())
    wave = sim.cfg.validate_wave_estimator()
    wave.update(max_bounces=0, grid_dx=0.0664e-6, pad=2.25)
    cap = sim.cfg.capillary
    theta = W._theta_max(cap, np.array([[0.0, 0.0]]), [ScreenGrid(cap.screen)])
    solver = W.BoreSolver(cap.bores[0], 0.117, 0.0525, theta, LAM, wave)
    u = solver.exit_field((0.0, 0.0), K, sim.delta_f, sim.beta_f, -1)
    z_in, L, R = 0.117, 0.0525, 24e-6

    def x1d(x):
        A = 1.0 / z_in + 1.0 / L
        c = x / (A * L)
        pre = np.exp(1j * K * x * x * (1.0 / L - 1.0 / (A * L * L)) / 2.0)
        s = math.sqrt(K * A / math.pi)
        s2, c2 = fresnel((R - c) * s)
        s1, c1 = fresnel((-R - c) * s)
        return pre * math.sqrt(math.pi / (K * A)) * ((c2 - c1) + 1j * (s2 - s1))

    worst = 0.0
    for tx, ty in ((0.0, 0.0), (5e-6, 3e-6), (-12e-6, 20e-6), (22e-6, -2e-6)):
        ix, iy = int(round(tx / solver.h)) - solver.win[0], int(round(ty / solver.h)) - solver.win[2]
        x, y = solver.xs[ix], solver.ys[iy]
        exact = x1d(x) * x1d(y) / ((1j * LAM * z_in) * (1j * LAM * L))
        worst = max(worst, abs(u[ix, iy] / solver.exit_mask[ix, iy] - exact) / abs(exact))
    assert worst < 2e-2


def test_reciprocity_source_and_detector_swap(square_point_solver):
    sim, wave, solver, grid, theta = square_point_solver
    xi, P = np.array([1.5e-6, -0.7e-6]), np.array([2.0e-6, 1.0e-6])
    d, z_in = 0.183 - 0.0525, 0.117
    uf = solver.exit_field(xi, K, sim.delta_f, sim.beta_f, -1)
    XS, YS = np.meshgrid(solver.xs, solver.ys, indexing="ij")
    e_fwd = np.sum(uf * np.exp(1j * K * ((P[0] - XS) ** 2 + (P[1] - YS) ** 2) / (2 * d))) * solver.h ** 2 / (1j * LAM * d)
    rev = W.BoreSolver(sim.cfg.capillary.bores[0], d, 0.0525, theta, LAM, dict(wave, grid_dx=solver.h))
    ur = rev.exit_field(P, K, sim.delta_f, sim.beta_f, -1)
    e_rev = np.sum(ur * np.exp(1j * K * ((xi[0] - XS) ** 2 + (xi[1] - YS) ** 2) / (2 * z_in))) * solver.h ** 2 / (1j * LAM * z_in)
    assert abs(e_fwd - e_rev) / abs(e_fwd) < 1e-2


def test_fresnel_r_limits():
    r = W.fresnel_r(np.array([0.0, 1e-3, 3.775e-3, 1e-2]), 7.1258e-6, 9.208e-8)
    assert abs(r[0] + 1.0) < 1e-12
    assert np.all(np.abs(r[:3]) > 0.85) and abs(r[3]) < 0.1


def test_broadband_grid_is_permutation_invariant(tmp_path):
    """The lattice follows the shortest wavelength; swapping the spectral lines
    changes nothing but the summation order."""
    def run(order):
        raw = _square(nx=11)
        raw["capillary"]["screen"].update(edge_x=6e-6, edge_y=6e-6, nx=11, ny=3)
        raw["screen"] = dict(raw["capillary"]["screen"])
        raw["spectrum"] = {"mode": "lines", "lines": [{"energy_kev": e, "weight": 1.0} for e in order]}
        raw["wave_estimator"] = {"observable": "point", "max_bounces": 0, "source_nodes": 1, "pad": 0.5}
        sim = Simulation.from_dict(raw)
        wave = sim.cfg.validate_wave_estimator()
        p = sim.cfg.precision
        lines = [(float(ln.k), float(ln.weight), float(sim.cfg.material.delta(ln.e_kev, precision=p)),
                  float(sim.cfg.material.beta(ln.e_kev, precision=p))) for ln in sim.lines]
        (label, res), = W._capillary_scene(sim, wave, lines, None, 1, lambda _: None)
        return res

    a, b = run((4.0, 12.0)), run((12.0, 4.0))
    assert a["meta"]["lattice_wavelength_m"] == b["meta"]["lattice_wavelength_m"]
    assert abs(a["meta"]["lattice_wavelength_m"] - 12.398419843320026e-10 / 12.0) < 1e-20
    assert np.allclose(a["mu"], b["mu"], rtol=0, atol=1e-12) and np.allclose(a["I"], b["I"], rtol=1e-12)


def test_pixel_intensity_keeps_cross_bore_interference():
    """I_pixel is the ordinary-detector intensity of the TOTAL wave: two bores with
    opposite fields cancel it, equal fields quadruple it."""
    raw = _two_bores()
    sim = Simulation.from_dict(raw)
    grid = ScreenGrid(sim.cfg.capillary.screen)
    samp = W.ScreenSampler(grid, (5e-6, 0.0), "coherent_cell", 2)
    xs = np.arange(-6e-6, 6e-6, 0.25e-6)
    u = np.exp(1j * K * (xs[:, None] ** 2 + xs[None, :] ** 2) / (2 * 0.08)) * 1e6
    kern = samp.kernels(xs, xs, K, 0.1, 0.25e-6)
    e, _, es = samp.fields(u, kern, K, 0.1, 0.25e-6)
    for sign, factor in ((-1.0, 0.0), (1.0, 4.0)):
        acc = W.Accumulator(samp, 2)
        acc.add(1.0, [e, sign * e], [None, None], [es, sign * es])
        single = W.Accumulator(samp, 1)
        single.add(1.0, [e], [None], [es])
        assert np.allclose(acc.I_pixel, factor * single.I_pixel, rtol=1e-12, atol=1e-30 * single.I_pixel.max())
        assert np.allclose(acc.I_bore[0], single.I) and np.allclose(acc.I_bore[1], single.I)
        assert np.allclose(acc.I, factor * single.I, rtol=1e-12, atol=1e-30 * single.I.max())


# ---------------------------------------------------------------- render

def test_heatmap_diverging_scale_keeps_default_unchanged():
    grid = [[-1.0, -0.1, 0.0, 0.5, 1.0]]
    plain = render.heatmap(grid, (0, 5, 0, 1), "t", "x", "y")
    assert plain == render.heatmap(grid, (0, 5, 0, 1), "t", "x", "y", diverging=False)
    div = render.heatmap(grid, (0, 5, 0, 1), "t", "x", "y", diverging=True)
    assert div != plain and ">-1<" in div["body"] and ">1<" in div["body"]
    assert render.diverging_color(-1.0) == (33, 102, 172) and render.diverging_color(0.0) == (247, 247, 247)
    assert render.diverging_color(1.0) == (178, 24, 43) and render.diverging_color(-0.5) != render.diverging_color(0.5)


# ---------------------------------------------------------------- end to end

def test_two_square_bores_run_and_interference_identity(tmp_path):
    raw = _two_bores()
    sim = Simulation.from_dict(raw)
    result = sim.run(str(tmp_path), stages=[16])
    out = tmp_path / "stage16"
    assert {"mu-wave.jsonl", "meta.json", "16-wave-mu-intensity.svg", "16-wave-phase-pixel.svg",
            "16-wave-bores.svg"} <= set(os.listdir(out))
    assert {"mu-wave.jsonl", "meta.json", "16-wave-mu.svg", "16-wave-intensity.svg",
            "16-wave-intensity-log.svg", "16-wave-interference.svg"} <= set(os.listdir(out / "screen-1"))
    rows = [json.loads(line) for line in (out / "mu-wave.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 41 * 5 and sum(r["is_reference"] for r in rows) == 1
    imax = max(r["I"] for r in rows)
    for r in rows:
        assert abs(r["I"] - r["I_bores"][0] - r["I_bores"][1] - r["I_int"]) <= 1e-12 * r["I"]
        assert abs(r["I_int"] - 2 * r["G12_re"]) <= 1e-9 * max(r["I"], 1e-300)
        assert r["observable"] == "coherent_cell" and r["I_pixel"] > 0
    centre = [r for r in rows if abs(r["x_um"]) < 0.5 and abs(r["y_um"]) < 0.5]
    assert centre and centre[0]["I"] > 0.02 * imax               # shadow lit by diffraction
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "unverified" and meta["provider"] == "uisk"
    assert meta["families"][0]["count"] == 13 and meta["wave_estimator"]["source_nodes"] == 4
    assert meta["screen"]["reference_cell_um"] is not None and meta["screen"]["reference_inside_window"]
    assert "units" in meta["conventions"]
    assert any("Stage 16" in line for line in sim.report)
    assert "stage16/mu-wave.jsonl" in result["files"]
    with pytest.raises(ValueError, match="publication conflict"):
        Simulation.from_dict(raw).run(str(tmp_path), stages=[16])


# ---------------------------------------------------------------- edge cases, preflight, diagnostics

def test_point_reference_outside_window_sets_the_lattice(tmp_path):
    """The chirp of an outside point reference must be resolved by the exit lattice."""
    raw = {"precision": 32, "energy_kev": 8.048,
           "capillary": {"bores": [{"center": [0.0, 0.0], "radius": 3e-6, "sides": 4}],
                         "z0": 0.0, "z1": 0.03,
                         "source": {"shape": "disk", "size": 0.5e-6, "position": [0.0, 0.0, -0.05],
                                    "n_modes": 1, "n_rays": 1},
                         "screen": {"z": 0.13, "center": [0.0, 0.0], "edge_x": 6e-6, "edge_y": 2e-6,
                                    "nx": 7, "ny": 3, "reference": [100e-6, 0.0]}},
           "wave_estimator": {"observable": "point", "source_nodes": 4, "max_bounces": 0}}
    sim = Simulation.from_dict(raw)
    cap = sim.cfg.capillary
    grid = ScreenGrid(cap.screen)
    nodes = np.zeros((1, 2))
    without = W._theta_max(cap, nodes, [grid])
    with_ref = W._theta_max(cap, nodes, [grid], [(100e-6, 0.0)])
    assert with_ref > 1.0e-3 > without                       # (100 + 3) um / 0.1 m
    sim.run(str(tmp_path), stages=[16])
    meta = json.loads((tmp_path / "stage16" / "meta.json").read_text(encoding="utf-8"))
    assert meta["theta_max_rad"] == pytest.approx(with_ref) and meta["lattice_h_m"] < 8e-8
    assert meta["screen"]["reference_inside_window"] is False


def test_no_trusted_pixels_still_publishes(tmp_path):
    """An all-untrusted 1D screen keeps its rows and intensity plots."""
    raw = {"precision": 32, "energy_kev": 8.048,
           "free": {"source": {"shape": "point", "size": 0.0, "position": [0.0, 0.0, -0.3],
                               "n_modes": 1, "n_rays": 1},
                    "screen": {"z": 0.0, "center": [0.0, 0.0], "edge_x": 40e-6, "edge_y": 20e-6,
                               "nx": 5, "ny": 1, "reference": [16e-6, 0.0]}},
           "wave_estimator": {"observable": "coherent_cell", "intensity_floor": 1.0}}
    sim = Simulation.from_dict(raw)
    sim.run(str(tmp_path), stages=[16])
    out = tmp_path / "stage16" / "free"
    files = set(os.listdir(out))
    assert {"mu-wave.jsonl", "meta.json", "16-wave-intensity.svg", "16-wave-intensity-log.svg"} <= files
    assert "16-wave-mu.svg" not in files                      # no trusted mu: no fake curve
    rows = [json.loads(l) for l in (out / "mu-wave.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(not r["trusted"] and r["mu_abs"] is not None and r["I"] > 0 for r in rows)


def test_phase_mask_and_log_rows():
    """W = 0 has no phase; exact zeros leave gaps in the log plot."""
    mu = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=complex)
    trusted = np.ones((2, 2), dtype=bool)
    assert W._phase_mask(mu, trusted).tolist() == [[True, True], [False, False]]
    assert W._phase_mask(np.array([[5e-4 + 0j]]), np.ones((1, 1), bool)).tolist() == [[False]]
    assert W._log_rows([1.0, 0.0, 1e-8], 1.0) == [0.0, None, -8.0]


def test_one_dimensional_gamma12_plots_distinguish_phase(tmp_path):
    """Opposite cross-bore phases give different 1D Gamma_12 figures."""
    def run(sign, out):
        raw = _two_bores(nx=21, ny=1)
        raw["capillary"].pop("screens")
        raw["capillary"]["screen"].update(edge_y=1e-6)
        sim = Simulation.from_dict(raw)
        wave = sim.cfg.validate_wave_estimator()
        grid = ScreenGrid(sim.cfg.capillary.screen)
        samp = W.ScreenSampler(grid, (5e-6, 0.0), "coherent_cell", 2)
        acc = W.Accumulator(samp, 2)
        e = np.exp(1j * np.linspace(0, 3, 21))[:, None] + 0j
        acc.add(1.0, [e, sign * e], [None, None], [np.ones((42, 2), complex), sign * np.ones((42, 2), complex)])
        res = W._finish(acc, samp, {"scene": "capillary", "provider": "uisk", "n_nodes": 1}, wave, 0.0)
        os.makedirs(out)
        names = W._maps(str(out), res, wave)
        return names, (out / "16-wave-gamma12.svg").read_text(encoding="utf-8")

    names_a, svg_a = run(1j, tmp_path / "a")
    names_b, svg_b = run(-1j, tmp_path / "b")
    assert "16-wave-gamma12.svg" in names_a and "16-wave-interference.svg" in names_a
    assert svg_a != svg_b and "arg Γ₁₂" in svg_a


def test_raw_negative_spectral_weights_are_rejected(tmp_path):
    """All-negative weights normalize to positive ones but stay invalid input."""
    raw = _free()
    raw["spectrum"] = {"mode": "lines", "lines": [{"energy_kev": 8.0, "weight": -1.0},
                                                   {"energy_kev": 16.0, "weight": -2.0}]}
    sim = Simulation.from_dict(raw)
    assert all(float(ln.weight) > 0 for ln in sim.lines)      # normalized: the trap
    with pytest.raises(ValueError, match="non-negative weights"):
        sim.run(str(tmp_path), stages=[16])


def test_input_preflight_precedes_stage14(tmp_path, monkeypatch):
    """Invalid Stage-16 inputs fail before Stage 14 does any work in a joint run."""
    import formula.capsysred.simulation as simulation

    def boom(*args, **kwargs):
        raise AssertionError("stage 14 must not run")

    monkeypatch.setattr(simulation, "run_stage14", boom)
    raw = _square(screen_z=0.0525)
    with pytest.raises(ValueError, match="outside the Stage-16 MVP"):
        Simulation.from_dict(raw).run(str(tmp_path), stages=[14, 16])
    assert not list(tmp_path.glob("stage14*"))


def test_companion_plots_survive_total_cancellation(tmp_path):
    """I = 0 everywhere still leaves I_b, I_pixel, I_int and Gamma_12 plots."""
    raw = _two_bores(nx=21, ny=1)
    raw["capillary"].pop("screens")
    sim = Simulation.from_dict(raw)
    wave = sim.cfg.validate_wave_estimator()
    grid = ScreenGrid(sim.cfg.capillary.screen)
    samp = W.ScreenSampler(grid, (5e-6, 0.0), "coherent_cell", 2)
    acc = W.Accumulator(samp, 2)
    e = np.ones((21, 1), complex)
    es = np.ones((42, 2), complex)
    acc.add(1.0, [e, -e], [None, None], [es, -es])
    res = W._finish(acc, samp, {"scene": "capillary", "provider": "uisk", "n_nodes": 1}, wave, 0.0)
    assert float(res["I"].max()) == 0.0 and res["I_bore"].max() == 1.0
    out = tmp_path / "zero"
    out.mkdir()
    names = W._maps(str(out), res, wave)
    assert {"16-wave-intensity.svg", "16-wave-interference.svg", "16-wave-gamma12.svg"} <= set(names)
    assert "16-wave-intensity-log.svg" not in names and "16-wave-mu.svg" not in names
    assert "I_1 / S_coh" in (out / "16-wave-intensity.svg").read_text(encoding="utf-8")


def test_explicit_coarse_grid_is_diagnosed():
    """An explicit grid_dx above the sampling requirement is flagged, never silent."""
    raw = {"precision": 32, "energy_kev": 8.048,
           "capillary": {"bores": [{"center": [0.0, 0.0], "radius": 3e-6, "sides": 4}],
                         "z0": 0.0, "z1": 0.03,
                         "source": {"shape": "point", "size": 0.0, "position": [0.0, 0.0, -0.05],
                                    "n_modes": 1, "n_rays": 1},
                         "screen": {"z": 0.13, "center": [0.0, 0.0], "edge_x": 6e-6, "edge_y": 2e-6,
                                    "nx": 7, "ny": 3, "reference": [100e-6, 0.0]}},
           "wave_estimator": {"observable": "point", "source_nodes": 1, "max_bounces": 0,
                              "grid_dx": 4e-7}}
    sim = Simulation.from_dict(raw)
    wave = sim.cfg.validate_wave_estimator()
    logs = []
    (label, res), = W._capillary_scene(sim, wave, _lines(sim), None, 1, logs.append)
    s = res["meta"]["sampling"]
    assert s["violation"] and s["sampling_ratio"] > 5 and s["h_auto_m"] < 4e-8
    assert any("WARNING" in m and "under-samples" in m for m in logs)
    assert any("ALIASING" in line for line in res["report"])
    wave["grid_dx"] = None
    logs.clear()
    (label, res), = W._capillary_scene(sim, wave, _lines(sim), None, 1, logs.append)
    assert not res["meta"]["sampling"]["violation"] and res["meta"]["sampling"]["sampling_ratio"] == pytest.approx(0.5)


def test_geometry_preflight_precedes_stage14(tmp_path, monkeypatch):
    """A cylinder bore with provider uisk fails before Stage 14 (auto now routes circular bores
    to the fb provider); free + free scene passes."""
    import formula.capsysred.simulation as simulation

    def boom(*args, **kwargs):
        raise AssertionError("stage 14 must not run")

    monkeypatch.setattr(simulation, "run_stage14", boom)
    raw = _square("disk", 3e-7)
    raw["capillary"]["bores"] = [{"center": [0.0, 0.0], "radius": 24e-6}]
    raw["wave_estimator"] = {"provider": "uisk"}
    with pytest.raises(ValueError, match="regular-polygon bores only"):
        Simulation.from_dict(raw).run(str(tmp_path / "cyl"), stages=[14, 16])
    raw["wave_estimator"] = {"provider": "uisk"}
    raw.pop("capillary")
    raw.update(_free())
    with pytest.raises(ValueError, match="needs a configured capillary"):
        Simulation.from_dict(raw).run(str(tmp_path / "nocap"), stages=[16])
    raw["wave_estimator"] = {"provider": "free", "source_nodes": 4}
    Simulation.from_dict(raw).run(str(tmp_path / "free"), stages=[16])
    assert (tmp_path / "free" / "stage16" / "free" / "mu-wave.jsonl").exists()


def test_intensity_profiles_use_separate_scales(tmp_path, monkeypatch):
    """Coherent I and ordinary I_pixel differ by the cell area and get their
    own normalizations; both profiles read [0.25, 1, 0.25] on their curves."""
    raw = _free()
    raw["free"]["screen"].update(edge_x=2e-6, edge_y=2e-6, nx=3, ny=1, reference=[0.0, 0.0])
    sim = Simulation.from_dict(raw)
    wave = sim.cfg.validate_wave_estimator()
    grid = ScreenGrid(sim.cfg.free_screen)
    samp = W.ScreenSampler(grid, (0.0, 0.0), "coherent_cell", 2)
    area = samp.hx * samp.hy
    prof = np.array([1.0, 2.0, 1.0])
    e = (area * prof)[:, None] + 0j
    es = np.repeat(np.repeat(prof[:, None], 2, 0), 2, 1) + 0j          # constant inside each cell
    acc = W.Accumulator(samp, 1)
    acc.add(1.0, [e], [None], [es])
    assert np.allclose(acc.I[:, 0], area ** 2 * prof ** 2) and np.allclose(acc.I_pixel[:, 0], area * prof ** 2)
    captured = {}
    real = render.line_chart

    def spy(series, title, *a, **k):
        if title.startswith("intensity (profiles"):
            captured.update({s["label"]: s["ys"] for s in series})
        return real(series, title, *a, **k)

    monkeypatch.setattr(render, "line_chart", spy)
    res = W._finish(acc, samp, {"scene": "free", "provider": "free", "n_nodes": 1}, wave, 0.0)
    W._maps(str(tmp_path), res, wave)
    coh = [k for k in captured if k.startswith("I / S_coh")][0]
    pix = [k for k in captured if k.startswith("I_pixel")][0]
    assert np.allclose(captured[coh], [0.25, 1.0, 0.25]) and np.allclose(captured[pix], [0.25, 1.0, 0.25])


def test_zero_coherent_intensity_caption(tmp_path, monkeypatch):
    """I_coherent = 0 with I_pixel > 0 is not announced as 'all zero', S_coh = 0."""
    raw = _free()
    raw["free"]["screen"].update(edge_x=2e-6, edge_y=2e-6, nx=3, ny=1, reference=[0.0, 0.0])
    sim = Simulation.from_dict(raw)
    wave = sim.cfg.validate_wave_estimator()
    samp = W.ScreenSampler(ScreenGrid(sim.cfg.free_screen), (0.0, 0.0), "coherent_cell", 2)
    es = np.array([[1.0, -1.0], [1.0, -1.0]] * 3, dtype=complex)        # +1/-1 halves in each cell
    acc = W.Accumulator(samp, 1)
    acc.add(1.0, [np.zeros((3, 1), complex)], [None], [es])
    assert acc.I.max() == 0.0 and acc.I_pixel.min() > 0.0
    captured = {}
    real = render.line_chart

    def spy(series, title, xlabel, ylabel, subtitle="", *a, **k):
        if title.startswith("intensity (profiles"):
            captured["sub"] = subtitle
            captured.update({s["label"]: s["ys"] for s in series})
        return real(series, title, xlabel, ylabel, subtitle, *a, **k)

    monkeypatch.setattr(render, "line_chart", spy)
    res = W._finish(acc, samp, {"scene": "free", "provider": "free", "n_nodes": 1}, wave, 0.0)
    W._maps(str(tmp_path), res, wave)
    assert "all intensities" not in captured["sub"] and "coherent intensities are zero" in captured["sub"]
    assert "S_coh = max(I, I_b) = 0.000e+00" in captured["sub"]
    assert np.allclose(captured["I / S_coh"], 0.0)
    assert np.allclose([v for k, v in captured.items() if k.startswith("I_pixel")][0], 1.0)


def test_far_tail_reference_is_flagged(tmp_path):
    """A reference deep in the diffraction tail (I_ref << max I) is reported, not trusted silently."""
    raw = _free(reference=(19e-6, 0.0))
    raw["free"]["source"].update(shape="point", size=0.0)
    raw["free"]["screen"].update(edge_x=4e-6, edge_y=2e-6, nx=5, ny=1, center=[0.0, 0.0])
    raw["wave_estimator"] = {"observable": "point"}
    sim = Simulation.from_dict(raw)
    wave = sim.cfg.validate_wave_estimator()
    res = W._free_scene(sim, wave, _lines(sim), None)
    assert res["meta"]["I_ref_over_max_I"] is not None and not res["meta"]["reference_in_far_tail"]
    fake = dict(res["meta"], I_ref_over_max_I=4.6e-5, reference_in_far_tail=True)
    lines = W._report_lines("free", dict(res, meta=fake), wave, [])
    assert any("far diffraction tail" in l for l in lines)

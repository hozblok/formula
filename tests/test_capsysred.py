"""Smoke and physics checks for the CAPSYSred package (tiny ray budgets)."""

import gzip
import json
import os
import math

import pytest

from formula.capsysred import Simulation
from formula.capsysred.stages.analytic import vcz_mu
from formula.capsysred.shared.nums import exp_i, lift, vunit
from formula.capsysred.spectrum import spectral_lines, wavevector
from formula.capsysred.surfaces import CapillaryBundle, Mirror, engine_hit_t
from formula.capsysred.walls.wall_cylinder import CylinderWall
from formula.capsysred.walls.wall_polygon import PolygonWall
from formula.capsysred.walls.wall_revolution import RevolutionWall
from formula.capsysred.walls.wall_torus import (_bisect_first, _float_seeds,
                                          _quartic_first)
from formula.capsysred.symbolic import (LineAmplitudes, ampl_template,
                                     ray_expression, ray_field_template)
from formula.capsysred.fresnel import FresnelAmplitude
from formula.capsysred.trace import trace_ray
from formula.capsysred.shared.types import HitMethod
from formula import xray
from formula.formula import Number, Solver

FREE_SOURCE = {
    "shape": "point",
    "size": 0.0,
    "position": [0.0, 0.0, -0.08],
    "n_modes": 4,
    "n_rays": 240,
}

CAPILLARY_SOURCE = {
    "shape": "point",
    "size": 3.0e-7,
    "position": [0.0, 0.0, -0.01],
    "n_modes": 3,
    "n_rays": 40,
}

TINY = {
    "screen": {"nx": 21},
    "free": {"source": FREE_SOURCE},
    "capillary": {
        "source": CAPILLARY_SOURCE,
        "screen": {"nx": 9, "ny": 9},
    },
}


def _record(sim, out, scenes="all"):
    """Record the sim's scenes into out/rays-modes with trace_v3 (jobs=1)."""
    import yaml
    from formula.capsysred.trace_v3 import trace as trace_v3
    if not isinstance(sim, Simulation):
        sim = Simulation.from_dict(sim)
    out = str(out)
    parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(parent, exist_ok=True)
    cfg = os.path.join(parent, os.path.basename(out) + ".trace-config.yaml")
    with open(cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(sim.cfg.raw, fh, sort_keys=False)
    return trace_v3(cfg, os.path.join(out, "rays-modes"), jobs=1, level=6,
                    log=lambda m: None, scenes=scenes)


def test_full_pipeline_files_and_point_source_coherence(tmp_path):
    sim = Simulation.from_dict(TINY)
    _record(sim, str(tmp_path))
    result = sim.run(str(tmp_path), stages=[1, 2, 3, 6])
    for name in result["files"]:
        assert (tmp_path / name).stat().st_size > 0
    # point source -> fully coherent: mu ~ 1 on well-lit pixels
    maps = sim.results["free"]["maps"]
    imax = max(maps["intensity"][0])
    lit = [m for m, i in zip(maps["mu"][0], maps["intensity"][0]) if i > 0.3 * imax]
    assert lit and min(lit) > 0.9
    assert max(max(row) for row in maps["mu"]) <= 1.0 + 1e-9


def test_stage3_uses_reference_row_for_mc_slice(tmp_path, monkeypatch):
    # An off-centre reference defines the y-row whose MC profile must be
    # compared with the analytic profile; the geometric middle row is unrelated.
    cfg = dict(TINY, free={
        "source": FREE_SOURCE,
        "screen": {
            "nx": 3, "ny": 3, "reference": [0.0, -0.9e-6],
        },
    })
    sim = Simulation.from_dict(cfg)
    from formula.capsysred.screen import ScreenGrid
    screen = ScreenGrid(sim.cfg.free_screen)
    ref = screen.ref_pixel(sim.cfg.free_screen.reference)
    assert ref // screen.nx == 0 and screen.ny // 2 == 1
    mu = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    zeros = [[0.0] * screen.nx for _ in range(screen.ny)]
    captured = {}

    def capture(series, *_args, **_kwargs):
        captured["series"] = series
        return {"w": 1, "h": 1, "body": ""}

    monkeypatch.setattr("formula.capsysred.simulation.render.line_chart", capture)
    sim.files, sim.report = [], []
    sim._stage3(str(tmp_path), {
        "screen": screen,
        "maps": {"ref_pixel": ref, "mu": mu, "mu_err": zeros,
                 "dubious": zeros},
        "src_cfg": sim.cfg.free_source,
    })
    assert captured["series"][0]["ys"] == mu[0]


def test_material_enum_selects_wall_glass():
    # OE 20:3975 glass: eps = 1 - 9.115e-6 + i*1.145e-7 at 8 keV -> 2*delta, 2*beta
    sim = Simulation.from_dict(dict(TINY, material="glass_oe2012"))
    assert sim.cfg.material is xray.OE2012_GLASS
    assert abs(2.0 * float(sim.cfg.material.delta("8.0", precision=32))
               - 9.115e-6) < 1e-9
    assert abs(2.0 * float(sim.cfg.material.beta("8.0", precision=32))
               - 1.145e-7) < 1e-12
    assert Simulation.from_dict(TINY).cfg.material is xray.FUSED_SILICA
    with pytest.raises(ValueError, match="unknown material"):
        Simulation.from_dict(dict(TINY, material="lead"))


def test_fresnel_matches_engine_reflect_amplitude():
    sim = Simulation.from_dict(TINY)
    p = sim.cfg.precision
    from formula.capsysred.shared.nums import solver
    s = solver("sin(x)", p).number({"x": "2.5e-4"})
    r_fast = sim.fresnel(s)
    r_ref = xray.reflect_amplitude("2.5e-4", str(sim.cfg.energy_kev),
                                   sim.cfg.material, precision=p)
    assert float(abs(r_fast - r_ref)) < 1e-25


def test_wall_hit_matches_raysurface_engine():
    sim = Simulation.from_dict(TINY)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = sim.cfg.precision
    d = vunit((lift(1.0e-3, p), lift(2.0e-4, p), lift(1.0, p)))
    origin = (cap.bores[0]["center"][0], cap.bores[0]["center"][1], cap.z0)
    kind, t_fast, _, _ = bundle.next_event(origin, d)
    assert kind == "reflect"
    a_um = float(cap.bores[0]["radius"]) * 1e6
    t_engine = engine_hit_t(f"x^2+y^2-({a_um})^2", origin, d, 0.08)
    assert abs(float(t_fast - t_engine)) / float(t_fast) < 1e-20


def test_flat_mirror_reflection_physics():
    sim = Simulation.from_dict(TINY)
    p = sim.cfg.precision
    mirror = Mirror(lift(0.0, p), lift(0.06, p))
    origin = (lift(1.0e-5, p), lift(0.0, p), lift(-0.08, p))
    slope = -1.0e-5 / (0.03 - float(origin[2]))
    d = vunit((lift(slope, p), lift(0.0, p), lift(1.0, p)))
    tr = trace_ray(origin, d, mirror, lift(0.06, p), 10)
    assert tr.fate == "screen" and len(tr.reflections) == 1
    r = complex(sim.fresnel(tr.reflections[0][1]))
    assert abs(r) > 0.99                        # far below the critical angle
    assert abs(abs(math.atan2(r.imag, r.real)) - math.pi) < 0.1   # arg r ~ pi
    # unit direction preserved -> opl is a true path length (>= straight line)
    straight = math.hypot(float(tr.point[0]) - float(origin[0]),
                          float(tr.point[2]) - float(origin[2]))
    assert float(tr.opl) >= straight - 1e-15


def test_capillary_multibounce_survives():
    # A ray on the wall after each reflection must stay inside its bore:
    # float rounding of the on-wall radius must not absorb it (coin-flip bug).
    sim = Simulation.from_dict(TINY)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = sim.cfg.precision
    a = float(cap.bores[0]["radius"])
    length = float(cap.z1) - float(cap.z0)
    slope = 3.5 * 2 * a / length          # ~3-4 wall crossings over the bore
    d = vunit((lift(slope, p), lift(0.0, p), lift(1.0, p)))
    origin = (cap.bores[0]["center"][0], cap.bores[0]["center"][1], cap.z0)
    tr = trace_ray(origin, d, bundle, cap.screen.z, 50)
    assert tr.fate == "screen" and len(tr.reflections) >= 3


def test_symbolic_templates_match_references():
    p = 30
    mat = xray.FUSED_SILICA
    fres = FresnelAmplitude(mat, Number("8.0", p))
    for e in ("8.0", "9.0", "10.0"):
        for theta in ("2.5e-4", "1.5e-3", "3.5e-3"):
            s = Number(f"sin({theta})", p)
            r_sym = ampl_template(1, mat, p).number({"s1": str(s), "E": e})
            r_ref = xray.reflect_amplitude(theta, e, mat, precision=p)
            assert float(abs(r_sym - r_ref)) < 1e-27
    sins = [Number(f"sin({t})", p) for t in ("8e-4", "1.2e-3", "2.1e-3")]
    chain = fres(sins[0]) * fres(sins[1]) * fres(sins[2])
    values = {f"s{j + 1}": str(s) for j, s in enumerate(sins)}
    values["E"] = "8.0"
    prod = ampl_template(3, mat, p).number(values)
    assert float(abs(chain - prod)) < 1e-27
    # a baked literal expression of E is the same function
    lit = Solver(ray_expression(sins, mat), p).number({"E": "8.0"})
    assert float(abs(lit - prod)) == 0.0
    # max_bounces-sized template parses and evaluates
    big = ampl_template(200, mat, p)
    values = {f"s{j + 1}": "1.0e-3" for j in range(200)}
    values["E"] = "8.0"
    assert float(abs(big.number(values))) > 0.0


def test_ray_field_template_carries_exact_phase():
    p = 30
    mat = xray.FUSED_SILICA
    e, opl = Number("8.0", p), Number("0.1", p)
    u0 = ray_field_template(0, mat, p).number({"E": str(e), "L": str(opl)})
    ref = exp_i(wavevector(e) * opl)
    assert float(abs(u0 - ref)) < 1e-20
    s = Number("sin(1.5e-3)", p)
    u1 = ray_field_template(1, mat, p).number(
        {"E": str(e), "L": str(opl), "s1": str(s)})
    r = ampl_template(1, mat, p).number({"s1": str(s), "E": str(e)})
    assert float(abs(u1 - ref * r)) < 1e-20


def test_cylinder_grazing_invariant_gives_r_pow_nb():
    # A straight cylinder preserves sin(theta) across bounces -> ampl = r(s)^nb.
    sim = Simulation.from_dict(TINY)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = sim.cfg.precision
    a = float(cap.bores[0]["radius"])
    length = float(cap.z1) - float(cap.z0)
    d = vunit((lift(3.5 * 2 * a / length, p), lift(0.0, p), lift(1.0, p)))
    origin = (cap.bores[0]["center"][0], cap.bores[0]["center"][1], cap.z0)
    tr = trace_ray(origin, d, bundle, cap.screen.z, 50)
    sins = [s for _, s in tr.reflections]
    assert len(sins) >= 3
    assert max(abs(float(s) - float(sins[0])) for s in sins) < 1e-15
    r1 = sim.fresnel(sins[0])
    assert float(abs(sim.fresnel.product(sins) - r1 ** len(sins))) < 1e-25


def test_spectral_lines_carry_energy_and_k(tmp_path):
    e0 = Number("8.0", 30)
    lines = spectral_lines({"mode": "gaussian", "rel_fwhm": 2e-4, "n_lines": 5,
                            "n_sigma": 3.0}, e0)
    assert len(lines) == 5
    assert abs(sum(l.weight for l in lines) - 1.0) < 1e-12
    for l in lines:
        assert float(abs(l.k - wavevector(l.e_kev))) == 0.0
    table = tmp_path / "sp.txt"
    table.write_text("# E w\n8.0 1\n8.5, 2\n\n", encoding="utf-8")
    lines = spectral_lines({"mode": "table", "file": str(table)}, e0)
    assert [float(l.e_kev) for l in lines] == [8.0, 8.5]
    assert abs(lines[1].weight - 2.0 / 3.0) < 1e-12


def test_line_amplitudes_energy_dependence():
    # Between the two critical angles reflection collapses at the higher energy.
    e0 = Number("8.0", 30)
    lines = spectral_lines({"mode": "lines", "lines": [
        {"energy_kev": 8.0}, {"energy_kev": 10.0}]}, e0)
    la = LineAmplitudes(xray.FUSED_SILICA, lines, 30)
    amps = la([str(Number("sin(3.4e-3)", 30))])
    assert float(abs(amps[0])) > 2.0 * float(abs(amps[1]))
    # no bounces -> unit amplitude on every line
    ones = la([])
    assert [float(abs(a)) for a in ones] == [1.0, 1.0]


def test_rays_jsonl_records(tmp_path):
    from formula.capsysred import rays_v3
    sim = Simulation.from_dict(TINY)
    _record(sim, tmp_path)
    archive = tmp_path / "rays-modes"
    index = rays_v3.load_index(archive)
    assert index.budgets["capillary"] == [3, 40]
    assert isinstance(rays_v3.read_fingerprint(archive)["geometry"], dict)
    sim.run(str(tmp_path), stages=[6])
    rows = [json.loads(line) for line in rays_v3.scene_lines(archive, index, "capillary")]
    assert len(rows) == sim.results["capillary"]["stats"]["emitted"]
    assert {"stage", "mode", "ray", "fate", "pixel", "opl", "sins"} <= set(rows[0])
    hit = next(row for row in rows if row["fate"] == "screen")
    assert {"x", "y", "dx", "dy"} <= set(hit)            # v2 float geometry
    assert any(not row["sins"] for row in rows)          # direct rays recorded too
    digits = rows[0]["opl"].replace(".", "").replace("-", "").lstrip("0")
    assert len(digits) >= 25                             # full-precision strings
    modes = [row["mode"] for row in rows]
    assert modes == sorted(modes)                        # grouped by mode


def test_replay_matches_direct_mono(tmp_path):
    sim = Simulation.from_dict(TINY)
    _record(sim, str(tmp_path))
    sim.run(str(tmp_path), stages=[6])
    direct = sim.results["capillary"]["maps"]
    sim.replay(str(tmp_path / "rays-modes"), str(tmp_path / "replay"))
    rep = sim.results["capillary"]["maps"]
    for key in ("mu", "intensity"):
        scale = max(max(abs(v) for v in row) for row in direct[key]) or 1.0
        diff = max(abs(x - y) for ra, rb in zip(direct[key], rep[key])
                   for x, y in zip(ra, rb))
        assert diff <= 1e-9 * scale, key


def test_replay_matches_direct_gaussian(tmp_path):
    cfg = dict(TINY, spectrum={"mode": "gaussian", "rel_fwhm": 2.0e-4,
                               "n_lines": 3, "n_sigma": 2.0})
    sim = Simulation.from_dict(cfg)
    assert sim.per_line
    _record(sim, str(tmp_path))
    sim.run(str(tmp_path), stages=[6])
    direct = sim.results["capillary"]["maps"]
    sim.replay(str(tmp_path / "rays-modes"), str(tmp_path / "replay"))
    rep = sim.results["capillary"]["maps"]
    for key in ("mu", "intensity"):
        scale = max(max(abs(v) for v in row) for row in direct[key]) or 1.0
        diff = max(abs(x - y) for ra, rb in zip(direct[key], rep[key])
                   for x, y in zip(ra, rb))
        assert diff <= 1e-12 * scale, key


def test_material_change_keeps_rays_file_valid(tmp_path):
    # rays are material-free: a glass_oe2012 re-run reuses the silica trace
    # and only the physics (Fresnel amplitudes -> intensity) changes
    silica = Simulation.from_dict(TINY)
    _record(silica, str(tmp_path))
    silica.run(str(tmp_path), stages=[6])
    oe = Simulation.from_dict(dict(TINY, material="glass_oe2012"))
    oe.run(str(tmp_path), stages=[6])
    assert oe.results["capillary"]["rays_from"] == "file"
    a = silica.results["capillary"]["maps"]
    b = oe.results["capillary"]["maps"]
    assert a["density"] == b["density"]              # same geometry
    assert a["intensity"] != b["intensity"]          # different Fresnel


def test_replay_with_other_material(tmp_path):
    # same recorded rays, glass_oe2012 wall on replay: ray bookkeeping identical,
    # reflected amplitudes differ
    sim = Simulation.from_dict(TINY)
    _record(sim, str(tmp_path))
    sim.run(str(tmp_path), stages=[6])
    direct = sim.results["capillary"]
    other = Simulation.from_dict(dict(TINY, material="glass_oe2012"))
    other.replay(str(tmp_path / "rays-modes"), str(tmp_path / "replay"))
    rep = other.results["capillary"]
    assert rep["rays_from"] == "file"
    assert rep["stats"]["emitted"] == direct["stats"]["emitted"]
    assert rep["stats"]["screen"] == direct["stats"]["screen"]
    assert rep["maps"]["intensity"] != direct["maps"]["intensity"]


def test_cli_trace_then_stages_reuse(tmp_path):
    import yaml
    from formula.capsysred.__main__ import main
    from formula.capsysred.trace_v3 import main as trace_main
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(TINY))
    out = tmp_path / "out"
    archive = out / "rays-modes"
    assert trace_main([str(cfg), "--archive", str(archive), "--jobs", "1"]) == 0
    assert (archive / "rays-index.jsonl").exists()
    with pytest.raises(ValueError, match="no rays recording"):
        main([str(cfg), "-o", str(tmp_path / "empty"), "--stages", "6"])
    assert main([str(cfg), "-o", str(out), "--stages", "6"]) == 0
    reports = list(out.glob("report-*.md"))
    assert any("no tracing" in r.read_text(encoding="utf-8") for r in reports)

    original = (archive / "rays-index.jsonl").read_bytes()
    changed = dict(TINY, capillary=dict(TINY["capillary"], z1=0.06))
    cfg.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="differs from the archive"):
        trace_main([str(cfg), "--archive", str(archive), "--jobs", "1"])
    with pytest.raises(ValueError, match="does not match this config"):
        main([str(cfg), "-o", str(out), "--stages", "6"])
    assert (archive / "rays-index.jsonl").read_bytes() == original
    for removed_option in ("--force", "--no-jackknife", "--trace", "--quick"):
        with pytest.raises(SystemExit):
            main([str(cfg), "-o", str(out), "--stages", "6", removed_option])


def _cap_sim(bores, z0=0.0, z1=0.05, **cap):
    over = dict(TINY)
    over["capillary"] = dict(TINY["capillary"], bores=bores, z0=z0, z1=z1, **cap)
    return Simulation.from_dict(over)


def test_cone_adiabatic_invariant_and_engine():
    # r(z) = 5e-6 - 3.5e-5*z: each bounce adds 2k to the grazing angle and
    # a(z)*sin(theta) stays ~const; the hit must match the root-finding engine.
    sim = _cap_sim([{"center": [0.0, 0.0],
                     "r2_poly": [2.5e-11, -3.5e-10, 1.225e-9]}], z1=0.1,
                    screen={"z": 0.101})
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = sim.cfg.precision
    d = vunit((lift(4.0e-4, p), lift(0.0, p), lift(1.0, p)))
    origin = (cap.bores[0]["center"][0], cap.bores[0]["center"][1], cap.z0)
    kind, t_fast, P, _ = bundle.next_event(origin, d)
    assert kind == "reflect"
    t_engine = engine_hit_t(bundle.walls[0].expr_um, origin, d, 0.12)
    assert abs(float(t_fast - t_engine)) / float(t_fast) < 1e-18
    tr = trace_ray(origin, d, bundle, cap.screen.z, 400)
    assert tr.fate == "screen" and len(tr.reflections) >= 3
    sins = [float(s) for _, s in tr.reflections]
    steps = [b - a for a, b in zip(sins, sins[1:])]
    assert all(5.0e-5 < s < 9.0e-5 for s in steps)      # ~2k = 7e-5 per bounce
    wall = bundle.walls[0]
    inv = [math.sqrt(wall.r2f(float(P[2]))) * float(s)
           for P, s in tr.reflections]
    assert max(inv) / min(inv) < 1.05                   # a·θ adiabatic invariant


def test_torus_whispering_and_engine():
    sim = _cap_sim([{"center": [0.0, 0.0], "radius": 3.0e-6,
                     "bend": {"radius": 1.5, "toward": [1.0, 0.0]}}], z1=0.1,
                    screen={"z": 0.101, "center": [3.40386e-3, 0.0]})
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = sim.cfg.precision
    d = vunit((lift(5.0e-4, p), lift(0.0, p), lift(1.0, p)))
    origin = (cap.bores[0]["center"][0], cap.bores[0]["center"][1], cap.z0)
    kind, t_fast, P, normal = bundle.next_event(origin, d)
    assert kind == "reflect"
    t_engine = engine_hit_t(bundle.walls[0].expr_um, origin, d, 0.12)
    assert abs(float(t_fast - t_engine)) / float(t_fast) < 1e-16
    # whispering gallery: many shallow bounces on the outer wall, z monotone
    tr = trace_ray(origin, d, bundle, cap.screen.z, 400)
    assert tr.fate == "screen" and len(tr.reflections) >= 5
    zs = [float(P[2]) for P, _ in tr.reflections]
    assert zs == sorted(zs)
    R, a = 1.5, 3.0e-6
    for P, sin_g in tr.reflections:
        xf, zf = float(P[0]), float(P[2])
        rho = math.hypot(xf - R, zf)
        assert abs((rho - R) ** 2 + float(P[1]) ** 2 - a * a) < 1e-9 * a * a
        assert float(sin_g) < 3.0e-3                    # stays shallow (< theta_c)


def test_torus_gentle_bend_keeps_on_wall_points():
    # R = 234 m: float rho-R cancellation in inside() is ~1e-8·a — the on-wall
    # slack must scale with R or every reflected ray dies at the next locate.
    sim = _cap_sim([{"center": [1.6e-5, 0.0], "radius": 3.0e-6,
                     "bend": {"radius": 234.375, "toward": [-1.0, 0.0]}}])
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = sim.cfg.precision
    origin = (lift(1.6e-5, p), lift(0.0, p), lift(-0.01, p))
    d = vunit((lift(1.9e-4, p), lift(0.0, p), lift(1.0, p)))
    tr = trace_ray(origin, d, bundle, cap.screen.z, 50)
    # the absorb-bug killed the ray right after its first reflection
    assert tr.fate == "screen" and len(tr.reflections) >= 1


def test_polygon_hex_flat_face_physics():
    sim = _cap_sim([{"center": [0.0, 0.0], "radius": 3.0e-6, "sides": 6}])
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = sim.cfg.precision
    d = vunit((lift(5.0e-4, p), lift(0.0, p), lift(1.0, p)))
    origin = (cap.bores[0]["center"][0], cap.bores[0]["center"][1], cap.z0)
    kind, t_fast, _, _ = bundle.next_event(origin, d)
    assert kind == "reflect"
    t_engine = engine_hit_t(bundle.walls[0].expr_um, origin, d, 0.08)
    assert abs(float(t_fast - t_engine)) / float(t_fast) < 1e-20
    # opposite faces are parallel: grazing angle is exactly preserved
    tr = trace_ray(origin, d, bundle, cap.screen.z, 50)
    assert tr.fate == "screen" and len(tr.reflections) >= 3
    sins = [float(s) for _, s in tr.reflections]
    assert max(sins) - min(sins) < 1e-15
    # a point beyond the flat (but inside the circumradius) is in the web
    x = lift(3.2e-6, p)
    event = bundle.next_event((x, lift(0.0, p), cap.z0 + lift(1e-6, p)), d)
    assert event[0] == "absorb"


def test_implicit_hex_product_matches_polygon():
    # one smooth string for a polygon bore: F = -prod(m_k·r - a) over the faces
    sim_t = _cap_sim([{"center": [0.0, 0.0], "radius": 3.0e-6, "sides": 6}])
    cap_t = sim_t.cfg.capillary
    bundle_t = CapillaryBundle(cap_t.bores, cap_t.z0, cap_t.z1)
    fac = [f"((x)*({mx})+(y)*({my})-(3))" for mx, my in bundle_t.walls[0].faces]
    sim_i = _cap_sim([{"center": [0.0, 0.0], "surface": "(0-1)*" + "*".join(fac),
                       "aim_radius": 3.4641016151377543e-06,
                       "engine_method": "subdivision"}])
    cap_i = sim_i.cfg.capillary
    bundle_i = CapillaryBundle(cap_i.bores, cap_i.z0, cap_i.z1)
    p = sim_t.cfg.precision
    d = vunit((lift(4.0e-4, p), lift(1.5e-4, p), lift(1.0, p)))
    origin = (lift(0.0, p), lift(0.0, p), cap_t.z0)
    (k1, t1, _, n1) = bundle_t.next_event(origin, d)
    (k2, t2, _, n2) = bundle_i.next_event(origin, d)
    assert k1 == k2 == "reflect"
    assert abs(float(t1 - t2)) / float(t1) < 1e-18
    assert max(abs(float(x - y)) for x, y in zip(n1, n2)) < 1e-15


def test_implicit_surface_matches_cylinder():
    a_um = 3.0
    sim_c = _cap_sim([{"center": [0.0, 0.0], "radius": 3.0e-6}])
    sim_i = _cap_sim([{"center": [0.0, 0.0], "surface": f"x^2+y^2-({a_um})^2",
                       "aim_radius": 3.0e-6,
                       "engine_method": "subdivision"}])
    p = sim_c.cfg.precision
    d = vunit((lift(1.0e-3, p), lift(2.0e-4, p), lift(1.0, p)))
    events = []
    for sim in (sim_c, sim_i):
        cap = sim.cfg.capillary
        bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
        origin = (cap.bores[0]["center"][0], cap.bores[0]["center"][1], cap.z0)
        events.append(bundle.next_event(origin, d))
    (k1, t1, P1, n1), (k2, t2, P2, n2) = events
    assert k1 == k2 == "reflect"
    assert abs(float(t1 - t2)) / float(t1) < 1e-18
    assert max(abs(float(x - y)) for x, y in zip(n1, n2)) < 1e-15


def test_ellipsoid_focus_opl_degeneracy():
    # Ellipse property: every F1 -> wall -> F2 path is exactly 2A; the traced
    # OPL to the focal-plane screen must reproduce 2A to full precision.
    A, b, zc = 0.03, 7.5e-5, 0.02
    k = (b / A) ** 2
    sim = _cap_sim([{"center": [0.0, 0.0],
                     "r2_poly": [b * b - k * zc * zc, 2 * k * zc, -k]}],
                   z0=0.015, z1=0.025)
    p = sim.cfg.precision
    f = Number(f"(({A})^2-({b})^2)^0.5", p)
    src = (lift(0.0, p), lift(0.0, p), lift(zc, p) - f)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    # aim at the wall band: z*=0.018, r(z*) = sqrt(r2(0.018))
    r_star = math.sqrt(b * b - k * (0.018 - zc) ** 2)
    target = (lift(r_star, p), lift(0.0, p), lift(0.018, p))
    d = vunit((target[0] - src[0], target[1] - src[1], target[2] - src[2]))
    screen_z = lift(zc, p) + f
    tr = trace_ray(src, d, bundle, screen_z, 10)
    assert tr.fate == "screen" and len(tr.reflections) == 1
    assert float(tr.reflections[0][1]) < 3.0e-3         # grazing < theta_c
    assert abs(float(tr.opl) - 2.0 * A) < 1e-20
    assert math.hypot(float(tr.point[0]), float(tr.point[1])) < 1e-12


def test_bore_config_validation():
    good = {"center": [0.0, 0.0], "radius": 1e-6}
    for bad in (
        {"center": [0.0, 0.0]},                                   # no geometry
        {"radius": 1e-6, "r2_poly": [1e-12]},                     # conflict
        {"surface": "x^2+y^2-1", "engine_method": "subdivision"},  # no aim_radius
        {"surface": "x^2+y^2-1", "aim_radius": 1e-6},            # no engine_method
        {"radius": 1e-6, "engine_method": "subdivision"},         # only for surface
        {"radius": 1e-6, "sides": 2},                             # sides < 3
        {"radius": 1e-6, "bend": {"radius": 1.0}},                # bend w/o toward
        {"radius": 1e-6, "bend": {"radius": 1.0, "toward": [1, 0]}, "sides": 6},
    ):
        with pytest.raises(ValueError):
            _cap_sim([good, bad])


def test_vcz_reference_sane():
    assert vcz_mu(0.0, "gaussian", 2e-6, 1.55e-10, 0.14) == 1.0
    assert vcz_mu(5e-6, "gaussian", 2e-6, 1.55e-10, 0.14) < 0.1


# --------------------------------------------- quartic solver & tolerance kit


def _monic_quartic(roots, p):
    """Monic Number quartic with the given integer roots (exact coefficients)."""
    cs = [1]
    for r in roots:
        cs = [a - r * b for a, b in zip(cs + [0], [0] + cs)]
    return tuple(Number(str(v), p) for v in cs)


def test_quartic_newton_accuracy_scales_with_precision():
    # stop threshold is 10^-max(24, p//2): a fixed 24-digit cap would leave
    # ~1e-48 error at every precision and fail the p >= 64 bounds
    for p, bound in ((32, "1e-45"), (64, "1e-60"), (256, "1e-200")):
        t = _quartic_first(_monic_quartic((1, 2, 3, 4), p), 10.0)
        assert t is not None
        assert abs(t - Number("1", p)) < Number(bound, p)


def test_quartic_bisection_scales_with_precision():
    # rescue path: 8 + p*log2(10) halvings shrink the bracket to ~10^-p;
    # the old fixed 90 gave only ~1e-27 of the bracket at any precision
    for p, bound in ((32, "1e-30"), (256, "1e-240")):
        t = _bisect_first(_monic_quartic((1, 2, 3, 4), p), 10.0)
        assert t is not None
        assert abs(t - Number("1", p)) < Number(bound, p)


def test_float_seeds_window_and_complex_filter():
    # (t^2+1)(t-3)(t-4): the genuinely complex pair is dropped, real kept
    seeds = _float_seeds([1.0, -7.0, 13.0, -7.0, 12.0], 10.0)
    assert [round(s) for s in seeds] == [3, 4]
    # window cap: only roots inside (eps/2, t_capf] survive
    seeds = _float_seeds([1.0, -10.0, 35.0, -50.0, 24.0], 2.5)
    assert [round(s) for s in seeds] == [1, 2]


def test_quartic_first_empty_window_returns_none():
    # all roots beyond t_capf, no sign change inside -> honest None
    assert _quartic_first(_monic_quartic((5, 6, 7, 8), 32), 2.0) is None


def test_on_wall_point_counts_inside():
    # _INSIDE_TOL: a reflection point sits exactly ON the wall; float rounding
    # of dx^2+dy^2 vs a^2 must not flip it outside (coin-flip absorption)
    p = 32
    zero = Number("0", p)
    a = float(Number("6e-6", p))
    cyl = CylinderWall((zero, zero), Number("6e-6", p))
    rev = RevolutionWall("revolution", (zero, zero),
                         (Number("6e-6", p) * Number("6e-6", p), zero, zero))
    poly = PolygonWall((zero, zero), Number("6e-6", p), 6, zero)
    for wall in (cyl, rev, poly):
        assert wall.inside(a, 0.0, 0.0)
        assert not wall.inside(a * (1 + 1e-6), 0.0, 0.0)


def test_implicit_engine_method_config_wiring():
    # Each implicit bore owns its root finder; typed walls reject that option.
    from formula.capsysred.config import load
    from formula.capsysred.rays import geometry_metadata
    assert "engine_method" not in load({}).raw["trace"]
    with pytest.raises(ValueError, match="trace.engine_method was removed"):
        load({"trace": {"engine_method": "subdivision"}})
    for unsupported in ("newton", "auto"):
        with pytest.raises(ValueError):
            load({"capillary": {
                "source": CAPILLARY_SOURCE,
                "bores": [{"surface": "x^2+y^2-36",
                           "aim_radius": 6.0e-6,
                           "engine_method": unsupported}],
            }})
    cfg = load({"capillary": {
        "source": CAPILLARY_SOURCE,
        "bores": [{"surface": "x^2+y^2-36",
                   "aim_radius": 6.0e-6,
                   "engine_method": "sturm"}],
    }})
    bundle = CapillaryBundle(cfg.capillary.bores, cfg.capillary.z0,
                             cfg.capillary.z1)
    assert bundle.walls[0].method == "sturm"
    subdivision = load({"capillary": {
        "source": CAPILLARY_SOURCE,
        "bores": [{
            "surface": "x^2+y^2-36", "aim_radius": 6.0e-6,
            "engine_method": "subdivision",
        }],
    }})
    assert geometry_metadata(cfg) != geometry_metadata(subdivision)


def test_removed_lloyd_config_is_rejected():
    from formula.capsysred.config import load

    with pytest.raises(ValueError, match="lloyd was removed.*stages 4 and 5"):
        load({"lloyd": {}})


def test_top_level_source_is_rejected():
    from formula.capsysred.config import load

    with pytest.raises(ValueError, match="top-level source was removed"):
        load({"source": FREE_SOURCE})


@pytest.mark.parametrize("text", ["false", "0", "[]", "''"])
def test_yaml_config_must_be_a_mapping(tmp_path, text):
    from formula.capsysred.config import load

    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="config must be a mapping"):
        load(path)


@pytest.mark.parametrize("scene", ["free", "capillary"])
def test_configured_scene_requires_source_mapping(scene):
    from formula.capsysred.config import load

    with pytest.raises(ValueError, match=rf"{scene}\.source must be a mapping"):
        load({scene: {}})


@pytest.mark.parametrize("scene", ["free", "capillary"])
@pytest.mark.parametrize(
    "missing", ["shape", "size", "position", "n_modes", "n_rays"]
)
def test_scene_source_common_fields_are_all_required(scene, missing):
    from formula.capsysred.config import load

    source = dict(FREE_SOURCE)
    source.pop(missing)
    with pytest.raises(
            ValueError,
            match=rf"{scene}\.source is missing required fields.*{missing}"):
        load({scene: {"source": source}})


@pytest.mark.parametrize("scene", ["free", "capillary"])
@pytest.mark.parametrize("missing", ["grid_n", "grid_step"])
def test_grid_source_requires_grid_fields(scene, missing):
    from formula.capsysred.config import load

    source = {
        **FREE_SOURCE,
        "shape": "grid",
        "grid_n": 3,
        "grid_step": 1.0e-6,
    }
    source.pop(missing)
    with pytest.raises(
            ValueError,
            match=rf"{scene}\.source is missing required fields.*{missing}"):
        load({scene: {"source": source}})


@pytest.mark.parametrize(("raw", "scenes"), [
    ({}, set()),
    ({"free": {"source": FREE_SOURCE}}, {"free"}),
    ({"capillary": {"source": CAPILLARY_SOURCE}}, {"capillary"}),
    (TINY, {"free", "capillary"}),
])
def test_scene_sections_control_raw_geometry_and_budgets(raw, scenes):
    from formula.capsysred.config import load
    from formula.capsysred.rays import budgets, geometry_metadata

    cfg = load(raw)
    geometry = geometry_metadata(cfg)
    assert "source" not in cfg.raw
    assert "source" not in geometry
    assert {s for s in ("free", "capillary") if s in cfg.raw} == scenes
    assert {s for s in ("free", "capillary") if s in geometry} == scenes
    assert set(budgets(cfg)) == scenes
    assert (cfg.free_source is not None) == ("free" in scenes)
    assert (cfg.free_screen is not None) == ("free" in scenes)
    assert (cfg.capillary is not None) == ("capillary" in scenes)
    for scene in scenes:
        assert geometry[scene]["source"] == cfg.raw[scene]["source"]


@pytest.mark.parametrize("stage", [2, 3, 12])
def test_free_stage_preflight_rejects_missing_free_scene(tmp_path, stage):
    sim = Simulation.from_dict({
        "capillary": {"source": CAPILLARY_SOURCE},
    })
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=r"configured free\.source"):
        sim.run(str(out), stages=[stage])
    assert not out.exists()


@pytest.mark.parametrize("stage", [6, 9, 10])
def test_capillary_stage_preflight_rejects_missing_capillary_scene(
        tmp_path, stage):
    sim = Simulation.from_dict({"free": {"source": FREE_SOURCE}})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=r"configured capillary\.source"):
        sim.run(str(out), stages=[stage])
    assert not out.exists()


@pytest.mark.parametrize("stage", [1, 7, 8, 11])
def test_scene_stage_preflight_rejects_empty_config(tmp_path, stage):
    sim = Simulation.from_dict({})
    out = tmp_path / "out"
    with pytest.raises(
            ValueError,
            match=r"require(?:s)? a free or capillary scene"):
        sim.run(str(out), stages=[stage])
    assert not out.exists()


def test_default_run_and_trace_preflight_reject_empty_config(tmp_path):
    sim = Simulation.from_dict({})
    run_out = tmp_path / "run"
    trace_out = tmp_path / "trace"
    with pytest.raises(ValueError, match="requires a free or capillary scene"):
        sim.run(str(run_out))
    with pytest.raises(ValueError, match="no scene to trace"):
        _record(sim, trace_out)
    assert not run_out.exists()
    assert not trace_out.exists()


@pytest.mark.parametrize("stage", [4, 5])
def test_removed_stages_are_rejected_by_api_and_cli(tmp_path, stage):
    from formula.capsysred.__main__ import main

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="unknown stages"):
        Simulation.from_dict(TINY).run(str(out), stages=[stage])
    assert not out.exists()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(TINY), encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(config_path), "-o", str(out), "--stages", str(stage)])


def test_validate_partial_override_keeps_defaults():
    from formula.capsysred.config import DEFAULTS, load

    cfg = load({"validate": {"n_rays": 17}})
    assert cfg.validate_rays == 17
    assert cfg.validate_reference is HitMethod.PYTHON_CLOSED_FORM
    assert cfg.validate_methods == (HitMethod.CPP_CLOSED_FORM,
                                    HitMethod.SUBDIVISION)
    assert load(DEFAULTS).validate_methods == cfg.validate_methods
    assert load(load({}).raw).validate_methods == cfg.validate_methods


def test_validate_rejects_duplicate_methods():
    from formula.capsysred.config import load

    with pytest.raises(ValueError, match="methods must not contain duplicates"):
        load({"validate": {"methods": ["sturm", "sturm"]}})


@pytest.mark.parametrize("n_rays", [0, -1])
def test_validate_rejects_nonpositive_ray_count(n_rays):
    from formula.capsysred.config import load

    with pytest.raises(ValueError, match="n_rays must be at least 1"):
        load({"validate": {"n_rays": n_rays}})


def test_per_line_fresnel_mono_rejected():
    # explicit key with a single line is a config error; multi-line modes accept it
    from formula.capsysred.config import load

    with pytest.raises(ValueError, match="per_line_fresnel"):
        load({"spectrum": {"mode": "monochromatic", "per_line_fresnel": True}})
    with pytest.raises(ValueError, match="per_line_fresnel"):
        load({"spectrum": {"per_line_fresnel": False}})  # default mode is mono
    band = {"mode": "gaussian", "rel_fwhm": 2.0e-4, "n_lines": 3, "n_sigma": 3.0}
    assert load({"spectrum": band}).per_line_fresnel
    assert not load({"spectrum": {**band, "per_line_fresnel": False}}).per_line_fresnel


def test_precision_target_config():
    # default p - 2 on straight bores; a torus bore subtracts the conditioning
    # loss ceil(2*log10(R/a) + log10(1/theta_c) + 2); explicit values above
    # the ceiling warn
    import warnings
    from formula.capsysred.config import load
    assert load({}).precision_target == 30
    bent = {"precision": 64, "capillary": {
        "source": CAPILLARY_SOURCE,
        "bores": [
            {"center": [0.0, 0.0], "radius": 6.0e-6,
             "bend": {"radius": 8625.0, "toward": [1.0, 0.0]}},
        ],
    }}
    cfg = load(bent)
    assert cfg.precision_target_auto and cfg.precision_target_loss == 23
    assert cfg.precision_target == 39
    # theta_c is taken at the hardest spectral line: 24 keV -> theta_c/3
    hard = load({**bent, "spectrum": {"mode": "lines", "lines": [
        {"energy_kev": 8.0}, {"energy_kev": 24.0, "weight": 0.2}]}})
    assert hard.precision_target_loss == 24 and hard.precision_target == 38
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load({**bent, "precision_target": 45})
    assert not cfg.precision_target_auto and cfg.precision_target == 45
    assert any("ceiling 39" in str(w.message) for w in caught)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert load({**bent, "precision_target": 32}).precision_target == 32
    assert not caught


def test_sturm_engine_matches_closed_form_hit():
    # the polynomial wall expr_um is exactly Sturm's domain: the cross-check
    # must agree with the closed-form hit as tightly as subdivision does
    sim = Simulation.from_dict(TINY)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = sim.cfg.precision
    d = vunit((lift(1.0e-3, p), lift(2.0e-4, p), lift(1.0, p)))
    origin = (cap.bores[0]["center"][0], cap.bores[0]["center"][1], cap.z0)
    kind, t_fast, _, _ = bundle.next_event(origin, d)
    assert kind == "reflect"
    t_engine = engine_hit_t(bundle.walls[0].expr_um, origin, d, 0.08,
                            method="sturm")
    assert abs(float(t_fast - t_engine)) / float(t_fast) < 1e-25


def test_stage9_hit_methods_agree_on_cylinder(tmp_path):
    # every method must reproduce the python hit t and its pass/reflect calls
    raw = {**TINY, "validate": {
        "n_rays": 100, "methods": [HitMethod.CPP_CLOSED_FORM,
                                   HitMethod.SUBDIVISION, HitMethod.STURM]}}
    yaml_file = tmp_path / "stage9.yaml"
    yaml_file.write_text(json.dumps(raw), encoding="utf-8")
    sim = Simulation.from_yaml(yaml_file)
    result = sim.run(str(tmp_path), stages=[9])
    assert "hit-validation/hit-validation.jsonl" in result["files"]
    assert "hit-validation/meta.json" in result["files"]
    with open(tmp_path / "hit-validation" / "hit-validation.jsonl") as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 100          # one record per emitted ray
    assert {row["reference_method"] for row in rows} == {
        "python-closed-form"}
    with open(tmp_path / "hit-validation" / "meta.json") as fh:
        meta = json.load(fh)
    from formula import __version__
    assert meta == {
        "capsysred_version": __version__,
        "yaml_file": "stage9.yaml",
        "validation": {
            "n_rays": 100,
            "reference": "python-closed-form",
            "methods": ["cpp-closed-form", "subdivision", "sturm"],
            "precision": 32,
            "precision_target": 30,
        },
    }
    res = sim.results["validate"]
    assert res["native"] and res["stats"]["hits"] > 0
    for name, s in res["per"].items():
        assert s["n"] == res["stats"]["hits"], name
        assert s["missing"] == 0 and s["extra"] == 0, name
        assert s["max_rel"] < 1e-20, name


def test_stage9_rejects_when_no_comparison_method_is_runnable():
    from formula.capsysred.stages.validate import run_validate_stage

    sim = Simulation.from_dict({
        "capillary": {
            "source": CAPILLARY_SOURCE,
            "bores": [{"surface": "x^2+y^2-36",
                       "aim_radius": 6.0e-6,
                       "engine_method": "subdivision"}],
        },
        "validate": {"n_rays": 1, "reference": "sturm",
                     "methods": ["cpp-closed-form"]},
    })
    with pytest.raises(
            ValueError,
            match="no runnable comparison methods"):
        run_validate_stage(sim, 1)


def test_stage9_mixed_bundle_disables_closed_forms_globally():
    # Stage 9 compares one common method set across the whole bundle: one
    # implicit wall therefore disables both closed-form methods everywhere.
    from formula.capsysred.stages.validate import run_validate_stage

    sim = Simulation.from_dict({
        "capillary": {
            "source": CAPILLARY_SOURCE,
            "bores": [
                {"center": [-6.0e-6, 0.0], "radius": 3.0e-6},
                {"center": [6.0e-6, 0.0], "surface": "(x-6)^2+y^2-9",
                 "aim_radius": 3.0e-6, "engine_method": "subdivision"},
            ],
        },
        "validate": {
            "reference": "sturm",
            "methods": ["python-closed-form", "cpp-closed-form",
                        "subdivision"],
        },
    })
    result = run_validate_stage(sim, 1)
    assert not result["native"]
    assert tuple(result["per"]) == (HitMethod.SUBDIVISION,)


def test_stage11_beamlet_point_source_fully_coherent(tmp_path):
    # point source -> one coherent field, honest estimator must give mu = 1
    # on every pixel the beamlets light up; mu never exceeds 1
    sim = Simulation.from_dict(TINY)
    _record(sim, str(tmp_path))
    result = sim.run(str(tmp_path), stages=[11])
    for name in result["files"]:
        assert (tmp_path / name).stat().st_size > 0
    assert "mu-beamlet.jsonl" in result["files"]
    maps = sim.results["beamlet:free"]["maps"]
    imax = max(maps["intensity"][0])
    lit = [m for m, i in zip(maps["mu"][0], maps["intensity"][0])
           if i > 0.3 * imax]
    assert lit and min(lit) > 0.999
    for key in ("beamlet:free", "beamlet:capillary"):
        mu = sim.results[key]["maps"]["mu"]
        assert max(max(r) for r in mu) <= 1.0 + 1e-9
        for row in sim.results[key]["maps"]["intensity"]:
            assert all(math.isfinite(v) and v >= 0.0 for v in row)


def test_stage11_beamlet_gaussian_matches_vcz(tmp_path):
    # extended gaussian source: the beamlet |mu| row must track the vCZ curve
    from formula.capsysred.stages.analytic import rms_diff
    sim = Simulation.from_dict({
        "screen": {"nx": 41},
        "free": {"source": {
            "shape": "gaussian", "size": 2.1e-6,
            "position": [0.0, 0.0, -0.08],
            "n_modes": 36, "n_rays": 200,
        }},
        "capillary": {
            "source": {
                "shape": "gaussian", "size": 3.0e-7,
                "position": [0.0, 0.0, -0.01],
                "n_modes": 2, "n_rays": 20,
            },
            "screen": {"nx": 5, "ny": 5},
        },
    })
    _record(sim, str(tmp_path))
    sim.run(str(tmp_path), stages=[11])
    res = sim.results["beamlet:free"]
    maps, screen = res["maps"], res["screen"]
    src = sim.cfg.free_source
    dist = float(screen.z) - float(src.position[2])
    ref_x = screen.pixel_xy(maps["ref_pixel"])[0]
    mu_th = [vcz_mu(x - ref_x, src.shape, float(src.size), float(sim.lam), dist)
             for x in screen.xs()]
    assert rms_diff(maps["mu"][0], mu_th) < 0.2


def test_stage11_beamlet_same_rays_as_stage6(tmp_path):
    # the rng stream matches _mc_stage: arrival-pixel densities are identical
    sim = Simulation.from_dict(TINY)
    _record(sim, str(tmp_path))
    sim.run(str(tmp_path), stages=[6, 11])
    d6 = sim.results["capillary"]["maps"]["density"]
    d11 = sim.results["beamlet:capillary"]["maps"]["density"]
    assert d6 == d11


def test_estimator_protocol_direct_drive_identical_modes():
    # the protocol lets tests feed estimators synthetic rays: no MC, no tracing
    from formula.capsysred.stages.coherence import CoherenceAccumulator
    from formula.capsysred.stages.jackknife import JackknifeCoherence
    from formula.capsysred.shared.types import RayRecord

    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    rec = lambda mode, ray, pixel, opl: RayRecord(
        mode, ray, "screen", pixel, None, None, opl, (), ())
    jack = JackknifeCoherence(lines, 0)
    for mode in range(3):
        jack.new_mode()
        for pixel in (0, 1):
            for ray in (0, 1):
                jack.add_ray(rec(mode, ray, pixel, 0.05), [1.0 + 0j])
        jack.fold_mode()
    maps = jack.finalize(2, 1)
    assert maps["mu"][0][0] == 1.0 and maps["mu"][0][1] == 1.0
    assert maps["mu_err"][0][1] == 0.0

    acc = CoherenceAccumulator(lines, 0, 32)
    one, opl = Number("1", 32), Number("0.05", 32)
    for mode in range(2):
        acc.new_mode()
        for pixel in (0, 1):
            for ray in (0, 1):
                acc.add_ray(rec(mode, ray, pixel, opl), one)
        acc.fold_mode()
    maps = acc.finalize(2, 1)
    assert maps["mu"][0][1] == 1.0 and maps["density"][0][0] == 4.0


def test_jackknife_direct_drive_pi_flip_decoheres():
    # ref phase fixed, pixel-1 phase flips by pi every other mode -> W sums to 0
    from formula.capsysred.stages.jackknife import JackknifeCoherence
    from formula.capsysred.shared.types import RayRecord

    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    k = float(lines[0].k)
    rec = lambda mode, ray, pixel, opl: RayRecord(
        mode, ray, "screen", pixel, None, None, opl, (), ())
    jack = JackknifeCoherence(lines, 0)
    for mode in range(4):
        jack.new_mode()
        for ray in (0, 1):
            jack.add_ray(rec(mode, ray, 0, 0.05), [1.0 + 0j])
            jack.add_ray(rec(mode, ray, 1, 0.05 + (mode % 2) * math.pi / k),
                         [1.0 + 0j])
        jack.fold_mode()
    maps = jack.finalize(2, 1)
    assert maps["mu"][0][0] == 1.0
    assert maps["mu"][0][1] < 1e-6


def test_beamlet_direct_drive_single_mode_fully_coherent():
    # one mode -> one coherent field: mu = 1 on every deposited pixel
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord

    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=7, ny=1, center=[0.0, 0.0],
                                     edge_x=1.0e-5, edge_y=2.0e-6))
    bf = BeamletField(lines, scr, 3, 5.0e-7, 3.0)
    bf.new_mode()
    bf.add_ray(RayRecord(0, 0, "screen", 3, (1.0e-6, 0.0, 0.06),
                         (1.0e-5, 0.0, 1.0), 0.06, (), ()), [1.0 + 0j])
    bf.fold_mode()
    maps = bf.finalize(7, 1)
    lit = [m for m, i in zip(maps["mu"][0], maps["intensity"][0]) if i > 0.0]
    assert lit and min(lit) > 1.0 - 1e-12


def test_stage10_from_file_equals_traced(tmp_path):
    # stage 6 records the capillary rays; stage 10 in the same run consumes
    # the file and must land on the traced maps exactly
    traced = Simulation.from_dict(TINY)
    _record(traced, str(tmp_path / "a"))
    traced.run(str(tmp_path / "a"), stages=[10])
    assert traced.results["jack:capillary"]["rays_from"] == "file"
    reused = Simulation.from_dict(TINY)
    _record(reused, str(tmp_path / "b"))
    reused.run(str(tmp_path / "b"), stages=[6, 10])
    assert reused.results["jack:capillary"]["rays_from"] == "file"
    for key in ("mu", "mu_err", "intensity", "density"):
        assert (traced.results["jack:capillary"]["maps"][key]
                == reused.results["jack:capillary"]["maps"][key]), key
    st_t = traced.results["jack:capillary"]["stats"]
    st_r = reused.results["jack:capillary"]["stats"]
    assert st_t["reflections"] > 0 and st_t["bounce_hist"]
    assert (st_t["reflections"], st_t["bounce_hist"]) == (st_r["reflections"], st_r["bounce_hist"])


def test_rays_file_reused_across_runs(tmp_path):
    # run 1 records the capillary scene; run 2 (same out dir, same config)
    # consumes it for stage 11 and appends the free scene it traces itself
    _record(Simulation.from_dict(TINY), str(tmp_path))
    sim = Simulation.from_dict(TINY)
    _record(sim, str(tmp_path))
    sim.run(str(tmp_path), stages=[11])
    assert sim.results["beamlet:capillary"]["rays_from"] == "file"
    assert sim.results["beamlet:free"]["rays_from"] == "file"
    from formula.capsysred import rays_v3
    assert set(rays_v3.load_index(str(tmp_path / "rays-modes")).budgets) == {"capillary", "free"}


def test_partial_recording_serves_only_its_scenes(tmp_path):
    # capillary-only recording of a yaml that also configures free: stages
    # needing capillary run; free and budget mismatches fail per scene
    _record(TINY, tmp_path, scenes=("capillary",))
    sim = Simulation.from_dict(TINY)
    sim.run(str(tmp_path), stages=[1, 6])
    assert sim.results["capillary"]["rays_from"] == "file"
    with pytest.raises(ValueError, match="scene 'free' is not in"):
        Simulation.from_dict(TINY).run(str(tmp_path), stages=[2])
    more = dict(TINY, capillary=dict(TINY["capillary"], source=dict(
        TINY["capillary"]["source"], n_rays=TINY["capillary"]["source"]["n_rays"] * 2)))
    with pytest.raises(ValueError, match="match n_modes/n_rays"):
        Simulation.from_dict(more).run(str(tmp_path), stages=[6])


def test_trace_command_records_all_scenes(tmp_path):
    # trace_v3 --scene all records every scene; a stage run with the same
    # config and out dir consumes it; a second trace is a no-op
    from formula.capsysred import rays_v3
    _record(TINY, tmp_path)
    archive = str(tmp_path / "rays-modes")
    budgets = rays_v3.load_index(archive).budgets
    assert set(budgets) == {"free", "capillary"}
    sim = Simulation.from_dict(TINY)
    sim.run(str(tmp_path), stages=[2, 6])
    for scene in ("free", "capillary"):
        assert sim.results[scene]["rays_from"] == "file", scene
    before = (tmp_path / "rays-modes" / "rays-index.jsonl").read_bytes()
    _record(TINY, tmp_path)
    assert (tmp_path / "rays-modes" / "rays-index.jsonl").read_bytes() == before


def test_trace_then_replay(tmp_path):
    # trace records rays.jsonl.gz; replay reads it back with no tracing
    _record(TINY, tmp_path)
    sim = Simulation.from_dict(TINY)
    sim.replay(str(tmp_path / "rays-modes"), str(tmp_path / "replay"))
    assert sim.results["capillary"]["stats"]["emitted"] > 0
    assert sim.results["capillary"]["rays_from"] == "file"


def test_rays_runtime_uses_sidecar_and_ignores_first_line(tmp_path):
    from formula.capsysred.config import load
    from formula.capsysred.rays import (RaysReader, scan, sidecar_metadata,
                                        write_metadata)

    path = tmp_path / "rays.jsonl"
    # The preamble is deliberately shaped exactly like a ray row.  It must not
    # affect either the scan counts or the records yielded to consumers.
    rows = [
        {"stage": "free", "mode": 99, "ray": 99, "fate": "lost",
         "pixel": None, "opl": "99", "sins": []},
        {"stage": "free", "mode": 0, "ray": 0, "fate": "lost",
         "pixel": None, "opl": "0.1", "sins": []},
        {"scene_end": "free", "rows": 1},
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    expected = sidecar_metadata(load({"free": {"source": FREE_SOURCE}}))
    write_metadata(path, expected)

    meta, done, clean = scan(str(path), expected_meta=expected)
    assert meta == expected
    assert done == {"free": 1}
    assert clean
    reader = RaysReader(str(path))
    assert reader.meta == expected
    assert [(row.mode, row.ray)
            for row in reader.scene_records("free")] == [(0, 0)]


def test_sidecar_can_retire_an_archived_scene_without_rewriting_rows(tmp_path):
    from formula.capsysred.config import load
    from formula.capsysred.rays import scan, sidecar_metadata, write_metadata

    path = tmp_path / "rays.jsonl.gz"
    cfg = load(TINY)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write("{}\n")
        fh.write(json.dumps({
            "stage": "retired", "mode": 0, "ray": 0,
            "fate": "absorbed", "pixel": None, "opl": "0", "sins": [],
        }) + "\n")
        fh.write(json.dumps({"scene_end": "retired", "rows": 1}) + "\n")
    write_metadata(path, sidecar_metadata(cfg))

    _, done, clean = scan(str(path))
    assert clean and done == {}


def test_rays_reader_refuses_partial_archive(tmp_path):
    from formula.capsysred.config import load
    from formula.capsysred.rays import (RaysReader, sidecar_metadata,
                                        write_metadata)

    path = tmp_path / "rays.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write("{}\n")
        fh.write(json.dumps({
            "stage": "free", "mode": 0, "ray": 0, "fate": "lost",
            "pixel": None, "opl": "0", "sins": [],
        }) + "\n")
    write_metadata(
        path,
        sidecar_metadata(load({"free": {"source": FREE_SOURCE}})),
    )

    with pytest.raises(ValueError, match="incomplete"):
        RaysReader(str(path))


def test_existing_rays_without_sidecar_is_refused_without_mutation(tmp_path):
    _record(TINY, tmp_path)
    archive = tmp_path / "rays-modes"
    index = archive / "rays-index.jsonl"
    (archive / "rays-fingerprint.yaml").unlink()
    before = index.read_bytes()

    with pytest.raises(ValueError, match="rays-fingerprint.yaml"):
        Simulation.from_dict(TINY).run(str(tmp_path), stages=[6])

    assert index.read_bytes() == before
    assert not (archive / "rays-fingerprint.yaml").exists()


def test_rays_file_geometry_change_is_never_overwritten(tmp_path):
    # A stale recording must survive a mismatched run or trace byte-for-byte.
    _record(TINY, tmp_path)
    archive = tmp_path / "rays-modes"
    index, fingerprint = archive / "rays-index.jsonl", archive / "rays-fingerprint.yaml"
    original, original_fp = index.read_bytes(), fingerprint.read_bytes()
    changed = dict(TINY, capillary=dict(TINY["capillary"], z1=0.06))
    with pytest.raises(ValueError, match="remove"):
        Simulation.from_dict(changed).run(str(tmp_path), stages=[11])
    with pytest.raises(ValueError, match="differs from the archive"):
        _record(changed, tmp_path)
    assert index.read_bytes() == original
    assert fingerprint.read_bytes() == original_fp


def test_unreadable_rays_file_is_never_overwritten(tmp_path):
    path = tmp_path / "rays.jsonl.gz"
    original = b"not a gzip stream"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="remove"):
        Simulation.from_dict(TINY).run(str(tmp_path), stages=[6])
    assert path.read_bytes() == original
    assert not (tmp_path / "rays-fingerprint.yaml").exists()


def test_truncated_rays_file_is_never_appended_or_overwritten(tmp_path):
    from formula.capsysred import rays_v3
    _record(TINY, tmp_path)
    archive = str(tmp_path / "rays-modes")
    entry = rays_v3.load_index(archive).sections("capillary", 0)[0]
    path = tmp_path / "rays-modes" / "modes" / entry.file
    path.write_bytes(path.read_bytes()[:-8])  # remove the gzip footer
    broken = path.read_bytes()
    index_bytes = (tmp_path / "rays-modes" / "rays-index.jsonl").read_bytes()

    with pytest.raises(ValueError, match="truncated or corrupt|sha256"):
        Simulation.from_dict(TINY).run(str(tmp_path), stages=[10])
    from formula.capsysred.convert_rays_v3 import verify
    with pytest.raises(ValueError, match="truncated or corrupt|sha256"):
        verify(archive, jobs=1, log=lambda m: None)

    assert path.read_bytes() == broken
    assert (tmp_path / "rays-modes" / "rays-index.jsonl").read_bytes() == index_bytes


def test_conflicting_sidecar_does_not_create_rays_file(tmp_path):
    from formula.capsysred import rays_v3
    archive = tmp_path / "rays-modes"
    rays_v3.write_fingerprint(str(archive), {"format": 3, "geometry": {"stale": True}})
    with pytest.raises(ValueError, match="remove it manually"):
        _record(TINY, tmp_path)
    assert not (archive / "rays-index.jsonl").exists()
    assert not (archive / "modes").exists() or not any((archive / "modes").iterdir())


def test_conflicting_sidecar_is_never_overwritten(tmp_path):
    import yaml
    from formula.capsysred import rays_v3
    _record(TINY, tmp_path)
    archive = str(tmp_path / "rays-modes")
    index_bytes = (tmp_path / "rays-modes" / "rays-index.jsonl").read_bytes()
    wrong = rays_v3.read_fingerprint(archive)
    wrong["geometry"]["seed"] += 1
    fingerprint = tmp_path / "rays-modes" / "rays-fingerprint.yaml"
    fingerprint.write_text(yaml.safe_dump(wrong, sort_keys=False), encoding="utf-8")
    wrong_bytes = fingerprint.read_bytes()

    with pytest.raises(ValueError, match="remove"):
        Simulation.from_dict(TINY).run(str(tmp_path), stages=[6])
    assert (tmp_path / "rays-modes" / "rays-index.jsonl").read_bytes() == index_bytes
    assert fingerprint.read_bytes() == wrong_bytes


def test_rays_metadata_sidecar_helpers(tmp_path):
    from formula.capsysred.rays import (metadata_equal, metadata_path,
                                        read_metadata, write_metadata)

    rays_path = tmp_path / "rays.jsonl.gz"
    first = {"format": 2, "geometry": "abc", "budgets": {"free": [2, 3]}}
    sidecar = tmp_path / "rays-fingerprint.yaml"
    assert metadata_path(rays_path) == str(sidecar)
    assert write_metadata(rays_path, first) == str(sidecar)
    assert read_metadata(rays_path) == first
    write_metadata(rays_path, first)  # Writing identical metadata is idempotent.

    second = dict(first, geometry="def")
    with pytest.raises(ValueError, match="remove it manually"):
        write_metadata(rays_path, second)
    assert read_metadata(rays_path) == first
    sidecar.unlink()  # Conflict resolution is always an explicit user action.
    write_metadata(rays_path, second)
    assert read_metadata(rays_path) == second
    assert not list(tmp_path.glob(".*.tmp"))
    assert not metadata_equal({"value": 1}, {"value": 1.0})
    assert not metadata_equal({"value": True}, {"value": 1})


def test_rays_sidecar_metadata_is_structured(tmp_path):
    from formula.capsysred.config import load
    from formula.capsysred.rays import (geometry_metadata, metadata_equal,
                                        read_metadata,
                                        sidecar_metadata, write_metadata)

    cfg = load(TINY)
    geometry = geometry_metadata(cfg)
    sidecar = sidecar_metadata(cfg)
    assert sidecar["geometry"] == geometry
    rays_path = tmp_path / "rays.jsonl.gz"
    write_metadata(rays_path, sidecar)
    assert metadata_equal(read_metadata(rays_path), sidecar)
    assert geometry["max_bounces"] == cfg.max_bounces
    assert geometry["screen"] == cfg.raw["screen"]
    assert "source" not in geometry
    assert geometry["free"]["source"] == cfg.raw["free"]["source"]
    assert (geometry["capillary"]["source"]
            == cfg.raw["capillary"]["source"])
    assert "lloyd" not in geometry
    assert "lloyd" not in sidecar["budgets"]
    assert "screens" not in geometry["capillary"]

    with_extra_screen = load({
        **TINY,
        "capillary": {
            **TINY["capillary"],
            "screens": [{"z": 0.052}],
        },
    })
    assert geometry_metadata(with_extra_screen) == geometry
    physics_change = load({
        **TINY,
        "material": "glass_oe2012",
        "trace": {"lean_rays": True},
    })
    assert geometry_metadata(physics_change) == geometry
    assert sidecar_metadata(physics_change)["lean"] is True

    geometry["screen"]["z"] = -1
    assert cfg.raw["screen"]["z"] != -1  # The sidecar owns a detached copy.


def test_rays_file_reused_within_run(tmp_path):
    # the run's own record is reused by stage 10 after stage 6
    sim = Simulation.from_dict(TINY)
    _record(sim, tmp_path)
    sim.run(str(tmp_path), stages=[6, 10])
    assert sim.results["jack:capillary"]["rays_from"] == "file"
    d6 = sim.results["capillary"]["maps"]["density"]
    assert d6 == sim.results["jack:capillary"]["maps"]["density"]


def test_stage6_from_file_equals_traced(tmp_path):
    # stage 10 run first records the capillary scene; a later stage-6 run
    # consumes it — the Number path from full-precision strings must land on
    # the traced maps exactly
    cfg = TINY
    traced = Simulation.from_dict(cfg)
    _record(traced, str(tmp_path / "a"))
    traced.run(str(tmp_path / "a"), stages=[6])
    assert traced.results["capillary"]["rays_from"] == "file"
    _record(Simulation.from_dict(cfg), str(tmp_path / "b"))
    Simulation.from_dict(cfg).run(str(tmp_path / "b"), stages=[10])
    reused = Simulation.from_dict(cfg)
    reused.run(str(tmp_path / "b"), stages=[6])
    assert (tmp_path / "b" / "rays-modes" / "rays-index.jsonl").exists()
    assert reused.results["capillary"]["rays_from"] == "file"
    assert traced.results["capillary"]["stats"] == reused.results["capillary"]["stats"]
    for key in ("mu", "intensity", "density"):
        assert (traced.results["capillary"]["maps"][key]
                == reused.results["capillary"]["maps"][key]), key


def test_stage10_extra_screens(tmp_path):
    # extra screens re-bin the same trace: the z-identical extra reproduces
    # the canonical maps exactly, the downstream plane still catches rays
    cap = dict(TINY["capillary"],
               screens=[{}, {"z": 0.08, "edge_x": 6.4e-5, "edge_y": 6.4e-5}])
    sim = Simulation.from_dict({**TINY, "capillary": cap})
    _record(sim, str(tmp_path))
    result = sim.run(str(tmp_path), stages=[10])
    base = sim.results["jack:capillary"]
    same, far = sim.results["jack:capillary-s1"], sim.results["jack:capillary-s2"]
    assert same["rays_from"] == "file" and far["rays_from"] == "file"
    for key in ("mu", "mu_err", "intensity", "density"):
        assert same["maps"][key] == base["maps"][key], key
    assert far["stats"]["screen"] > 0
    assert {"10-capillary-s1-jack-mu.svg",
            "10-capillary-s2-jack-mu.svg"} <= set(result["files"])


def test_rays_file_survives_added_screens(tmp_path):
    # extra screens are post-trace re-binning: the fingerprint ignores them
    _record(Simulation.from_dict(TINY), str(tmp_path))
    cap = dict(TINY["capillary"], screens=[{"z": 0.08}])
    sim = Simulation.from_dict({**TINY, "capillary": cap})
    sim.run(str(tmp_path), stages=[10])
    assert sim.results["jack:capillary"]["rays_from"] == "file"
    assert sim.results["jack:capillary-s1"]["rays_from"] == "file"


def test_stage1_one_scheme_shows_extra_screens(tmp_path):
    cap = dict(TINY["capillary"], screens=[{"z": 0.08}])
    result = Simulation.from_dict({**TINY, "capillary": cap}).run(
        str(tmp_path), stages=[1])
    schemes = sorted(f for f in result["files"] if "scheme" in f)
    assert schemes == ["01-scheme.svg", "01a-scheme-traced.svg"]
    assert "screen 1" in (tmp_path / "01a-scheme-traced.svg").read_text()


def test_extra_screen_inside_optic_rejected():
    cap = dict(TINY["capillary"], screens=[{"z": 0.01}])
    with pytest.raises(ValueError, match="screens"):
        Simulation.from_dict({**TINY, "capillary": cap})


def test_gamma_free_drift_reduces_to_scalar_q():
    # no bounces: Q = (q0+L)*I, no coupling, amplitude = q0/q (w0/w, Gouy)
    import cmath
    from formula.capsysred.gamma import det2, propagate
    k = 2.0 * math.pi / 1.55e-10
    zr = 0.5 * (5e-7) ** 2 * k
    L = 0.14
    q, amp = propagate(zr, [L], [])
    q_scalar = complex(L, zr)
    assert q[1] == 0 and q[0] == q[2]
    assert cmath.isclose(q[0], q_scalar, rel_tol=1e-12)
    assert cmath.isclose(amp, complex(0, zr) / q_scalar, rel_tol=1e-12)


def test_gamma_meridional_reduces_to_two_scalar_q():
    # phi in {0, pi} bounces keep Gamma diagonal: each axis is its own
    # scalar-q chain (doc/capsysred-results.ru.md §5в reduction)
    import cmath
    from formula.capsysred.gamma import propagate
    k = 2.0 * math.pi / 1.55e-10
    zr = 0.5 * (5e-7) ** 2 * k
    segs = [0.01, 0.006, 0.002]
    inv_fs = 1.0 / 1.5e-3
    q, _ = propagate(zr, segs, [(0.0, 0.0, inv_fs), (math.pi, 0.0, inv_fs)])
    assert abs(q[1]) < 1e-12 * abs(q[0])       # sin(pi) float noise only

    def scalar(inv_f):
        qs = complex(0.0, zr)
        for seg, invf in zip(segs, [inv_f, inv_f, 0.0]):
            qs += seg
            if invf:
                qs = 1.0 / (1.0 / qs - invf)
        return qs
    assert cmath.isclose(q[0], scalar(0.0), rel_tol=1e-12)      # tangential
    assert cmath.isclose(q[2], scalar(inv_fs), rel_tol=1e-12)   # sagittal


def test_gamma_skew_bounces_couple_planes():
    # a precessing azimuth mixes the axes: off-diagonal Gamma appears
    from formula.capsysred.gamma import propagate
    k = 2.0 * math.pi / 1.55e-10
    zr = 0.5 * (5e-7) ** 2 * k
    inv_fs = 1.0 / 1.5e-3
    q, _ = propagate(zr, [0.01, 0.006, 0.002],
                     [(0.0, 0.0, inv_fs), (math.pi / 3, 0.0, inv_fs)])
    assert abs(q[1]) > 0.0


def test_gamma_normal_incidence_isotropic():
    # theta = 90 deg: f_t = f_s = R/2, the bounce must not depend on phi
    from formula.capsysred.gamma import propagate
    k = 2.0 * math.pi / 1.55e-10
    zr = 0.5 * (5e-7) ** 2 * k
    inv_f = 2.0 / 0.01
    import cmath
    outs = [propagate(zr, [0.01, 0.02], [(phi, inv_f, inv_f)])[0]
            for phi in (0.0, 0.7, 2.0)]
    for q in outs[1:]:
        assert cmath.isclose(q[0], outs[0][0], rel_tol=1e-12)
        assert cmath.isclose(q[2], outs[0][2], rel_tol=1e-12)
        assert abs(q[1]) < 1e-12 * abs(q[0])


def test_bounce_lenses_cylinder_wall():
    # straight cylinder: f_t flat, 1/f_s = 2 sin/a, phi from the hit azimuth
    from formula.capsysred.gamma import bounce_lenses
    sim = Simulation.from_dict(TINY)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    a = float(cap.bores[0]["radius"])
    s = 2.0e-3
    [(phi, inv_ft, inv_fs)] = bounce_lenses(bundle, [(0.0, a, 0.02)], [s])
    assert phi == pytest.approx(math.pi / 2)
    assert inv_ft == 0.0
    assert inv_fs == pytest.approx(2.0 * s / a)


def test_bounce_lenses_funnel_wall():
    # degenerate funnel (g = f = 1) must reproduce the cylinder lens; a
    # parabolic centered funnel adds the r0*f''(z) meridional curvature
    from formula.capsysred.gamma import bounce_lenses
    r0, s, z = 6.0e-6, 2.0e-3, 0.01
    flat = _cap_sim([{"center": [0.0, 0.0], "radius": r0,
                      "funnel": {"g": [0.0, 0.0]}}])
    bundle = CapillaryBundle(flat.cfg.capillary.bores, flat.cfg.capillary.z0,
                             flat.cfg.capillary.z1)
    [(phi, ift, ifs)] = bounce_lenses(bundle, [(0.0, r0, z)], [s])
    assert phi == pytest.approx(math.pi / 2)
    assert ift == 0.0 and ifs == pytest.approx(2.0 * s / r0)

    bf = -2.0e2                       # r(z) = r0*(1 + bf*z^2): waist profile
    para = _cap_sim([{"center": [0.0, 0.0], "radius": r0,
                      "funnel": {"g": [0.0, 0.0], "f": [0.0, bf]}}])
    bundle = CapillaryBundle(para.cfg.capillary.bores, para.cfg.capillary.z0,
                             para.cfg.capillary.z1)
    ff = 1.0 + bf * z * z
    rp = r0 * 2.0 * bf * z
    rpp = 2.0 * r0 * bf
    [(phi, ift, ifs)] = bounce_lenses(bundle, [(r0 * ff, 0.0, z)], [s])
    assert phi == pytest.approx(0.0)
    assert ifs == pytest.approx(2.0 * s / (r0 * ff))
    assert ift == pytest.approx(-2.0 * rpp / ((1.0 + rp * rp) ** 1.5 * s))
    assert ift > 0.0                  # waist wall curves toward the ray: focusing


def test_bounce_lenses_unknown_kind_falls_flat():
    # future wall kinds must degrade to the flat (scalar-q) model, not crash
    from formula.capsysred.gamma import bounce_lenses
    wall = type("OddWall", (), {"kind": "odd", "_cxf": 0.0, "_cyf": 0.0})()
    optic = type("Optic", (), {"walls": [wall]})()
    assert bounce_lenses(optic, [(1e-6, 0.0, 0.01)], [1e-3]) == [(0.0, 0.0, 0.0)]


_IMPLICIT_PAIR = [
    {"center": [0.0, 0.0], "surface": "x^2+y^2-9", "aim_radius": 3.0e-6,
     "engine_method": "subdivision"},
    {"center": [1.2e-5, 0.0], "surface": "(x-12)^2+y^2-9",
     "aim_radius": 3.0e-6, "engine_method": "subdivision"},
]


def test_bounce_lenses_implicit_multibore_flat():
    # regression: nearest-bore selection read _cxf off ImplicitWall and
    # crashed; implicit bounces come out flat whichever bore is hit
    from formula.capsysred.gamma import bounce_lenses
    cap = _cap_sim(_IMPLICIT_PAIR).cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    pts = [(3.0e-6, 0.0, 0.01), (1.5e-5, 0.0, 0.02)]
    assert bounce_lenses(bundle, pts, [2.0e-3, 2.0e-3]) == [(0.0, 0.0, 0.0)] * 2


def test_bounce_lenses_mixed_kinds_nearest_center():
    # an implicit bore in the bundle: the closed-form neighbour still wins
    # the center-distance pick, the implicit hit falls flat
    from formula.capsysred.gamma import bounce_lenses
    a, s = 3.0e-6, 2.0e-3
    cap = _cap_sim([{"center": [0.0, 0.0], "radius": a},
                    _IMPLICIT_PAIR[1]]).cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    (phi, ift, ifs), flat = bounce_lenses(
        bundle, [(0.0, a, 0.01), (1.5e-5, 0.0, 0.02)], [s, s])
    assert phi == pytest.approx(math.pi / 2)
    assert ift == 0.0 and ifs == pytest.approx(2.0 * s / a)
    assert flat == (0.0, 0.0, 0.0)


def test_beamlet_deposit_implicit_multibore():
    # the stage-11 path of the same regression: add_ray over an implicit
    # bundle deposits a flat-bounce beamlet and flags flat_walls
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord
    cap = _cap_sim(_IMPLICIT_PAIR).cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=21, ny=5, center=[0.0, 0.0],
                                     edge_x=1.2e-5, edge_y=6.0e-6))
    field = BeamletField(lines, scr, 52, 5.0e-7, 3.0, bundle)
    assert field.flat_walls
    field.new_mode()
    rec = RayRecord(0, 0, "screen", scr.pixel((2.0e-6, 1.0e-6)),
                    (2.0e-6, 1.0e-6, 0.06), (5.0e-5, -3.0e-5, 1.0), 0.0612,
                    (2.0e-3,), ((1.4e-5, 0.0, 0.03),))
    field.add_ray(rec, [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(scr.nx, scr.ny)
    assert maps["gamma_bad"] == 0
    total = sum(v for row in maps["intensity"] for v in row)
    assert math.isfinite(total) and total > 0.0


def test_stage11_funnel_bore_runs(tmp_path):
    # taper: bore radius shrinks 6 -> ~4.2 um over 5 cm; stage 11 must
    # deposit finite maps with the funnel meridional lens engaged
    sim = _cap_sim([{"center": [0.0, 0.0], "radius": 6.0e-6,
                     "funnel": {"g": [0.0, 0.0], "f": [-6.0, 0.0]}}])
    _record(sim, str(tmp_path))
    sim.run(str(tmp_path), stages=[11])
    maps = sim.results["beamlet:capillary"]["maps"]
    assert not maps["flat_walls"]
    assert max(max(r) for r in maps["mu"]) <= 1.0 + 1e-9
    for row in maps["intensity"]:
        assert all(math.isfinite(v) and v >= 0.0 for v in row)


def test_beamlet_native_deposit_matches_python():
    # the C++ BeamletGrid mirrors the Python window loop op-for-op: same
    # records, native on/off -> the same maps to float64 roundoff
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.native import make_beamlet_grid
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord

    if make_beamlet_grid(1, 1, 0.0, 0.0, 1.0, 1.0, [1.0], [1.0], [1.0], 3.0) is None:
        pytest.skip("BeamletGrid missing from the built .so")
    lines = spectral_lines({"mode": "gaussian", "rel_fwhm": 1.0e-3,
                            "n_lines": 3, "n_sigma": 2.0}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=21, ny=5, center=[0.0, 0.0],
                                     edge_x=1.2e-5, edge_y=6.0e-6))
    rays = [
        (0.0, 0.0, 0.0, 0.0, 0.06, ()),
        (2.0e-6, 1.0e-6, 5.0e-5, -3.0e-5, 0.0612, ((1.0e-6, 0.0, 0.03),)),
        (-3.0e-6, -2.0e-6, -1.0e-4, 2.0e-5, 0.0605,
         ((2.0e-6, 1.0e-6, 0.02), (-1.0e-6, 5.0e-7, 0.045))),
    ]
    maps = []
    for use_native in (True, False):
        field = BeamletField(lines, scr, 52, 5.0e-7, 3.0, None,
                             use_native=use_native)
        assert (field.native is not None) == use_native
        field.new_mode()
        for i, (x, y, dx, dy, opl, refl) in enumerate(rays):
            rec = RayRecord(0, i, "screen", scr.pixel((x, y)), (x, y, 0.06),
                            (dx, dy, 1.0), opl,
                            tuple(1.0e-3 * (j + 1) for j in range(len(refl))),
                            refl)
            field.add_ray(rec, [1.0 + 0.5j] * len(lines))
        field.fold_mode()
        maps.append(field.finalize(21, 5))
    nat, ref = maps
    imax = max(max(r) for r in ref["intensity"])
    for key in ("mu", "intensity"):
        scale = 1.0 if key == "mu" else imax
        diff = max(abs(a - b) for ra, rb in zip(nat[key], ref[key])
                   for a, b in zip(ra, rb))
        assert diff <= 1e-12 * scale, key
    assert nat["density"] == ref["density"]


# ---------------------------------------------------------------- beamlets


def _beamlet_field_1d(lines, nx=161, edge=8.0e-6, z=0.1, w0=2.5e-7):
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    scr = ScreenGrid(SimpleNamespace(z=z, nx=nx, ny=1, center=[0.0, 0.0],
                                     edge_x=edge, edge_y=2.0e-6))
    return BeamletField(lines, scr, nx // 2, w0, 3.0, None), scr


def test_beamlet_fan_reconstructs_diffraction_limited_focus():
    # the raison d'etre of beamlets (beamlets.ru.md §7B): a converging fan
    # is FINITE at the caustic and its coherent sum narrows to the
    # diffraction limit FWHM = 0.886*lam*f/(2a), where rays give infinity
    import cmath
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    k = float(lines[0].k)
    lam = 2.0 * math.pi / k
    f, a, n = 0.1, 5.0e-6, 81
    field, scr = _beamlet_field_1d(lines, z=f)
    field.new_mode()
    for i in range(n):
        ai = -a + 2.0 * a * i / (n - 1)
        rec = RayRecord(0, i, "screen", scr.pixel((0.0, 0.0)), (0.0, 0.0, f),
                        (-ai / f, 0.0, 1.0), f + ai * ai / (2.0 * f), (), ())
        # lens at the aperture: the converging wave cancels the chirp
        field.add_ray(rec, [cmath.exp(-1j * k * ai * ai / (2.0 * f))])
    field.fold_mode()
    maps = field.finalize(scr.nx, 1)
    prof = maps["intensity"][0]
    xs = scr.xs()
    peak = max(prof)
    assert math.isfinite(peak) and prof.index(peak) == scr.nx // 2
    above = [x for x, v in zip(xs, prof) if v > 0.5 * peak]
    fwhm = max(above) - min(above)
    expected = 0.886 * lam * f / (2.0 * a)
    assert abs(fwhm - expected) < 0.1 * expected
    # first sinc zero at lam*f/(2a): the profile must dip deeply there
    zero_px = min(range(scr.nx), key=lambda i: abs(xs[i] - lam * f / (2 * a)))
    assert prof[zero_px] < 0.05 * peak


def test_beamlet_two_beam_fringes_period():
    # two crossed beamlets interfere with period lam/(2*theta): the direct
    # regression for the tilt-phase term of the deposit
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    lam = 2.0 * math.pi / float(lines[0].k)
    theta = 5.0e-5
    field, scr = _beamlet_field_1d(lines, z=0.06, w0=5.0e-7)
    field.new_mode()
    for i, sgn in enumerate((1.0, -1.0)):
        rec = RayRecord(0, i, "screen", scr.pixel((0.0, 0.0)),
                        (0.0, 0.0, 0.06), (sgn * theta, 0.0, 1.0), 0.06, (), ())
        field.add_ray(rec, [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(scr.nx, 1)
    prof, xs = maps["intensity"][0], scr.xs()
    peak = prof[scr.nx // 2]
    dx = lam / (2.0 * theta)
    at = lambda x: prof[min(range(scr.nx), key=lambda i: abs(xs[i] - x))]
    assert at(0.5 * dx) < 0.03 * peak          # dark fringe
    assert at(dx) > 0.6 * peak                 # next bright fringe


def test_gamma_amp_through_focus_matches_scalar_ratios():
    # a strong isotropic lens puts the focus inside the next drift: the
    # sub-stepped principal square roots must telescope to the exact
    # per-segment ratios q_start/q_end (branch-safe: q stays in the upper
    # half-plane), Gouy pi-jump included
    import cmath
    from formula.capsysred.gamma import propagate
    k = 2.0 * math.pi / 1.55e-10
    zr = 0.5 * (2.0e-7) ** 2 * k
    f = 5.0e-3
    segs, inv_f = [0.01, 0.02], 1.0 / f
    q, amp = propagate(zr, segs, [(0.7, inv_f, inv_f)])
    qs, expected = complex(0.0, zr), complex(1.0, 0.0)
    for seg, invf in zip(segs, [inv_f, 0.0]):
        expected *= qs / (qs + seg)
        qs += seg
        if invf:
            qs = 1.0 / (1.0 / qs - invf)
    assert cmath.isclose(q[0], qs, rel_tol=1e-9) and q[1] == 0
    assert cmath.isclose(amp, expected, rel_tol=1e-9)
    assert abs(cmath.phase(amp) - cmath.phase(1.0 / (1.0 + segs[0] / complex(0, zr)))) > 2.0


def test_gamma_marginal_channel_hundred_bounces_stays_physical():
    # s/f_s = 4*cos(theta) is the stability boundary of the periodic lens
    # chain: after 100 meridional bounces Im(G) must stay negative-definite
    # and the amplitude finite — the long-chain regression for reflect/inv2
    from formula.capsysred.gamma import inv2, propagate
    k = 2.0 * math.pi / 1.55e-10
    zr = 0.5 * (5.0e-7) ** 2 * k
    a, theta = 6.0e-6, 2.0e-3
    seg = 2.0 * a / math.tan(theta)
    inv_fs = 2.0 * math.sin(theta) / a
    lenses = [(0.0 if i % 2 else math.pi, 0.0, inv_fs) for i in range(100)]
    q, amp = propagate(zr, [seg] * 101, lenses)
    gm = inv2(q)
    mean = 0.5 * (gm[0].imag + gm[2].imag)
    dev = math.hypot(0.5 * (gm[0].imag - gm[2].imag), gm[1].imag)
    assert mean + dev < 0.0
    assert 0.0 < abs(amp) < math.inf
    assert abs(q[1]) < 1e-9 * abs(q[0])       # sin(pi) float noise only


def test_beamlet_edge_deposits():
    # (a) center off the window deposits the tail only, density stays empty;
    # (b) far beyond the 3w window deposits nothing and must not crash;
    # (c) a bounce exactly at the source (zero first segment) stays finite
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    field, scr = _beamlet_field_1d(lines, nx=21, edge=1.2e-5, z=0.06, w0=5e-7)
    field.new_mode()
    field.add_ray(RayRecord(0, 0, "screen", None, (8.0e-6, 0.0, 0.06),
                            (0.0, 0.0, 1.0), 0.06, (), ()), [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(21, 1)
    assert all(v == 0.0 for v in maps["density"][0])
    prof = maps["intensity"][0]
    assert prof[-1] > 100.0 * prof[0] > 0.0    # tail lights the near edge

    field, scr = _beamlet_field_1d(lines, nx=21, edge=1.2e-5, z=0.06, w0=5e-7)
    field.new_mode()
    field.add_ray(RayRecord(0, 0, "screen", None, (4.0e-5, 0.0, 0.06),
                            (0.0, 0.0, 1.0), 0.06, (), ()), [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(21, 1)
    assert all(v == 0.0 for v in maps["intensity"][0])

    field, scr = _beamlet_field_1d(lines, nx=21, edge=1.2e-5, z=0.06, w0=5e-7)
    field.new_mode()
    hit = (1.0e-6, 0.0, 0.03)
    opl = math.dist(hit, (2.0e-6, 0.0, 0.06))  # source ON the wall: L0 = 0
    field.add_ray(RayRecord(0, 0, "screen", scr.pixel((2.0e-6, 0.0)),
                            (2.0e-6, 0.0, 0.06), (3.3e-5, 0.0, 1.0), opl,
                            (1.0e-3,), (hit,)), [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(21, 1)
    assert all(math.isfinite(v) for v in maps["intensity"][0])
    assert max(maps["intensity"][0]) > 0.0


def test_beamlet_single_pixel_screen():
    # nx = ny = 1: the window clamps to one cell, mu of the lit cell is 1
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=1, ny=1, center=[0.0, 0.0],
                                     edge_x=2.0e-6, edge_y=2.0e-6))
    field = BeamletField(lines, scr, 0, 5.0e-7, 3.0, None)
    field.new_mode()
    field.add_ray(RayRecord(0, 0, "screen", 0, (0.0, 0.0, 0.06),
                            (0.0, 0.0, 1.0), 0.06, (), ()), [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(1, 1)
    assert maps["mu"] == [[1.0]] and maps["intensity"][0][0] > 0.0


# ------------------------------------------------------------------ funnel


def test_funnel_linear_taper_equals_revolution_twin():
    # r(z) = r0*(1 + af*z) is exactly r2_poly (r0^2, 2 r0^2 af, r0^2 af^2):
    # two independent _wall_lens branches must produce one lens
    from formula.capsysred.gamma import bounce_lenses
    r0, af, s = 5.0e-6, -8.0, 1.5e-3
    fun = _cap_sim([{"center": [0.0, 0.0], "radius": r0,
                     "funnel": {"g": [0.0, 0.0], "f": [af, 0.0]}}])
    rev = _cap_sim([{"center": [0.0, 0.0],
                     "r2_poly": [r0 * r0, 2 * r0 * r0 * af,
                                 r0 * r0 * af * af]}])
    bf = CapillaryBundle(fun.cfg.capillary.bores, fun.cfg.capillary.z0,
                         fun.cfg.capillary.z1)
    br = CapillaryBundle(rev.cfg.capillary.bores, rev.cfg.capillary.z0,
                         rev.cfg.capillary.z1)
    for z in (0.005, 0.02, 0.04):
        pt = (r0 * (1.0 + af * z), 0.0, z)
        (pf, tf, sf), = bounce_lenses(bf, [pt], [s])
        (pr, tr, sr), = bounce_lenses(br, [pt], [s])
        assert pf == pr == pytest.approx(0.0)
        assert tf == pytest.approx(tr, rel=1e-12)
        assert sf == pytest.approx(sr, rel=1e-12)


def test_funnel_bent_axis_matches_torus_arc():
    # a parabolic axis bend c*g(z) with 2*c*bg = 1/R approximates the torus
    # arc: the meridional lens must agree on BOTH walls — focusing on the
    # outer wall of the bend, defocusing on the inner (sign included)
    from formula.capsysred.gamma import bounce_lenses
    R, a, c, s, z = 1.0, 5.0e-6, 1.0e-3, 2.0e-3, 0.01
    tor = _cap_sim([{"center": [0.0, 0.0], "radius": a,
                     "bend": {"radius": R, "toward": [1.0, 0.0]}}])
    fun = _cap_sim([{"center": [c, 0.0], "radius": a,
                     "funnel": {"g": [0.0, 1.0 / (2.0 * c * R)],
                                "f": [0.0, 0.0]}}])
    bt = CapillaryBundle(tor.cfg.capillary.bores, tor.cfg.capillary.z0,
                         tor.cfg.capillary.z1)
    bfu = CapillaryBundle(fun.cfg.capillary.bores, fun.cfg.capillary.z0,
                          fun.cfg.capillary.z1)
    xc_t = R - math.sqrt(R * R - z * z)            # torus centerline at z
    xc_f = c * (1.0 + z * z / (2.0 * c * R))       # funnel axis at z
    for side in (-1.0, +1.0):                      # outer (-x) / inner (+x)
        (_, tt, ts), = bounce_lenses(bt, [(xc_t + side * a, 0.0, z)], [s])
        (_, ft, fs), = bounce_lenses(bfu, [(xc_f + side * a, 0.0, z)], [s])
        expected = -side * 2.0 / (R * s)
        assert tt == pytest.approx(expected, rel=1e-3)
        assert ft == pytest.approx(expected, rel=1e-3)
        assert ts == fs == pytest.approx(2.0 * s / a)


def test_funnel_azimuth_from_local_axis():
    # g(z) shifts the bore axis: phi must follow the LOCAL axis, not the
    # entrance center
    from formula.capsysred.gamma import bounce_lenses
    r0, c, z = 2.0e-6, 1.0e-3, 0.01
    fun = _cap_sim([{"center": [c, 0.0], "radius": r0,
                     "funnel": {"g": [0.0, 500.0]}}])
    b = CapillaryBundle(fun.cfg.capillary.bores, fun.cfg.capillary.z0,
                        fun.cfg.capillary.z1)
    axis_x = c * (1.0 + 500.0 * z * z)
    (phi, _, _), = bounce_lenses(b, [(axis_x, r0, z)], [2.0e-3])
    assert phi == pytest.approx(math.pi / 2)


def test_funnel_true_cone_has_no_meridional_lens():
    # straight generatrix r = r0*(1 + af*z): f'' = 0 exactly -> 1/f_t = 0
    from formula.capsysred.gamma import bounce_lenses
    r0, af = 5.0e-6, -6.0
    fun = _cap_sim([{"center": [0.0, 0.0], "radius": r0,
                     "funnel": {"g": [0.0, 0.0], "f": [af, 0.0]}}])
    b = CapillaryBundle(fun.cfg.capillary.bores, fun.cfg.capillary.z0,
                        fun.cfg.capillary.z1)
    (_, inv_ft, _), = bounce_lenses(b, [(r0 * (1.0 + af * 0.03), 0.0, 0.03)],
                                    [1.0e-3])
    assert inv_ft == 0.0


def test_funnel_multibore_picks_nearest_axis():
    # two funnels: the hit near bore 2 must use ITS local axis for phi
    from formula.capsysred.gamma import bounce_lenses
    r0, z = 2.0e-6, 0.02
    sim = _cap_sim([{"center": [-4.0e-6, 0.0], "radius": r0,
                     "funnel": {"g": [0.0, 0.0]}},
                    {"center": [4.0e-6, 0.0], "radius": r0,
                     "funnel": {"g": [0.0, 0.0]}}])
    b = CapillaryBundle(sim.cfg.capillary.bores, sim.cfg.capillary.z0,
                        sim.cfg.capillary.z1)
    (phi, _, ifs), = bounce_lenses(b, [(4.0e-6 - r0, 0.0, z)], [1.0e-3])
    assert phi == pytest.approx(math.pi)
    assert ifs == pytest.approx(2.0 * 1.0e-3 / r0)


def test_stage11_extra_screens_rebin_same_records(tmp_path):
    # an extra screen on the SAME plane must reproduce the main maps exactly
    # (rescreen is a zero-length flight); a farther screen re-bins the same
    # records into a finite, bounded map
    cfg = dict(TINY)
    cfg["capillary"] = dict(TINY["capillary"], screens=[
        {"z": 0.051},
        {"z": 0.06, "edge_x": 4.0e-5, "edge_y": 4.0e-5},
    ])
    sim = Simulation.from_dict(cfg)
    _record(sim, str(tmp_path))
    result = sim.run(str(tmp_path), stages=[11])
    main = sim.results["beamlet:capillary"]["maps"]
    s1 = sim.results["beamlet:capillary-s1"]["maps"]
    assert s1["mu"] == main["mu"]
    assert s1["mu_err"] == main["mu_err"]
    assert s1["intensity"] == main["intensity"]
    assert s1["density"] == main["density"]
    s2 = sim.results["beamlet:capillary-s2"]["maps"]
    assert max(max(r) for r in s2["mu"]) <= 1.0 + 1e-9
    assert any(v > 0.0 for row in s2["intensity"] for v in row)
    for name in ("11-capillary-s1-beamlet-mu.svg", "11-capillary-s2-beamlet-mu.svg"):
        assert name in result["files"]


def test_universal_replay_runs_any_streaming_stage(tmp_path):
    # record once (trace-only), then replay stages 6, 10 and 11 from the
    # file on a fresh Simulation: maps must equal the directly-run ones
    rec = Simulation.from_dict(TINY)
    _record(rec, str(tmp_path / "rec"))
    direct = Simulation.from_dict(TINY)
    _record(direct, str(tmp_path / "direct"))
    direct.run(str(tmp_path / "direct"), stages=[6, 10, 11])
    rep = Simulation.from_dict(TINY)
    rep.replay(str(tmp_path / "rec" / "rays-modes"),
               str(tmp_path / "rep"), stages=[6, 10, 11])
    for key, maps_key in (("capillary", "mu"), ("jack:capillary", "mu_err"),
                          ("beamlet:capillary", "mu")):
        assert rep.results[key]["rays_from"] == "file", key
        assert (rep.results[key]["maps"][maps_key]
                == direct.results[key]["maps"][maps_key]), key


def test_universal_replay_default_stages_and_guards(tmp_path):
    # default stages come from the scenes present; stage 9 and absent
    # scenes are refused
    _record({k: v for k, v in TINY.items() if k != "free"}, tmp_path)   # capillary only
    path = str(tmp_path / "rays-modes")
    sim = Simulation.from_dict(TINY)
    sim.replay(path, str(tmp_path / "rep"))     # -> stage 6 by default
    assert sim.results["capillary"]["rays_from"] == "file"
    assert "free" not in sim.results
    with pytest.raises(ValueError, match="stage 9"):
        Simulation.from_dict(TINY).replay(path, str(tmp_path / "r9"),
                                          stages=[9])
    with pytest.raises(ValueError, match="scene 'free'"):
        Simulation.from_dict(TINY).replay(path, str(tmp_path / "r2"),
                                          stages=[2])


def test_replay_defaults_intersect_recorded_and_configured_scenes(tmp_path):
    recorded = tmp_path / "recorded"
    _record(Simulation.from_dict(TINY), str(recorded))
    capillary_only = {
        "screen": TINY["screen"],
        "capillary": TINY["capillary"],
    }
    sim = Simulation.from_dict(capillary_only)
    sim.replay(str(recorded / "rays-modes"), str(tmp_path / "replay"))
    assert sim.results["capillary"]["rays_from"] == "file"
    assert "free" not in sim.results


def test_universal_replay_new_spectrum(tmp_path):
    # the point of replay: same geometry, different physics — a gaussian
    # band replayed over rays recorded monochromatically
    _record(Simulation.from_dict(TINY), str(tmp_path))
    band = dict(TINY, spectrum={"mode": "gaussian", "rel_fwhm": 2.0e-4,
                                "n_lines": 3, "n_sigma": 2.0})
    sim = Simulation.from_dict(band)
    sim.replay(str(tmp_path / "rays-modes"), str(tmp_path / "rep"),
               stages=[6, 11])
    assert sim.results["capillary"]["rays_from"] == "file"
    assert sim.results["beamlet:capillary"]["rays_from"] == "file"
    mu = sim.results["capillary"]["maps"]["mu"]
    assert max(max(r) for r in mu) <= 1.0 + 1e-9


def test_replay_rebins_records_onto_the_config_grid(tmp_path):
    # file pixel ids belong to the recording grid; a replay onto a config
    # with another screen grid must re-bin them, not trust them
    _record(Simulation.from_dict(TINY), str(tmp_path / "rec"))
    other = dict(TINY, capillary=dict(TINY["capillary"],
                                      screen={"nx": 5, "ny": 7}))
    direct = Simulation.from_dict(other)
    _record(direct, str(tmp_path / "direct"))
    direct.run(str(tmp_path / "direct"), stages=[10])
    rep = Simulation.from_dict(other)
    rep.replay(str(tmp_path / "rec" / "rays-modes"),
               str(tmp_path / "rep"), stages=[10])
    assert rep.results["jack:capillary"]["rays_from"] == "file"
    a, b = (s.results["jack:capillary"]["maps"] for s in (direct, rep))
    for key in ("mu", "intensity", "density", "solid"):
        assert a[key] == b[key], key


def test_gamma_anisotropic_launch():
    # isotropic triple == scalar for any psi; an elliptic launch drifts as
    # two independent scalar axes; psi = pi/2 swaps the axes
    import cmath
    from formula.capsysred.gamma import propagate
    k = 2.0 * math.pi / 1.55e-10
    zrt, zrs = 0.5 * (3.0e-6) ** 2 * k, 0.5 * (5.0e-7) ** 2 * k
    L = 0.1
    assert (propagate((zrs, zrs, 1.234), [L], [])
            == propagate(zrs, [L], []))
    q, _ = propagate((zrt, zrs, 0.0), [L], [])
    assert q[1] == 0
    assert cmath.isclose(q[0], complex(L, zrt), rel_tol=1e-12)
    assert cmath.isclose(q[2], complex(L, zrs), rel_tol=1e-12)
    q90, _ = propagate((zrt, zrs, math.pi / 2), [L], [])
    assert cmath.isclose(q90[0], complex(L, zrs), rel_tol=1e-9)
    assert cmath.isclose(q90[2], complex(L, zrt), rel_tol=1e-9)


def test_beamlet_anisotropic_spot_flips_aspect():
    # launch narrow sagittal (y), wide tangential (x): the far field flips
    # the aspect — the spot lands WIDER along y than along x
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=41, ny=41, center=[0.0, 0.0],
                                     edge_x=2.4e-5, edge_y=2.4e-5))
    field = BeamletField(lines, scr, scr.ref_pixel(None), 5.0e-7, 3.0, None,
                         w0_t=3.0e-6)
    field.new_mode()
    field.add_ray(RayRecord(0, 0, "screen", scr.ref_pixel(None),
                            (0.0, 0.0, 0.06), (0.0, 0.0, 1.0), 0.06, (), ()),
                  [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(41, 41)
    mid = 20
    peak = maps["intensity"][mid][mid]
    row = [maps["intensity"][mid][ix] for ix in range(41)]   # along x
    col = [maps["intensity"][iy][mid] for iy in range(41)]   # along y
    wx = sum(1 for v in row if v > 0.5 * peak)
    wy = sum(1 for v in col if v > 0.5 * peak)
    assert wy > 1.3 * wx


def test_beamlet_native_parity_anisotropic():
    # the C++ elliptic launch mirrors the Python one, bounces included
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "gaussian", "rel_fwhm": 1.0e-3,
                            "n_lines": 2, "n_sigma": 2.0}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=21, ny=5, center=[0.0, 0.0],
                                     edge_x=1.2e-5, edge_y=6.0e-6))
    rays = [
        (1.0e-6, 0.0, 3.0e-5, -1.0e-5, 0.0611, ((0.0, 1.0e-6, 0.03),)),
        (-2.0e-6, 1.0e-6, -5.0e-5, 2.0e-5, 0.0604, ()),
    ]
    maps = []
    for use_native in (True, False):
        field = BeamletField(lines, scr, 52, 5.0e-7, 3.0, None,
                             use_native=use_native, w0_t=2.5e-6)
        if use_native and field.native is None:
            pytest.skip("BeamletGrid missing from the built .so")
        field.new_mode()
        for i, (x, y, dx, dy, opl, refl) in enumerate(rays):
            field.add_ray(RayRecord(0, i, "screen", scr.pixel((x, y)),
                                    (x, y, 0.06), (dx, dy, 1.0), opl,
                                    tuple(1.0e-3 for _ in refl), refl),
                          [1.0 + 0.5j] * len(lines))
        field.fold_mode()
        maps.append(field.finalize(21, 5))
    nat, ref = maps
    imax = max(max(r) for r in ref["intensity"])
    for key in ("mu", "intensity"):
        scale = 1.0 if key == "mu" else imax
        diff = max(abs(a - b) for ra, rb in zip(nat[key], ref[key])
                   for a, b in zip(ra, rb))
        assert diff <= 1e-12 * scale, key


def test_stage11_w0t_auto(tmp_path):
    # w0_t: "auto" resolves to the scene's Fresnel scale sqrt(lam*L/pi)
    cfg = dict(TINY, beamlet={"w0_t": "auto"})
    sim = Simulation.from_dict(cfg)
    _record(sim, str(tmp_path))
    sim.run(str(tmp_path), stages=[11])
    res = sim.results["beamlet:capillary"]
    src = sim.cfg.capillary.source
    flight = float(res["screen"].z) - float(src.position[2])
    expected = math.sqrt(float(sim.lam) * flight / math.pi)
    assert res["w0_t"] == pytest.approx(expected)
    assert res["w0_t"] != sim.cfg.beamlet_w0
    mu = res["maps"]["mu"]
    assert max(max(r) for r in mu) <= 1.0 + 1e-9


def test_beamlet_jackknife_identical_modes_pinned():
    # identical modes: mu = 1 with sigma = 0 everywhere lit — pinned at the
    # clamp, so every lit pixel must carry the don't-trust flag
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    field, scr = _beamlet_field_1d(lines, nx=21, edge=1.2e-5, z=0.06, w0=5e-7)
    rec = RayRecord(0, 0, "screen", scr.pixel((0.0, 0.0)), (0.0, 0.0, 0.06),
                    (0.0, 0.0, 1.0), 0.06, (), ())
    for _ in range(4):
        field.new_mode()
        field.add_ray(rec, [1.0 + 0j])
        field.fold_mode()
    maps = field.finalize(21, 1)
    lit = [i for i, v in enumerate(maps["intensity"][0]) if v > 0.0]
    assert lit
    assert all(maps["mu"][0][i] > 1.0 - 1e-12 for i in lit)
    assert all(maps["mu_err"][0][i] < 1e-9 for i in lit)
    # at ref the numerator and denominator share the float path: mu is
    # exactly 1 with sigma exactly 0 — pinned at the clamp, don't-trust
    ref = maps["ref_pixel"]
    assert maps["mu"][0][ref] == 1.0 and maps["mu_err"][0][ref] == 0.0
    assert maps["dubious"][0][ref] == 1.0


def test_beamlet_jackknife_matches_bruteforce_leave_one_out():
    # the incremental rows (W - W_s, I - I_s) must land on the sigma computed
    # the hard way: an independent field rebuilt from every K-1 mode subset
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    K = 5

    def mode_rays(mode):
        ph = 0.9 * mode
        return [(-2.0e-6 + 1.0e-6 * mode, 0.0, 3.0e-5 * math.cos(ph),
                 1.0e-5 * math.sin(ph), 0.06 + mode * 3.0e-11),
                (1.5e-6, -1.0e-6 * (mode % 2), -2.0e-5, 0.0,
                 0.0600000002 + mode * 1.0e-11)]

    def build(skip=None):
        field, scr = _beamlet_field_1d(lines, nx=21, edge=1.2e-5, z=0.06,
                                       w0=5.0e-7)
        for mode in range(K):
            if mode == skip:
                continue
            field.new_mode()
            for i, (x, y, dx, dy, opl) in enumerate(mode_rays(mode)):
                field.add_ray(RayRecord(mode, i, "screen", scr.pixel((x, y)),
                                        (x, y, 0.06), (dx, dy, 1.0), opl,
                                        (), ()), [1.0 + 0j])
            field.fold_mode()
        return field.finalize(21, 1)

    full = build()
    for pixel in (6, 10, 14):
        loo = [min(build(skip=s)["mu"][0][pixel], 1.0) for s in range(K)]
        mean = sum(loo) / len(loo)
        sigma = math.sqrt(sum((v - mean) ** 2 for v in loo)
                          * (len(loo) - 1) / len(loo))
        # rows are float32: the incremental sigma matches the float64
        # brute force to single precision
        assert full["mu_err"][0][pixel] == pytest.approx(sigma, abs=1e-5), pixel


def test_beamlet_jackknife_outlier_mode_inflates_sigma():
    # ref phased identically every mode; the probe pixel flips phase by pi in
    # ONE of three modes: the loo set is asymmetric and sigma is large
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    k = float(lines[0].k)
    field, scr = _beamlet_field_1d(lines, nx=21, edge=1.2e-5, z=0.06,
                                   w0=3.0e-6)
    probe_x = 4.0e-6
    for mode in range(3):
        field.new_mode()
        field.add_ray(RayRecord(mode, 0, "screen", scr.pixel((-4.0e-6, 0.0)),
                                (-4.0e-6, 0.0, 0.06), (0.0, 0.0, 1.0),
                                0.06, (), ()), [1.0 + 0j])
        flip = (mode == 2) * math.pi / k
        field.add_ray(RayRecord(mode, 1, "screen", scr.pixel((probe_x, 0.0)),
                                (probe_x, 0.0, 0.06), (0.0, 0.0, 1.0),
                                0.06 + flip, (), ()), [1.0 + 0j])
        field.fold_mode()
    maps = field.finalize(21, 1)
    ref_i = maps["ref_pixel"]
    probe = scr.pixel((probe_x, 0.0))
    assert maps["mu_err"][0][probe] > 0.1          # overlap tails soften it
    assert maps["dubious"][0][probe] == 0.0        # sigma < 1: trusted, just wide
    assert maps["mu_err"][0][probe] > 5.0 * maps["mu_err"][0][ref_i]


def test_beamlet_jackknife_single_mode_pixel_guard():
    # a pixel lit by exactly one mode: its own leave-out term is skipped
    # (denominator would vanish) and sigma stays finite
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    field, scr = _beamlet_field_1d(lines, nx=41, edge=4.0e-5, z=0.01,
                                   w0=3.0e-6)
    lone_x = 1.5e-5
    for mode in range(3):
        field.new_mode()
        field.add_ray(RayRecord(mode, 0, "screen", scr.pixel((0.0, 0.0)),
                                (0.0, 0.0, 0.01), (0.0, 0.0, 1.0),
                                0.01, (), ()), [1.0 + 0j])
        if mode == 1:      # the lone contributor to the far pixel
            field.add_ray(RayRecord(mode, 1, "screen", scr.pixel((lone_x, 0.0)),
                                    (lone_x, 0.0, 0.01), (0.0, 0.0, 1.0),
                                    0.01, (), ()), [1.0 + 0j])
        field.fold_mode()
    maps = field.finalize(41, 1)
    lone = scr.pixel((lone_x, 0.0))
    assert maps["intensity"][0][lone] > 0.0
    assert math.isfinite(maps["mu_err"][0][lone])
    assert 0.0 <= maps["mu_err"][0][lone] <= 1.0


def test_beamlet_aniso_free_widths_match_gaussian():
    # quantitative anisotropy: the deposited spot's second moments must land
    # on the per-axis Gaussian widths w(L) of the two launch waists
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    k = float(lines[0].k)
    w0s, w0t, L = 5.0e-7, 3.0e-6, 0.06
    scr = ScreenGrid(SimpleNamespace(z=L, nx=61, ny=61, center=[0.0, 0.0],
                                     edge_x=4.0e-5, edge_y=4.0e-5))
    field = BeamletField(lines, scr, scr.ref_pixel(None), w0s, 3.0, None,
                         w0_t=w0t)
    field.new_mode()
    field.add_ray(RayRecord(0, 0, "screen", scr.ref_pixel(None),
                            (0.0, 0.0, L), (0.0, 0.0, 1.0), L, (), ()),
                  [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(61, 61)
    xs, ys = scr.xs(), scr.ys()
    tot = sx2 = sy2 = 0.0
    for iy in range(61):
        for ix in range(61):
            v = maps["intensity"][iy][ix]
            tot += v
            sx2 += v * xs[ix] * xs[ix]
            sy2 += v * ys[iy] * ys[iy]
    wx = 2.0 * math.sqrt(sx2 / tot)      # I ~ exp(-2x^2/w^2): <x^2> = w^2/4
    wy = 2.0 * math.sqrt(sy2 / tot)
    zrt, zrs = 0.5 * w0t * w0t * k, 0.5 * w0s * w0s * k
    assert wx == pytest.approx(w0t * math.hypot(1.0, L / zrt), rel=0.05)
    assert wy == pytest.approx(w0s * math.hypot(1.0, L / zrs), rel=0.05)


def test_beamlet_aniso_ellipse_follows_direction_azimuth():
    # a bounce-free ray at azimuth 45 deg carries its launch ellipse with it:
    # the far field is wide along the anti-diagonal (the narrow sagittal
    # launch axis) and narrow along the diagonal
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=61, ny=61, center=[0.0, 0.0],
                                     edge_x=4.0e-5, edge_y=4.0e-5))
    field = BeamletField(lines, scr, scr.ref_pixel(None), 5.0e-7, 3.0, None,
                         w0_t=3.0e-6)
    field.new_mode()
    d = 7.0e-5
    field.add_ray(RayRecord(0, 0, "screen", scr.ref_pixel(None),
                            (0.0, 0.0, 0.06), (d, d, 1.0), 0.06, (), ()),
                  [1.0 + 0j])
    field.fold_mode()
    maps = field.finalize(61, 61)
    r = 4.0e-6 / math.sqrt(2.0)
    at = lambda x, y: maps["intensity"][
        min(range(61), key=lambda i: abs(scr.ys()[i] - y))][
        min(range(61), key=lambda i: abs(scr.xs()[i] - x))]
    assert at(-r, r) > 3.0 * at(r, r)      # anti-diagonal wide, diagonal narrow


def test_gamma_aniso_amp_squared_is_per_axis_product():
    # psi = 0 keeps the axes independent even through an astigmatic bounce:
    # amp^2 must equal the product of the two scalar per-axis chains
    import cmath
    from formula.capsysred.gamma import propagate
    k = 2.0 * math.pi / 1.55e-10
    zrt, zrs = 0.5 * (2.0e-6) ** 2 * k, 0.5 * (3.0e-7) ** 2 * k
    segs, ift, ifs = [0.01, 0.02], 1.0 / 0.04, 1.0 / 0.004
    q, amp = propagate((zrt, zrs, 0.0), segs, [(0.0, ift, ifs)])

    def chain(zr0, inv_f):
        qs, prod = complex(0.0, zr0), complex(1.0, 0.0)
        for seg, invf in zip(segs, [inv_f, 0.0]):
            prod *= qs / (qs + seg)
            qs += seg
            if invf:
                qs = 1.0 / (1.0 / qs - invf)
        return qs, prod

    qx, px = chain(zrt, ift)
    qy, py = chain(zrs, ifs)
    assert cmath.isclose(q[0], qx, rel_tol=1e-9)
    assert cmath.isclose(q[2], qy, rel_tol=1e-9)
    assert cmath.isclose(amp * amp, px * py, rel_tol=1e-9)


def test_stage11_aniso_auto_shrinks_spot_and_reports_sigma(tmp_path):
    # w0_t auto narrows the mean deposited spot vs the isotropic default,
    # and the jackknife maps ride along in maps and mu-beamlet.jsonl
    iso = Simulation.from_dict(TINY)
    _record(iso, str(tmp_path / "iso"))
    iso.run(str(tmp_path / "iso"), stages=[11])
    aniso = Simulation.from_dict(dict(TINY, beamlet={"w0_t": "auto"}))
    _record(aniso, str(tmp_path / "aniso"))
    aniso.run(str(tmp_path / "aniso"), stages=[11])
    w_iso = iso.results["beamlet:capillary"]["maps"]["w_mean"]
    w_ani = aniso.results["beamlet:capillary"]["maps"]["w_mean"]
    assert w_ani < w_iso
    maps = aniso.results["beamlet:capillary"]["maps"]
    flat = [v for row in maps["mu_err"] for v in row]
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in flat)
    rows = [json.loads(l)
            for l in (tmp_path / "aniso" / "mu-beamlet.jsonl").read_text().splitlines()]
    assert all("mu_err" in r and "dubious" in r for r in rows)


def test_grid_source_draws_weighted_nodes():
    # importance draws land on the fixed lattice with gaussian-weighted rates
    import random
    from formula.capsysred.config import SourceCfg
    from formula.capsysred.source import Source

    cfg = SourceCfg({"shape": "grid", "size": 2.1e-6, "grid_n": 7,
                     "grid_step": 1.5085e-6, "position": [0.0, 0.0, -0.03],
                     "n_modes": 49, "n_rays": 10}, 32)
    src = Source(cfg, random.Random(7))
    counts = {}
    for _ in range(20000):
        o = src.mode_origin()
        counts[(float(o[0]), float(o[1]))] = counts.get((float(o[0]), float(o[1])), 0) + 1
    xs = sorted({x for x, _ in counts})
    assert len(counts) <= 49 and len(xs) <= 7
    step = xs[1] - xs[0]
    assert math.isclose(step, 1.5085e-6, rel_tol=1e-9)
    # corner node sits at the printed 6.4 um max radial position
    assert math.isclose(math.hypot(min(xs), min(xs)), 6.4e-6, rel_tol=1e-3)
    # center outdraws the corner by the weight ratio exp(dr^2/2s^2) ~ 104
    assert counts[(0.0, 0.0)] > 20 * counts.get((min(xs), min(xs)), counts[(0.0, 0.0)] // 100 + 1)


# ------------------------------------------------ stage-11 closed-form amp


def _dense_amp_reference(zr, segs, lenses, n=100000):
    """Brute-force branch tracking: n sub-steps per drift, principal sqrt
    per step — the exact continuous branch up to float rounding."""
    import cmath
    from formula.capsysred.gamma import det2, reflect
    zrt, zrs, psi = zr
    c, sn = math.cos(psi), math.sin(psi)
    q = (complex(0.0, zrt * c * c + zrs * sn * sn),
         complex(0.0, (zrt - zrs) * c * sn),
         complex(0.0, zrt * sn * sn + zrs * c * c))
    amp = 1.0 + 0j
    for j, seg in enumerate(segs):
        step = seg / n
        for _ in range(n):
            pre = det2(q)
            q = (q[0] + step, q[1], q[2] + step)
            amp *= cmath.sqrt(pre / det2(q))
        if j < len(lenses):
            q = reflect(q, *lenses[j])
    return q, amp


def test_gamma_amp_closed_form_matches_dense_reference():
    # closed-form drift amplitude vs brute-force branch tracking on an
    # astigmatic chain with skew lenses and on a long free drift
    from formula.capsysred.gamma import propagate
    cases = [
        ((0.2, 0.004, 0.6), [0.05, 0.03, 0.4],
         [(0.3, 1.0 / 0.02, 1.0 / 0.008), (1.2, 0.0, 1.0 / 0.01)]),
        ((0.5, 0.005, 0.3), [0.5], []),
    ]
    for zr, segs, lenses in cases:
        q, amp = propagate(zr, segs, lenses)
        q_ref, amp_ref = _dense_amp_reference(zr, segs, lenses)
        assert abs(amp - amp_ref) <= 1e-9 * abs(amp_ref)
        assert max(abs(a - b) for a, b in zip(q, q_ref)) <= 1e-9 * abs(q_ref[0])


def test_gamma_amp_tight_focus_keeps_the_branch():
    # seg/z_R ~ 5e4: the old 256-sub-step cap lost exactly pi through the
    # lens focus (the whole beamlet flipped sign); the closed form must not
    from formula.capsysred.gamma import propagate
    zr, segs = (2.0e-4, 2.0e-4, 0.0), [0.01, 0.5]
    lenses = [(0.7, 1.0 / 0.005, 1.0 / 0.005)]
    _, amp = propagate(zr, segs, lenses)
    _, amp_ref = _dense_amp_reference(zr, segs, lenses, n=200000)
    assert abs(amp - amp_ref) <= 1e-6 * abs(amp_ref)


def test_gamma_amp_segment_split_invariant():
    # zero-curvature bounces split a drift: q and amp must not move, even
    # when the split points straddle a focus
    from formula.capsysred.gamma import propagate
    zr = (0.3, 0.002, 0.9)
    lens = (0.4, 1.0 / 0.05, 1.0 / 0.006)
    none = (0.0, 0.0, 0.0)
    q1, a1 = propagate(zr, [0.03, 0.5], [lens])
    q2, a2 = propagate(zr, [0.03, 0.1, 0.15, 0.25], [lens, none, none])
    assert abs(a1 - a2) <= 1e-12 * abs(a1)
    assert max(abs(a - b) for a, b in zip(q1, q2)) <= 1e-12 * abs(q1[0])


# ------------------------------------------------ stage-11 wall selection


def test_bounce_lenses_tapered_bundle_picks_the_true_bore():
    # two bores whose axes shrink as g(z): near the exit the outer bore's
    # wall sits nearer the INNER bore's entrance center — selection must
    # follow the axis at z, not the entrance
    from formula.capsysred.gamma import bounce_lenses
    g = [-12.0, 0.0]
    cap = _cap_sim([{"center": [4.0e-5, 0.0], "radius": 2.0e-6,
                     "funnel": {"g": g}},
                    {"center": [1.6e-5, 0.0], "radius": 3.0e-6,
                     "funnel": {"g": g}}]).cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    z, s = 0.049, 2.0e-3
    gg = 1.0 - 12.0 * z
    hit = (4.0e-5 * gg + 2.0e-6, 0.0, z)     # on the outer bore, phi = 0
    [(phi, _, ifs)] = bounce_lenses(bundle, [hit], [s])
    assert phi == pytest.approx(0.0)
    # radius follows the taper (f defaults to g): r = r0*gg — and NOT the
    # 3 um neighbour's 2s/(3e-6*gg) the entrance-center pick would give
    assert ifs == pytest.approx(2.0 * s / (2.0e-6 * gg))


def test_axis_dist2_measures_distance_to_the_bent_axis():
    # torus: zero on the tube-center circle; offsets along the tube radius
    # and along the bend normal come back squared
    from types import SimpleNamespace
    from formula.capsysred.gamma import _axis_dist2
    wall = SimpleNamespace(kind="torus", _Cf=(0.03, 0.0, 0.02),
                           _nf=(0.0, 1.0), _Rf=0.5)
    a, b = math.cos(0.3), math.sin(0.3)
    on = (0.03 + 0.5 * a, 0.0, 0.02 + 0.5 * b)
    assert _axis_dist2(wall, *on) == pytest.approx(0.0, abs=1e-24)
    off = (on[0] + 2.0e-6 * a, on[1], on[2] + 2.0e-6 * b)
    assert _axis_dist2(wall, *off) == pytest.approx(4.0e-12)
    side = (on[0], 5.0e-6, on[2])
    assert _axis_dist2(wall, *side) == pytest.approx(2.5e-11)


# ------------------------------------------- stage-11 estimator internals


def test_beamlet_jackknife_skips_sole_mode_pixels():
    # two modes lighting disjoint screen regions: on a pixel lit by one
    # mode the delete-that-mode replicate is pure float32 residue and must
    # be skipped — sigma exactly 0 plus the don't-trust flag, not noise
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.native import make_beamlet_grid
    from formula.capsysred.screen import ScreenGrid
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.1, nx=401, ny=1, center=[0.0, 0.0],
                                     edge_x=2.0e-4, edge_y=2.0e-6))
    ref = scr.pixel((-6.0e-5, 0.0))
    natives = [False] + ([True] if make_beamlet_grid(
        1, 1, 0.0, 0.0, 1.0, 1.0, [1.0], [1.0], [1.0], 3.0) else [])
    for use_native in natives:
        field = BeamletField(lines, scr, ref, 2.5e-7, 3.0, None,
                             use_native=use_native)
        for x in (-6.0e-5, 6.0e-5):     # one mode per side, no overlap
            field.new_mode()
            field.deposit(x, 0.0, 0.0, 0.0, 0.1, 0.0, [0.1], [], [1.0 + 0j],
                          scr.pixel((x, 0.0)))
            field.fold_mode()
        maps = field.finalize(scr.nx, scr.ny)
        checked = 0
        for ix in range(scr.nx):
            if maps["intensity"][0][ix] <= 0.0 or scr.xs()[ix] > -1.0e-5:
                continue
            checked += 1
            assert maps["mu"][0][ix] == pytest.approx(1.0)
            assert maps["mu_err"][0][ix] == 0.0
            assert maps["dubious"][0][ix] == 1.0
        assert checked > 50


def test_beamlet_window_is_the_ellipse_bounding_box():
    # a skewed anisotropic spot: the windowed deposit must equal the
    # full-frame one inside its support, and everything it drops must sit
    # below the ns-sigma envelope cut
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.screen import ScreenGrid
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.1, nx=201, ny=101, center=[0.0, 0.0],
                                     edge_x=2.0e-4, edge_y=1.0e-4))

    def spot(ns):
        field = BeamletField(lines, scr, 0, 2.5e-7, ns, None,
                             use_native=False, w0_t=5.0e-6)
        field.new_mode()
        field.deposit(0.0, 0.0, 0.0, 0.0, 0.1, 0.7, [0.1], [], [1.0 + 0j], 0)
        return field._g[0]

    cut, full = spot(3.0), spot(24.0)
    peak = max(abs(v) for v in full.values())
    assert cut and len(cut) < len(full)
    for pix, v in cut.items():
        assert v == full[pix]
    dropped = [abs(v) for pix, v in full.items() if pix not in cut]
    assert dropped and max(dropped) <= 2.0e-3 * peak


def test_beamlet_tail_ray_deposit_stays_finite():
    # center far off-window (a tail ray): the C++ row recurrence must sweep
    # outward from the envelope crest — an edge start overflows the ratio
    # to inf and poisons the row with NaN; both paths must agree and stay
    # finite across an e^-30 dynamic range
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.native import make_beamlet_grid
    from formula.capsysred.screen import ScreenGrid
    lines = spectral_lines({"mode": "monochromatic"}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.5, nx=101, ny=3, center=[0.0, 0.0],
                                     edge_x=1.0e-4, edge_y=6.0e-6))
    fields = {}
    for use_native in (True, False):
        if use_native and make_beamlet_grid(1, 1, 0.0, 0.0, 1.0, 1.0, [1.0],
                                            [1.0], [1.0], 3.0) is None:
            pytest.skip("BeamletGrid missing from the built .so")
        field = BeamletField(lines, scr, 151, 5.0e-7, 3.0, None,
                             use_native=use_native)
        field.new_mode()
        field.deposit(-1.8e-4, 0.0, 1.0e-4, 0.0, 0.5, 0.2, [0.5], [],
                      [1.0 + 0j], None)
        field.fold_mode()
        fields[use_native] = field.finalize(scr.nx, scr.ny)
    for maps in fields.values():
        vals = [v for row in maps["intensity"] for v in row]
        assert all(math.isfinite(v) and v >= 0.0 for v in vals)
        assert max(vals) > 0.0          # the window clip did deposit
    nat = [v for row in fields[True]["intensity"] for v in row]
    ref = [v for row in fields[False]["intensity"] for v in row]
    for a, b in zip(nat, ref):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-300)


def test_beamlet_native_lensed_bounces_match_python():
    # nonzero wall lenses through BOTH paths: cylinder (sagittal-only) and
    # a parabolic funnel (meridional f_t too), skew azimuths couple the
    # planes — the C++ reflect branch had no cross-check before
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.gamma import bounce_lenses
    from formula.capsysred.native import make_beamlet_grid
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord
    if make_beamlet_grid(1, 1, 0.0, 0.0, 1.0, 1.0, [1.0], [1.0], [1.0],
                         3.0) is None:
        pytest.skip("BeamletGrid missing from the built .so")
    lines = spectral_lines({"mode": "gaussian", "rel_fwhm": 1.0e-3,
                            "n_lines": 3, "n_sigma": 2.0}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=21, ny=5, center=[0.0, 0.0],
                                     edge_x=1.2e-5, edge_y=6.0e-6))
    a = 3.0e-6
    cyl = _cap_sim([{"center": [0.0, 0.0], "radius": a}]).cfg.capillary
    fun = _cap_sim([{"center": [0.0, 0.0], "radius": 6.0e-6,
                     "funnel": {"g": [0.0, 0.0],
                                "f": [0.0, -2.0e2]}}]).cfg.capillary
    r1 = 6.0e-6 * (1.0 - 2.0e2 * 0.01 * 0.01)
    r2 = 6.0e-6 * (1.0 - 2.0e2 * 0.035 * 0.035)
    cases = [
        (CapillaryBundle(cyl.bores, cyl.z0, cyl.z1),
         [((0.0, a, 0.01), (a * 0.6, -a * 0.8, 0.03)),
          ((-a, 0.0, 0.02),)]),
        (CapillaryBundle(fun.bores, fun.z0, fun.z1),
         [((r1 * 0.28, r1 * 0.96, 0.01), (-r2 * 0.8, r2 * 0.6, 0.035)),
          ((0.0, -r2, 0.035),)]),
    ]
    for bundle, mode_refls in cases:
        # premise: these hits really produce nonzero lenses
        lens = bounce_lenses(bundle, list(mode_refls[0]),
                             [2.0e-3] * len(mode_refls[0]))
        assert all(ifs != 0.0 for _, _, ifs in lens)
        maps = []
        for use_native in (True, False):
            field = BeamletField(lines, scr, 52, 5.0e-7, 3.0, bundle,
                                 use_native=use_native)
            for m, refl in enumerate(mode_refls):
                field.new_mode()
                x, y = 1.0e-6 * (m + 1), -0.5e-6 * m
                rec = RayRecord(m, 0, "screen", scr.pixel((x, y)),
                                (x, y, 0.06), (5.0e-5, -3.0e-5, 1.0), 0.0612,
                                tuple(2.0e-3 * (j + 1)
                                      for j in range(len(refl))), refl)
                field.add_ray(rec, [1.0 + 0.5j, 0.8 - 0.1j, 0.9 + 0.2j])
                field.fold_mode()
            maps.append(field.finalize(21, 5))
        nat, ref = maps
        imax = max(max(r) for r in ref["intensity"])
        for key, scale in (("mu", 1.0), ("mu_err", 1.0),
                           ("intensity", imax)):
            diff = max(abs(p - q) for rp, rq in zip(nat[key], ref[key])
                       for p, q in zip(rp, rq))
            assert diff <= 1e-9 * scale, key
        assert nat["dubious"] == ref["dubious"]
        assert nat["density"] == ref["density"]


def test_beamlet_native_multimode_fold_matches_python():
    # three modes through the C++ fold/totals path vs the pure-Python
    # dicts: mu, sigma_jack, dubious, intensity, density must coincide
    from types import SimpleNamespace
    from formula.capsysred.stages.beamlet import BeamletField
    from formula.capsysred.native import make_beamlet_grid
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.shared.types import RayRecord
    if make_beamlet_grid(1, 1, 0.0, 0.0, 1.0, 1.0, [1.0], [1.0], [1.0],
                         3.0) is None:
        pytest.skip("BeamletGrid missing from the built .so")
    lines = spectral_lines({"mode": "gaussian", "rel_fwhm": 1.0e-3,
                            "n_lines": 3, "n_sigma": 2.0}, Number("8.0", 32))
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=21, ny=5, center=[0.0, 0.0],
                                     edge_x=1.2e-5, edge_y=6.0e-6))
    modes = [
        [(0.0, 0.0, 0.0, 0.0, 0.06, ()),
         (2.0e-6, 1.0e-6, 5.0e-5, -3.0e-5, 0.0612, ((1.0e-6, 0.0, 0.03),))],
        [(-3.0e-6, -2.0e-6, -1.0e-4, 2.0e-5, 0.0605,
          ((2.0e-6, 1.0e-6, 0.02), (-1.0e-6, 5.0e-7, 0.045)))],
        [(1.0e-6, -1.0e-6, 3.0e-5, 4.0e-5, 0.0608,
          ((5.0e-7, -5.0e-7, 0.04),))],
    ]
    maps = []
    for use_native in (True, False):
        field = BeamletField(lines, scr, 52, 5.0e-7, 3.0, None,
                             use_native=use_native)
        for m, rays in enumerate(modes):
            field.new_mode()
            for i, (x, y, dx, dy, opl, refl) in enumerate(rays):
                rec = RayRecord(m, i, "screen", scr.pixel((x, y)),
                                (x, y, 0.06), (dx, dy, 1.0), opl,
                                tuple(1.0e-3 * (j + 1)
                                      for j in range(len(refl))), refl)
                field.add_ray(rec, [1.0 + 0.5j, 0.8 - 0.1j, 0.9 + 0.2j])
            field.fold_mode()
        maps.append(field.finalize(21, 5))
    nat, ref = maps
    imax = max(max(r) for r in ref["intensity"])
    for key, scale in (("mu", 1.0), ("mu_err", 1.0), ("intensity", imax)):
        diff = max(abs(a - b) for ra, rb in zip(nat[key], ref[key])
                   for a, b in zip(ra, rb))
        assert diff <= 1e-9 * scale, key
    assert nat["dubious"] == ref["dubious"]
    assert nat["density"] == ref["density"]
    assert nat["i_ref"] == pytest.approx(ref["i_ref"], rel=1e-12)


def test_file_records_scene_name_prefix_no_cross_pickup(tmp_path):
    # scene names sharing a prefix must not leak rows into each other
    from formula.capsysred.rays import _file_records
    path = str(tmp_path / "rays.jsonl")
    rows = [
        {},
        {"stage": "cap", "mode": 0, "ray": 0, "fate": "absorbed",
         "pixel": None, "opl": "0.1", "sins": []},
        {"stage": "capillary", "mode": 0, "ray": 1, "fate": "screen",
         "pixel": 5, "opl": "0.2", "sins": ["1.0e-3"], "x": 1.0e-6,
         "y": 0.0, "dx": 0.0, "dy": 0.0, "refl": [[0.0, 0.0, 0.01]]},
        {"scene_end": "cap", "rows": 1},
        {"scene_end": "capillary", "rows": 1},
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    short = list(_file_records(path, "cap"))
    assert [(r.mode, r.ray, r.fate) for r in short] == [(0, 0, "absorbed")]
    full = list(_file_records(path, "capillary"))
    assert [(r.ray, r.fate, r.pixel) for r in full] == [(1, "screen", 5)]
    assert full[0].refl == ((0.0, 0.0, 0.01),)
    assert list(_file_records(path, "missing")) == []


# ---------------------------------------- tracer audit xfails (2026-07-31)


@pytest.mark.xfail(strict=True, reason="audit 2026-07-31: a near-zero (not "
                   "exactly zero) leading coefficient collapses the DK seed "
                   "tier in _poly_first; the 48-point bisect grid cannot see "
                   "the grazing pair — hit() returns None on a real crossing")
def test_funnel_near_zero_lead_finds_grazing_chord():
    # straight taper with a bf = 1e-20 residue; tangent chord dipping
    # 1e-5*r0 (60 pm, grazing sine ~4.5 mrad — physical range) into the
    # wall at z = 0.01. The exact-sign scan of the same polynomial gives
    # the crossing pair [2.7398251603e-4, 3.2601316396e-4].
    from formula.capsysred.shared.nums import lift, vadd, vdot, vscale, vsub, vunit
    from formula.capsysred.walls.wall_funnel import FunnelWall

    def N(v, p=96):
        return lift(str(v), p)

    r0, af, bf = N("6e-6"), N("-6.0"), N("1e-20")
    zt, s_az, delta_rel, back = N("0.01"), N("1e-3"), N("1e-5"), N("3e-4")
    one, zero = N(1), N(0)
    f = one + zt * (af + zt * bf)
    fp = af + 2 * bf * zt
    rt = r0 * f
    Pt = (rt, zero, zt)
    nhat = vunit((rt, zero, zero - r0 * r0 * f * fp))
    d0 = (zero, s_az, one)
    dhat = vunit(vsub(d0, vscale(nhat, vdot(d0, nhat))))
    O = vsub(vadd(Pt, vscale(nhat, zero - r0 * delta_rel)),
             vscale(dhat, back))
    n32 = lambda v: lift(str(v), 32)
    wall = FunnelWall((n32(0), n32(0)), n32("6e-6"), (n32(0), n32(0)),
                      (n32("-6.0"), n32("1e-20")), n32(0))
    hit = wall.hit(tuple(n32(c) for c in O), tuple(n32(c) for c in dhat),
                   n32("1e-3"))
    assert hit is not None
    assert abs(float(hit[0]) - 2.7398251603e-4) < 1e-9


@pytest.mark.xfail(strict=True, reason="audit 2026-07-31: pinch in span is "
                   "silently constructible; the exactly-axial ray tunnels "
                   "through the closed throat (double root, no sign change) "
                   "and reaches the screen with 0 bounces")
def test_funnel_pinch_axial_ray_does_not_tunnel():
    # taper radius r0*f crosses zero at z = 1/6 inside [0, 0.3]: a closed
    # throat. The on-axis ray must not come out the other side.
    sim = _cap_sim([{"center": [0.0, 0.0], "radius": 6.0e-6,
                     "funnel": {"g": [0.0, 0.0], "f": [-6.0, 0.0]}}],
                   z0=0.0, z1=0.3)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    p = cap.z0.precision
    zero, one = Number("0", p), Number("1", p)
    res = trace_ray((zero, zero, Number("-0.01", p)), (zero, zero, one),
                    bundle, Number("0.4", p), 200)
    assert res.fate == "absorbed"


@pytest.mark.xfail(strict=True, reason="audit 2026-07-31: past the pinch "
                   "f < 0 but r^2 > 0 — the mirror ghost cone counts as "
                   "bore interior")
def test_funnel_inside_stays_closed_past_pinch():
    from formula.capsysred.shared.nums import lift
    from formula.capsysred.walls.wall_funnel import FunnelWall
    n32 = lambda v: lift(str(v), 32)
    wall = FunnelWall((n32(0), n32(0)), n32("6e-6"), (n32(0), n32(0)),
                      (n32("-6.0"), n32(0)), n32(0))
    assert not wall.inside(1.0e-6, 0.0, 0.2)   # r(0.2) = -1.2e-6: ghost


@pytest.mark.xfail(strict=True, reason="audit 2026-07-31: below ~2.4e-21*a "
                   "chord depth both pair seeds stall in Newton and the "
                   "bisect grid is ~10 orders too coarse — hit() returns "
                   "the far outer-wall root or None instead of the pair")
def test_torus_graze_pair_survives_seed_floor():
    # in-plane grazing chord dipping 2.371e-21*a into the inner wall of a
    # bent bore (R = 0.5, a = 6e-6): the exact quartic has the crossing
    # pair at t ~ 2.4495e-3 (dense exact-sign scan); at delta = 1e-20*a
    # hit() still returns it, one step below it tunnels.
    from formula.capsysred.shared.nums import lift, sqrt as nsqrt
    from formula.capsysred.walls.wall_torus import TorusWall
    N = lambda v: lift(str(v), 32)
    a, R = N("6e-6"), N("0.5")
    wall = TorusWall((N(0), N(0)), a, R, (N(1), N(0)), N(0))
    delta = N("2.371e-21") * a
    dz = (R - a - delta) / R
    dx = nsqrt(N(1) - dz * dz)
    O, d = (N(0), N(0), N(0)), (dx, N(0), dz)
    for cap in ("0.1", "4e-3"):
        hit = wall.hit(O, d, N(cap))
        assert hit is not None
        assert abs(float(hit[0]) - 2.4494823940579796e-3) < 1e-8


@pytest.mark.xfail(strict=True, reason="audit 2026-07-31: python filters "
                   "roots with float(t) > 1e-12 (decimal eps), the C++ twin "
                   "compares in mp against the binary double image — a root "
                   "in the half-ulp gap picks different branches")
def test_native_quartic_first_matches_python_at_eps_t_edge():
    from formula import _formula
    from formula.capsysred.walls import wall_torus as wt
    from formula.capsysred.shared.nums import lift
    if not hasattr(_formula, "trace_dbg_quartic_first"):
        pytest.skip("debug binding missing from the built .so")
    P = 32
    r = lift("0.99999999999999999e-12", P)   # binary 1e-12 < r <= decimal
    assert float(r) == 1e-12
    roots = [r] + [lift(v, P) for v in ("0.002", "-0.001", "0.05")]
    c = [Number("1", P)]
    for rt in roots:
        nc = [Number("0", P) for _ in range(len(c) + 1)]
        for i, v in enumerate(c):
            nc[i] = nc[i] + v
            nc[i + 1] = nc[i + 1] - v * rt
        c = nc
    py = wt._quartic_first(tuple(c), 0.1)
    nat = _formula.trace_dbg_quartic_first(tuple(v._value for v in c), 0.1)
    assert (py is None) == (nat is None)
    assert float(py) == float(Number._wrap(nat, P, False))


def _run_jack(modes, ref=3, npx=7):
    from types import SimpleNamespace
    from formula.capsysred.stages.jackknife import JackknifeCoherence
    from formula.capsysred.shared.types import RayRecord
    line = SimpleNamespace(k=4.05e10, weight=1.0)
    jack = JackknifeCoherence([line], ref)
    for rays in modes:
        jack.new_mode()
        for i, (pixel, opl) in enumerate(rays):
            rec = RayRecord(0, i, "screen", pixel, (0.0, 0.0, 0.5),
                            (0.0, 0.0, 1.0), opl, (), ())
            jack.add_ray(rec, [1.0 + 0j])
        jack.fold_mode()
    return jack.finalize(npx, 1)


def test_jackknife_trust_flags_cover_no_data_cases():
    # C1: the ref never gets >= 2 same-mode rays -> the whole map is
    # masked zeros; every solid pixel must carry the don't-trust flag
    maps = _run_jack([[(3, 0.5 + 1e-7 * m)]
                      + [(1, 0.5 + 1e-7 * m + 1e-9 * r) for r in range(4)]
                      for m in range(5)])
    assert maps["solid"][0][1] == 1.0 and maps["solid"][0][3] == 0.0
    assert maps["mu"][0][1] == 0.0 and maps["mu_err"][0][1] == 0.0
    assert maps["dubious"][0][1] == 1.0

    # C2: pixel 1 lit only in modes where the ref was dark: mu = 0 with
    # zero cross data must be flagged, not reported as confident
    maps = _run_jack([[(3 if m < 3 else 1, 0.5 + 1e-7 * m)] * 4
                      for m in range(6)])
    assert maps["solid"][0][1] == 1.0
    assert maps["mu"][0][1] == 0.0 and maps["dubious"][0][1] == 1.0
    assert maps["mu"][0][3] == 1.0          # the ref itself stays exact

    # single cross-mode: the delete-one loses its only usable replicate
    maps = _run_jack([[(3, 0.5), (3, 0.5), (1, 0.5), (1, 0.5)],
                      [(3, 0.5 + 1e-7), (3, 0.5 + 1e-7)]])
    assert maps["solid"][0][1] == 1.0 and maps["dubious"][0][1] == 1.0

    # control: modes co-lighting ref and pixel 5 with mode-varying
    # relative phase stay unflagged: 0 < mu < 1, 0 < sigma <= 1
    maps = _run_jack([[(3, 0.5 + 1e-7 * m)] * 2
                      + [(5, 0.5 + 1e-7 * m + 3.9e-11 * m)] * 2
                      for m in range(6)])
    assert maps["solid"][0][5] == 1.0 and maps["dubious"][0][5] == 0.0
    assert 0.0 < maps["mu"][0][5] < 1.0
    assert 0.0 < maps["mu_err"][0][5] <= 1.0


def test_scene_stream_refuses_thinned_recording(tmp_path):
    # The sidecar promises 2x3 rows but the scene holds 5: replay must
    # refuse loudly, and a partially consumed stream must count-check
    from types import SimpleNamespace
    from formula.capsysred.config import load
    from formula.capsysred.rays import (RaysReader, _counted,
                                        geometry_metadata, scene_stream,
                                        write_metadata)
    path = tmp_path / "rays.jsonl"
    rows = [{}]
    rows += [{"stage": "free", "mode": 0, "ray": i, "fate": "lost",
              "pixel": None, "opl": "0.1", "sins": []} for i in range(5)]
    rows.append({"scene_end": "free", "rows": 5})
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    write_metadata(path, {
        "format": 2,
        "geometry": geometry_metadata(
            load({"free": {"source": FREE_SOURCE}})
        ),
        "budgets": {"free": [2, 3]},
    })
    sim = SimpleNamespace(rays=RaysReader(str(path)))
    src = SimpleNamespace(budget=lambda: (2, 3))
    with pytest.raises(ValueError, match="thinned"):
        scene_stream(sim, "free", src, None, None, None, 0)
    with pytest.raises(ValueError, match="rewritten or truncated"):
        list(_counted(iter([1, 2]), 3, "free", str(path)))


def test_multi_rays_reader_propagates_lean_from_every_part(tmp_path):
    from formula.capsysred.config import load
    from formula.capsysred.rays import (MultiRaysReader, geometry_metadata,
                                        require_full_rows, write_metadata)

    def recording(name, lean):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "rays.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
            fh.write("{}\n")
            fh.write(json.dumps({
                "stage": "free", "mode": 0, "ray": 0, "fate": "lost",
                "pixel": None, "opl": 0.1 if lean else "0.1", "sins": [],
            }) + "\n")
            fh.write(json.dumps({"scene_end": "free", "rows": 1}) + "\n")
        meta = {
            "format": 2,
            "geometry": geometry_metadata(
                load({"free": {"source": FREE_SOURCE}})
            ),
            "budgets": {"free": [1, 1]},
        }
        if lean:
            meta["lean"] = True
        write_metadata(path, meta)
        return str(path)

    full = recording("full", False)
    lean = recording("lean", True)
    for paths in ([full, lean], [lean, full]):
        rays = MultiRaysReader(paths)
        with pytest.raises(ValueError, match="lean"):
            require_full_rows(rays, "file", "number stage")


def _hex_grazing_case():
    from formula.capsysred.shared.units import m_to_um
    from formula.intersect import RaySurface

    p = 32
    expr = ("((x-(0))*(1)+(y-(0))*(0)-(24))"
            "*((x-(0))*(0.49999999999999999999999999999997)"
            "+(y-(0))*(0.86602540378443864676372317075295)-(24))"
            "*((x-(0))*(-0.49999999999999999999999999999997)"
            "+(y-(0))*(0.86602540378443864676372317075295)-(24))"
            "*((x-(0))*(-1)"
            "+(y-(0))*(2.8841971693993751058209749445923e-33)-(24))"
            "*((x-(0))*(-0.49999999999999999999999999999997)"
            "+(y-(0))*(-0.86602540378443864676372317075295)-(24))"
            "*((x-(0))*(0.49999999999999999999999999999997)"
            "+(y-(0))*(-0.86602540378443864676372317075296)-(24))")
    O = tuple(Number(s, p) for s in
              ("-1.2167386983331492e-08", "2.4448041063369702e-05",
               "-7.67726609566741e-42"))
    d = tuple(Number(s, p) for s in
              ("2.272177873466233779743747043025e-06",
               "0.00061157517247929563405253923964793",
               "0.99999981298530532006755016828724"))
    t_exit = Number("0.052500009818273306857838382314289", p)
    t_ref = Number("0.0053382644074804444925814276487202", p)
    rs = RaySurface(expr, p)
    scale = lift(m_to_um(1), p)
    return rs, scale, O, d, t_exit, t_ref


def test_stage9_sturm_root_pair_hex_grazing():
    from formula.capsysred.stages.validate import _engine_t

    rs, scale, O, d, t_exit, t_ref = _hex_grazing_case()
    t_sturm = _engine_t(rs, scale, O, d, t_exit, HitMethod.STURM)
    assert t_sturm is not None
    assert abs(float((t_sturm - t_ref) / t_ref)) < 1e-25


@pytest.mark.xfail(
    reason="subdivision drops same-sign root pairs (ray 44218); fix pending",
    strict=True)
def test_stage9_subdivision_root_pair_hex_grazing():
    """Pinned bug (2026-08-12): a grazing hex-bore ray loses its wall hit.

    Ray 44218 of A-hex-fold-facet-scatter-100m stage 9: the physical hit and
    an extended-plane crossing share one merged candidate region, g(t) keeps
    one sign at both region ends, and the subdivision refine drops the pair.
    Sturm isolates it. Constants come from replaying the stage-9 rng stream
    (seed 12345) to ray 44218 and printing O/d/full_expr_um at p = 32.
    """
    from formula.capsysred.stages.validate import _engine_t

    rs, scale, O, d, t_exit, t_ref = _hex_grazing_case()
    t_sub = _engine_t(rs, scale, O, d, t_exit, HitMethod.SUBDIVISION)
    assert t_sub is not None, "subdivision dropped the root pair (known bug)"
    assert abs(float((t_sub - t_ref) / t_ref)) < 1e-25

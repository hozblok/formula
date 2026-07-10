"""Smoke and physics checks for the CAPSYSred package (tiny ray budgets)."""

import json
import math
import os

import pytest

from formula.capsysred import Simulation
from formula.capsysred.analytic import lloyd_reference, vcz_mu
from formula.capsysred.nums import exp_i, lift, vunit
from formula.capsysred.spectrum import spectral_lines, wavevector
from formula.capsysred.surfaces import CapillaryBundle, Mirror, engine_hit_t
from formula.capsysred.wall_cylinder import CylinderWall
from formula.capsysred.wall_polygon import PolygonWall
from formula.capsysred.wall_revolution import RevolutionWall
from formula.capsysred.wall_torus import (_bisect_first, _float_seeds,
                                          _quartic_first)
from formula.capsysred.symbolic import (LineAmplitudes, ampl_template,
                                     ray_expression, ray_field_template)
from formula.capsysred.fresnel import FresnelAmplitude
from formula.capsysred.trace import trace_ray
from formula import xray
from formula.formula import Number, Solver

TINY = {
    "source": {"n_modes": 4, "n_rays": 240, "size": 0.0, "shape": "point"},
    "screen": {"nx": 21},
    "lloyd": {"source": {"n_modes": 3, "n_rays": 60}, "screen": {"nx": 31}},
    "capillary": {
        "source": {"n_modes": 3, "n_rays": 40},
        "screen": {"nx": 9, "ny": 9},
    },
}


def test_full_pipeline_files_and_point_source_coherence(tmp_path):
    sim = Simulation.from_dict(TINY)
    result = sim.run(str(tmp_path), stages=[1, 2, 3, 4, 5, 6])
    for name in result["files"]:
        assert (tmp_path / name).stat().st_size > 0
    # point source -> fully coherent: mu ~ 1 on well-lit pixels
    maps = sim.results["free"]["maps"]
    imax = max(maps["intensity"][0])
    lit = [m for m, i in zip(maps["mu"][0], maps["intensity"][0]) if i > 0.3 * imax]
    assert lit and min(lit) > 0.9
    assert max(max(row) for row in maps["mu"]) <= 1.0 + 1e-9


def test_fresnel_matches_engine_reflect_amplitude():
    sim = Simulation.from_dict(TINY)
    p = sim.cfg.precision
    from formula.capsysred.nums import solver
    s = solver("sin(x)", p).number({"x": "2.5e-4"})
    r_fast = sim.fresnel(s)
    r_ref = xray.reflect_amplitude("2.5e-4", str(sim.cfg.energy_kev),
                                   sim.cfg.material, p)
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


def test_lloyd_mirror_reflection_physics():
    sim = Simulation.from_dict(TINY)
    lloyd = sim.cfg.lloyd
    p = sim.cfg.precision
    mirror = Mirror(lloyd.z0, lloyd.z1)
    origin = lloyd.source.position
    slope = -float(lloyd.height) / (0.03 - float(origin[2]))
    d = vunit((lift(slope, p), lift(0.0, p), lift(1.0, p)))
    tr = trace_ray(origin, d, mirror, sim.cfg.lloyd.screen.z, 10)
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
            r_ref = xray.reflect_amplitude(theta, e, mat, p)
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
    sim = Simulation.from_dict(TINY)
    result = sim.run(str(tmp_path), stages=[4])
    assert "rays.jsonl" in result["files"]
    rows = [json.loads(line)
            for line in (tmp_path / "rays.jsonl").read_text().splitlines()]
    assert len(rows) == sim.results["lloyd"]["stats"]["emitted"]
    assert {"stage", "mode", "ray", "fate", "pixel", "opl", "sins"} <= set(rows[0])
    assert any(not row["sins"] for row in rows)          # direct rays recorded too
    digits = rows[0]["opl"].replace(".", "").replace("-", "").lstrip("0")
    assert len(digits) >= 25                             # full-precision strings
    modes = [row["mode"] for row in rows]
    assert modes == sorted(modes)                        # grouped by mode


def test_sample_every_thins_records(tmp_path):
    sim = Simulation.from_dict(dict(TINY, trace={"sample_every": 3}))
    sim.run(str(tmp_path), stages=[4])
    rows = (tmp_path / "rays.jsonl").read_text().splitlines()
    st = sim.results["lloyd"]["stats"]
    n_modes = sim.results["lloyd"]["n_modes"]
    per_mode = st["emitted"] // n_modes
    expected = n_modes * len(range(0, per_mode, 3))
    assert len(rows) == expected


def test_replay_matches_direct_mono(tmp_path):
    sim = Simulation.from_dict(TINY)
    sim.run(str(tmp_path), stages=[4])
    direct = sim.results["lloyd"]["maps"]
    sim.replay(str(tmp_path / "rays.jsonl"), str(tmp_path / "replay"))
    rep = sim.results["replay:lloyd"]["maps"]
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
    sim.run(str(tmp_path), stages=[4])
    direct = sim.results["lloyd"]["maps"]
    sim.replay(str(tmp_path / "rays.jsonl"), str(tmp_path / "replay"))
    rep = sim.results["replay:lloyd"]["maps"]
    for key in ("mu", "intensity"):
        scale = max(max(abs(v) for v in row) for row in direct[key]) or 1.0
        diff = max(abs(x - y) for ra, rb in zip(direct[key], rep[key])
                   for x, y in zip(ra, rb))
        assert diff <= 1e-12 * scale, key


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
                       "aim_radius": 3.4641016151377543e-06}])
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
                       "aim_radius": 3.0e-6}])
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
        {"surface": "x^2+y^2-1"},                                 # no aim_radius
        {"radius": 1e-6, "sides": 2},                             # sides < 3
        {"radius": 1e-6, "bend": {"radius": 1.0}},                # bend w/o toward
        {"radius": 1e-6, "bend": {"radius": 1.0, "toward": [1, 0]}, "sides": 6},
    ):
        with pytest.raises(ValueError):
            _cap_sim([good, bad])


def test_analytic_references_sane():
    assert vcz_mu(0.0, "gaussian", 2e-6, 1.55e-10, 0.14) == 1.0
    assert vcz_mu(5e-6, "gaussian", 2e-6, 1.55e-10, 0.14) < 0.1
    ref = lloyd_reference([1e-6 * i for i in range(1, 12)], 5e-6, "point", 0.0,
                          1e-5, -0.08, 0.0, 0.06, 0.06, 1.55e-10,
                          7.1e-6, 1.6e-7)
    assert max(ref["mu"]) <= 1.0 + 1e-9
    v = ref["intensity"]
    assert max(v) / (min(v) + 1e-12) > 5.0      # fringes present


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


def test_engine_method_config_wiring():
    # trace.engine_method: validated at the config boundary, reaches the
    # ImplicitWall of a `surface:` bore through CapillaryBundle
    from formula.capsysred.config import load
    assert load({}).engine_method == "subdivision"
    with pytest.raises(ValueError):
        load({"trace": {"engine_method": "newton"}})
    cfg = load({"trace": {"engine_method": "sturm"},
                "capillary": {"bores": [{"surface": "x^2+y^2-36",
                                         "aim_radius": 6.0e-6}]}})
    bundle = CapillaryBundle(cfg.capillary.bores, cfg.capillary.z0,
                             cfg.capillary.z1, cfg.engine_method)
    assert bundle.walls[0].method == "sturm"


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

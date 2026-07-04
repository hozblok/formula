"""Smoke and physics checks for the capsim package (tiny ray budgets)."""

import json
import math
import os

import pytest

from formula.capsim import Simulation
from formula.capsim.analytic import lloyd_reference, vcz_mu
from formula.capsim.nums import lift, vunit
from formula.capsim.surfaces import CapillaryBundle, Mirror, engine_hit_t
from formula.capsim.trace import FresnelAmplitude, trace_ray
from formula import xray

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
    with open(tmp_path / "reflections.jsonl", encoding="utf-8") as fh:
        rec = json.loads(next(fh))
    assert {"stage", "mode", "ray", "bounce", "x", "y", "z",
            "grazing_rad", "r_abs"} <= set(rec)


def test_reflections_jsonl_flag_off(tmp_path):
    sim = Simulation.from_dict(dict(TINY, trace={"reflections_jsonl": False}))
    result = sim.run(str(tmp_path), stages=[4])
    assert "reflections.jsonl" not in result["files"]
    assert not (tmp_path / "reflections.jsonl").exists()


def test_fresnel_matches_engine_reflect_amplitude():
    sim = Simulation.from_dict(TINY)
    p = sim.cfg.precision
    from formula.capsim.nums import solver
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
    tr = trace_ray(origin, d, mirror, sim.cfg.lloyd.screen.z, sim.fresnel, 10, 1e-9)
    assert tr.fate == "screen" and len(tr.reflections) == 1
    r = complex(tr.reflections[0][2])
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
    tr = trace_ray(origin, d, bundle, cap.screen.z, sim.fresnel, 50, 1e-9)
    assert tr.fate == "screen" and len(tr.reflections) >= 3


def test_analytic_references_sane():
    assert vcz_mu(0.0, "gaussian", 2e-6, 1.55e-10, 0.14) == 1.0
    assert vcz_mu(5e-6, "gaussian", 2e-6, 1.55e-10, 0.14) < 0.1
    ref = lloyd_reference([1e-6 * i for i in range(1, 12)], 5e-6, "point", 0.0,
                          1e-5, -0.08, 0.0, 0.06, 0.06, 1.55e-10,
                          7.1e-6, 1.6e-7)
    assert max(ref["mu"]) <= 1.0 + 1e-9
    v = ref["intensity"]
    assert max(v) / (min(v) + 1e-12) > 5.0      # fringes present

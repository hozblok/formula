"""Interval DRT: explicit fallback, unchanged MP geometry, real C-2 paths."""

from decimal import Decimal as D, localcontext
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

native = pytest.importorskip("formula._stage19_transport")

from formula.formula import Number
from formula.capsysred import Simulation
from formula.capsysred.native import make_tracer
from formula.capsysred.shared.nums import vsub, vunit
from formula.capsysred.surfaces import CapillaryBundle
from formula.capsysred.stages import _s19_transport as mp
from formula.capsysred.stages._s19_fast_transport import (
    certify_transport, enclosing_interval, interval_payload,
    transport_with_fallback, verify_ray_geometry,
)
from formula.capsysred.stages._s19_mesh import _mp_bore


def interval_matrix(matrix):
    return [[enclosing_interval(v) for v in row] for row in matrix]


def classify(a, b, q=None):
    return native.classify_segment(interval_matrix(q or [[1, 0], [0, 1]]),
        interval_matrix([[a, 0], [0, b]]), enclosing_interval(1), enclosing_interval("1e-20"))


@pytest.mark.parametrize("value", ["0", ".1", "-.1", "1e-300", "1e-324", "-1e-324", "1e200"])
def test_decimal_inputs_are_enclosed(value):
    low, high = enclosing_interval(value)
    assert D.from_float(low) <= D(value) <= D.from_float(high)


def test_runtime_environment_is_compatible():
    assert native.environment_status()["supported"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CRT control test")
def test_ftz_daz_environment_is_rejected_in_isolated_process():
    # Never modify the test runner's floating-point state or other threads.
    code = r'''
import ctypes
from formula import _stage19_transport as m
control=ctypes.CDLL('ucrtbase')._controlfp_s
control.argtypes=[ctypes.POINTER(ctypes.c_uint),ctypes.c_uint,ctypes.c_uint]
control.restype=ctypes.c_int
old=ctypes.c_uint(); current=ctypes.c_uint()
assert control(ctypes.byref(old),0,0)==0
assert m.environment_status()['supported']
try:
    assert control(ctypes.byref(current),0x01000000,0x03000000)==0
    status=m.environment_status()
    q=[[[1,1],[0,0]],[[0,0],[1,1]]]
    result=m.classify_segment(q,q,[1,1],[1e-20,1e-20])
finally:
    assert control(ctypes.byref(current),old.value,0x03000000)==0
assert not status['supported'] and status['reason']=='unsupported_ftz_daz'
assert not result['certified'] and result['reason']=='unsupported_ftz_daz'
assert m.environment_status()['supported']
'''
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nonfinite_input_never_gets_certificate(value):
    with pytest.raises(ValueError):
        enclosing_interval(value)


def test_distinct_and_absent_foci_are_certified():
    assert classify(1, 1)["maslov"] == 0
    one = classify(-2, 1)
    assert one["certified"] and one["maslov"] == 1
    two = classify(-2, -3)
    assert two["certified"] and two["maslov"] == 2
    linear = classify(-2, 0)
    assert linear["certified"] and linear["maslov"] == 1


def test_endpoint_and_multiple_foci_request_mp():
    assert not classify(-1, 1)["certified"]
    assert not classify(-2, -2)["certified"]
    with localcontext() as context:
        context.prec = 64
        for delta in (D("-1e-17"), D("1e-17"), D(0)):
            assert not classify(-1/(1+delta), 1)["certified"]
    # A nearly double focus behind the entrance is irrelevant. Bernstein
    # coefficients certify the entire extended segment without root division.
    behind = classify(1, 1, [[1, "1e-30"], ["-1e-30", 1]])
    assert behind["certified"] and behind["maslov"] == 0
    assert behind["reason"] == "bernstein_no_root"


def test_broad_interval_and_nan_are_rejected():
    q = interval_matrix([[1, 0], [0, 1]])
    velocity = interval_matrix([[-1, 0], [0, 1]])
    velocity[0][0] = [-1.2, -.8]
    assert not native.classify_segment(q, velocity, [1, 1], [1e-20, 1e-20])["certified"]
    velocity[0][0] = [float("nan"), float("nan")]
    assert not native.classify_segment(q, velocity, [1, 1], [1e-20, 1e-20])["certified"]


def test_mp_geometry_rejects_small_error_hidden_by_binary64_interval():
    # At unit coordinates a 1e-20 mutation is invisible in binary64 but exceeds
    # the original MP 1e-24 residual tolerance. The wrapper must still reject.
    with localcontext() as context:
        context.prec = 64
        source = list(map(D, [".1", ".2", "-1"]))
        entrance = list(map(D, [".1", ".2"]))
        ray = SimpleNamespace(point=[D(".1"), D(".2"), D(1)],
                              direction=[D(0), D(0), D(1)], reflections=[])
        bore = {"center": [0, 0], "radius": 1}
        good = certify_transport(source, entrance, ray, bore, 0)
        assert good["certified"] and good["mp_geometry_verified"]
        ray.point[0] += D("1e-20")
        bad = certify_transport(source, entrance, ray, bore, 0)
        assert not bad["certified"] and not bad["mp_geometry_verified"]
        assert bad["reason"] == "mp_transport_residual"


def test_analytic_free_transport_for_distinct_nonaxial_paths():
    with localcontext() as context:
        context.prec = 64
        for source, q, z0, z1 in [([".1", "-.2", "-1"], [".3", ".4"], "0", "2"),
                                   (["-.4", ".7", "-2"], [".1", "-.2"], ".3", ".8"),
                                   (["1", "2", "-3"], [".5", "-.7"], "0", "0")]:
            source, q, z0, z1 = list(map(D, source)), list(map(D, q)), D(z0), D(z1)
            difference = [q[0]-source[0], q[1]-source[1], z0-source[2]]
            direction = [v/mp.norm(difference) for v in difference]
            ratio = (z1-source[2])/(z0-source[2])
            point = [source[i]+(q[i]-source[i])*ratio for i in range(2)]+[z1]
            ray = SimpleNamespace(point=point, direction=direction, reflections=[])
            bore = {"center": [0, 0], "radius": 100}
            reference = mp.transport_ray(source, q, ray, bore, z0)
            fast = transport_with_fallback(source, q, ray, bore, z0)
            assert fast["backend"] == "analytic_free" and fast["valid"]
            assert fast["determinant_bounds"] == (fast["determinant"], fast["determinant"])
            assert fast["maslov"] == reference["maslov"] == 0
            for i in range(2):
                for j in range(2):
                    assert abs(fast["Q"][i][j]-reference["Q"][i][j]) < D("1e-60")
            assert abs(fast["determinant"]-reference["determinant"]) < D("1e-60")


def _scene():
    path = Path(__file__).resolve().with_name("C-2-625.yaml")
    return Simulation.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _trace(sim, bore, source, q):
    optic = CapillaryBundle([_mp_bore(bore, 64)], Number(str(sim.cfg.capillary.z0), 64),
                            Number(str(sim.cfg.capillary.z1), 64))
    tracer = make_tracer(optic)
    typed_source = tuple(Number(str(v), 64) for v in source)
    target = tuple(Number(str(v), 64) for v in (*q, D(str(optic.z0))))
    ray = tracer(typed_source, vunit(vsub(target, typed_source)), optic, optic.z1, 200)
    assert ray.fate == "screen"
    return ray


def test_1470_real_c2_paths_interval_contains_mp_drt():
    sim = _scene()
    # Three distinct archived C-2-625 header positions: modes 0, 312 and 624.
    # A small stratified validation sample, not the full source ensemble.
    sources = [[D("-1.9850001681262593e-7"), D("-7.062065632157497e-7"), D("-.675")],
               [D("8.538815908083625e-8"), D("-3.1537460829011746e-8"), D("-.675")],
               [D("-4.612148327185258e-7"), D("5.073456600494262e-7"), D("-.675")]]
    rng = random.Random(1933017)
    certified = fallback = 0
    with localcontext() as context:
        context.prec = 64
        for bore in sim.cfg.capillary.bores:
            for j in range(210):
                source = sources[j%3]
                radius = D(str(math.sqrt(rng.random())*.999999))*D(str(bore["radius"]))
                angle = rng.random()*2*math.pi
                q = [D(str(bore["center"][i]))+radius*D(str(f(angle)))
                     for i, f in enumerate((math.cos, math.sin))]
                ray = _trace(sim, bore, source, q)
                reference = mp.transport_ray(source, q, ray, bore, 0)
                assert reference["valid"], reference
                result = certify_transport(source, q, ray, bore, 0)
                if not result["certified"]:
                    fallback += 1
                    checked = transport_with_fallback(source, q, ray, bore, 0)
                    assert checked["maslov"] == reference["maslov"]
                    assert checked["valid"] == reference["valid"]
                    continue
                certified += 1
                assert result["maslov"] == reference["maslov"]
                assert result["det_sign"] == (1 if reference["determinant"] > 0 else -1)
                lo, hi = result["determinant_interval"]
                assert D.from_float(lo) <= reference["determinant"] <= D.from_float(hi)
                for key in ("Q", "P"):
                    for i in range(2):
                        for j in range(2):
                            lo, hi = result[key+"_intervals"][i][j]
                            assert D.from_float(lo) <= reference[key][i][j] <= D.from_float(hi)
                geometry = result["geometry"]
                for key in ("position_error", "wall_error", "direction_error", "minimum_grazing"):
                    assert geometry[key] == reference[key]
    assert certified > 0
    assert certified+fallback == 1470
    print({"certified": certified, "fallback": fallback, "total": 1470})


def test_adapter_preserves_certified_determinant_bounds():
    sim = _scene()
    bore = sim.cfg.capillary.bores[1]
    source = list(map(D, ["-1.9850001681262593e-7", "-7.062065632157497e-7", "-.675"]))
    q = list(map(D, [".0000723512301422911614082341348913224",
                      "-.00000300348427303964203780026851723078"]))
    with localcontext() as context:
        context.prec = 64
        ray = _trace(sim, bore, source, q)
        reference = mp.transport_ray(source, q, ray, bore, 0)
        result = transport_with_fallback(source, q, ray, bore, 0)
        assert result["valid"] and result["backend"] == "interval_drt"
        low, high = result["determinant_bounds"]
        assert low <= reference["determinant"] <= high < 0
        assert (low, high) == tuple(D.from_float(v) for v in result["certificate"]["determinant_interval"])


def test_real_c2_focus_boundaries_fall_back(monkeypatch):
    sim = _scene()
    bore = sim.cfg.capillary.bores[1]
    source = list(map(D, ["1e-7", "-2e-7", "-.675"]))
    q = list(map(D, ["79e-6", "8e-6"]))
    events = []
    original = mp.count_free_caustics

    def record(q, velocity, distance, **kwargs):
        v = [[x*distance for x in row] for row in velocity]
        a, c = mp.determinant(v), mp.determinant(q)
        b = q[0][0]*v[1][1]+v[0][0]*q[1][1]-q[0][1]*v[1][0]-v[0][1]*q[1][0]
        disc = b*b-4*a*c
        roots = []
        if a and disc >= 0:
            radical = disc.sqrt()
            stable = -(b+(radical if b >= 0 else -radical))/2
            roots = [stable/a, c/stable] if stable else [-b/(2*a)]*2
        elif not a and b:
            roots = [-c/b]
        events.append((distance, [r for r in roots if 0 < r < 1]))
        return original(q, velocity, distance, **kwargs)

    with localcontext() as context:
        context.prec = 64
        ray = _trace(sim, bore, source, q)
        monkeypatch.setattr(mp, "count_free_caustics", record)
        mp.transport_ray(source, q, ray, bore, 0)
        monkeypatch.setattr(mp, "count_free_caustics", original)
        segment, (distance, roots) = next((i, event) for i, event in enumerate(events) if event[1])
        start = [*q, D(0)] if segment == 0 else list(map(mp.decimal, ray.reflections[segment-1][0]))
        end = list(map(mp.decimal, ray.reflections[segment][0] if segment < len(ray.reflections) else ray.point))
        velocity = [(end[i]-start[i])/distance for i in range(3)]
        direction = [x/mp.norm(velocity) for x in velocity]
        for offset in (D("-1e-17"), D(0), D("1e-17")):
            point = [start[i]+velocity[i]*(distance*roots[0]+offset) for i in range(3)]
            near = SimpleNamespace(point=point, direction=direction, reflections=ray.reflections[:segment])
            result = certify_transport(source, q, near, bore, 0)
            assert not result["certified"]
            expected = mp.transport_ray(source, q, near, bore, 0)
            actual = transport_with_fallback(source, q, near, bore, 0)
            assert actual["backend"] == "decimal_fallback"
            assert actual["maslov"] == expected["maslov"] and actual["valid"] == expected["valid"]
            if actual.get("determinant") is not None:
                assert actual["determinant_bounds"] == (expected["determinant"], expected["determinant"])

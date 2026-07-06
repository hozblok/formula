"""Bit-exact equality of the C++ tracer twin against the Python reference.

Every assertion is exact — full-precision value strings, identical event
kinds, identical floats. Any mismatch is a twin bug, never tolerance-worthy.
"""

import math
import random

import pytest

from formula import _formula
from formula.capsysred.native import (compile_optic, make_tracer,
                                      trace_ray_native)
from formula.capsysred.nums import lift, vadd, vdot, vscale, vsub, vunit
from formula.capsysred.surfaces import CapillaryBundle, Mirror
from formula.capsysred.trace import trace_ray
from formula.capsysred.wall_revolution import RevolutionWall
from formula.capsysred.wall_torus import _dk_roots, _quartic_first
from formula.formula import Number

FMT = _formula.FmtFlags.default
PRECISIONS = (16, 30)          # storage 16 and 32


def full(value):
    """Full-precision canonical string of a Number or a raw mp value."""
    if isinstance(value, Number):
        value = value._value
    return type(value).__name__ + ":" + value.str(0, FMT)


def same_vec(py_vec, c_vec):
    assert len(py_vec) == len(c_vec)
    for a, b in zip(py_vec, c_vec):
        assert full(a) == full(b)


def assert_hit_equal(py_hit, c_hit):
    if py_hit is None or c_hit is None:
        assert py_hit is None and c_hit is None
        return
    assert full(py_hit[0]) == full(c_hit[0])
    same_vec(py_hit[1], c_hit[1])
    same_vec(py_hit[2], c_hit[2])


def assert_event_equal(py_ev, c_ev):
    assert py_ev[0] == c_ev[0]
    if py_ev[0] == "exit":
        assert py_ev[1] is None and c_ev[1] is None
        return
    assert full(py_ev[1]) == full(c_ev[1])
    if py_ev[0] == "reflect":
        same_vec(py_ev[2], c_ev[2])
        same_vec(py_ev[3], c_ev[3])


def assert_trace_equal(py_tr, c_tr):
    assert py_tr.fate == c_tr.fate
    same_vec(py_tr.point, c_tr.point)
    assert full(py_tr.opl) == full(c_tr.opl)
    assert len(py_tr.reflections) == len(c_tr.reflections)
    for (pa, sa), (pb, sb) in zip(py_tr.reflections, c_tr.reflections):
        same_vec(pa, pb)
        assert full(sa) == full(sb)


# ------------------------------------------------------------- geometry kit

def make_bundle(kind, p):
    N = lambda s: Number(s, p)
    z0, z1 = N("0"), N("0.05")
    if kind == "cylinder":
        bores = [{"kind": "cylinder", "center": (N("0"), N("0")),
                  "radius": N("6e-6")}]
    elif kind == "revolution":
        bores = [{"kind": "revolution", "center": (N("1e-6"), N("-2e-6")),
                  "r2_poly": (N("3.6e-11"), N("-4e-10"), N("2e-9"))}]
    elif kind == "polygon":
        bores = [{"kind": "polygon", "center": (N("0"), N("0")),
                  "radius": N("5e-6"), "sides": 6,
                  "rotation": Number("15*pi/180", p)}]
    elif kind == "torus":
        bores = [{"kind": "torus", "center": (N("0"), N("0")),
                  "radius": N("6e-6"),
                  "bend": {"radius": N("0.5"), "toward": (N("1"), N("0"))}}]
    else:  # three bores of mixed kinds, spaced along x
        bores = [
            {"kind": "cylinder", "center": (N("0"), N("0")),
             "radius": N("5e-6")},
            {"kind": "polygon", "center": (N("2e-5"), N("0")),
             "radius": N("4e-6"), "sides": 4, "rotation": N("0")},
            {"kind": "torus", "center": (N("-2e-5"), N("0")),
             "radius": N("5e-6"),
             "bend": {"radius": N("0.8"), "toward": (N("0"), N("1"))}},
        ]
    return CapillaryBundle(bores, z0, z1)


def make_revolution_bundle(r2_strs, p):
    N = lambda s: Number(s, p)
    bores = [{"kind": "revolution", "center": (N("0"), N("0")),
              "r2_poly": tuple(N(s) for s in r2_strs)}]
    return CapillaryBundle(bores, N("0"), N("0.05"))


def rays(bundle, p, seed, count, slope_scale=3.0, wall_index=0):
    """Random rays aimed into a bore: entrance points and grazing slopes."""
    rng = random.Random(seed)
    cxf, cyf, af = bundle.walls[wall_index].aim
    z0f = float(bundle.z0)
    length = float(bundle.z1) - z0f
    for _ in range(count):
        r = af * math.sqrt(rng.random()) * 0.9
        phi = rng.uniform(0.0, 2.0 * math.pi)
        ox, oy = cxf + r * math.cos(phi), cyf + r * math.sin(phi)
        oz = z0f if rng.random() < 0.7 else z0f - 0.01
        slope = slope_scale * 2.0 * af / length
        sx, sy = rng.uniform(-slope, slope), rng.uniform(-slope, slope)
        origin = (lift(ox, p), lift(oy, p), lift(oz, p))
        direction = vunit((lift(sx, p), lift(sy, p), lift(1.0, p)))
        yield origin, direction


def t_exit_of(bundle, origin, direction):
    return (bundle.z1 - origin[2]) / direction[2]


# ------------------------------------------------------------- bridges

def test_repr_bridge_matches_python_repr():
    rng = random.Random(7)
    vals = [0.0, -0.0, 1.0, -1.0, 0.1, 2.0 / 3.0, 1e16, 1e-16, 5e-324,
            1.7976931348623157e308, 2.2250738585072014e-308, 6e-6, math.pi]
    for _ in range(20000):
        vals.append(rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-320, 308))
    for v in vals:
        assert _formula.trace_dbg_repr(v) == repr(v)


def test_pyfloat_and_cmp_key_bridges():
    exprs = ["1/3", "sqrt(2)", "2/30000000", "1e-400", "1e400", "-1e-401",
             "6e-6", "-0.05", "sin(1)/1e20", "1+1e-25", "0"]
    for p in PRECISIONS:
        for expr in exprs:
            x = Number(expr, p)
            assert repr(_formula.trace_dbg_pyfloat(x._value)) == repr(float(x))
            assert full(_formula.trace_dbg_cmp_key(x._value)) == \
                full(Number(x.cmp_key))


def test_lift_bridge():
    rng = random.Random(11)
    vals = [0.0, -0.0, 1e-12, 5e-13, 2e-6, 0.05, 1.0 + 2.0 ** -52]
    for _ in range(2000):
        vals.append(rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-30, 3))
    for p in PRECISIONS:
        for v in vals:
            assert full(_formula.trace_dbg_lift(v, p)) == full(lift(v, p))


def test_dk_roots_bitwise():
    rng = random.Random(3)
    for _ in range(300):
        cf = [1.0] + [rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-9, 3)
                      for _ in range(4)]
        py = _dk_roots(cf)
        cc = _formula.trace_dbg_dk_roots(cf)
        assert len(py) == len(cc)
        for a, b in zip(py, cc):
            assert repr(a.real) == repr(b.real)
            assert repr(a.imag) == repr(b.imag)


def quartic_from_roots(r_mult3, s, p):
    """Monic (t-r)^3 (t-s) in Number — triple root stalls Newton, forcing the
    exact-sign bisection fallback in _quartic_first."""
    r, s = lift(r_mult3, p), lift(s, p)
    three, one = Number("3", p), Number("1", p)
    c3 = Number("0", p) - three * r - s
    c2 = three * r * r + three * r * s
    c1 = Number("0", p) - r * r * r - three * r * r * s
    c0 = r * r * r * s
    return (one, c3, c2, c1, c0)


def test_quartic_first_bitwise():
    rng = random.Random(5)
    for p in PRECISIONS:
        cases = []
        for _ in range(60):
            cs = tuple(lift(rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-11, -4), p)
                       for _ in range(4))
            cases.append(((Number("1", p),) + cs, 10.0 ** rng.randint(-4, -1)))
        cases.append((quartic_from_roots(2e-6, -1e-3, p), 1e-3))   # bisection
        cases.append((quartic_from_roots(4e-5, 9e-3, p), 1e-3))    # root beyond cap
        for c, t_capf in cases:
            py = _quartic_first(c, t_capf)
            cc = _formula.trace_dbg_quartic_first([x._value for x in c], t_capf)
            if py is None or cc is None:
                assert py is None and cc is None
            else:
                assert full(py) == full(cc)


# ------------------------------------------------------------- walls

@pytest.mark.parametrize("kind", ["cylinder", "revolution", "polygon", "torus"])
@pytest.mark.parametrize("p", PRECISIONS)
def test_wall_hit_bitwise(kind, p):
    bundle = make_bundle(kind, p)
    native = compile_optic(bundle)
    assert native is not None and native.kind == "bundle"
    wall = bundle.walls[0]
    count = 12 if kind == "torus" else 30
    for origin, direction in rays(bundle, p, seed=101, count=count):
        t_exit = t_exit_of(bundle, origin, direction)
        py_hit = wall.hit(origin, direction, t_exit)
        c_hit = _formula.trace_wall_hit(
            native, 0, tuple(c._value for c in origin),
            tuple(c._value for c in direction), t_exit._value)
        assert_hit_equal(py_hit, c_hit)


@pytest.mark.parametrize("p", PRECISIONS)
def test_wall_hit_none_paths(p):
    bundle = make_bundle("cylinder", p)
    native = compile_optic(bundle)
    axis = (lift(0.0, p), lift(0.0, p), lift(0.0, p))
    dz = (lift(0.0, p), lift(0.0, p), lift(1.0, p))     # axis-parallel: A ~ 0
    t_exit = t_exit_of(bundle, axis, dz)
    assert bundle.walls[0].hit(axis, dz, t_exit) is None
    assert _formula.trace_wall_hit(
        native, 0, tuple(c._value for c in axis),
        tuple(c._value for c in dz), t_exit._value) is None


@pytest.mark.parametrize("kind", ["cylinder", "revolution", "polygon", "torus"])
def test_wall_inside_bitwise(kind):
    p = 30
    bundle = make_bundle(kind, p)
    native = compile_optic(bundle)
    wall = bundle.walls[0]
    cxf, cyf, af = wall.aim
    rng = random.Random(13)
    for _ in range(400):
        x = cxf + rng.uniform(-1.6, 1.6) * af
        y = cyf + rng.uniform(-1.6, 1.6) * af
        z = rng.uniform(-0.001, 0.051)
        assert wall.inside(x, y, z) == \
            _formula.trace_wall_inside(native, 0, x, y, z)


# The C++ side has no cylinder wall: CylinderWall runs as Revolution with
# c1 = c2 = 0. These tests pin that mapping and widen revolution coverage.

@pytest.mark.parametrize("p", PRECISIONS)
def test_cylinder_equals_revolution_bitwise(p):
    """Python cylinder ≡ Python revolution(a², 0, 0) ≡ native, bit for bit."""
    bundle = make_bundle("cylinder", p)
    native = compile_optic(bundle)
    cyl = bundle.walls[0]
    zero = Number("0", p)
    rev = RevolutionWall("revolution", cyl.center, (cyl._a2, zero, zero))
    hits = 0
    for origin, direction in rays(bundle, p, seed=503, count=25):
        t_exit = t_exit_of(bundle, origin, direction)
        h_cyl = cyl.hit(origin, direction, t_exit)
        h_rev = rev.hit(origin, direction, t_exit)
        h_cc = _formula.trace_wall_hit(
            native, 0, tuple(c._value for c in origin),
            tuple(c._value for c in direction), t_exit._value)
        assert_hit_equal(h_cyl, h_rev)
        assert_hit_equal(h_cyl, h_cc)
        hits += h_cyl is not None
    assert hits
    rng = random.Random(19)
    for _ in range(200):
        x = rng.uniform(-1.5, 1.5) * 6e-6
        y = rng.uniform(-1.5, 1.5) * 6e-6
        z = rng.uniform(-0.001, 0.051)
        assert (cyl.inside(x, y, z) == rev.inside(x, y, z)
                == _formula.trace_wall_inside(native, 0, x, y, z))


R2_SHAPES = [("3.6e-11", "-4e-10", "0"),      # narrowing cone
             ("2.5e-11", "3e-10", "0"),       # widening cone
             ("3.6e-11", "2e-10", "-4e-9"),   # barrel (c2 < 0)
             ("2.5e-11", "-4e-10", "6e-9")]   # waist (c2 > 0)


@pytest.mark.parametrize("r2", R2_SHAPES)
@pytest.mark.parametrize("p", PRECISIONS)
def test_revolution_shapes_hit_bitwise(r2, p):
    bundle = make_revolution_bundle(r2, p)
    native = compile_optic(bundle)
    wall = bundle.walls[0]
    hits = 0
    for origin, direction in rays(bundle, p, seed=401, count=25,
                                  slope_scale=3.5):
        t_exit = t_exit_of(bundle, origin, direction)
        py_hit = wall.hit(origin, direction, t_exit)
        c_hit = _formula.trace_wall_hit(
            native, 0, tuple(c._value for c in origin),
            tuple(c._value for c in direction), t_exit._value)
        assert_hit_equal(py_hit, c_hit)
        hits += py_hit is not None
    assert hits


@pytest.mark.parametrize("p", PRECISIONS)
def test_axis_parallel_branches_bitwise(p):
    """d = (0,0,1): cylinder → None both sides; narrowing cone → the exact
    linear branch (A == 0, B ≠ 0) with a bit-equal hit."""
    N = lambda s: Number(s, p)
    d = (N("0"), N("0"), N("1"))
    cyl_bundle = make_bundle("cylinder", p)
    native_cyl = compile_optic(cyl_bundle)
    O = (N("2e-6"), N("1e-6"), N("0"))
    t_exit = t_exit_of(cyl_bundle, O, d)
    assert cyl_bundle.walls[0].hit(O, d, t_exit) is None
    assert _formula.trace_wall_hit(
        native_cyl, 0, tuple(c._value for c in O),
        tuple(c._value for c in d), t_exit._value) is None

    cone = make_revolution_bundle(("3.6e-11", "-4e-10", "0"), p)
    native_cone = compile_optic(cone)
    O = (N("5e-6"), N("0"), N("0"))
    t_exit = t_exit_of(cone, O, d)
    py_hit = cone.walls[0].hit(O, d, t_exit)
    assert py_hit is not None            # the wall narrows onto the ray
    c_hit = _formula.trace_wall_hit(
        native_cone, 0, tuple(c._value for c in O),
        tuple(c._value for c in d), t_exit._value)
    assert_hit_equal(py_hit, c_hit)
    screen_z = Number("0.051", p)
    tracer = make_tracer(cone)
    assert_trace_equal(trace_ray(O, d, cone, screen_z, 50),
                       tracer(O, d, cone, screen_z, 50))


@pytest.mark.parametrize("kind", ["cylinder", "revolution"])
@pytest.mark.parametrize("p", PRECISIONS)
def test_on_wall_event_chain_bitwise(kind, p):
    """next_event parity at every bounce: origins land exactly on the wall."""
    bundle = make_bundle(kind, p)
    native = compile_optic(bundle)
    two = Number("2", p)
    reflects = 0
    for origin, direction in rays(bundle, p, seed=307, count=6,
                                  slope_scale=5.0):
        O, d = origin, direction
        for _ in range(8):
            ev_py = bundle.next_event(O, d)
            ev_cc = _formula.trace_next_event(
                native, tuple(c._value for c in O),
                tuple(c._value for c in d))
            assert_event_equal(ev_py, ev_cc)
            if ev_py[0] in ("exit", "absorb"):
                break
            if ev_py[0] == "pass":
                O = vadd(O, vscale(d, ev_py[1]))
                continue
            _, t, P, n = ev_py
            dot = vdot(d, n)
            d = vsub(d, vscale(n, two * dot))
            O = P
            reflects += 1
    assert reflects


# ------------------------------------------------------------- events, traces

@pytest.mark.parametrize("p", PRECISIONS)
def test_bundle_next_event_bitwise(p):
    bundle = make_bundle("mixed", p)
    native = compile_optic(bundle)
    probes = []
    for i in range(len(bundle.walls)):
        probes += list(rays(bundle, p, seed=17 + i, count=8, wall_index=i))
    web = (lift(1e-5, p), lift(0.0, p), lift(0.0, p))       # between bores
    up = vunit((lift(1e-4, p), lift(0.0, p), lift(1.0, p)))
    back = vunit((lift(1e-4, p), lift(0.0, p), lift(-1.0, p)))
    beyond = (lift(0.0, p), lift(0.0, p), lift(0.06, p))    # past the exit
    probes += [(web, up), (web, back), (beyond, up)]
    for origin, direction in probes:
        assert_event_equal(
            bundle.next_event(origin, direction),
            _formula.trace_next_event(
                native, tuple(c._value for c in origin),
                tuple(c._value for c in direction)))


@pytest.mark.parametrize("p", PRECISIONS)
def test_mirror_events_and_traces_bitwise(p):
    N = lambda s: Number(s, p)
    mirror = Mirror(N("0"), N("0.06"))
    native = compile_optic(mirror)
    assert native is not None and native.kind == "mirror"
    screen_z = N("0.08")
    rng = random.Random(29)
    tracer = make_tracer(mirror)
    for _ in range(30):
        ox = rng.uniform(0.0, 2e-5)
        oz = rng.uniform(-0.09, -0.01)
        sx = rng.uniform(-4e-4, 1e-4)
        origin = (lift(ox, p), lift(0.0, p), lift(oz, p))
        direction = vunit((lift(sx, p), lift(rng.uniform(-1e-5, 1e-5), p),
                           lift(1.0, p)))
        assert_event_equal(
            mirror.next_event(origin, direction),
            _formula.trace_next_event(
                native, tuple(c._value for c in origin),
                tuple(c._value for c in direction)))
        py_tr = trace_ray(origin, direction, mirror, screen_z, 10)
        assert_trace_equal(py_tr, tracer(origin, direction, mirror,
                                         screen_z, 10))


@pytest.mark.parametrize("kind", ["cylinder", "revolution", "polygon",
                                  "torus", "mixed"])
@pytest.mark.parametrize("p", PRECISIONS)
def test_trace_ray_bitwise(kind, p):
    bundle = make_bundle(kind, p)
    tracer = make_tracer(bundle)
    assert tracer is not trace_ray          # the native path really engaged
    screen_z = Number("0.051", p)
    count = 8 if kind in ("torus", "mixed") else 20
    fates = set()
    for origin, direction in rays(bundle, p, seed=211, count=count,
                                  slope_scale=4.0):
        py_tr = trace_ray(origin, direction, bundle, screen_z, 50)
        assert_trace_equal(py_tr, tracer(origin, direction, bundle,
                                         screen_z, 50))
        fates.add(py_tr.fate)
    assert "screen" in fates                # the sample is not degenerate


def test_trace_free_space_bitwise():
    p = 30
    tracer = make_tracer(None)
    screen_z = Number("0.06", p)
    rng = random.Random(31)
    for _ in range(20):
        origin = (lift(rng.uniform(-1e-6, 1e-6), p),
                  lift(rng.uniform(-1e-6, 1e-6), p), lift(-0.08, p))
        direction = vunit((lift(rng.uniform(-1e-4, 1e-4), p),
                           lift(rng.uniform(-1e-4, 1e-4), p),
                           lift(rng.choice([1.0, -1.0]), p)))
        assert_trace_equal(
            trace_ray(origin, direction, None, screen_z, 10),
            tracer(origin, direction, None, screen_z, 10))


def test_python_trace_env_escape(monkeypatch):
    monkeypatch.setenv("CAPSYSRED_PYTHON_TRACE", "1")
    assert make_tracer(None) is trace_ray


# ------------------------------------------------------------- full pipeline

TINY = {
    "precision": 26,
    "source": {"n_modes": 2, "n_rays": 60, "size": 3e-7, "shape": "gaussian"},
    "screen": {"nx": 9, "ny": 1},
    "lloyd": {"source": {"n_modes": 2, "n_rays": 40}, "screen": {"nx": 11}},
    "capillary": {
        "bores": [{"center": [0.0, 0.0], "radius": 6.0e-6},
                  {"center": [1.5e-5, 0.0], "radius": 6.0e-6,
                   "bend": {"radius": 0.5, "toward": [1.0, 0.0]}}],
        "source": {"n_modes": 2, "n_rays": 24},
        "screen": {"nx": 7, "ny": 7},
    },
}


def test_simulation_native_equals_python(tmp_path, monkeypatch):
    from formula.capsysred import Simulation
    sim_native = Simulation.from_dict(TINY)
    sim_native.run(str(tmp_path / "native"), stages=[2, 4, 6])
    monkeypatch.setenv("CAPSYSRED_PYTHON_TRACE", "1")
    sim_python = Simulation.from_dict(TINY)
    sim_python.run(str(tmp_path / "python"), stages=[2, 4, 6])
    native_rays = (tmp_path / "native" / "rays.jsonl").read_bytes()
    python_rays = (tmp_path / "python" / "rays.jsonl").read_bytes()
    assert native_rays == python_rays
    for stage in ("free", "lloyd", "capillary"):
        rn, rp = sim_native.results[stage], sim_python.results[stage]
        assert rn["stats"] == rp["stats"]
        for name in ("mu", "intensity", "density"):
            assert rn["maps"][name] == rp["maps"][name]

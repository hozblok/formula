"""Parity of the C++ tracer twin against the Python reference.

The C++ side runs the same algorithms over the same mp type but is not
bit-identical (plain boost arithmetic, direct double bridges). Structure is
asserted exactly — fates, event kinds, branch choices, reflection counts,
inside() booleans — and values within tolerances far below any physical
scale yet far above the twins' rounding noise.
"""

import json
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

PRECISIONS = (16, 30)          # storage 16 and 32
HIT_TOL = 1e-15                # single hit / root / event
TRACE_TOL = 1e-10              # multi-bounce chains


def num(value):
    return value if isinstance(value, Number) else Number(value)


def assert_close(py_val, c_val, tol):
    a, b = num(py_val), num(c_val)
    fa, fb = float(a), float(b)
    scale = max(abs(fa), abs(fb))
    if scale == 0.0:
        assert fa == 0.0 and fb == 0.0
        return
    assert float(abs(a - b)) <= tol * scale, f"{a!r} vs {b!r} (tol {tol})"


def same_vec(py_vec, c_vec, tol):
    assert len(py_vec) == len(c_vec)
    for a, b in zip(py_vec, c_vec):
        assert_close(a, b, tol)


def assert_hit_equal(py_hit, c_hit, tol=HIT_TOL):
    if py_hit is None or c_hit is None:
        assert py_hit is None and c_hit is None
        return
    assert_close(py_hit[0], c_hit[0], tol)
    same_vec(py_hit[1], c_hit[1], tol)
    same_vec(py_hit[2], c_hit[2], tol)


def assert_event_equal(py_ev, c_ev, tol=HIT_TOL):
    assert py_ev[0] == c_ev[0]
    if py_ev[0] == "exit":
        assert py_ev[1] is None and c_ev[1] is None
        return
    assert_close(py_ev[1], c_ev[1], tol)
    if py_ev[0] == "reflect":
        same_vec(py_ev[2], c_ev[2], tol)
        same_vec(py_ev[3], c_ev[3], tol)


def assert_trace_equal(py_tr, c_tr, tol=TRACE_TOL):
    assert py_tr.fate == c_tr.fate
    same_vec(py_tr.point, c_tr.point, tol)
    assert_close(py_tr.opl, c_tr.opl, tol)
    assert len(py_tr.reflections) == len(c_tr.reflections)
    for (pa, sa), (pb, sb) in zip(py_tr.reflections, c_tr.reflections):
        same_vec(pa, pb, tol)
        assert_close(sa, sb, tol)
    same_vec(py_tr.direction, c_tr.direction, tol)


def raw_vec(vec):
    return tuple(c._value for c in vec)


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


# ------------------------------------------------------------- root finders

def test_dk_roots_close():
    assert _formula.trace_dbg_dk_roots([]) == []
    assert _formula.trace_dbg_dk_roots([2.0]) == []
    rng = random.Random(3)
    for _ in range(300):
        cf = [1.0] + [rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-9, 3)
                      for _ in range(4)]
        py = _dk_roots(cf)
        cc = list(_formula.trace_dbg_dk_roots(cf))
        assert len(py) == len(cc)
        for a in py:  # nearest-match: sorting mispairs x±iy conjugates
            b = min(cc, key=lambda z: abs(a - z))
            cc.remove(b)
            assert abs(a - b) <= 1e-8 * max(1.0, abs(a))


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


def test_quartic_first_close():
    rng = random.Random(5)
    for p in PRECISIONS:
        cases = []
        for _ in range(60):
            cs = tuple(lift(rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-11, -4), p)
                       for _ in range(4))
            cases.append(((Number("1", p),) + cs, 10.0 ** rng.randint(-4, -1),
                          HIT_TOL))
        # Triple roots: conditioning is ~ eps^(1/3), so the twins only agree
        # to the noise basin of the root, not to working precision.
        cases.append((quartic_from_roots(2e-6, -1e-3, p), 1e-3, 1e-7))
        cases.append((quartic_from_roots(4e-5, 9e-3, p), 1e-3, 1e-7))
        for c, t_capf, tol in cases:
            py = _quartic_first(c, t_capf)
            cc = _formula.trace_dbg_quartic_first([x._value for x in c], t_capf)
            if py is None or cc is None:
                assert py is None and cc is None
            else:
                assert_close(py, cc, tol)


# ------------------------------------------------------------- walls

@pytest.mark.parametrize("kind", ["cylinder", "revolution", "polygon", "torus"])
@pytest.mark.parametrize("p", PRECISIONS)
def test_wall_hit_parity(kind, p):
    bundle = make_bundle(kind, p)
    native = compile_optic(bundle)
    assert native is not None and native.kind == "bundle"
    wall = bundle.walls[0]
    count = 12 if kind == "torus" else 30
    for origin, direction in rays(bundle, p, seed=101, count=count):
        t_exit = t_exit_of(bundle, origin, direction)
        py_hit = wall.hit(origin, direction, t_exit)
        c_hit = _formula.trace_wall_hit(native, 0, raw_vec(origin),
                                        raw_vec(direction), t_exit._value)
        assert_hit_equal(py_hit, c_hit)


@pytest.mark.parametrize("p", PRECISIONS)
def test_wall_hit_none_paths(p):
    bundle = make_bundle("cylinder", p)
    native = compile_optic(bundle)
    axis = (lift(0.0, p), lift(0.0, p), lift(0.0, p))
    dz = (lift(0.0, p), lift(0.0, p), lift(1.0, p))     # axis-parallel: A == 0
    t_exit = t_exit_of(bundle, axis, dz)
    assert bundle.walls[0].hit(axis, dz, t_exit) is None
    assert _formula.trace_wall_hit(native, 0, raw_vec(axis), raw_vec(dz),
                                   t_exit._value) is None


@pytest.mark.parametrize("kind", ["cylinder", "revolution", "polygon", "torus"])
def test_wall_inside_parity(kind):
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
def test_cylinder_equals_revolution(p):
    """Python cylinder ≡ Python revolution(a², 0, 0) ≡ native."""
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
        h_cc = _formula.trace_wall_hit(native, 0, raw_vec(origin),
                                       raw_vec(direction), t_exit._value)
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


@pytest.mark.parametrize("p", PRECISIONS)
def test_cylinder_degenerate_linear_branch(p):
    """|A| < eps with |B| >= eps takes the linear branch on all three twins
    (legacy CylinderWall returned None here and disagreed with revolution)."""
    bundle = make_bundle("cylinder", p)
    native = compile_optic(bundle)
    cyl = bundle.walls[0]
    zero = Number("0", p)
    rev = RevolutionWall("revolution", cyl.center, (cyl._a2, zero, zero))
    O = (lift(5.9999e-6, p), lift(0.0, p), lift(0.0, p))
    d = vunit((lift(1e-16, p), lift(0.0, p), lift(1.0, p)))
    t_exit = t_exit_of(bundle, O, d)
    h_cyl = cyl.hit(O, d, t_exit)
    assert h_cyl is not None      # A ~ 1e-32 < eps, B ~ 1e-21 -> linear root
    assert_hit_equal(h_cyl, rev.hit(O, d, t_exit))
    assert_hit_equal(h_cyl, _formula.trace_wall_hit(
        native, 0, raw_vec(O), raw_vec(d), t_exit._value))


R2_SHAPES = [("3.6e-11", "-4e-10", "0"),      # narrowing cone
             ("2.5e-11", "3e-10", "0"),       # widening cone
             ("3.6e-11", "2e-10", "-4e-9"),   # barrel (c2 < 0)
             ("2.5e-11", "-4e-10", "6e-9")]   # waist (c2 > 0)


@pytest.mark.parametrize("r2", R2_SHAPES)
@pytest.mark.parametrize("p", PRECISIONS)
def test_revolution_shapes_hit_parity(r2, p):
    bundle = make_revolution_bundle(r2, p)
    native = compile_optic(bundle)
    wall = bundle.walls[0]
    hits = 0
    for origin, direction in rays(bundle, p, seed=401, count=25,
                                  slope_scale=3.5):
        t_exit = t_exit_of(bundle, origin, direction)
        py_hit = wall.hit(origin, direction, t_exit)
        c_hit = _formula.trace_wall_hit(native, 0, raw_vec(origin),
                                        raw_vec(direction), t_exit._value)
        assert_hit_equal(py_hit, c_hit)
        hits += py_hit is not None
    assert hits


@pytest.mark.parametrize("p", PRECISIONS)
def test_axis_parallel_branches(p):
    """d = (0,0,1): cylinder → None both sides; narrowing cone → the exact
    linear branch (A == 0, B ≠ 0) with a matching hit."""
    N = lambda s: Number(s, p)
    d = (N("0"), N("0"), N("1"))
    cyl_bundle = make_bundle("cylinder", p)
    native_cyl = compile_optic(cyl_bundle)
    O = (N("2e-6"), N("1e-6"), N("0"))
    t_exit = t_exit_of(cyl_bundle, O, d)
    assert cyl_bundle.walls[0].hit(O, d, t_exit) is None
    assert _formula.trace_wall_hit(native_cyl, 0, raw_vec(O), raw_vec(d),
                                   t_exit._value) is None

    cone = make_revolution_bundle(("3.6e-11", "-4e-10", "0"), p)
    native_cone = compile_optic(cone)
    O = (N("5e-6"), N("0"), N("0"))
    t_exit = t_exit_of(cone, O, d)
    py_hit = cone.walls[0].hit(O, d, t_exit)
    assert py_hit is not None            # the wall narrows onto the ray
    c_hit = _formula.trace_wall_hit(native_cone, 0, raw_vec(O), raw_vec(d),
                                    t_exit._value)
    assert_hit_equal(py_hit, c_hit)
    screen_z = Number("0.051", p)
    tracer = make_tracer(cone)
    assert_trace_equal(trace_ray(O, d, cone, screen_z, 50),
                       tracer(O, d, cone, screen_z, 50))


@pytest.mark.parametrize("p", PRECISIONS)
def test_polygon_tangent_rays_parity(p):
    """Directions nearly parallel to a face (md ~ 0 skipped or huge t) still
    pick the same face and hit on both sides."""
    bundle = make_bundle("polygon", p)
    native = compile_optic(bundle)
    wall = bundle.walls[0]
    mxf, myf = wall._facesf[0]
    hits = 0
    for scale, off in ((1e-3, 0.0), (1e-3, 1e-7), (5e-4, -1e-7)):
        # tangent of face 0 (+ small normal offset), unit z slope
        d = vunit((lift(-myf * scale + mxf * off, p),
                   lift(mxf * scale + myf * off, p), lift(1.0, p)))
        O = (lift(0.0, p), lift(0.0, p), lift(0.0, p))
        t_exit = t_exit_of(bundle, O, d)
        py_hit = wall.hit(O, d, t_exit)
        c_hit = _formula.trace_wall_hit(native, 0, raw_vec(O), raw_vec(d),
                                        t_exit._value)
        assert_hit_equal(py_hit, c_hit)
        hits += py_hit is not None
    assert hits


# ------------------------------------------------------------- events, traces

@pytest.mark.parametrize("kind", ["cylinder", "revolution"])
@pytest.mark.parametrize("p", PRECISIONS)
def test_on_wall_event_chain_parity(kind, p):
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
            ev_cc = _formula.trace_next_event(native, raw_vec(O), raw_vec(d))
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


@pytest.mark.parametrize("p", PRECISIONS)
def test_bundle_next_event_parity(p):
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
            _formula.trace_next_event(native, raw_vec(origin),
                                      raw_vec(direction)))


@pytest.mark.parametrize("p", PRECISIONS)
def test_bundle_tangent_bores_reflect(p):
    """Close-packed bores (pitch = 2r): a reflection at the tangency line is on
    two walls at once; the ray must reflect off the far wall, not pass through."""
    N = lambda s: Number(s, p)
    bores = [{"kind": "cylinder", "center": (N("0"), N("0")), "radius": N("1e-5")},
             {"kind": "cylinder", "center": (N("-2e-5"), N("0")), "radius": N("1e-5")}]
    bundle = CapillaryBundle(bores, N("0"), N("0.5"))
    origin = (lift(-2e-5, p), lift(0.0, p), lift(0.0, p))    # side-bore axis
    direction = vunit((lift(6e-4, p), lift(0.0, p), lift(1.0, p)))
    py_tr = trace_ray(origin, direction, bundle, N("0.5"), 400)
    c_tr = trace_ray_native(compile_optic(bundle), origin, direction,
                            N("0.5"), 400)
    assert py_tr.fate == "screen" and len(py_tr.reflections) > 2
    assert all(-3e-5 <= float(pt[0]) <= -1e-5 + 1e-9
               for pt, _ in py_tr.reflections)
    assert_trace_equal(py_tr, c_tr)


@pytest.mark.parametrize("p", PRECISIONS)
def test_mirror_branches_parity(p):
    """One deterministic ray per Mirror branch: reflect, absorb (leading
    edge), exit beyond z1, exit moving away."""
    N = lambda s: Number(s, p)
    mirror = Mirror(N("0"), N("0.06"))
    native = compile_optic(mirror)
    O = (N("1e-5"), N("0"), N("-0.05"))
    cases = [("reflect", "-1e-4"), ("absorb", "-2.5e-4"),
             ("exit", "-1e-7"), ("exit", "1e-4")]
    for kind, slope in cases:
        d = vunit((N(slope), N("0"), N("1")))
        ev_py = mirror.next_event(O, d)
        assert ev_py[0] == kind
        assert_event_equal(ev_py,
                           _formula.trace_next_event(native, raw_vec(O),
                                                     raw_vec(d)))


@pytest.mark.parametrize("p", PRECISIONS)
def test_mirror_events_and_traces_parity(p):
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
            _formula.trace_next_event(native, raw_vec(origin),
                                      raw_vec(direction)))
        py_tr = trace_ray(origin, direction, mirror, screen_z, 10)
        assert_trace_equal(py_tr, tracer(origin, direction, mirror,
                                         screen_z, 10))


@pytest.mark.parametrize("kind", ["cylinder", "revolution", "polygon",
                                  "torus", "mixed"])
@pytest.mark.parametrize("p", PRECISIONS)
def test_trace_ray_parity(kind, p):
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


@pytest.mark.parametrize("p", PRECISIONS)
def test_absorb_paths_parity(p):
    """Web point → absorb at t=0; backward ray → absorb; ray from before z0
    into the web → pass then absorb at the entrance face."""
    bundle = make_bundle("mixed", p)
    native = compile_optic(bundle)
    tracer = make_tracer(bundle)
    screen_z = Number("0.051", p)
    web = (lift(1e-5, p), lift(0.0, p), lift(0.01, p))
    up = vunit((lift(0.0, p), lift(0.0, p), lift(1.0, p)))
    down = vunit((lift(1e-4, p), lift(0.0, p), lift(-1.0, p)))
    for O, d in ((web, up), (web, down)):
        assert_event_equal(bundle.next_event(O, d),
                           _formula.trace_next_event(native, raw_vec(O),
                                                     raw_vec(d)))
    before = (lift(1e-5, p), lift(0.0, p), lift(-0.01, p))
    for O, d in ((web, up), (web, down), (before, up)):
        py_tr = trace_ray(O, d, bundle, screen_z, 50)
        assert py_tr.fate == "absorbed"
        assert_trace_equal(py_tr, tracer(O, d, bundle, screen_z, 50))


@pytest.mark.parametrize("p", PRECISIONS)
def test_exit_at_z1_parity(p):
    """Origin exactly at z1: immediate exit, then the free leg to the screen."""
    bundle = make_bundle("cylinder", p)
    tracer = make_tracer(bundle)
    screen_z = Number("0.051", p)
    O = (lift(0.0, p), lift(0.0, p), lift(0.05, p))
    d = vunit((lift(1e-4, p), lift(0.0, p), lift(1.0, p)))
    py_tr = trace_ray(O, d, bundle, screen_z, 50)
    assert py_tr.fate == "screen" and not py_tr.reflections
    assert_trace_equal(py_tr, tracer(O, d, bundle, screen_z, 50))


@pytest.mark.parametrize("p", PRECISIONS)
def test_max_bounces_lost_parity(p):
    bundle = make_bundle("cylinder", p)
    tracer = make_tracer(bundle)
    screen_z = Number("0.051", p)
    fates = set()
    for origin, direction in rays(bundle, p, seed=907, count=6,
                                  slope_scale=6.0):
        py_tr = trace_ray(origin, direction, bundle, screen_z, 2)
        assert_trace_equal(py_tr, tracer(origin, direction, bundle,
                                         screen_z, 2))
        fates.add(py_tr.fate)
    assert "lost" in fates                  # the budget really ran out


def test_trace_free_space_parity():
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


def test_free_space_edges_parity():
    p = 30
    tracer = make_tracer(None)
    N = lambda s: Number(s, p)
    O = (N("1e-6"), N("-2e-6"), N("0"))
    cases = [
        (vunit((N("0"), N("0"), N("-1"))), N("0.05")),      # backward: lost
        (vunit((N("1e-3"), N("0"), N("1"))), N("-0.01")),   # screen behind: lost
        (vunit((N("1"), N("0"), N("0"))), N("0.05")),       # d.z == 0: lost
    ]
    for d, screen_z in cases:
        py_tr = trace_ray(O, d, None, screen_z, 10)
        assert py_tr.fate == "lost"
        assert_trace_equal(py_tr, tracer(O, d, None, screen_z, 10))


# ------------------------------------------------------------- API contract

def test_python_trace_env_escape(monkeypatch):
    monkeypatch.setenv("CAPSYSRED_PYTHON_TRACE", "1")
    assert make_tracer(None) is trace_ray
    monkeypatch.setenv("CAPSYSRED_PYTHON_TRACE", "0")
    assert make_tracer(None) is not trace_ray


def test_tracer_optic_mismatch_falls_back():
    p = 16
    b1 = make_bundle("cylinder", p)
    b2 = make_bundle("polygon", p)
    tracer = make_tracer(b1)
    screen_z = Number("0.051", p)
    (origin, direction), = list(rays(b2, p, seed=5, count=1))
    ref = trace_ray(origin, direction, b2, screen_z, 50)
    assert_trace_equal(ref, tracer(origin, direction, b2, screen_z, 50),
                       tol=0.0)            # same Python path, identical values


def test_float_screen_z_accepted():
    p = 16
    tracer = make_tracer(None)
    O = (lift(0.0, p), lift(0.0, p), lift(-0.08, p))
    d = vunit((lift(1e-4, p), lift(0.0, p), lift(1.0, p)))
    assert_trace_equal(trace_ray(O, d, None, Number(0.06, p), 10),
                       tracer(O, d, None, 0.06, 10))


def test_precision_mismatch_raises():
    bundle16 = make_bundle("cylinder", 16)
    native16 = compile_optic(bundle16)
    O30 = (lift(0.0, 30), lift(0.0, 30), lift(0.0, 30))
    d30 = vunit((lift(1e-4, 30), lift(0.0, 30), lift(1.0, 30)))
    with pytest.raises(ValueError):
        trace_ray_native(native16, O30, d30, Number("0.051", 30), 10)


def test_native_deterministic():
    p = 30
    bundle = make_bundle("torus", p)
    tracer = make_tracer(bundle)
    screen_z = Number("0.051", p)
    (origin, direction), = list(rays(bundle, p, seed=71, count=1))
    a = tracer(origin, direction, bundle, screen_z, 50)
    b = tracer(origin, direction, bundle, screen_z, 50)
    assert a.fate == b.fate
    assert [str(c) for c in a.point] == [str(c) for c in b.point]
    assert str(a.opl) == str(b.opl)
    assert len(a.reflections) == len(b.reflections)


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
    p = TINY["precision"]
    def ray_rows(sub):  # skip the v2 meta line and scene trailers
        lines = (tmp_path / sub / "rays.jsonl").read_text().splitlines()
        return [row for row in map(json.loads, lines) if "stage" in row]

    n_rows, p_rows = ray_rows("native"), ray_rows("python")
    assert n_rows and len(n_rows) == len(p_rows)
    keys = ("stage", "mode", "ray", "fate", "pixel")
    for rn, rp in zip(n_rows, p_rows):
        assert {k: rn[k] for k in keys} == {k: rp[k] for k in keys}
        assert_close(Number(rp["opl"], p), Number(rn["opl"], p), TRACE_TOL)
        assert len(rn["sins"]) == len(rp["sins"])
        for sn, sp in zip(rn["sins"], rp["sins"]):
            assert_close(Number(sp, p), Number(sn, p), TRACE_TOL)
    for stage in ("free", "lloyd", "capillary"):
        rn, rp = sim_native.results[stage], sim_python.results[stage]
        assert rn["stats"] == rp["stats"]
        for name in ("mu", "intensity", "density"):
            for row_n, row_p in zip(rn["maps"][name], rp["maps"][name]):
                assert row_n == pytest.approx(row_p, rel=1e-9, abs=1e-12)

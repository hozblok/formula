"""A reflected ray whose last event is the pass to the exit plane z1 == screen_z
lands at z1 + O(ulp); the screen step must read that residue as arrival, not loss."""

import math

import pytest

from formula.capsysred.native import compile_optic, trace_ray_native
from formula.capsysred.shared.nums import lift, vsub, vunit
from formula.capsysred.surfaces import CapillaryBundle
from formula.capsysred.trace import trace_ray

# exact 60-digit geometry of this ray: one reflection, then the exit plane
ORIGIN = ("-5.415494935424336E-7", "-1.989637853228523E-7", "-0.675")
ENTRANCE = ("-4.312725056622851e-06", "-1.9772160399759857e-05")
PRECISION = 64


def _bore(p):
    return CapillaryBundle([dict(kind="cylinder", center=(lift("0", p), lift("0", p)),
                                 radius=lift("2.4e-5", p))], lift("0", p), lift("0.23", p))


def _tracer(backend, optic):
    if backend == "python":
        return trace_ray
    native = compile_optic(optic) if optic is not None else None
    assert optic is None or native is not None

    def tracer(origin, direction, traced_optic, screen_z, max_bounces):
        assert traced_optic is optic
        return trace_ray_native(native, origin, direction, screen_z, max_bounces)

    return tracer


def _floats(vec):
    return [float(c) for c in vec]


@pytest.mark.parametrize("backend", ["python", "native"])
def test_reflected_ray_ending_on_exit_plane_reaches_screen(backend):
    p = PRECISION
    optic = _bore(p)
    origin = tuple(lift(v, p) for v in ORIGIN)
    target = (lift(ENTRANCE[0], p), lift(ENTRANCE[1], p), optic.z0)
    ray = _tracer(backend, optic)(origin, vunit(vsub(target, origin)), optic, optic.z1, 200)
    assert len(ray.reflections) == 1
    assert ray.fate == "screen"
    assert abs(float(ray.point[2]) - 0.23) < 1e-15


@pytest.mark.parametrize("backend", ["python", "native"])
def test_residue_past_the_plane_is_arrival_but_more_is_lost(backend):
    p = 32
    screen_z = lift("0.3", p)
    d = vunit((lift("2e-3", p), lift("0", p), lift("1", p)))
    free = _tracer(backend, None)
    near = (lift("0", p), lift("0", p), lift("0.3", p) + lift("1e-13", p))
    on_plane = free(near, d, None, screen_z, 10)
    assert on_plane.fate == "screen" and float(on_plane.opl) == 0.0
    assert _floats(on_plane.point) == _floats(near)
    beyond = (lift("0", p), lift("0", p), lift("0.3", p) + lift("1e-11", p))
    assert free(beyond, d, None, screen_z, 10).fate == "lost"


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("precision", [16, 32, 64])
@pytest.mark.parametrize("screen_t, expected_fate", [
    pytest.param("-1e-13", "screen", id="inside-tolerance"),
    pytest.param(repr(math.nextafter(-1e-12, 0.0)), "screen",
                 id="binary64-neighbor-inside"),
    pytest.param("-1e-12", "screen", id="decimal-boundary"),
    pytest.param(repr(math.nextafter(-1e-12, -math.inf)), "lost",
                 id="binary64-neighbor-outside"),
    pytest.param("-1e-11", "lost", id="outside-tolerance"),
])
def test_screen_tolerance_boundary(backend, precision, screen_t, expected_fate):
    zero, one = lift("0", precision), lift("1", precision)
    origin = (zero, zero, -lift(screen_t, precision))
    direction = (zero, zero, one)
    ray = _tracer(backend, None)(origin, direction, None, zero, 10)

    assert ray.fate == expected_fate
    assert ray.opl == zero
    assert ray.point == origin


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("screen_t, expected_fate", [
    pytest.param("-9.9999999999999999e-13", "screen", id="inside-tolerance"),
    pytest.param("-1.00000000000000001e-12", "lost", id="outside-tolerance"),
])
def test_screen_tolerance_retains_multiprecision(backend, precision, screen_t,
                                               expected_fate):
    # Both values become -1e-12 in binary64, but straddle the decimal tolerance.
    assert float(screen_t) == -1e-12
    zero, one = lift("0", precision), lift("1", precision)
    origin = (zero, zero, -lift(screen_t, precision))
    ray = _tracer(backend, None)(origin, (zero, zero, one), None, zero, 10)

    assert ray.fate == expected_fate
    assert ray.opl == zero
    assert ray.point == origin

"""C++ fast path for the tracer: NativeOptic twins built from the Python
optics. Results match trace.trace_ray to working precision — same algorithms
and branches, not bit-for-bit (tests/test_native_trace.py).

Wall specs are (kind, mp_values, float_values) tuples; the layouts must match
the reader in src/cpp/bindings_trace.cpp (make_wall).
"""

import os

from .. import _formula
from ..formula import Number
from .surfaces import CapillaryBundle, Mirror
from .trace import TraceResult, trace_ray
from .types import _INSIDE_TOL
from .walls.wall_cylinder import CylinderWall
from .walls.wall_funnel import FunnelWall
from .walls.wall_polygon import PolygonWall
from .walls.wall_revolution import RevolutionWall
from .walls.wall_torus import TorusWall


def _raw(number: Number):
    return number._value


def _wall_spec(wall):
    """Native spec from a wall's already-derived attributes; None = unsupported."""
    if type(wall) is CylinderWall:
        # Cylinder = revolution with r²(z) = a² (c1 = c2 = 0); the native
        # side skips the known-zero terms but keeps the same branches.
        zero = Number("0", wall.center[0].precision)
        return ("revolution",
                (_raw(wall.center[0]), _raw(wall.center[1]),
                 _raw(wall._a2), _raw(zero), _raw(zero)),
                (wall.eps, wall._cxf, wall._cyf, wall._a2f, 0.0, 0.0))
    if type(wall) is RevolutionWall:
        return ("revolution",
                (_raw(wall.center[0]), _raw(wall.center[1]),
                 _raw(wall.c0), _raw(wall.c1), _raw(wall.c2)),
                (wall.eps, wall._cxf, wall._cyf,
                 wall._c0f, wall._c1f, wall._c2f))
    if type(wall) is PolygonWall:
        mp = [_raw(wall.apothem), _raw(wall.center[0]), _raw(wall.center[1])]
        fl = [wall._af, wall._cxf, wall._cyf]
        for (mx, my), (mxf, myf) in zip(wall.faces, wall._facesf):
            mp += [_raw(mx), _raw(my)]
            fl += [mxf, myf]
        return ("polygon", tuple(mp), tuple(fl))
    if type(wall) is TorusWall:
        return ("torus",
                tuple(_raw(v) for v in (*wall.C, *wall.nhat,
                                        wall.R, wall.K, wall.fourR2)),
                (*wall._Cf, *wall._nf, wall._Rf, wall._in2))
    if type(wall) is FunnelWall:
        return ("funnel",
                tuple(_raw(v) for v in (wall.center[0], wall.center[1],
                                        wall.r0, wall.r02, wall.ag, wall.bg,
                                        wall.af, wall.bf, wall.z0)),
                (wall._cxf, wall._cyf, wall._r0f, wall._agf, wall._bgf,
                 wall._aff, wall._bff, wall._z0f, 1.0 + _INSIDE_TOL))
    return None


def compile_optic(optic):
    """NativeOptic twin of a Python optic; None when a wall is unsupported."""
    if type(optic) is Mirror:
        return _formula.trace_make_mirror(_raw(optic.z0), _raw(optic.z1),
                                          optic._z0f, optic._z1f)
    if type(optic) is CapillaryBundle:
        specs = [_wall_spec(wall) for wall in optic.walls]
        if any(spec is None for spec in specs):
            return None
        return _formula.trace_make_bundle(_raw(optic.z0), _raw(optic.z1),
                                          optic._z0f, optic._z1f, specs)
    return None


def trace_ray_native(native, origin, direction, screen_z, max_bounces):
    """trace_ray twin on a compiled optic (native=None traces free space)."""
    p = origin[0].precision
    if not isinstance(screen_z, Number):
        screen_z = Number(screen_z, p)
    fate, point, opl, refl, direction = _formula.trace_ray_native(
        native, tuple(_raw(c) for c in origin),
        tuple(_raw(c) for c in direction), _raw(screen_z), max_bounces)

    def num(value):
        return Number._wrap(value, p, False)

    return TraceResult(fate, tuple(num(c) for c in point), num(opl),
                       [(tuple(num(c) for c in pt), num(s)) for pt, s in refl],
                       tuple(num(c) for c in direction))


def make_tracer(optic):
    """trace_ray-signature callable; the C++ twin when the optic supports it.

    An optic other than the compiled one falls back to the Python reference;
    CAPSYSRED_PYTHON_TRACE=1 forces it globally.
    """
    if os.environ.get("CAPSYSRED_PYTHON_TRACE", "0") not in ("", "0"):
        return trace_ray
    native = None
    if optic is not None:
        native = compile_optic(optic)
        if native is None:
            return trace_ray

    def tracer(origin, direction, traced_optic, screen_z, max_bounces):
        if traced_optic is not optic:
            return trace_ray(origin, direction, traced_optic, screen_z,
                             max_bounces)
        return trace_ray_native(native, origin, direction, screen_z,
                                max_bounces)
    return tracer


def make_beamlet_grid(nx, ny, x0, y0, ex, ey, kms, zrs, zrs_t, ns):
    """Native stage-11 deposit grids (propagate + window loop per ray); None
    when the .so predates BeamletGrid — the Python path is the reference and
    the fallback."""
    cls = getattr(_formula, "BeamletGrid", None)
    if cls is None or not hasattr(cls, "jackknife"):   # stale .so: old API
        return None
    try:
        return cls(nx, ny, x0, y0, ex, ey, kms, zrs, zrs_t, ns)
    except TypeError:      # stale .so without the anisotropic launch
        return None

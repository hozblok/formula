"""FunnelWall closed form: judge cross-check and the cylinder degeneracy."""

from formula.formula import Number
from formula.capsysred.shared.nums import vunit
from formula.capsysred.surfaces import engine_hit_t
from formula.capsysred.walls.wall_cylinder import CylinderWall
from formula.capsysred.walls.wall_funnel import FunnelWall

P = 32
AG, BG = 1.0 / 0.03, -1.0 / (2 * 0.03 * 0.23)   # OE 20:3975 collimating conditions, SI


def _n(v):
    return Number(repr(float(v)), P)


def _funnel(cx, f_coeffs, g_coeffs=None):
    g = g_coeffs if g_coeffs is not None else (_n(AG), _n(BG))
    return FunnelWall((_n(cx), _n(0.0)), _n(6e-6), g, f_coeffs, _n(0.0))


def test_funnel_hit_matches_engine():
    w = _funnel(12e-6, (_n(AG), _n(BG)))
    for k in range(5):
        O = (_n(12e-6 + (k - 2) * 2e-6), _n(1e-6), _n(0.0))
        d = vunit((_n(3e-4 + k * 2e-4), _n(1e-4), _n(1.0)))
        hit = w.hit(O, d, _n(0.3))
        assert hit is not None
        t_engine = engine_hit_t(w.expr_um, O, d, 0.3)
        assert t_engine is not None
        assert abs(float(hit[0] - t_engine)) / float(hit[0]) < 1e-25


def test_funnel_cylinder_degeneracy():
    zero = _n(0.0)
    # f = g = 1: an exact straight cylinder through the quartic padding path
    w = _funnel(0.0, (zero, zero), g_coeffs=(zero, zero))
    cyl = CylinderWall((zero, _n(0.0)), _n(6e-6))
    for k in range(4):
        O = (_n((k - 1.5) * 2e-6), _n(0.5e-6), _n(0.0))
        d = vunit((_n(2e-4 + k * 1e-4), _n(-1e-4), _n(1.0)))
        hf = w.hit(O, d, _n(0.3))
        hc = cyl.hit(O, d, _n(0.3))
        assert (hf is None) == (hc is None)
        if hf is not None:
            assert abs(float(hf[0] - hc[0])) / float(hf[0]) < 1e-28


def test_funnel_inside_brackets_wall():
    w = _funnel(12e-6, (_n(AG), _n(BG)))
    O = (_n(12e-6), _n(0.0), _n(0.0))
    d = vunit((_n(5e-4), _n(0.0), _n(1.0)))
    t, point, n = w.hit(O, d, _n(0.3))
    xf, yf, zf = (float(c) for c in point)
    nx, ny, nz = (float(c) for c in n)
    eps = 1e-9
    assert w.inside(xf - eps * nx, yf - eps * ny, zf - eps * nz)
    assert not w.inside(xf + eps * nx, yf + eps * ny, zf + eps * nz)

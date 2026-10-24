"""Stage 11 beamlet.self_pairs: the ray self-pair diagonal and its
U-statistic subtraction (native grid, Python fallback, config, run)."""

import math
import random
from types import SimpleNamespace

import pytest

from formula.capsysred import Simulation
from formula.capsysred.native import make_beamlet_grid
from formula.capsysred.screen import ScreenGrid
from formula.capsysred.shared.types import RayRecord
from formula.capsysred.spectrum import spectral_lines
from formula.capsysred.stages.beamlet import BeamletField, _stage11_contract
from formula.formula import Number

from tests.test_capsysred import TINY, _record


def _lines(n=3):
    return spectral_lines({"mode": "gaussian", "rel_fwhm": 1.0e-3,
                           "n_lines": n, "n_sigma": 2.0}, Number("8.0", 32))


def _grid_ok():
    g = make_beamlet_grid(1, 1, 0.0, 0.0, 1.0, 1.0, [1.0], [1.0], [1.0], 3.0)
    return g is not None and hasattr(g, "set_self_pairs")


def _rec(i, scr, x, y, dx, dy, opl, refl=()):
    return RayRecord(0, i, "screen", scr.pixel((x, y)), (x, y, float(scr.z)),
                     (dx, dy, 1.0), opl,
                     tuple(1.0e-3 * (j + 1) for j in range(len(refl))), refl)


def _maps_diff(a, b, key, scale=1.0):
    return max(abs(u - v) for ra, rb in zip(a[key], b[key])
               for u, v in zip(ra, rb)) / scale


def test_subtract_native_matches_python():
    # the C++ self-pair diagonal mirrors the Python dicts: same records,
    # native on/off -> the same U-statistic maps to float64 roundoff
    if not _grid_ok():
        pytest.skip("BeamletGrid without self pairs in the built .so")
    lines = _lines()
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=21, ny=5, center=[0.0, 0.0],
                                     edge_x=1.2e-5, edge_y=6.0e-6))
    rays = [
        (0.0, 0.0, 0.0, 0.0, 0.06, ()),
        (2.0e-6, 1.0e-6, 5.0e-5, -3.0e-5, 0.0612, ((1.0e-6, 0.0, 0.03),)),
        (-3.0e-6, -2.0e-6, -1.0e-4, 2.0e-5, 0.0605,
         ((2.0e-6, 1.0e-6, 0.02), (-1.0e-6, 5.0e-7, 0.045))),
        (9.0e-6, 2.0e-6, 2.0e-4, 1.0e-5, 0.0601, ()),   # ref outside its window
    ]
    maps = []
    for use_native in (True, False):
        field = BeamletField(lines, scr, 52, 5.0e-7, 3.0, None,
                             use_native=use_native, self_pairs="subtract")
        assert (field.native is not None) == use_native and field.subtract
        for mode in range(2):
            field.new_mode()
            for i, (x, y, dx, dy, opl, refl) in enumerate(rays):
                field.add_ray(_rec(i, scr, x, y, dx, dy, opl, refl),
                              [(1.0 + 0.5j) * (1 + 0.1 * mode)] * len(lines))
            field.fold_mode(n_rays=len(rays))
        maps.append(field.finalize(21, 5))
    nat, ref = maps
    imax = max(abs(v) for r in ref["intensity"] for v in r)
    assert imax > 0.0
    assert _maps_diff(nat, ref, "mu") <= 1e-12
    assert _maps_diff(nat, ref, "intensity", imax) <= 1e-12
    assert _maps_diff(nat, ref, "mu_err") <= 1e-6
    assert nat["dubious"] == ref["dubious"]
    # subtract differs from keep: the diagonal is not zero
    keep = BeamletField(lines, scr, 52, 5.0e-7, 3.0, None, use_native=False)
    for mode in range(2):
        keep.new_mode()
        for i, (x, y, dx, dy, opl, refl) in enumerate(rays):
            keep.add_ray(_rec(i, scr, x, y, dx, dy, opl, refl),
                         [(1.0 + 0.5j) * (1 + 0.1 * mode)] * len(lines))
        keep.fold_mode(n_rays=len(rays))
    kmaps = keep.finalize(21, 5)
    assert _maps_diff(kmaps, ref, "intensity", imax) > 1e-3


def test_keep_default_unchanged_by_n_rays():
    # self_pairs keep ignores n_rays: fold_mode(n) == fold_mode()
    if not _grid_ok():
        pytest.skip("BeamletGrid without self pairs in the built .so")
    lines = _lines(2)
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=11, ny=3, center=[0.0, 0.0],
                                     edge_x=6.0e-6, edge_y=3.0e-6))
    out = []
    for n in (0, 7):
        field = BeamletField(lines, scr, 16, 5.0e-7, 3.0, None)
        assert not field.subtract
        field.new_mode()
        field.add_ray(_rec(0, scr, 0.0, 0.0, 0.0, 0.0, 0.06), [1.0] * 2)
        field.add_ray(_rec(1, scr, 1.0e-6, 0.0, 2.0e-5, 0.0, 0.06), [1.0] * 2)
        field.fold_mode(n_rays=n)
        out.append(field.finalize(11, 3))
    assert out[0]["intensity"] == out[1]["intensity"]
    assert out[0]["mu"] == out[1]["mu"]


def test_subtract_removes_incoherent_floor():
    # n rays of random phase at one point: the true mean intensity of the
    # ensemble is 0 (|mean e|^2 = 0); keep returns the self-pair floor
    # n|e|^2, subtract averages to 0 over many modes
    if not _grid_ok():
        pytest.skip("BeamletGrid without self pairs in the built .so")
    lines = _lines(1)
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=11, ny=1, center=[0.0, 0.0],
                                     edge_x=6.0e-6, edge_y=1.0e-6))
    rng = random.Random(7)
    n, modes = 16, 400
    phases = [[rng.uniform(0.0, 2.0 * math.pi) for _ in range(n)]
              for _ in range(modes)]
    i_at = {}
    for sp in ("keep", "subtract"):
        field = BeamletField(lines, scr, 5, 5.0e-7, 3.0, None, self_pairs=sp)
        for mode in range(modes):
            field.new_mode()
            for i in range(n):
                field.add_ray(_rec(i, scr, 0.0, 0.0, 0.0, 0.0, 0.06),
                              [complex(math.cos(phases[mode][i]),
                                       math.sin(phases[mode][i]))])
            field.fold_mode(n_rays=n)
        i_at[sp] = field.finalize(11, 1)["intensity"][0][5] / modes
    # per-mode sigma of the U-statistic ~ n|e|^2; the mean over `modes`
    # shrinks by sqrt(modes): |mean| < 4 sigma / sqrt(modes)
    assert i_at["keep"] > 0.0
    assert abs(i_at["subtract"]) < 4.0 * i_at["keep"] / math.sqrt(modes)


def test_subtract_point_source_stays_coherent():
    # one point source, a fan of rays: the coherent field survives the
    # subtraction (mu ~ 1 in the lit center, I_U close to I)
    if not _grid_ok():
        pytest.skip("BeamletGrid without self pairs in the built .so")
    lines = _lines(1)
    nx, n = 81, 60
    scr = ScreenGrid(SimpleNamespace(z=0.1, nx=nx, ny=1, center=[0.0, 0.0],
                                     edge_x=4.0e-6, edge_y=1.0e-6))
    maps = {}
    for sp in ("keep", "subtract"):
        field = BeamletField(lines, scr, nx // 2, 2.5e-7, 3.0, None, self_pairs=sp)
        field.new_mode()
        for i in range(n):
            # a fan from the origin at z = 0 to x in [-3, 3] um on the screen
            x = -3.0e-6 + 6.0e-6 * i / (n - 1)
            dx = x / 0.1
            field.add_ray(_rec(i, scr, x, 0.0, dx, 0.0, math.hypot(0.1, x)), [1.0])
        field.fold_mode(n_rays=n)
        maps[sp] = field.finalize(nx, 1)
    ik, iu = maps["keep"]["intensity"][0], maps["subtract"]["intensity"][0]
    imax = max(ik)
    lit = [p for p in range(nx) if ik[p] > 0.3 * imax]
    assert lit
    for p in lit:
        assert abs(iu[p] - ik[p]) < 5.0 / n * ik[p]
        assert maps["subtract"]["mu"][0][p] > 0.99


def test_config_self_pairs():
    cfg = Simulation.from_dict(TINY).cfg
    assert cfg.beamlet_self_pairs == "keep"
    cfg.validate_beamlet()
    sub = dict(TINY, beamlet={"self_pairs": "subtract"})
    sim = Simulation.from_dict(sub)
    assert sim.cfg.beamlet_self_pairs == "subtract"
    planes = [{"z": 0.1, "nx": 3, "ny": 1, "center": [0.0, 0.0],
               "edge_x": 1e-6, "edge_y": 1e-6, "reference": None}]
    assert _stage11_contract(sim, "free", planes, 1)["beamlet"]["self_pairs"] == "subtract"
    bad = Simulation.from_dict(dict(TINY, beamlet={"self_pairs": "drop"}))
    with pytest.raises(ValueError, match="self_pairs"):
        bad.cfg.validate_beamlet()


def test_stage11_subtract_runs(tmp_path):
    # the shared path under subtract: finite maps, the point source still
    # fully coherent where lit, the report names the U-statistic
    if not _grid_ok():
        pytest.skip("BeamletGrid without self pairs in the built .so")
    sim = Simulation.from_dict(dict(TINY, beamlet={"self_pairs": "subtract"}))
    _record(sim, str(tmp_path))
    result = sim.run(str(tmp_path), stages=[11])
    assert "mu-beamlet.jsonl" in result["files"]
    maps = sim.results["beamlet:free"]["maps"]
    imax = max(maps["intensity"][0])
    lit = [m for m, i in zip(maps["mu"][0], maps["intensity"][0])
           if i > 0.3 * imax]
    assert lit and min(lit) > 0.99
    for key in ("beamlet:free", "beamlet:capillary"):
        for row in sim.results[key]["maps"]["intensity"]:
            assert all(math.isfinite(v) for v in row)
    assert any("U-statistic" in line for line in sim.report)


def test_subtract_keeps_the_self_cross_term_at_a_null_reference():
    # mode 1: one ray twice with opposite amplitude -> the summed field is exactly 0
    # everywhere, its self-pair diagonal is not: W must carry -sum e_i conj(e_i,ref)
    # (|mu| cannot show it: a negative-weight component pins the ratio at >= 1).
    # Native totals = Python dicts, and mode 1 does change W.
    if not _grid_ok():
        pytest.skip("BeamletGrid without self pairs in the built .so")
    import numpy as np
    lines = _lines(1)
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=21, ny=5, center=[0.0, 0.0],
                                     edge_x=1.2e-5, edge_y=6.0e-6))
    a_ray = (0.0, 0.0, 0.0, 0.0, 0.06)
    b_ray = (2.0e-6, 1.0e-6, 5.0e-5, -3.0e-5, 0.0612)

    def totals(use_native, with_null_mode):
        field = BeamletField(lines, scr, 52, 5.0e-7, 3.0, None,
                             use_native=use_native, self_pairs="subtract")
        if with_null_mode:
            field.new_mode()
            field.add_ray(_rec(0, scr, *b_ray), [1.0 + 0.5j])
            field.add_ray(_rec(1, scr, *b_ray), [-1.0 - 0.5j])
            field.fold_mode(n_rays=2)
        field.new_mode()
        for i in range(3):
            field.add_ray(_rec(i, scr, *a_ray), [1.0 + 0.5j])
        field.fold_mode(n_rays=3)
        npix = scr.nx * scr.ny
        if use_native:
            w_b, i_b = field.native.totals()
            return np.frombuffer(w_b, np.complex128), np.frombuffer(i_b, np.float64)
        w, i = np.zeros(npix, complex), np.zeros(npix)
        for pix, v in field.W.items():
            w[pix] = v
        for pix, v in field.I.items():
            i[pix] = v
        return w, i

    w_nat, i_nat = totals(True, True)
    w_py, i_py = totals(False, True)
    w_two, _ = totals(False, False)
    scale = abs(w_py).max()
    assert abs(w_two - w_py).max() > 1e-3 * scale      # the null mode is not inert
    assert abs(w_nat - w_py).max() <= 1e-12 * scale
    assert abs(i_nat - i_py).max() <= 1e-12 * abs(i_py).max()


def test_negative_u_intensity_is_dubious_in_every_path():
    # subtract can leave I < 0 at a lit pixel: no estimate, flagged dubious (never a
    # confident mu = 0); never-deposited pixels stay unflagged. Python, retained
    # native jackknife and the production tile agree.
    if not _grid_ok():
        pytest.skip("BeamletGrid without self pairs in the built .so")
    import numpy as np
    from formula.capsysred.native import jackknife_tile
    lines = _lines(1)
    scr = ScreenGrid(SimpleNamespace(z=0.06, nx=21, ny=5, center=[0.0, 0.0],
                                     edge_x=1.2e-5, edge_y=6.0e-6))
    a_ray = (0.0, 0.0, 0.0, 0.0, 0.06)
    b_ray = (2.0e-6, 1.0e-6, 5.0e-5, -3.0e-5, 0.0612)
    maps = []
    for use_native in (True, False):
        field = BeamletField(lines, scr, 52, 5.0e-7, 3.0, None,
                             use_native=use_native, self_pairs="subtract")
        field.new_mode()
        field.add_ray(_rec(0, scr, *b_ray), [1.0 + 0.5j])
        field.add_ray(_rec(1, scr, *b_ray), [-1.0 - 0.5j])
        field.fold_mode(n_rays=2)
        field.new_mode()
        for i in range(3):
            field.add_ray(_rec(i, scr, *a_ray), [1.0 + 0.5j])
        field.fold_mode(n_rays=3)
        maps.append(field.finalize(21, 5))
    for m in maps:
        I, mu, dub = (np.array(m[k]) for k in ("intensity", "mu", "dubious"))
        assert m["i_ref"] > 0.0 and (I < 0).sum() > 0
        assert (dub[I < 0] == 1).all() and (mu[I < 0] == 0).all()
        assert (dub[I == 0] == 0).all()
    assert maps[0]["dubious"] == maps[1]["dubious"]
    # the production tile: I = [1, -1, 0, 2], reference lit -> pixel 1 dubious only
    w = np.array([0.5, 0.5, 0.0, 1.0], complex).tobytes()
    i_tot = np.array([1.0, -1.0, 0.0, 2.0]).tobytes()
    rows_w = np.zeros(2 * 2 * 4, np.float32).tobytes()
    rows_i = np.zeros(2 * 4, np.float32).tobytes()
    _, _, dub, _ = jackknife_tile(w, i_tot, 1.0, rows_w, rows_i, [0.0, 0.0], 1e-6, 0.0)
    assert list(dub) == [0, 1, 0, 0]
    _, _, dub, _ = jackknife_tile(w, i_tot, -1.0, rows_w, rows_i, [0.0, 0.0], 1e-6, 0.0)
    assert list(dub) == [1, 1, 0, 1]

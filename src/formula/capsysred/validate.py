"""Stage 9: cross-validation of the ray-hit methods on one random ray set.

Rays are emitted exactly as in the capillary stage; the first wall hit of
each ray is computed three ways — Python closed form, the C++ twin, and the
RaySurface engine subdivision backend (the only grazing-safe one, see
doc/2026-07-10-hit-method-backends.ru.md) — and every hit parameter t is
compared against the Python reference.
"""

import random
import time

from ..intersect import RaySurface
from .native import compile_optic, trace_ray_native
from .nums import lift, vadd, vscale
from .progress import Progress
from .rays import _SCENE_SEED_STRIDE, SceneSeed
from .source import Source
from .surfaces import CapillaryBundle
from .types import _EPS_T, _M_TO_UM, _ONWALL_TOL, _TCAP_TOL

METHOD_LABELS = {"cpp": "C++ analytic", "subdivision": "implicit subdivision"}


def full_expr_um(wall):
    """Whole-surface F=0 in µm; a polygon is the product of its face planes."""
    if wall.kind != "polygon":
        return wall.expr_um
    um = lift(_M_TO_UM, wall.center[0].precision)
    return "*".join(
        f"((x-({wall.center[0] * um}))*({mx})"
        f"+(y-({wall.center[1] * um}))*({my})-({wall.apothem * um}))"
        for mx, my in wall.faces)


def _engine_t(rs, scale, O, d, t_exit, method):
    """First-hit t via a prebuilt RaySurface (ImplicitWall.hit's root path)."""
    eps_um = _EPS_T * _M_TO_UM
    ts = rs.intersect(tuple(c * scale for c in O), d,
                      t_max=float(t_exit) * _M_TO_UM * (1.0 + _TCAP_TOL),
                      t_min=eps_um, method=method)
    ts = [t for t in ts if float(t) > _ONWALL_TOL * eps_um]
    if not ts:
        return None
    t = ts[0] / scale
    return None if t_exit <= t else t


def run_validate_stage(sim, n_rays: int):
    """Emit n_rays into the capillary scene; compare first-hit t per method."""
    cfg = sim.cfg
    cap = cfg.capillary
    p = cfg.precision
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1, cfg.engine_method)
    native = compile_optic(bundle)
    scale = lift(_M_TO_UM, p)
    engines = {id(w): RaySurface(full_expr_um(w), p)
               for w in bundle.walls if w.expr_um is not None}
    rng = random.Random(cfg.seed * _SCENE_SEED_STRIDE + SceneSeed.VALIDATE)
    source = Source(cap.source, rng)
    aim = sim._aim_capillary(source, None, rng)
    methods = tuple(m for m in METHOD_LABELS if m != "cpp" or native is not None)
    per = {m: {"n": 0, "missing": 0, "extra": 0, "max_rel": 0.0,
               "sum_sq": 0.0, "rels": [], "seconds": 0.0} for m in methods}
    stats = {"rays": n_rays, "skipped": 0, "hits": 0, "passes": 0,
             "py_seconds": 0.0}
    rows = []
    progress = Progress("9 hit validation", n_rays)
    t_start = time.time()
    for i in range(n_rays):
        O = source.mode_origin()
        d = aim(O)
        O = vadd(O, vscale(d, (cap.z0 - O[2]) / d[2]))  # to the entrance plane
        wall = bundle._locate(O, d)
        if wall is None or id(wall) not in engines:
            stats["skipped"] += 1   # entrance web, or `surface:` bore (engine-only)
            rows.append({"ray": i, "fate": "skipped"})
            progress.step()
            continue
        t_exit = (cap.z1 - O[2]) / d[2]
        t0 = time.time()
        hit = wall.hit(O, d, t_exit)
        stats["py_seconds"] += time.time() - t0
        t_py = hit[0] if hit is not None and hit[0] < t_exit else None
        stats["hits" if t_py is not None else "passes"] += 1
        ts = {}
        if native is not None:
            t0 = time.time()
            tr = trace_ray_native(native, O, d, cap.screen.z, 1)
            per["cpp"]["seconds"] += time.time() - t0
            ts["cpp"] = ((tr.reflections[0][0][2] - O[2]) / d[2]
                         if tr.reflections else None)
        t0 = time.time()
        ts["subdivision"] = _engine_t(engines[id(wall)], scale, O, d, t_exit,
                                      "subdivision")
        per["subdivision"]["seconds"] += time.time() - t0
        row = {"ray": i, "fate": "hit" if t_py is not None else "pass",
               "kind": wall.kind,
               "t_python": None if t_py is None else str(t_py)}
        for m, t_m in ts.items():
            st = per[m]
            row[f"t_{m}"] = None if t_m is None else str(t_m)
            if t_py is None or t_m is None:
                if t_m is not None:
                    st["extra"] += 1
                elif t_py is not None:
                    st["missing"] += 1
                continue
            rel = abs(float((t_m - t_py) / t_py))
            st["n"] += 1
            st["sum_sq"] += rel * rel
            st["max_rel"] = max(st["max_rel"], rel)
            st["rels"].append(rel)
        rows.append(row)
        progress.step()
    progress.finish(f"wall hits {stats['hits']:,}")
    for st in per.values():
        st["rms"] = (st["sum_sq"] / st["n"]) ** 0.5 if st["n"] else 0.0
    return {"stats": stats, "per": per, "rows": rows,
            "native": native is not None, "seconds": time.time() - t_start}

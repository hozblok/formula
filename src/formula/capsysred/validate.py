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
from .types import _EPS_T, _ONWALL_TOL, _TCAP_TOL, HitMethod
from .units import m_to_um

METHOD_LABELS = {
    HitMethod.PYTHON_CLOSED_FORM: "Python closed form (reference)",
    HitMethod.CPP_CLOSED_FORM: "C++ closed form",
    HitMethod.STURM: "Sturm isolation",
    HitMethod.SUBDIVISION: "implicit subdivision",
    HitMethod.CHEBYSHEV: "Chebyshev interpolant (experimental)",
    HitMethod.SAMPLING: "grid sampling (experimental)",
}


def full_expr_um(wall):
    """Whole-surface F=0 in µm; a polygon is the product of its face planes."""
    if wall.kind != "polygon":
        return wall.expr_um
    um = lift(m_to_um(1), wall.center[0].precision)
    return "*".join(
        f"((x-({wall.center[0] * um}))*({mx})"
        f"+(y-({wall.center[1] * um}))*({my})-({wall.apothem * um}))"
        for mx, my in wall.faces)


def _engine_t(rs, scale, O, d, t_exit, method):
    """First-hit t via a prebuilt RaySurface (ImplicitWall.hit's root path)."""
    eps_um = m_to_um(_EPS_T)
    ts = rs.intersect(tuple(c * scale for c in O), d,
                      t_max=m_to_um(t_exit) * (1.0 + _TCAP_TOL),
                      t_min=eps_um, method=method)
    ts = [t for t in ts if float(t) > _ONWALL_TOL * eps_um]
    if not ts:
        return None
    t = ts[0] / scale
    return None if t_exit <= t else t


def run_validate_stage(sim, n_rays: int):
    """Emit n_rays into the capillary scene; compare every validate.methods
    first-hit t against the validate.reference one."""
    cfg = sim.cfg
    cap = cfg.capillary
    p = cfg.precision
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1, cfg.engine_method)
    native = compile_optic(bundle)
    scale = lift(m_to_um(1), p)
    engines = {id(w): w.rs if w.kind == "implicit"
               else RaySurface(full_expr_um(w), p)
               for w in bundle.walls
               if w.kind == "implicit" or w.expr_um is not None}
    ref = cfg.validate_reference
    has_implicit = any(w.kind == "implicit" for w in bundle.walls)
    if ref is HitMethod.PYTHON_CLOSED_FORM and has_implicit:
        raise ValueError(
            "validate: on `surface:` bores the python closed form IS the "
            "engine — set validate.reference to an independent method "
            "(e.g. sturm)")
    if ref is HitMethod.CPP_CLOSED_FORM and native is None:
        raise ValueError("validate: cpp-closed-form reference has no native "
                         "twin for this wall kind")
    rng = random.Random(cfg.seed * _SCENE_SEED_STRIDE + SceneSeed.VALIDATE)
    source = Source(cap.source, rng)
    aim = sim._aim_capillary(source, None, rng)
    methods = tuple(m for m in cfg.validate_methods
                    if (m is not HitMethod.CPP_CLOSED_FORM
                        or native is not None)
                    and (m is not HitMethod.PYTHON_CLOSED_FORM
                         or not has_implicit))
    per = {m: {"n": 0, "missing": 0, "extra": 0, "max_rel": 0.0,
               "sum_sq": 0.0, "rels": [], "seconds": 0.0} for m in methods}
    stats = {"rays": n_rays, "skipped": 0, "hits": 0, "passes": 0,
             "reference": str(ref), "ref_seconds": 0.0}
    rows = []
    progress = Progress("9 hit validation", n_rays)
    t_start = time.time()
    for i in range(n_rays):
        O = source.mode_origin()
        d = aim(O)
        O = vadd(O, vscale(d, (cap.z0 - O[2]) / d[2]))  # to the entrance plane
        wall = bundle._locate(O, d)
        if wall is None or id(wall) not in engines:
            stats["skipped"] += 1   # entrance web
            rows.append({"ray": i, "fate": "skipped"})
            progress.step()
            continue
        t_exit = (cap.z1 - O[2]) / d[2]
        t0 = time.time()
        if ref is HitMethod.PYTHON_CLOSED_FORM:
            hit = wall.hit(O, d, t_exit)
            t_ref = hit[0] if hit is not None and hit[0] < t_exit else None
        elif ref is HitMethod.CPP_CLOSED_FORM:
            tr = trace_ray_native(native, O, d, cap.screen.z, 1)
            t_ref = ((tr.reflections[0][0][2] - O[2]) / d[2]
                     if tr.reflections else None)
        else:
            t_ref = _engine_t(engines[id(wall)], scale, O, d, t_exit, ref)
        stats["ref_seconds"] += time.time() - t0
        stats["hits" if t_ref is not None else "passes"] += 1
        ts = {}
        for m in methods:
            t0 = time.time()
            if m is HitMethod.CPP_CLOSED_FORM:
                tr = trace_ray_native(native, O, d, cap.screen.z, 1)
                t_m = ((tr.reflections[0][0][2] - O[2]) / d[2]
                       if tr.reflections else None)
            elif m is HitMethod.PYTHON_CLOSED_FORM:
                hit = wall.hit(O, d, t_exit)
                t_m = hit[0] if hit is not None and hit[0] < t_exit else None
            else:
                t_m = _engine_t(engines[id(wall)], scale, O, d, t_exit, m)
            per[m]["seconds"] += time.time() - t0
            ts[m] = t_m
        row = {"ray": i, "fate": "hit" if t_ref is not None else "pass",
               "kind": wall.kind,
               "t_reference": None if t_ref is None else str(t_ref)}
        if ref is HitMethod.PYTHON_CLOSED_FORM:
            row["t_python"] = row["t_reference"]  # legacy readers
        for m, t_m in ts.items():
            st = per[m]
            row[f"t_{m}"] = None if t_m is None else str(t_m)
            if t_ref is None or t_m is None:
                if t_m is not None:
                    st["extra"] += 1
                elif t_ref is not None:
                    st["missing"] += 1
                continue
            rel = abs(float((t_m - t_ref) / t_ref))
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

"""Trace scenes straight into a v3 archive, or top them up.

    python -m formula.capsysred.trace_v3 config.yaml --archive ARCHIVE_DIR
        [--jobs J] [--level 6]

Scenes and budgets come from the yaml alone.  Per configured scene: absent
from the archive -> trace rays 0..n_rays-1 (one section per mode — no
shards, no merge, resumable per section); recorded below the yaml budget
-> top up; at the budget -> no-op; above it -> error.  A missing archive
is created (fingerprint from the yaml, ``rng: lattice-v1``).  On legacy
sequential-v2 archives (converted Z-26) scenes the archive does not hold
are skipped with a log line — only capillary can top up there.

lattice-v1: mode m of a scene draws its origin, then a fixed number of
draws per ray (capillary aim 3, free aim 2) from
``stream_rng(seed, <scene>, m)``, so ``--jobs``, one-shot and top-up give
identical rays (as did the retired sequential v2 writer).  Legacy sequential-v2
archives (converted Z-26) top up their capillary scene from the tail
substream ``Random((seed*STRIDE + CAPILLARY_TOPUP) * 2**32 + m)`` at
position 3*r.  Sections publish tmp -> rename; the index is replaced
atomically once every mode is done, so an interrupted run leaves the
archive at its old budget.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import sys
import time

from ..formula import Number
from . import rays, rays_v3
from .shared.common import tlog as _log
from .native import make_tracer
from .screen import ScreenGrid
from .source import Source
from .surfaces import CapillaryBundle
from .shared.types import ray_record

# aim draws per ray, asserted by tests (rng skip counts)
SCENES = {"capillary": {"tag": rays.SceneSeed.CAPILLARY, "draws": 3},
          "free": {"tag": rays.SceneSeed.FREE, "draws": 2}}
AIM_DRAWS = SCENES["capillary"]["draws"]
LEGACY_SCHEME = "per-mode-substream-v1"
MAX_JOBS = 1024


def tail_rng(seed: int, mode: int) -> random.Random:
    return random.Random(
        (seed * rays._SCENE_SEED_STRIDE + rays.SceneSeed.CAPILLARY_TOPUP) * 2**32 + mode)


def _scene_parts(sim, scene: str):
    """(source cfg, screen grid, optic, aim factory) of one scene."""
    cfg = sim.cfg
    if scene == "capillary":
        cap = cfg.capillary
        if cap is None:
            raise ValueError("config has no capillary scene")
        return cap.source, ScreenGrid(cap.screen), CapillaryBundle(cap.bores, cap.z0, cap.z1), \
            sim._aim_capillary
    if scene == "free":
        if cfg.free_source is None:
            raise ValueError("config has no free scene")
        return cfg.free_source, ScreenGrid(cfg.free_screen), None, sim._aim_free
    raise ValueError(f"unknown scene {scene!r}")


_W = {}


def _init_worker(config_path):
    from .simulation import Simulation
    _W.clear()
    _W["sim"] = Simulation.from_yaml(config_path)


def _worker_scene(scene: str):
    if scene not in _W:
        src, screen, optic, aim_factory = _scene_parts(_W["sim"], scene)
        _W[scene] = (src, screen, optic, aim_factory, make_tracer(optic))
    return _W[scene]


def _trace_section(archive, scene, mode, r0, r1, origin_str, lean, level, lattice):
    sim = _W["sim"]
    src, screen, optic, aim_factory, tracer = _worker_scene(scene)
    cfg = sim.cfg
    draws = SCENES[scene]["draws"]
    name = rays_v3.section_name(scene, mode, r0, r1)
    final = os.path.join(archive, rays_v3.MODES_DIR, name)
    if os.path.exists(final):
        size, digest = rays_v3._hash_file(final)
        entry = rays_v3.Section(scene, mode, r0, r1, r1 - r0, size, digest, name)
        rays_v3.verify_section(archive, entry)
        return entry.as_dict(), True
    tmp = final + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    if lattice:
        rng = rays.stream_rng(cfg.seed, SCENES[scene]["tag"], mode)
        origin = Source(src, rng).mode_origin()
        if origin_str is None:
            origin_str = [str(c) for c in origin]
        elif [str(c) for c in origin] != list(origin_str):
            raise ValueError(f"{scene} mode {mode}: lattice origin differs from the header")
        extra = {"rng": {"scheme": rays.RNG_SCHEME, "seed": cfg.seed, "aim_draws": draws}}
    else:
        p = cfg.precision
        origin = tuple(Number(c, p) for c in origin_str)
        rng = tail_rng(cfg.seed, mode)
        extra = {"rng": {"scheme": LEGACY_SCHEME, "tag": int(rays.SceneSeed.CAPILLARY_TOPUP),
                         "seed": cfg.seed, "aim_draws": draws}}
    draw = rng.random
    for _ in range(draws * r0):
        draw()
    aim = aim_factory(None, screen, rng)
    writer = rays_v3.SectionWriter(archive, scene, mode, r0, r1, origin_str, extra, level)
    try:
        for ray in range(r0, r1):
            tr = tracer(origin, aim(origin), optic, screen.z, cfg.max_bounces)
            rec = ray_record(tr, screen, mode, ray, tr.fate)
            writer.write_row(rays.row_of(scene, rec, lean))
        return writer.close().as_dict(), False
    except BaseException:
        writer.abort()
        raise


def _new_archive(cfg, archive: str) -> dict:
    """Create an empty lattice-v1 archive for the yaml geometry."""
    if os.path.lexists(rays_v3.fingerprint_path(archive)):
        raise ValueError(f"{archive}: fingerprint exists without an index; remove it manually")
    modes_dir = os.path.join(archive, rays_v3.MODES_DIR)
    if os.path.isdir(modes_dir) and any(os.scandir(modes_dir)):
        # Sections of unknown provenance must never be adopted silently.
        raise ValueError(
            f"{archive}: orphan sections exist without an index; delete the "
            "whole archive directory and re-record")
    meta = {"format": rays_v3.FORMAT, "geometry": rays.geometry_metadata(cfg),
            "rng": {"scheme": rays.RNG_SCHEME,
                    "aim_draws": {name: spec["draws"] for name, spec in SCENES.items()}}}
    if cfg.lean_rays:
        meta["lean"] = True
    os.makedirs(os.path.join(archive, rays_v3.MODES_DIR), exist_ok=True)
    rays_v3.write_fingerprint(archive, meta)
    return meta


def trace(config_path: str, archive: str, jobs: int,
          level: int, log=_log, scenes="all") -> dict:
    from .simulation import Simulation
    sim = Simulation.from_yaml(config_path)
    cfg = sim.cfg
    if scenes == "all":
        scenes = tuple(s for s in SCENES
                       if (cfg.capillary if s == "capillary" else cfg.free_source) is not None)
    scenes = tuple(scenes)
    if not scenes:
        raise ValueError("no scene to trace")
    fresh = not os.path.lexists(rays_v3.index_path(archive))
    if fresh:
        meta = _new_archive(cfg, archive)
        index = None
        old_entries = []
    else:
        meta = rays_v3.read_fingerprint(archive)
        index = rays_v3.load_index(archive)
        old_entries = index.entries
    if not rays.metadata_equal(rays.geometry_core(meta["geometry"]),
                               rays.geometry_core(rays.geometry_metadata(cfg))):
        raise ValueError("config geometry differs from the archive")
    if int(meta["geometry"].get("seed")) != cfg.seed:
        raise ValueError("config seed differs from the archive seed")
    lean = bool(meta.get("lean"))
    if lean != bool(cfg.lean_rays):
        raise ValueError("config trace.lean_rays must match the archive's lean flag")
    scheme = (meta.get("rng") or {}).get("scheme")
    if scheme not in (rays.RNG_SCHEME, "sequential-v2"):
        raise ValueError(f"{archive}: unknown rng scheme {scheme!r}; cannot trace into it")
    lattice = scheme == rays.RNG_SCHEME

    plan = []
    for scene in scenes:
        src, _, _, _ = _scene_parts(sim, scene)
        n_modes, target = src.budget()
        if not lattice and scene not in (index.budgets if index else ()):
            log(f"{scene}: not in this legacy archive, skipped")
            continue
        if index is not None and scene in index.budgets:
            n_modes_recorded, n0 = index.budgets[scene]
            if n_modes_recorded != n_modes:
                raise ValueError(f"{archive} holds {n_modes_recorded} {scene} modes, "
                                 f"config gives {n_modes}")
            if target == n0:
                log(f"{scene}: already recorded at {n0} rays per mode, nothing to do")
                continue
            if target < n0:
                raise ValueError(f"{scene}: target {target} is below the recorded "
                                 f"{n0} rays per mode")
            origins = rays_v3.origins(archive, index, scene)
            missing = [m for m, o in enumerate(origins) if not o]
            if missing:
                raise ValueError(f"{archive}: {len(missing)} {scene} modes have no recorded "
                                 f"origin (first {missing[:5]}); reconvert with origin recovery")
        else:
            n0 = 0
            origins = [None] * n_modes
        if not lattice and scene != "capillary":
            raise ValueError("legacy sequential-v2 archives can only top up capillary")
        plan.append((scene, n_modes, n0, target, origins))
        log(f"{'trace' if n0 == 0 else 'top-up'} {archive} ({scheme}) {scene}: "
            f"{n_modes} modes, rays {n0} -> {target}, jobs {jobs}")

    started = time.time()
    new_entries = []
    reused = 0
    total = sum(n_modes for _, n_modes, _, _, _ in plan)
    tasks = [(scene, mode, n0, target, origins[mode])
             for scene, n_modes, n0, target, origins in plan
             for mode in range(n_modes)]
    if jobs == 1:
        # In-process: no spawn cost, same code path as the pool workers.
        _init_worker(config_path)
        results = (_trace_section(archive, scene, mode, n0, target, origin,
                                  lean, level, lattice)
                   for scene, mode, n0, target, origin in tasks)
        done = 0
        for entry, was_reused in results:
            new_entries.append(rays_v3.Section(**entry))
            reused += was_reused
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                log(f"  {done}/{total} modes ({(time.time() - started) / 60:.1f} min)")
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, initializer=_init_worker,
                initargs=(config_path,)) as pool:
            futures = [pool.submit(_trace_section, archive, scene, mode, n0,
                                   target, origin, lean, level, lattice)
                       for scene, mode, n0, target, origin in tasks]
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                entry, was_reused = fut.result()
                new_entries.append(rays_v3.Section(**entry))
                reused += was_reused
                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    log(f"  {done}/{total} modes ({(time.time() - started) / 60:.1f} min)")
    index = (rays_v3.write_index(archive, old_entries + new_entries)
             if new_entries else rays_v3.load_index(archive))
    summary = {"scenes": [(scene, n0, target) for scene, _, n0, target, _ in plan],
               "reused_sections": reused, "budgets": index.budgets,
               "seconds": time.time() - started}
    log(f"done: budgets {index.budgets}, {summary['seconds'] / 60:.1f} min")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m formula.capsysred.trace_v3")
    ap.add_argument("config")
    ap.add_argument("--archive", required=True, help="v3 archive directory (created if absent)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--level", type=int, default=rays_v3.DEFAULT_LEVEL)
    args = ap.parse_args(argv)
    max_jobs = os.cpu_count() or MAX_JOBS
    if args.jobs < 1 or args.jobs > max_jobs:
        ap.error(f"--jobs must be within 1..{max_jobs}")
    trace(os.path.abspath(args.config), os.path.abspath(args.archive),
          args.jobs, args.level)
    return 0


if __name__ == "__main__":
    sys.exit(main())

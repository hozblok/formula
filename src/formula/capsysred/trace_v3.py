"""Trace scenes straight into a v3 archive, or top them up.

    python -m formula.capsysred.trace_v3 config.yaml --archive ARCHIVE_DIR
        [--scene capillary|free|all] [--rays R1] [--jobs J] [--level 6]

No archive yet: it is created (fingerprint from the yaml, ``rng:
lattice-v1``) and every mode of the selected scenes traces rays 0..R1-1
into its own section, one process per mode — no shards, no merge,
resumable per section.  Archive present: the same command adds rays
n0..R1-1 to every existing mode (top-up) or a whole new scene.  ``--rays``
defaults to the yaml budget and needs a single scene; the yaml `n_rays`
must equal the target (Stage 14's "recorded == configured").

lattice-v1: mode m of a scene draws its origin, then a fixed number of
draws per ray (capillary aim 3, free aim 2) from
``stream_rng(seed, <scene>, m)``, so ``--jobs``, one-shot vs top-up and the
sequential v2 ``--trace`` all give identical rays.  Legacy sequential-v2
archives (converted Z-26) top up their capillary scene from the tail
substream ``Random((seed*STRIDE + CAPILLARY_TOPUP) * 2**32 + m)`` at
position 3*r.  Sections publish tmp -> rename; the index is replaced
atomically once every mode is done, so an interrupted run leaves the
archive at its old budget.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import os
import random
import sys
import time

from ..formula import Number
from . import rays, rays_v3
from .native import make_tracer
from .screen import ScreenGrid
from .source import Source
from .surfaces import CapillaryBundle
from .types import ray_record

# aim draws per ray, asserted by tests (rng skip counts)
SCENES = {"capillary": {"tag": rays.SceneSeed.CAPILLARY, "draws": 3},
          "free": {"tag": rays.SceneSeed.FREE, "draws": 2}}
AIM_DRAWS = SCENES["capillary"]["draws"]
LEGACY_SCHEME = "per-mode-substream-v1"


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


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


def _geometry_core(geometry: dict) -> dict:
    """Recording geometry without the per-scene budgets."""
    core = copy.deepcopy(geometry)
    for scene in ("capillary", "free"):
        source = (core.get(scene) or {}).get("source")
        if isinstance(source, dict):
            source.pop("n_modes", None)
            source.pop("n_rays", None)
    return core


def _new_archive(cfg, archive: str) -> dict:
    """Create an empty lattice-v1 archive for the yaml geometry."""
    if os.path.lexists(rays_v3.fingerprint_path(archive)):
        raise ValueError(f"{archive}: fingerprint exists without an index; remove it manually")
    meta = {"format": rays_v3.FORMAT, "geometry": rays.geometry_metadata(cfg),
            "rng": {"scheme": rays.RNG_SCHEME,
                    "aim_draws": {name: spec["draws"] for name, spec in SCENES.items()}}}
    if cfg.lean_rays:
        meta["lean"] = True
    os.makedirs(os.path.join(archive, rays_v3.MODES_DIR), exist_ok=True)
    rays_v3.write_fingerprint(archive, meta)
    return meta


def trace(config_path: str, archive: str, r1: int | None, jobs: int,
          level: int, log=_log, scenes=("capillary",)) -> dict:
    from .simulation import Simulation
    sim = Simulation.from_yaml(config_path)
    cfg = sim.cfg
    if scenes == "all":
        scenes = tuple(s for s in SCENES
                       if (cfg.capillary if s == "capillary" else cfg.free_source) is not None)
    scenes = tuple(scenes)
    if not scenes:
        raise ValueError("no scene to trace")
    if r1 is not None and len(scenes) != 1:
        raise ValueError("--rays needs a single --scene")
    fresh = not os.path.lexists(rays_v3.index_path(archive))
    if fresh:
        meta = _new_archive(cfg, archive)
        index = None
        old_entries = []
    else:
        meta = rays_v3.read_fingerprint(archive)
        index = rays_v3.load_index(archive)
        old_entries = index.entries
    if not rays.metadata_equal(_geometry_core(meta["geometry"]),
                               _geometry_core(rays.geometry_metadata(cfg))):
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
        n_modes, budget_rays = src.budget()
        target = budget_rays if r1 is None else r1
        if src.budget() != (n_modes, target):
            raise ValueError(f"config {scene} budget {src.budget()} must equal "
                             f"[{n_modes}, {target}] (modes, target rays)")
        if index is not None and scene in index.budgets:
            n_modes_recorded, n0 = index.budgets[scene]
            if n_modes_recorded != n_modes:
                raise ValueError(f"{archive} holds {n_modes_recorded} {scene} modes, "
                                 f"config gives {n_modes}")
            if target <= n0:
                raise ValueError(f"{scene}: target {target} must exceed the current "
                                 f"{n0} rays per mode")
            origins = rays_v3.origins(archive, index, scene)
            missing = [m for m, o in enumerate(origins) if not o]
            if missing:
                raise ValueError(f"{archive}: {len(missing)} {scene} modes have no recorded "
                                 f"origin (first {missing[:5]}); reconvert with origin recovery")
        else:
            n0 = 0
            origins = [None] * n_modes
            if not lattice:
                raise ValueError(f"{archive}: a new scene needs a lattice-v1 archive")
        if not lattice and scene != "capillary":
            raise ValueError("legacy sequential-v2 archives can only top up capillary")
        plan.append((scene, n_modes, n0, target, origins))
        log(f"{'trace' if n0 == 0 else 'top-up'} {archive} ({scheme}) {scene}: "
            f"{n_modes} modes, rays {n0} -> {target}, jobs {jobs}")

    started = time.time()
    new_entries = []
    reused = 0
    total = sum(n_modes for _, n_modes, _, _, _ in plan)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_worker,
            initargs=(config_path,)) as pool:
        futures = [pool.submit(_trace_section, archive, scene, mode, n0, target,
                               origins[mode], lean, level, lattice)
                   for scene, n_modes, n0, target, origins in plan
                   for mode in range(n_modes)]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            entry, was_reused = fut.result()
            new_entries.append(rays_v3.Section(**entry))
            reused += was_reused
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                log(f"  {done}/{total} modes ({(time.time() - started) / 60:.1f} min)")
    index = rays_v3.write_index(archive, old_entries + new_entries)
    summary = {"scenes": [(scene, n0, target) for scene, _, n0, target, _ in plan],
               "reused_sections": reused, "budgets": index.budgets,
               "seconds": time.time() - started}
    log(f"done: budgets {index.budgets}, {summary['seconds'] / 60:.1f} min")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m formula.capsysred.trace_v3")
    ap.add_argument("config")
    ap.add_argument("--archive", required=True, help="v3 archive directory (created if absent)")
    ap.add_argument("--scene", default="capillary", choices=[*SCENES, "all"],
                    help="scene(s) to trace (default capillary)")
    ap.add_argument("--rays", type=int, default=None,
                    help="target rays per mode (default: the yaml budget; single scene only)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--level", type=int, default=rays_v3.DEFAULT_LEVEL)
    args = ap.parse_args(argv)
    if args.jobs < 1 or args.jobs > 8:
        ap.error("--jobs must be within 1..8")
    scenes = "all" if args.scene == "all" else (args.scene,)
    trace(os.path.abspath(args.config), os.path.abspath(args.archive), args.rays,
          args.jobs, args.level, scenes=scenes)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Trace the capillary scene straight into a v3 archive, or top it up.

    python -m formula.capsysred.trace_v3 config.yaml --archive ARCHIVE_DIR
        [--rays R1] [--jobs J] [--level 6]

No archive yet: it is created (fingerprint from the yaml, ``rng:
lattice-v1``) and every mode traces rays 0..R1-1 into its own section, one
process per mode range — no shards, no merge, resumable per section.
Archive present: the same command adds rays n0..R1-1 to every existing mode
(top-up).  ``--rays`` defaults to the yaml budget; the yaml `n_rays` must
equal the target (Stage 14's "recorded == configured").

lattice-v1: mode m draws origin then 3 draws per ray from
``stream_rng(seed, CAPILLARY, m)``, so ``--jobs``, one-shot vs top-up and
the sequential v2 ``--trace`` all give identical rays.  Legacy
sequential-v2 archives (converted Z-26) top up from the tail substream
``Random((seed*STRIDE + CAPILLARY_TOPUP) * 2**32 + m)`` at position 3*r.
Sections publish tmp -> rename; the index is replaced atomically once every
mode is done, so an interrupted run leaves the archive at its old budget.
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
from .native import make_tracer
from .screen import ScreenGrid
from .source import Source
from .surfaces import CapillaryBundle
from .types import ray_record

AIM_DRAWS = 3
LEGACY_SCHEME = "per-mode-substream-v1"


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def tail_rng(seed: int, mode: int) -> random.Random:
    return random.Random(
        (seed * rays._SCENE_SEED_STRIDE + rays.SceneSeed.CAPILLARY_TOPUP) * 2**32 + mode)


_W = {}


def _init_worker(config_path):
    from .simulation import Simulation
    sim = Simulation.from_yaml(config_path)
    cap = sim.cfg.capillary
    optic = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    _W.update(sim=sim, cap=cap, screen=ScreenGrid(cap.screen), optic=optic,
              tracer=make_tracer(optic))


def _trace_tail(archive, mode, r0, r1, origin_str, lean, level, lattice):
    sim, cap = _W["sim"], _W["cap"]
    screen, optic, tracer = _W["screen"], _W["optic"], _W["tracer"]
    cfg = sim.cfg
    name = rays_v3.section_name("capillary", mode, r0, r1)
    final = os.path.join(archive, rays_v3.MODES_DIR, name)
    if os.path.exists(final):
        size, digest = rays_v3._hash_file(final)
        entry = rays_v3.Section("capillary", mode, r0, r1, r1 - r0, size, digest, name)
        rays_v3.verify_section(archive, entry)
        return entry.as_dict(), True
    tmp = final + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    if lattice:
        rng = rays.stream_rng(cfg.seed, rays.SceneSeed.CAPILLARY, mode)
        origin = Source(cap.source, rng).mode_origin()
        if origin_str is None:
            origin_str = [str(c) for c in origin]
        elif [str(c) for c in origin] != list(origin_str):
            raise ValueError(f"mode {mode}: lattice origin differs from the recorded header")
        extra = {"rng": {"scheme": rays.RNG_SCHEME, "seed": cfg.seed, "aim_draws": AIM_DRAWS}}
    else:
        p = cfg.precision
        origin = tuple(Number(c, p) for c in origin_str)
        rng = tail_rng(cfg.seed, mode)
        extra = {"rng": {"scheme": LEGACY_SCHEME, "tag": int(rays.SceneSeed.CAPILLARY_TOPUP),
                         "seed": cfg.seed, "aim_draws": AIM_DRAWS}}
    draw = rng.random
    for _ in range(AIM_DRAWS * r0):
        draw()
    aim = sim._aim_capillary(None, screen, rng)
    writer = rays_v3.SectionWriter(archive, "capillary", mode, r0, r1, origin_str,
                                   extra, level)
    try:
        for ray in range(r0, r1):
            tr = tracer(origin, aim(origin), optic, screen.z, cfg.max_bounces)
            rec = ray_record(tr, screen, mode, ray, tr.fate)
            writer.write_row(rays.row_of("capillary", rec, lean))
        return writer.close().as_dict(), False
    except BaseException:
        writer.abort()
        raise


def _new_archive(cfg, archive: str) -> dict:
    """Create an empty lattice-v1 archive for the yaml geometry."""
    if os.path.lexists(rays_v3.fingerprint_path(archive)):
        raise ValueError(f"{archive}: fingerprint exists without an index; remove it manually")
    meta = {"format": rays_v3.FORMAT, "geometry": rays.geometry_metadata(cfg),
            "rng": {"scheme": rays.RNG_SCHEME, "aim_draws": AIM_DRAWS}}
    if cfg.lean_rays:
        meta["lean"] = True
    os.makedirs(os.path.join(archive, rays_v3.MODES_DIR), exist_ok=True)
    rays_v3.write_fingerprint(archive, meta)
    return meta


def trace(config_path: str, archive: str, r1: int | None, jobs: int,
          level: int, log=_log) -> dict:
    from .simulation import Simulation
    from .stage14 import _trace_core
    sim = Simulation.from_yaml(config_path)
    cfg = sim.cfg
    if cfg.capillary is None:
        raise ValueError("trace_v3 requires a configured capillary.source")
    n_modes, budget_rays = cfg.capillary.source.budget()
    if r1 is None:
        r1 = budget_rays
    fresh = not os.path.lexists(rays_v3.index_path(archive))
    if fresh:
        meta = _new_archive(cfg, archive)
        n0 = 0
        old_entries = []
    else:
        meta = rays_v3.read_fingerprint(archive)
        index = rays_v3.load_index(archive)
        if "capillary" not in index.budgets:
            raise ValueError(f"{archive}: no capillary scene to top up")
        n_modes_recorded, n0 = index.budgets["capillary"]
        if n_modes_recorded != n_modes:
            raise ValueError(f"{archive} holds {n_modes_recorded} modes, config "
                             f"gives {n_modes}")
        if r1 <= n0:
            raise ValueError(f"--rays {r1} must exceed the current {n0} rays per mode")
        old_entries = index.entries
    expected_core = _trace_core(rays.geometry_metadata(cfg))
    if not rays.metadata_equal(_trace_core(meta["geometry"]), expected_core):
        raise ValueError("config trace geometry differs from the archive")
    if int(meta["geometry"].get("seed")) != cfg.seed:
        raise ValueError("config seed differs from the archive seed")
    if cfg.capillary.source.budget() != (n_modes, r1):
        raise ValueError(
            f"config budget {cfg.capillary.source.budget()} must equal "
            f"[{n_modes}, {r1}] (archive modes, target rays)")
    lean = bool(meta.get("lean"))
    if lean != bool(cfg.lean_rays):
        raise ValueError("config trace.lean_rays must match the archive's lean flag")
    scheme = (meta.get("rng") or {}).get("scheme")
    if scheme not in (rays.RNG_SCHEME, "sequential-v2"):
        raise ValueError(f"{archive}: unknown rng scheme {scheme!r}; cannot top up")
    lattice = scheme == rays.RNG_SCHEME
    if fresh:
        origins = [None] * n_modes
    else:
        origins = rays_v3.origins(archive, index, "capillary")
        missing = [m for m, o in enumerate(origins) if not o]
        if missing:
            raise ValueError(f"{archive}: {len(missing)} modes have no recorded origin "
                             f"(first {missing[:5]}); reconvert with origin recovery")
    log(f"{'trace' if fresh else 'top-up'} {archive} ({scheme}): {n_modes} modes, "
        f"rays {n0} -> {r1}, jobs {jobs}")
    started = time.time()
    new_entries = []
    reused = 0
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_worker,
            initargs=(config_path,)) as pool:
        futures = {pool.submit(_trace_tail, archive, mode, n0, r1, origins[mode],
                               lean, level, lattice): mode for mode in range(n_modes)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            entry, was_reused = fut.result()
            new_entries.append(rays_v3.Section(**entry))
            reused += was_reused
            done += 1
            if done % max(1, n_modes // 20) == 0 or done == n_modes:
                log(f"  {done}/{n_modes} modes ({(time.time() - started) / 60:.1f} min)")
    index = rays_v3.write_index(archive, old_entries + new_entries)
    summary = {"modes": n_modes, "r0": n0, "r1": r1, "reused_sections": reused,
               "budgets": index.budgets, "seconds": time.time() - started}
    log(f"done: budgets {index.budgets}, {summary['seconds'] / 60:.1f} min")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m formula.capsysred.trace_v3")
    ap.add_argument("config")
    ap.add_argument("--archive", required=True, help="v3 archive directory (created if absent)")
    ap.add_argument("--rays", type=int, default=None,
                    help="target rays per mode (default: the yaml budget)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--level", type=int, default=rays_v3.DEFAULT_LEVEL)
    args = ap.parse_args(argv)
    if args.jobs < 1 or args.jobs > 8:
        ap.error("--jobs must be within 1..8")
    trace(os.path.abspath(args.config), os.path.abspath(args.archive), args.rays,
          args.jobs, args.level)
    return 0


if __name__ == "__main__":
    sys.exit(main())

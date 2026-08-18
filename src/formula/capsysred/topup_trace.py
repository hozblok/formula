"""In-mode top-up: trace rays n0..R1-1 into the existing modes of a v3 archive.

    python -m formula.capsysred.topup_trace config.yaml --archive ARCHIVE_DIR
        --rays R1 [--jobs J] [--quick N] [--level 6]

Origins come from the section headers (recorded at conversion or tracing).
lattice-v1 archives: the tail simply continues the mode's own stream
``stream_rng(seed, CAPILLARY, m)`` (origin, then 3 draws per ray), so head +
tail equals a single-piece trace bit for bit.  Legacy sequential-v2
archives (converted Z-26): ray r of mode m draws from the tail substream
``Random((seed*STRIDE + CAPILLARY_TOPUP) * 2**32 + m)`` at position 3*r.
Either way one top-up n0->R1 equals two top-ups n0->R->R1.  Sections are
published tmp -> rename; the index is replaced atomically once every mode
has its tail, so an interrupted run leaves the archive at its old budget.
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


def _init_worker(config_path, quick):
    from .simulation import Simulation
    sim = Simulation.from_yaml(config_path)
    cap = sim.cfg.capillary
    optic = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    _W.update(sim=sim, cap=cap, screen=ScreenGrid(cap.screen), optic=optic,
              tracer=make_tracer(optic), quick=quick)


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
        if [str(c) for c in origin] != list(origin_str):
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


def topup(config_path: str, archive: str, r1: int, jobs: int, quick: int,
          level: int, log=_log) -> dict:
    from .simulation import Simulation
    from .stage14 import _trace_core
    sim = Simulation.from_yaml(config_path)
    cfg = sim.cfg
    if cfg.capillary is None:
        raise ValueError("top-up requires a configured capillary.source")
    meta = rays_v3.read_fingerprint(archive)
    index = rays_v3.load_index(archive)
    if "capillary" not in index.budgets:
        raise ValueError(f"{archive}: no capillary scene to top up")
    n_modes, n0 = index.budgets["capillary"]
    if r1 <= n0:
        raise ValueError(f"--rays {r1} must exceed the current {n0} rays per mode")
    expected_core = _trace_core(rays.geometry_metadata(cfg))
    if not rays.metadata_equal(_trace_core(meta["geometry"]), expected_core):
        raise ValueError("config trace geometry differs from the archive")
    if int(meta["geometry"].get("seed")) != cfg.seed:
        raise ValueError("config seed differs from the archive seed")
    if cfg.capillary.source.budget(quick) != (n_modes, r1):
        raise ValueError(
            f"config/--quick budget {cfg.capillary.source.budget(quick)} must equal "
            f"[{n_modes}, {r1}] (archive modes, target rays)")
    lean = bool(meta.get("lean"))
    if lean != bool(cfg.lean_rays):
        raise ValueError("config trace.lean_rays must match the archive's lean flag")
    scheme = (meta.get("rng") or {}).get("scheme")
    if scheme not in (rays.RNG_SCHEME, "sequential-v2"):
        raise ValueError(f"{archive}: unknown rng scheme {scheme!r}; cannot top up")
    lattice = scheme == rays.RNG_SCHEME
    origins = rays_v3.origins(archive, index, "capillary")
    missing = [m for m, o in enumerate(origins) if not o]
    if missing:
        raise ValueError(f"{archive}: {len(missing)} modes have no recorded origin "
                         f"(first {missing[:5]}); reconvert with origin recovery")
    log(f"top-up {archive} ({scheme}): {n_modes} modes, rays {n0} -> {r1}, jobs {jobs}")
    started = time.time()
    new_entries = []
    reused = 0
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_worker,
            initargs=(config_path, quick)) as pool:
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
    index = rays_v3.write_index(archive, index.entries + new_entries)
    summary = {"modes": n_modes, "r0": n0, "r1": r1, "reused_sections": reused,
               "budgets": index.budgets, "seconds": time.time() - started}
    log(f"done: budgets {index.budgets}, {summary['seconds'] / 60:.1f} min")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m formula.capsysred.topup_trace")
    ap.add_argument("config")
    ap.add_argument("--archive", required=True, help="v3 archive directory")
    ap.add_argument("--rays", type=int, required=True, help="target rays per mode")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", type=int, default=1)
    ap.add_argument("--level", type=int, default=rays_v3.DEFAULT_LEVEL)
    args = ap.parse_args(argv)
    if args.jobs < 1 or args.jobs > 8:
        ap.error("--jobs must be within 1..8")
    topup(os.path.abspath(args.config), os.path.abspath(args.archive), args.rays,
          args.jobs, max(1, args.quick), args.level)
    return 0


if __name__ == "__main__":
    sys.exit(main())

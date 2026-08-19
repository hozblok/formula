"""Convert a v2 rays.jsonl.gz recording into a v3 per-mode archive.

    python -m formula.capsysred.convert_rays_v3 RUN_DIR [--out RUN_DIR/rays-modes]
        [--jobs 6] [--level 6] [--shards K] [--no-origins]
    python -m formula.capsysred.convert_rays_v3 --verify ARCHIVE_DIR [--jobs 6]

One strict pass over the v2 stream splits every scene listed in the sidecar
budgets into sections of one mode each; rows are copied byte for byte.  For
the capillary scene the mode origins are recovered by replaying the recorded
rng streams — lattice-v1 recordings (sidecar ``rng: lattice-v1``): one
stream per global mode; legacy sequential-v2 recordings: shard-k/config.yaml
gives seed and mode count per shard, a single stream under the base seed
otherwise — and every recovered origin is verified by re-tracing ray 0
against the recorded row before the section is written.  ``--verify``
re-reads a v3 archive completely: hashes, counts and canonical mode/ray ids.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import io
import json
import math
import os
import random
import sys
import time

import yaml

from . import rays, rays_v3
from .common import tlog as _log
from .native import make_tracer
from .screen import ScreenGrid
from .source import Source
from .surfaces import CapillaryBundle
from .types import ray_record

CAPILLARY_AIM_DRAWS = 3     # bore choice + disk point; asserted by tests
XY_TOL = 1e-9               # m, ray-0 re-trace against the recorded row
OPL_REL_TOL = 1e-9


def _chunks(total, jobs):
    base, extra = divmod(total, jobs)
    return [base + (k < extra) for k in range(jobs)]


def shard_layout(run_dir: str, meta: dict, shards: int | None) -> list[tuple[int, int]]:
    """[(seed, n_modes)] per capillary shard, in global mode order."""
    n_modes = meta["budgets"]["capillary"][0]
    base_seed = int(meta["geometry"]["seed"])
    layout = []
    k = 0
    while True:
        cfg_path = os.path.join(run_dir, f"shard-{k}", "config.yaml")
        if not os.path.exists(cfg_path):
            break
        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        layout.append((int(raw["seed"]), int(raw["capillary"]["source"]["n_modes"])))
        k += 1
    if layout:
        if shards is not None and shards != len(layout):
            raise ValueError(f"--shards {shards} contradicts {len(layout)} shard configs")
        if layout[0][0] != base_seed:
            raise ValueError("shard-0 seed differs from the recording seed")
    elif shards:
        layout = [(base_seed + k, n) for k, n in enumerate(_chunks(n_modes, shards))]
    else:
        layout = [(base_seed, n_modes)]
    if sum(n for _, n in layout) != n_modes:
        raise ValueError(f"shard layout covers {sum(n for _, n in layout)} modes, "
                         f"recording has {n_modes}")
    return layout


def _geometry_sim(geometry: dict, seed: int):
    from .simulation import Simulation
    raw = {"seed": seed, "precision": geometry["precision"],
           "screen": copy.deepcopy(geometry["screen"]),
           "capillary": copy.deepcopy(geometry["capillary"]),
           "trace": {"max_bounces": geometry["max_bounces"]}}
    return Simulation.from_dict(raw)


def _check_record(rec) -> dict:
    check = {"fate": rec.fate, "pixel": rec.pixel, "opl": float(rec.opl),
             "bounces": len(rec.sins)}
    if rec.point is not None:
        check["x"], check["y"] = float(rec.point[0]), float(rec.point[1])
    return check


def recover_origins(meta: dict, layout, log=_log) -> list[dict]:
    """Per global capillary mode: origin strings and the re-traced ray 0.

    ``layout`` is the legacy sequential-v2 shard list; ``None`` selects the
    lattice-v1 streams (one per global mode, no shard knowledge needed).
    """
    geometry = meta["geometry"]
    n_modes, n_rays = meta["budgets"]["capillary"]
    out = []
    started = time.time()
    if layout is None:
        seed = int(geometry["seed"])
        sim = _geometry_sim(geometry, seed)
        cfg = sim.cfg
        cap = cfg.capillary
        source = Source(cap.source, None)
        screen = ScreenGrid(cap.screen)
        optic = CapillaryBundle(cap.bores, cap.z0, cap.z1)
        tracer = make_tracer(optic)
        for mode in range(n_modes):
            rng = rays.stream_rng(seed, rays.SceneSeed.CAPILLARY, mode)
            source.rng = rng
            origin = source.mode_origin()
            d0 = sim._aim_capillary(source, screen, rng)(origin)
            tr = tracer(origin, d0, optic, screen.z, cfg.max_bounces)
            rec = ray_record(tr, screen, mode, 0, tr.fate)
            out.append({"origin": [str(c) for c in origin], "check": _check_record(rec),
                        "seed": seed})
        log(f"  origins: {len(out)} lattice modes recovered ({time.time() - started:.0f} s)")
        return out
    for seed, n_local in layout:
        sim = _geometry_sim(geometry, seed)
        cfg = sim.cfg
        cap = cfg.capillary
        rng = random.Random(seed * rays._SCENE_SEED_STRIDE + rays.SceneSeed.CAPILLARY)
        source = Source(cap.source, rng)
        screen = ScreenGrid(cap.screen)
        aim = sim._aim_capillary(source, screen, rng)
        optic = CapillaryBundle(cap.bores, cap.z0, cap.z1)
        tracer = make_tracer(optic)
        skip = CAPILLARY_AIM_DRAWS * (n_rays - 1)
        draw = rng.random
        for _ in range(n_local):
            origin = source.mode_origin()
            d0 = aim(origin)
            for _ in range(skip):
                draw()
            tr = tracer(origin, d0, optic, screen.z, cfg.max_bounces)
            rec = ray_record(tr, screen, len(out), 0, tr.fate)
            out.append({"origin": [str(c) for c in origin], "check": _check_record(rec),
                        "seed": seed})
        log(f"  origins: {len(out)} modes recovered ({time.time() - started:.0f} s)")
    return out


def check_ray0(row: dict, expect: dict, mode: int) -> tuple[float, float]:
    """Raise unless the recorded ray 0 matches the re-trace; return deviations."""
    check = expect["check"]
    if row.get("fate") != check["fate"]:
        raise ValueError(f"mode {mode}: recorded ray 0 fate {row.get('fate')!r} != "
                         f"re-traced {check['fate']!r}; wrong seed/shard layout?")
    opl = float(row["opl"])
    d_opl = abs(opl - check["opl"]) / max(abs(check["opl"]), 1e-300)
    d_xy = 0.0
    if check["fate"] == "screen":
        d_xy = max(abs(float(row["x"]) - check["x"]), abs(float(row["y"]) - check["y"]))
    if d_xy > XY_TOL or d_opl > OPL_REL_TOL:
        raise ValueError(f"mode {mode}: recorded ray 0 differs from the re-trace "
                         f"(dxy={d_xy:.3e} m, dopl/opl={d_opl:.3e}); wrong seed/shard layout?")
    return d_xy, d_opl


def _write_section(out_dir, scene, mode, n, origin, extra, payload, rows, level):
    final = os.path.join(out_dir, rays_v3.MODES_DIR,
                         rays_v3.section_name(scene, mode, 0, n))
    if os.path.exists(final):
        # Resume: an already published section is re-verified, not rewritten.
        size, digest = rays_v3._hash_file(final)
        entry = rays_v3.Section(scene, mode, 0, n, n, size, digest,
                                os.path.basename(final))
        rays_v3.verify_section(out_dir, entry)
        return entry, True
    tmp = final + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    writer = rays_v3.SectionWriter(out_dir, scene, mode, 0, n, origin, extra, level)
    try:
        writer.write_lines(payload, rows)
        return writer.close(), False
    except BaseException:
        writer.abort()
        raise


def _mode_of(line: bytes) -> int:
    i = line.find(b'"mode": ') + 8
    return int(line[i:line.index(b",", i)])


def _ray_of(line: bytes) -> int:
    i = line.find(b'"ray": ') + 7
    return int(line[i:line.index(b",", i)])


def convert(run_dir: str, out_dir: str, jobs: int, level: int, shards: int | None,
            origins: bool, log=_log) -> dict:
    src = os.path.join(run_dir, "rays.jsonl.gz")
    meta = rays.read_metadata(src)
    rays._validate_stream_metadata(meta, src)
    budgets = meta["budgets"]
    if os.path.lexists(rays_v3.index_path(out_dir)):
        raise ValueError(f"{out_dir}: v3 index already exists; refusing to convert again")
    os.makedirs(os.path.join(out_dir, rays_v3.MODES_DIR), exist_ok=True)
    src_stat = os.stat(src)
    log(f"convert {src} ({src_stat.st_size / 2**30:.2f} GiB) -> {out_dir}; "
        f"budgets {budgets}; jobs {jobs}, level {level}")

    layout = None
    recovered = None
    lattice = meta.get("rng") == rays.RNG_SCHEME
    if "capillary" in budgets and origins:
        if not lattice:
            layout = shard_layout(run_dir, meta, shards)
            log(f"  shard layout: {layout}")
        recovered = recover_origins(meta, layout, log)

    entries = []
    reused = 0
    dev_xy = dev_opl = 0.0
    counts = {scene: 0 for scene in budgets}
    trailers = {}
    skipped_scenes = {}
    started = time.time()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
    pending = []

    def collect(block: bool):
        nonlocal reused
        while pending and (block or pending[0].done() or len(pending) > jobs + 2):
            fut = pending.pop(0)
            entry, was_reused = fut.result()
            entries.append(entry)
            reused += was_reused

    def flush(scene, mode, lines):
        n = budgets[scene][1]
        if len(lines) != n:
            raise ValueError(f"{src}: scene {scene!r} mode {mode} holds {len(lines)} rows, "
                             f"budget promises {n}")
        origin = extra = None
        if scene == "capillary" and recovered is not None:
            origin = recovered[mode]["origin"]
            extra = {"origin_check": "retrace-ray0", "seed": recovered[mode]["seed"]}
        payload = b"".join(lines)
        pending.append(pool.submit(_write_section, out_dir, scene, mode, n,
                                   origin, extra, payload, len(lines), level))
        collect(False)

    current = None
    lines = []
    expect_ray = 0
    fast_prefix = None      # b'{"stage": "<scene>", "mode": <m>, "ray": ' of the open mode
    fast_len = 0
    modes_done = {scene: 0 for scene in budgets}
    try:
        with open(src, "rb") as raw_fh:
            gz = gzip.GzipFile(fileobj=io.BufferedReader(raw_fh, 1 << 20), mode="rb")
            preamble = gz.readline()
            if not preamble.endswith(b"\n"):
                raise ValueError(f"{src}: missing preamble line")
            for line in gz:
                if fast_prefix is not None and line.startswith(fast_prefix):
                    # Hot path: another row of the open mode.
                    if (not line.startswith(b"%d, " % expect_ray, fast_len)
                            or not line.endswith(b"\n")):
                        raise ValueError(f"{src}: scene {current[0]!r} mode {current[1]} "
                                         f"ray {_ray_of(line)} where {expect_ray} expected")
                    expect_ray += 1
                    lines.append(line)
                    continue
                if not line.endswith(b"\n"):
                    raise ValueError(f"{src}: unterminated line")
                if line.startswith(b'{"scene_end"'):
                    row = json.loads(line)
                    scene = row["scene_end"]
                    if scene in trailers:
                        raise ValueError(f"{src}: duplicate trailer for {scene!r}")
                    trailers[scene] = row["rows"]
                    if current is not None and current[0] == scene:
                        counts[scene] += len(lines)
                        flush(current[0], current[1], lines)
                        modes_done[scene] += 1
                        current, lines, fast_prefix = None, [], None
                    continue
                if not line.startswith(b'{"stage": "'):
                    raise ValueError(f"{src}: unexpected row {line[:60]!r}")
                end = line.index(b'"', 11)
                scene = line[11:end].decode("ascii")
                if scene not in budgets:
                    skipped_scenes[scene] = skipped_scenes.get(scene, 0) + 1
                    continue
                if scene in trailers:
                    raise ValueError(f"{src}: {scene!r} row after its trailer")
                mode = _mode_of(line)
                if current is not None:
                    if current[0] != scene:
                        raise ValueError(f"{src}: scene {current[0]!r} interrupted by {scene!r}")
                    counts[scene] += len(lines)
                    flush(current[0], current[1], lines)
                    modes_done[scene] += 1
                if mode != modes_done[scene]:
                    raise ValueError(f"{src}: scene {scene!r} mode {mode} out of order "
                                     f"(expected {modes_done[scene]})")
                current, lines, expect_ray = (scene, mode), [], 0
                fast_prefix = b'{"stage": "%s", "mode": %d, "ray": ' % (scene.encode("ascii"), mode)
                fast_len = len(fast_prefix)
                if not line.startswith(fast_prefix) or _ray_of(line) != 0:
                    raise ValueError(f"{src}: scene {scene!r} mode {mode} does not start at ray 0")
                if scene == "capillary" and recovered is not None:
                    d_xy, d_opl = check_ray0(json.loads(line), recovered[mode], mode)
                    dev_xy, dev_opl = max(dev_xy, d_xy), max(dev_opl, d_opl)
                if mode % 25 == 0:
                    elapsed = time.time() - started
                    log(f"  {scene} mode {mode}/{budgets[scene][0]} "
                        f"({elapsed / 60:.1f} min, pending {len(pending)})")
                expect_ray = 1
                lines.append(line)
            if current is not None:
                raise ValueError(f"{src}: scene {current[0]!r} has no trailer")
        collect(True)
    except BaseException:
        for fut in pending:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)

    for scene, (n_modes, n_rays) in budgets.items():
        if trailers.get(scene) != counts[scene] or counts[scene] != n_modes * n_rays:
            raise ValueError(f"{src}: scene {scene!r} rows {counts[scene]}, trailer "
                             f"{trailers.get(scene)}, budget {n_modes * n_rays}")
    index = rays_v3.write_index(out_dir, entries)
    if index.budgets != {scene: list(b) for scene, b in budgets.items()}:
        raise ValueError(f"v3 index budgets {index.budgets} differ from {budgets}")
    fingerprint = {"format": rays_v3.FORMAT, "geometry": _json(meta["geometry"])}
    if meta.get("lean"):
        fingerprint["lean"] = True
    if lattice:
        fingerprint["rng"] = {"scheme": rays.RNG_SCHEME, "aim_draws": CAPILLARY_AIM_DRAWS}
    elif layout is not None:
        fingerprint["rng"] = {
            "scheme": "sequential-v2",
            "scene_seed_stride": rays._SCENE_SEED_STRIDE,
            "capillary": {"tag": int(rays.SceneSeed.CAPILLARY),
                          "aim_draws": CAPILLARY_AIM_DRAWS,
                          "shards": [{"seed": seed, "modes": n} for seed, n in layout],
                          "head_rays": budgets["capillary"][1]},
        }
    fingerprint["converted_from"] = {
        "path": os.path.abspath(src), "bytes": src_stat.st_size,
        "mtime_ns": src_stat.st_mtime_ns,
    }
    rays_v3.write_fingerprint(out_dir, fingerprint)
    summary = {
        "sections": len(entries), "reused_sections": reused,
        "bytes": index.total_bytes(), "seconds": time.time() - started,
        "skipped_scene_rows": skipped_scenes,
        "origin_check": ({"modes": len(recovered), "max_dxy_m": dev_xy,
                          "max_rel_dopl": dev_opl} if recovered is not None else None),
    }
    with open(os.path.join(out_dir, "conversion.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log(f"done: {len(entries)} sections, {index.total_bytes() / 2**30:.2f} GiB, "
        f"{summary['seconds'] / 60:.1f} min; origin check {summary['origin_check']}")
    return summary


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _verify_one(archive: str, entry_dict: dict) -> dict:
    entry = rays_v3.Section(**entry_dict)
    header = rays_v3.verify_section(archive, entry)
    return {"file": entry.file, "origin": header.get("origin") is not None}


def verify(archive: str, jobs: int, log=_log) -> dict:
    index = rays_v3.load_index(archive)
    meta = rays_v3.metadata(archive)
    started = time.time()
    with_origin = 0
    n = len(index.entries)

    def progress(k):
        if k % 100 == 0 or k == n:
            log(f"  verified {k}/{n} sections ({time.time() - started:.0f} s)")

    if jobs == 1:
        # In-process: no spawn cost.
        for k, e in enumerate(index.entries, 1):
            with_origin += _verify_one(archive, e.as_dict())["origin"]
            progress(k)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_verify_one, archive, e.as_dict()) for e in index.entries]
            for k, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                with_origin += fut.result()["origin"]
                progress(k)
    summary = {"sections": len(index.entries), "budgets": index.budgets,
               "bytes": index.total_bytes(), "sections_with_origin": with_origin,
               "seconds": time.time() - started, "lean": bool(meta.get("lean"))}
    log(f"verify ok: {summary}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m formula.capsysred.convert_rays_v3")
    ap.add_argument("run_dir", nargs="?", help="directory holding rays.jsonl.gz")
    ap.add_argument("--out", default=None, help="v3 archive directory (default RUN_DIR/rays-modes)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--level", type=int, default=rays_v3.DEFAULT_LEVEL)
    ap.add_argument("--shards", type=int, default=None,
                    help="shard count when shard-k/config.yaml files are absent")
    ap.add_argument("--no-origins", action="store_true",
                    help="do not recover/verify capillary mode origins")
    ap.add_argument("--verify", metavar="ARCHIVE_DIR", default=None,
                    help="re-read a v3 archive completely instead of converting")
    args = ap.parse_args(argv)
    if args.jobs < 1 or args.jobs > 8:
        ap.error("--jobs must be within 1..8")
    if args.verify:
        verify(os.path.abspath(args.verify), args.jobs)
        return 0
    if not args.run_dir:
        ap.error("run_dir is required unless --verify is given")
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(run_dir, "rays-modes")
    convert(run_dir, out_dir, args.jobs, args.level, args.shards, not args.no_origins)
    return 0


if __name__ == "__main__":
    sys.exit(main())

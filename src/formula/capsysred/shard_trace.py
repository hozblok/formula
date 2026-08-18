"""Shard-parallel tracing: split the capillary modes across processes and
merge the records into one canonical rays.jsonl.gz.

DEPRECATED for capillary recordings: trace_v3 writes the per-mode v3
archive directly (parallel, resumable, no merge) with identical rays.

    python -m formula.capsysred.shard_trace config.yaml -o out/RUN \
        --jobs 7 [--quick N] [--keep-shards] [--no-merge]

Shard k traces the global modes [start_k, start_k + n_k) of the one rng
lattice (source.mode_start = start_k, same seed) into out/RUN/shard-k/
(derived config and log sit next to the record); shards with a complete
record are skipped on restart. The merge writes the canonical empty
preamble, copies an optional free scene from shard 0, renumbers the
capillary modes globally, recomputes the trailers, scans the body, and
publishes the structured metadata sidecar. Consumers then run the ORIGINAL
config (with the same --quick) against out/RUN and reuse the file.

Reproducibility: the merged record is bit-equal to a sequential trace of
the same config for any --jobs (lattice-v1 streams are per global mode).
Each tracer holds ~1-2 GB: pick --jobs for the RAM, not just the cores.
Disk peak: all shard records + the growing merge; consumed shards are
deleted unless --keep-shards. --no-merge stops after tracing: shard
records stay put for a multi-file --replay (halves the disk peak).
"""
import argparse
import copy
import gzip
import json
import os
import subprocess
import sys
import zlib

import yaml

from . import rays
from .config import Config


def _chunks(total, jobs):
    base, extra = divmod(total, jobs)
    return [base + (k < extra) for k in range(jobs)]


def _shard_raw(raw, budgets_q, cap_modes, mode_start):
    shard = copy.deepcopy(raw)
    if "free" in budgets_q:
        source = shard["free"]["source"]
        source["n_modes"], source["n_rays"] = budgets_q["free"]
    cap = shard["capillary"]["source"]
    cap["n_modes"], cap["n_rays"] = cap_modes, budgets_q["capillary"][1]
    cap["mode_start"] = mode_start
    return shard


def _capillary_done(path, expected_meta):
    if not os.path.exists(path):
        return False
    try:
        meta, done, clean = rays.scan(path, expected_meta=expected_meta)
    except (OSError, EOFError, zlib.error, UnicodeError, ValueError, KeyError,
            IndexError, AttributeError, TypeError):
        return False
    expected_counts = {
        scene: budget[0] * budget[1]
        for scene, budget in expected_meta["budgets"].items()
    }
    return (meta is not None and clean
            and rays.metadata_equal(meta, expected_meta)
            and done == expected_counts)


def _mode_span(line):
    i = line.find('"mode": ') + 8
    return i, line.index(",", i)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m formula.capsysred.shard_trace",
        description="trace the capillary scene in parallel shards and merge")
    ap.add_argument("config")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--jobs", type=int, required=True)
    ap.add_argument("--quick", type=int, default=1)
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--no-merge", action="store_true",
                    help="stop after tracing: leave shard records for a "
                         "multi-file --replay instead of one rays.jsonl.gz")
    args = ap.parse_args(argv)

    dst_path = os.path.join(args.out, "rays.jsonl.gz")
    dst_sidecar = rays.metadata_path(dst_path)
    if not args.no_merge and os.path.lexists(dst_path):
        ap.error(f"{dst_path} already exists; refusing to overwrite it "
                 "(remove it manually or choose another --out)")
    if not args.no_merge and os.path.lexists(dst_sidecar):
        ap.error(f"{dst_sidecar} already exists; refusing to overwrite it "
                 "(remove it manually or choose another --out)")

    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg = Config(raw)
    if cfg.capillary is None:
        sys.exit("config has no capillary scene")
    quick = max(1, args.quick)
    budgets_q = rays.budgets(cfg, quick)
    merged_sidecar = rays.sidecar_metadata(cfg, quick)
    cap_modes = budgets_q["capillary"][0]
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    max_jobs = cap_modes // 2
    if args.jobs > max_jobs:
        ap.error(
            f"--jobs {args.jobs} would create shards with fewer than 2 "
            f"modes; use at most {max_jobs} for {cap_modes} capillary modes"
        )
    chunks = _chunks(cap_modes, args.jobs)
    os.makedirs(args.out, exist_ok=True)

    shards = []
    expected = {}
    mode_start = 0
    for k, n in enumerate(chunks):
        sdir = os.path.join(args.out, f"shard-{k}")
        shard_raw = _shard_raw(raw, budgets_q, n, mode_start)
        mode_start += n
        shard_cfg = Config(shard_raw)
        expected[k] = rays.sidecar_metadata(shard_cfg, 1)
        shard_path = os.path.join(sdir, "rays.jsonl.gz")
        complete = _capillary_done(shard_path, expected[k])
        shard_sidecar = rays.metadata_path(shard_path)
        if ((os.path.lexists(shard_path) or os.path.lexists(shard_sidecar))
                and not complete):
            sys.exit(f"shard {k}: {shard_path} is incompatible or incomplete; "
                     "remove it and its sidecar manually before retrying")
        if not complete:
            conflicts = [
                os.path.join(sdir, name) for name in ("config.yaml", "trace.log")
                if os.path.lexists(os.path.join(sdir, name))
            ]
            if conflicts:
                sys.exit(f"shard {k}: auxiliary files already exist: "
                         f"{conflicts}; remove them manually or choose another "
                         "--out")
        shards.append((k, n, sdir, shard_raw, complete))

    procs = {}
    for k, n, sdir, shard_raw, complete in shards:
        if complete:
            print(f"shard {k}: complete, skipping", flush=True)
            continue
        os.makedirs(sdir, exist_ok=True)
        scfg = os.path.join(sdir, "config.yaml")
        with open(scfg, "x") as fh:
            yaml.safe_dump(shard_raw, fh, sort_keys=False)
        log = open(os.path.join(sdir, "trace.log"), "x")
        cmd = [sys.executable, "-m", "formula.capsysred", scfg,
               "-o", sdir, "--trace"]
        procs[k] = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"shard {k}: tracing {n} modes from mode "
              f"{shard_raw['capillary']['source']['mode_start']} (pid {procs[k].pid})",
              flush=True)

    fails = [k for k, p in procs.items() if p.wait() != 0]
    if fails:
        sys.exit(f"shards failed: {fails} (see shard-*/trace.log)")
    for k in range(args.jobs):
        if not _capillary_done(
                os.path.join(args.out, f"shard-{k}", "rays.jsonl.gz"),
                expected[k]):
            sys.exit(f"shard {k}: record incomplete")

    if args.no_merge:
        recs = " ".join(os.path.join(args.out, f"shard-{k}", "rays.jsonl.gz")
                        for k in range(args.jobs))
        print(f"shards complete, merge skipped; --replay {recs}", flush=True)
        return

    counts = {scene: 0 for scene in merged_sidecar["budgets"]}
    gmode = -1
    # newline="\n": no \r\n translation on Windows text-mode writes
    with gzip.open(dst_path, "xt", encoding="utf-8", newline="\n") as dst:
        dst.write("{}\n")
        for k in range(args.jobs):
            rec = os.path.join(args.out, f"shard-{k}", "rays.jsonl.gz")
            last = None
            for line in rays._body_lines(rec):
                if '"stage": "capillary"' in line:
                    i, j = _mode_span(line)
                    if line[i:j] != last:
                        gmode += 1
                        last = line[i:j]
                    dst.write(line[:i] + str(gmode) + line[j:])
                    counts["capillary"] += 1
                elif (k == 0 and "free" in counts
                      and '"stage": "free"' in line):
                    dst.write(line)
                    counts["free"] += 1
            print(f"shard {k}: merged; modes {gmode + 1}, "
                  f"capillary rows {counts['capillary']:,}", flush=True)
        for scene, n in counts.items():
            if n:
                dst.write(json.dumps({"scene_end": scene, "rows": n}) + "\n")

    done, clean = rays._scan_rows(dst_path)
    expected_counts = {
        scene: budget[0] * budget[1]
        for scene, budget in merged_sidecar["budgets"].items()
    }
    ok = (clean and counts == expected_counts and done == expected_counts
          and gmode + 1 == budgets_q["capillary"][0])
    if ok:
        rays.write_metadata(dst_path, merged_sidecar)
        if not args.keep_shards:
            for k in range(args.jobs):
                rec = os.path.join(args.out, f"shard-{k}", "rays.jsonl.gz")
                os.remove(rec)
                sidecar = rays.metadata_path(rec)
                if os.path.exists(sidecar):
                    os.remove(sidecar)
    print(f"merged {dst_path}: modes {gmode + 1}/{budgets_q['capillary'][0]}, "
          f"scenes {done}, scan {'clean' if clean else 'DIRTY'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

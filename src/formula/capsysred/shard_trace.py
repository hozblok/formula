"""Shard-parallel tracing: split the capillary modes across processes and
merge the records into one canonical rays.jsonl.gz.

    python -m formula.capsysred.shard_trace config.yaml -o out/RUN \
        --jobs 7 [--quick N] [--keep-shards] [--no-merge]

Shard k traces its slice of the modes under seed+k into out/RUN/shard-k/
(derived config and log sit next to the record); shards with a complete
record are skipped on restart. The merge writes the canonical empty
preamble, copies the free and lloyd scenes from shard 0 (seed+0 keeps
their canonical streams), renumbers the capillary modes globally,
recomputes the trailers, scans the body, and publishes the structured
metadata sidecar. Consumers then run the ORIGINAL config (with the same
--quick) against out/RUN and reuse the file.

Reproducibility: the seed set {seed .. seed+jobs-1} plus this command —
not bit-equal to a sequential trace (modes are iid across seeds).
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


def _shard_raw(raw, budgets_q, k, cap_modes):
    shard = copy.deepcopy(raw)
    shard["seed"] = int(raw.get("seed", 12345)) + k
    shard.setdefault("free", {}).setdefault("source", {})
    shard["free"]["source"]["n_modes"], shard["free"]["source"]["n_rays"] = budgets_q["free"]
    shard.setdefault("lloyd", {}).setdefault("source", {})
    shard["lloyd"]["source"]["n_modes"], shard["lloyd"]["source"]["n_rays"] = budgets_q["lloyd"]
    cap = shard["capillary"]["source"]
    cap["n_modes"], cap["n_rays"] = cap_modes, budgets_q["capillary"][1]
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
    chunks = _chunks(budgets_q["capillary"][0], args.jobs)
    os.makedirs(args.out, exist_ok=True)

    shards = []
    expected = {}
    for k, n in enumerate(chunks):
        sdir = os.path.join(args.out, f"shard-{k}")
        shard_raw = _shard_raw(raw, budgets_q, k, n)
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
        print(f"shard {k}: tracing {n} modes under seed {raw.get('seed', 12345) + k} "
              f"(pid {procs[k].pid})", flush=True)

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

    counts = {"free": 0, "lloyd": 0, "capillary": 0}
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
                elif k == 0 and '"stage": "free"' in line:
                    dst.write(line)
                    counts["free"] += 1
                elif k == 0 and '"stage": "lloyd"' in line:
                    dst.write(line)
                    counts["lloyd"] += 1
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

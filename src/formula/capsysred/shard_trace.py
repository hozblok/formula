"""Shard-parallel tracing: split the capillary modes across processes and
merge the records into one canonical rays.jsonl.gz.

    python -m formula.capsysred.shard_trace config.yaml -o out/RUN \
        --jobs 7 [--quick N] [--keep-shards]

Shard k traces its slice of the modes under seed+k into out/RUN/shard-k/
(derived config and log sit next to the record); shards with a complete
record are skipped on restart. The merge writes the canonical header
(fingerprint of the untouched config, its budgets at --quick), copies the
free and lloyd scenes from shard 0 (seed+0 keeps their canonical
streams), renumbers the capillary modes globally, recomputes the
trailers, and engine-scans the result. Consumers then run the ORIGINAL
config (with the same --quick) against out/RUN and reuse the file.

Reproducibility: the seed set {seed .. seed+jobs-1} plus this command —
not bit-equal to a sequential trace (modes are iid across seeds).
Each tracer holds ~1-2 GB: pick --jobs for the RAM, not just the cores.
Disk peak: all shard records + the growing merge; consumed shards are
deleted unless --keep-shards.
"""
import argparse
import copy
import gzip
import json
import os
import subprocess
import sys

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


def _capillary_done(path):
    if not os.path.exists(path):
        return False
    meta, done, clean = rays.scan(path)
    return meta is not None and clean and "capillary" in done


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
    args = ap.parse_args(argv)

    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg = Config(raw)
    if cfg.capillary is None:
        sys.exit("config has no capillary scene")
    budgets_q = rays.budgets(cfg, max(1, args.quick))
    chunks = _chunks(budgets_q["capillary"][0], args.jobs)
    os.makedirs(args.out, exist_ok=True)

    procs = {}
    for k, n in enumerate(chunks):
        sdir = os.path.join(args.out, f"shard-{k}")
        if _capillary_done(os.path.join(sdir, "rays.jsonl.gz")):
            print(f"shard {k}: complete, skipping", flush=True)
            continue
        os.makedirs(sdir, exist_ok=True)
        scfg = os.path.join(sdir, "config.yaml")
        with open(scfg, "w") as fh:
            yaml.safe_dump(_shard_raw(raw, budgets_q, k, n), fh, sort_keys=False)
        log = open(os.path.join(sdir, "trace.log"), "w")
        procs[k] = subprocess.Popen(
            [sys.executable, "-m", "formula.capsysred", scfg, "-o", sdir, "--trace"],
            stdout=log, stderr=subprocess.STDOUT)
        print(f"shard {k}: tracing {n} modes under seed {raw.get('seed', 12345) + k} "
              f"(pid {procs[k].pid})", flush=True)

    fails = [k for k, p in procs.items() if p.wait() != 0]
    if fails:
        sys.exit(f"shards failed: {fails} (see shard-*/trace.log)")
    for k in range(args.jobs):
        if not _capillary_done(os.path.join(args.out, f"shard-{k}", "rays.jsonl.gz")):
            sys.exit(f"shard {k}: record incomplete")

    meta = {"format": rays.FORMAT, "geometry": rays.fingerprint(cfg),
            "budgets": budgets_q}
    if cfg.lean_rays:
        meta["lean"] = True
    dst_path = os.path.join(args.out, "rays.jsonl.gz")
    counts = {"free": 0, "lloyd": 0, "capillary": 0}
    gmode = -1
    # newline="\n": no \r\n translation on Windows text-mode writes
    with gzip.open(dst_path, "wt", encoding="utf-8", newline="\n") as dst:
        dst.write(json.dumps(meta) + "\n")
        for k in range(args.jobs):
            rec = os.path.join(args.out, f"shard-{k}", "rays.jsonl.gz")
            last = None
            with gzip.open(rec, "rt", encoding="utf-8") as fh:
                for line in fh:
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
            if not args.keep_shards:
                os.remove(rec)
        for scene, n in counts.items():
            if n:
                dst.write(json.dumps({"scene_end": scene, "rows": n}) + "\n")

    _, done, clean = rays.scan(dst_path)
    ok = clean and gmode + 1 == budgets_q["capillary"][0]
    print(f"merged {dst_path}: modes {gmode + 1}/{budgets_q['capillary'][0]}, "
          f"scenes {done}, scan {'clean' if clean else 'DIRTY'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

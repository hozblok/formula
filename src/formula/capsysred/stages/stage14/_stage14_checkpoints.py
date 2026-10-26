"""Stage-14 range checkpoints: the on-disk contract of a parallel cache build.

A build decodes disjoint mode ranges [m0, m1); each completed range is the
triple rows+aggregates+manifest, committed by the manifest (written last,
carrying byte sizes and sha256 of both files as the worker wrote them from
memory).  The partition is a pure function of n_modes — a range file is
valid independently of the split — so resumes survive any change among
jobs > 1 and jobs only caps pool concurrency (jobs = 1 selects the strict
serial pass, which refuses a leftover partial).  Content sha256 is verified
against the manifests at assembly time; this module knows nothing about
stage14 proper.
"""

import json
import os

# Useful-concurrency ceiling: the fixed number of range slots.
RANGE_SLOTS = 128


def mode_ranges(n_modes: int) -> list[tuple[int, int]]:
    """Deterministic balanced partition of [0, n_modes) into contiguous ranges."""
    slots = min(RANGE_SLOTS, n_modes)
    base, extra = divmod(n_modes, slots)
    ranges, start = [], 0
    for i in range(slots):
        size = base + (1 if i < extra else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def range_rows_name(m0: int, m1: int) -> str:
    return f"rows-m{m0:06d}-{m1:06d}.f64"


def range_agg_name(m0: int, m1: int) -> str:
    return f"agg-m{m0:06d}-{m1:06d}.bin"


def range_manifest_name(m0: int, m1: int) -> str:
    return f"done-m{m0:06d}-{m1:06d}.json"


def read_range_manifest(partial: str, m0: int, m1: int) -> dict:
    """The range triple's commit record: bytes and sha256 of rows/aggregates
    as the worker wrote them from memory."""
    path = os.path.join(partial, range_manifest_name(m0, m1))
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    for kind in ("rows", "aggregates"):
        entry = manifest.get(kind) if isinstance(manifest, dict) else None
        if (not isinstance(entry, dict)
                or type(entry.get("bytes")) is not int
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64):
            raise ValueError(f"{path}: invalid range manifest")
    return manifest


def range_done(partial: str, m0: int, m1: int,
               rows_size: int, agg_size: int) -> bool:
    """Committed = manifest present and both files at the expected sizes;
    the content sha256 is verified against the manifests during assembly."""
    expected = {"rows": (rows_size, range_rows_name(m0, m1)),
                "aggregates": (agg_size, range_agg_name(m0, m1))}
    try:
        manifest = read_range_manifest(partial, m0, m1)
        for kind, (size, name) in expected.items():
            if (manifest[kind]["bytes"] != size
                    or os.path.getsize(os.path.join(partial, name)) != size):
                return False
    except (OSError, ValueError):
        return False
    return True

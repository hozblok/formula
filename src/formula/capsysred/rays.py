"""Ray recordings: the contract shared by trace_v3 (the writer) and the
stages (readers).  A recording is a v3 per-mode archive — a directory with
fingerprint, index and one gzip section per mode and ray range (rays_v3).
Stages never trace: a run without a recording is an error.

Rows carry the PRE-threshold fate — amplitude_min is physics and every
consumer applies its own.  Screen-fate rows add x, y, dx, dy and the refl
bounce points in float64 (enough for the float estimators, stages
2/7/8/11/14); opl/sins stay full-precision strings.

trace.lean_rays writes opl/sins as float64 json numbers and drops refl
(the fingerprint gains "lean": true): the estimators and rescreen are
bit-identical (they float() these fields anyway), the archive shrinks ~4x
on bounce-heavy scenes; the beamlet stage refuses such an archive
(require_full_rows).
"""

import copy
import enum
import json
import math
import random

from . import rays_v3
from .shared.types import RayRecord


class SceneSeed(enum.IntEnum):
    """Per-scene rng-stream tag: its lowercase name keys the lattice streams."""
    FREE = 2         # no-optics scene (stages 2, 7, 8, 11)
    CAPILLARY = 4    # capillary (stages 7, 8, 11, 14)
    VALIDATE = 9     # stage 9 hit-method cross-check


RNG_SCHEME = "lattice-v1"


def stream_rng(seed: int, tag: SceneSeed, *parts) -> random.Random:
    """The rng of one named stream: ``"<seed>/<scene>[/<mode>]"``.

    Mode m of a scene draws its origin first, then 3 draws per ray, so the
    ray lattice (seed, scene, mode, ray) is addressable: shards, top-ups
    and single-piece traces produce identical rays.  Strings seed Python's
    Random through SHA-512, so distinct keys never alias.
    """
    key = f"{seed}/{tag.name.lower()}" + "".join(f"/{part}" for part in parts)
    return random.Random(key)


def geometry_metadata(cfg) -> dict:
    """YAML-native inputs that completely identify a traced ray stream.

    Spectrum and material are absent because rays are energy-free.  Extra
    capillary screens are re-binned post-trace and therefore do not count.
    The JSON round-trip detaches the result from ``cfg.raw`` and normalizes
    string enums and other string-compatible values.
    """
    geo = {k: cfg.raw[k] for k in ("seed", "precision", "screen")}
    if "free" in cfg.raw:
        geo["free"] = cfg.raw["free"]
    if "capillary" in cfg.raw:
        geo["capillary"] = {
            k: v for k, v in cfg.raw["capillary"].items()
            if k != "screens"
        }
    geo["max_bounces"] = cfg.max_bounces
    return json.loads(json.dumps(geo, sort_keys=True, default=str))


def geometry_core(geometry: dict) -> dict:
    """Recording geometry without the per-scene budgets."""
    core = copy.deepcopy(geometry)
    for scene in ("capillary", "free"):
        source = (core.get(scene) or {}).get("source")
        if isinstance(source, dict):
            source.pop("n_modes", None)
            source.pop("n_rays", None)
    return core


def metadata_equal(left: dict, right: dict) -> bool:
    """Strict, order-independent comparison preserving JSON scalar types."""
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
    return canonical(left) == canonical(right)


class RaysReader:
    """Read-only v3 archive (a directory) for --replay: stages consume,
    nothing traces or writes. Geometry fingerprint is NOT checked — replaying
    on a different spectrum/material/precision is the point; budgets are
    (scene_stream)."""

    readonly = True

    def __init__(self, path):
        if not rays_v3.is_v3(path):
            raise ValueError(f"{path}: not a v3 rays archive directory")
        self._index = rays_v3.load_index(path)
        meta = rays_v3.metadata(path)
        done = {scene: math.prod(budget)
                for scene, budget in meta["budgets"].items()}
        self.path, self.meta, self.done = path, meta, done

    @property
    def index(self):
        """The v3 section index."""
        return self._index

    def scene_records(self, scene):
        return rays_v3.scene_records(self.path, self._index, scene)

    def write(self, scene, rec):
        pass

    def finish_scene(self, scene):
        pass

    def close(self):
        pass


class MultiRaysReader:
    """Union of read-only rays archives for --replay: scenes stream archive
    by archive with mode ids offset, so k recordings act as one. Per scene
    the mode counts add up; n_rays must agree across archives."""

    readonly = True

    def __init__(self, paths):
        self.parts = [RaysReader(p) for p in paths]
        self.path = " + ".join(p.path for p in self.parts)
        self.meta = dict(self.parts[0].meta)
        if any(part.meta.get("lean") for part in self.parts):
            self.meta["lean"] = True
        else:
            self.meta.pop("lean", None)
        scenes = set(self.parts[0].done)
        for part in self.parts[1:]:
            scenes &= set(part.done)
        budgets, done = {}, {}
        for sc in scenes:
            ms, rs = zip(*(p.meta["budgets"][sc] for p in self.parts))
            if len(set(rs)) != 1:
                raise ValueError(f"--replay union: scene {sc!r} n_rays differ "
                                 f"across files: {sorted(set(rs))}")
            budgets[sc] = [sum(ms), rs[0]]
            done[sc] = sum(p.done[sc] for p in self.parts)
        self.meta["budgets"], self.done = budgets, done

    def scene_records(self, scene):
        offset = 0
        for part in self.parts:
            for rec in part.scene_records(scene):
                yield rec._replace(mode=rec.mode + offset) if offset else rec
            offset += part.meta["budgets"][scene][0]

    def write(self, scene, rec):
        pass

    def finish_scene(self, scene):
        pass

    def close(self):
        pass


def row_of(scene: str, rec: RayRecord, lean: bool) -> dict:
    """The row of one record: key order is part of the file contract."""
    row = {"stage": scene, "mode": rec.mode, "ray": rec.ray,
           "fate": rec.fate, "pixel": rec.pixel}
    if lean:
        row["opl"] = float(rec.opl)
        row["sins"] = [float(s) for s in rec.sins]
    else:
        row["opl"] = str(rec.opl)
        row["sins"] = [str(s) for s in rec.sins]
    if rec.fate == "screen":
        row["x"], row["y"] = float(rec.point[0]), float(rec.point[1])
        row["dx"] = float(rec.direction[0])
        row["dy"] = float(rec.direction[1])
        if rec.refl and not lean:
            row["refl"] = [[float(c) for c in p] for p in rec.refl]
    return row


def require_full_rows(rays, rays_from: str, what: str):
    """Consumers of refl / full-precision opl+sins refuse a lean rays file."""
    if rays_from == "file" and rays is not None and rays.meta.get("lean"):
        raise ValueError(f"{what}: rays file is lean (trace.lean_rays) — no "
                         "refl / full-precision rows; re-trace without it")


def rescreen(records, z0f: float, grid):
    """Screen-fate records re-projected from the plane z0 onto grid's plane:
    straight vacuum flight, pixel re-binned, opl extended in float64."""
    zf = float(grid.z)
    for rec in records:
        if rec.fate != "screen":
            yield rec
            continue
        dxf, dyf, dzf = (float(c) for c in rec.direction)
        if dzf <= 0.0:               # never reaches another plane
            yield rec._replace(pixel=None)
            continue
        s = (zf - z0f) / dzf
        x = float(rec.point[0]) + dxf * s
        y = float(rec.point[1]) + dyf * s
        yield rec._replace(point=(x, y, zf), direction=(dxf, dyf, dzf),
                           pixel=grid.pixel((x, y)), opl=float(rec.opl) + s)


def _counted(records, expected: int, scene: str, path: str):
    """Guard on full consumption: a rewritten file (foreign key order defeats
    the prefix skip) or a thinned recording must fail loudly, not thin the
    statistics silently."""
    n = 0
    for rec in records:
        n += 1
        yield rec
    if n != expected:
        raise ValueError(f"{path}: scene {scene!r} yielded {n} rows of "
                         f"{expected} recorded — file rewritten or truncated")


def scene_stream(sim, scene, src_cfg, scr_cfg, optic, aim_factory):
    """(records, "file"): the run's reader must hold the scene at the
    configured budgets; stages never trace (trace_v3 is the recording
    command).  ``scr_cfg``/``optic``/``aim_factory`` are kept for the stage
    call sites and identify the scene only."""
    n_modes, n_rays = src_cfg.budget()
    w = sim.rays
    if w is None:
        raise ValueError(
            f"scene {scene!r}: no rays recording to read; stages never trace — "
            "run trace_v3 first or pass --replay")
    if scene not in w.done:
        raise ValueError(f"scene {scene!r} is not in {w.path} "
                         f"(recorded: {sorted(w.done)})")
    if w.meta["budgets"].get(scene) != [n_modes, n_rays]:
        raise ValueError(
            f"scene {scene!r} budgets {w.meta['budgets'].get(scene)} != config "
            f"{[n_modes, n_rays]} — match n_modes/n_rays to the recording")
    if w.done[scene] != n_modes * n_rays:
        raise ValueError(
            f"scene {scene!r} holds {w.done[scene]} rows, "
            f"budgets promise {n_modes * n_rays} — thinned recording")
    return _counted(w.scene_records(scene), w.done[scene], scene, w.path), "file"

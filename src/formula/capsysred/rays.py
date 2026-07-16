"""rays.jsonl v2: one traced-geometry stream shared by the stages.

The tracer's work (geometry, no physics) is recorded once and reused: a
stage consumes the file instead of re-tracing when the geometry fingerprint,
the per-scene budgets and sample_every == 1 all match; otherwise it traces
and tees the records into the run's writer. Rows carry the PRE-threshold
fate — amplitude_min is physics and every consumer applies its own.

Layout: meta line {"format": 2, ...}, one row per ray, and a
{"scene_end": scene, "rows": n} trailer per completed scene. Screen-fate
rows add x, y, dx, dy and the refl bounce points in float64 (enough for the
float estimators, stages 7/8/10/11); opl/sins stay full-precision strings
for the Number path (--replay of stages 2/4/6). rays_gzip writes/reads
.jsonl.gz transparently.
"""

import gzip
import hashlib
import json
import math
import os
import random

from .native import make_tracer
from .screen import ScreenGrid
from .source import Source
from .types import RayRecord, ray_record

FORMAT = 2


def fingerprint(cfg) -> str:
    """Geometry-only config fingerprint: spectrum/material changes keep the
    file valid (rays are energy-free), geometry/seed changes invalidate it.
    Extra capillary screens are re-binned post-trace, so they don't count."""
    geo = {k: cfg.raw[k] for k in ("seed", "precision", "source", "screen",
                                   "free", "lloyd", "capillary")}
    geo["capillary"] = {k: v for k, v in geo["capillary"].items()
                        if k != "screens"}
    geo["max_bounces"] = cfg.max_bounces
    geo["engine_method"] = cfg.engine_method
    raw = json.dumps(geo, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def budgets(cfg, quick: int) -> dict:
    """Scene -> [n_modes, n_rays], the same clamps as the stage loops."""
    def per(src):
        return [max(2, src.n_modes // quick), max(20, src.n_rays // quick)]
    out = {"free": per(cfg.free_source), "lloyd": per(cfg.lloyd.source)}
    if cfg.capillary is not None:
        out["capillary"] = per(cfg.capillary.source)
    return out


def _open(path, mode):
    return (gzip.open(path, mode + "t", encoding="utf-8")
            if path.endswith(".gz") else open(path, mode, encoding="utf-8"))


def _lines(path):
    """Complete rows; tolerates a live gzip tail (no end-of-stream marker)."""
    with _open(path, "r") as fh:
        try:
            for line in fh:
                if line.endswith("\n"):
                    yield line
        except EOFError:
            return


def scan(path):
    """(meta, {scene: rows} complete, no-partial-scenes flag)."""
    meta, counts, trailers = None, {}, {}
    for i, line in enumerate(_lines(path)):
        try:
            row = json.loads(line)
        except ValueError:
            return None, {}, False
        if i == 0:
            if row.get("format") != FORMAT:
                return None, {}, False
            meta = row
        elif "scene_end" in row:
            trailers[row["scene_end"]] = row["rows"]
        elif "stage" in row:
            counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    done = {s: n for s, n in trailers.items() if counts.get(s, 0) == n}
    clean = set(counts) == set(trailers) and len(done) == len(trailers)
    return meta, done, clean


class RaysFile:
    """The run's rays file: appends to a matching existing file, else starts
    fresh; `done` scenes are complete and readable back mid-run."""

    def __init__(self, path, cfg, quick):
        self.path = path
        self.meta = {"format": FORMAT, "geometry": fingerprint(cfg),
                     "sample_every": cfg.sample_every,
                     "budgets": budgets(cfg, quick)}
        self.done = {}
        mode = "w"
        if os.path.exists(path):
            meta, done, clean = scan(path)
            if meta is not None and clean and all(
                    meta.get(k) == v for k, v in self.meta.items()):
                mode, self.done = "a", done
        self._fh = _open(path, mode)
        if mode == "w":
            self._fh.write(json.dumps(self.meta) + "\n")
        self._scene, self._count = None, 0

    def write(self, scene: str, rec: RayRecord):
        if scene in self.done or rec.ray % self.meta["sample_every"]:
            return
        if scene != self._scene:
            self._scene, self._count = scene, 0
        row = {"stage": scene, "mode": rec.mode, "ray": rec.ray,
               "fate": rec.fate, "pixel": rec.pixel, "opl": str(rec.opl),
               "sins": [str(s) for s in rec.sins]}
        if rec.fate == "screen":
            row["x"], row["y"] = float(rec.point[0]), float(rec.point[1])
            row["dx"] = float(rec.direction[0])
            row["dy"] = float(rec.direction[1])
            if rec.refl:
                row["refl"] = [[float(c) for c in p] for p in rec.refl]
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._count += 1

    def finish_scene(self, scene: str):
        if scene in self.done:
            return
        n = self._count if self._scene == scene else 0
        self._fh.write(json.dumps({"scene_end": scene, "rows": n}) + "\n")
        self._fh.flush()
        self.done[scene] = n
        self._scene, self._count = None, 0

    def close(self):
        self._fh.close()


def _file_records(path, scene):
    """Scene rows -> RayRecords. opl/sins stay strings (float() at use);
    point z is unknown-by-design (nan), direction dz rebuilt (unit, dz > 0)."""
    for line in _lines(path):
        row = json.loads(line)
        if row.get("stage") != scene:
            continue
        point = direction = None
        if "x" in row:
            point = (row["x"], row["y"], float("nan"))
            dx, dy = row["dx"], row["dy"]
            direction = (dx, dy, math.sqrt(max(1.0 - dx * dx - dy * dy, 0.0)))
        yield RayRecord(row["mode"], row["ray"], row["fate"], row["pixel"],
                        point, direction, row["opl"], tuple(row["sins"]),
                        tuple(tuple(p) for p in row.get("refl", ())))


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


def _traced_records(sim, scene, src_cfg, scr_cfg, optic, aim_factory,
                    seed_offset, quick):
    """The stage-2/6 rng stream; every record is teed into the run's writer."""
    cfg = sim.cfg
    rng = random.Random(cfg.seed * 1000003 + seed_offset)
    source = Source(src_cfg, rng)
    screen = ScreenGrid(scr_cfg)
    n_modes = max(2, src_cfg.n_modes // quick)
    n_rays = max(20, src_cfg.n_rays // quick)
    aim = aim_factory(source, screen, rng)
    tracer = make_tracer(optic)
    writer = sim.rays
    for mode in range(n_modes):
        origin = source.mode_origin()
        for ray in range(n_rays):
            tr = tracer(origin, aim(origin), optic, screen.z, cfg.max_bounces)
            rec = ray_record(tr, screen, mode, ray, tr.fate)
            if writer is not None:
                writer.write(scene, rec)
            yield rec
    if writer is not None:
        writer.finish_scene(scene)


def scene_stream(sim, scene, src_cfg, scr_cfg, optic, aim_factory,
                 seed_offset: int, quick: int):
    """(records, "file"|"trace"): the file when the run's rays file already
    holds this scene at these budgets, else tracing (teeing into the file)."""
    n_modes = max(2, src_cfg.n_modes // quick)
    n_rays = max(20, src_cfg.n_rays // quick)
    w = sim.rays
    if (w is not None and scene in w.done and w.meta["sample_every"] == 1
            and w.meta["budgets"].get(scene) == [n_modes, n_rays]):
        return _file_records(w.path, scene), "file"
    return _traced_records(sim, scene, src_cfg, scr_cfg, optic, aim_factory,
                           seed_offset, quick), "trace"

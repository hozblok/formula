"""rays.jsonl v2: one traced-geometry stream shared by the stages.

The tracer's work (geometry, no physics) is recorded once and reused: a
stage consumes the file instead of re-tracing when the geometry fingerprint
and the per-scene budgets match; otherwise it traces and tees the records
into the run's writer. Rows carry the PRE-threshold
fate — amplitude_min is physics and every consumer applies its own.

Layout: meta line {"format": 2, ...}, one row per ray, and a
{"scene_end": scene, "rows": n} trailer per completed scene. Screen-fate
rows add x, y, dx, dy and the refl bounce points in float64 (enough for the
float estimators, stages 7/8/10/11); opl/sins stay full-precision strings
for the Number path (--replay of stages 2/4/6). The file is gzipped
(rays.jsonl.gz).

trace.lean_rays writes opl/sins as float64 json numbers and drops refl
(meta gains "lean": true): stage 10 and rescreen are bit-identical (they
float() these fields anyway), the file shrinks ~4x on bounce-heavy scenes;
the Number path and the beamlet stage refuse such a file (require_full_rows).
"""

import enum
import gzip
import hashlib
import json
import math
import os
import random
import tempfile

import yaml

from .native import make_tracer
from .screen import ScreenGrid
from .source import Source
from .types import RayRecord, ray_record

FORMAT = 2
METADATA_NAME = "rays-fingerprint.yaml"

# Mixes cfg.seed with a per-scene offset into an independent rng stream.
_SCENE_SEED_STRIDE = 1000003


class SceneSeed(enum.IntEnum):
    """Per-scene rng-stream tag added to cfg.seed*_SCENE_SEED_STRIDE; stages
    reusing a scene's rays pass its tag, stage 9 gets its own."""
    FREE = 2         # no-optics scene (stages 2, 7, 8, 11, 12)
    LLOYD = 3        # Lloyd mirror (stage 4)
    CAPILLARY = 4    # capillary (stages 6, 7, 8, 10, 11)
    VALIDATE = 9     # stage 9 hit-method cross-check


def geometry_metadata(cfg) -> dict:
    """YAML-native inputs that completely identify a traced ray stream.

    Spectrum and material are absent because rays are energy-free.  Extra
    capillary screens are re-binned post-trace and therefore do not count.
    The JSON round-trip detaches the result from ``cfg.raw`` and normalizes
    string enums and other string-compatible values.
    """
    geo = {k: cfg.raw[k] for k in ("seed", "precision", "source", "screen",
                                   "free", "lloyd", "capillary")}
    geo["capillary"] = {k: v for k, v in geo["capillary"].items()
                        if k != "screens"}
    geo["max_bounces"] = cfg.max_bounces
    return json.loads(json.dumps(geo, sort_keys=True, default=str))


def fingerprint(cfg) -> str:
    """Compact digest of :func:`geometry_metadata` for legacy headers."""
    raw = json.dumps(geometry_metadata(cfg), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def budgets(cfg, quick: int) -> dict:
    """Scene -> [n_modes, n_rays], the same clamps as the stage loops."""
    def per(src):
        return list(src.budget(quick))
    out = {"free": per(cfg.free_source), "lloyd": per(cfg.lloyd.source)}
    if cfg.capillary is not None:
        out["capillary"] = per(cfg.capillary.source)
    return out


def metadata(cfg, quick: int) -> dict:
    """Legacy first-line metadata describing a rays recording."""
    meta = {"format": FORMAT, "geometry": fingerprint(cfg),
            "budgets": budgets(cfg, quick)}
    if cfg.lean_rays:
        meta["lean"] = True
    return meta


def sidecar_metadata(cfg, quick: int) -> dict:
    """Structured metadata for ``rays-fingerprint.yaml``."""
    meta = metadata(cfg, quick)
    meta["geometry"] = geometry_metadata(cfg)
    return meta


def metadata_equal(left: dict, right: dict) -> bool:
    """Strict, order-independent comparison preserving JSON scalar types."""
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
    return canonical(left) == canonical(right)


def metadata_path(rays_path: str | os.PathLike) -> str:
    """Return the metadata sidecar beside a canonical rays recording."""
    return os.path.join(os.path.dirname(os.fspath(rays_path)), METADATA_NAME)


def read_metadata(rays_path: str | os.PathLike) -> dict:
    """Read and validate a rays metadata sidecar."""
    path = metadata_path(rays_path)
    try:
        with open(path, encoding="utf-8") as fh:
            meta = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid rays metadata YAML") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: rays metadata must be a mapping")
    return meta


def write_metadata(rays_path: str | os.PathLike, meta: dict,
                   force: bool = False) -> str:
    """Atomically write a sidecar, refusing a conflicting one by default."""
    if not isinstance(meta, dict):
        raise TypeError("rays metadata must be a mapping")
    path = metadata_path(rays_path)
    if os.path.exists(path):
        try:
            existing = read_metadata(rays_path)
        except (OSError, ValueError) as exc:
            if not force:
                raise ValueError(f"{path}: existing metadata is unreadable; "
                                 "remove it manually before retrying") from exc
        else:
            if metadata_equal(existing, meta):
                return path
        if not force:
            raise ValueError(f"{path}: metadata already exists and differs; "
                             "remove it manually before retrying")

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{METADATA_NAME}.", suffix=".tmp",
                               dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


def _open(path, mode):
    # newline="\n": no \r\n translation on Windows text-mode writes
    return (gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
            if path.endswith(".gz") else open(path, mode, encoding="utf-8",
                                              newline="\n"))


def _lines(path):
    """Complete rows; tolerates a live gzip tail (no end-of-stream marker)."""
    with _open(path, "r") as fh:
        try:
            for line in fh:
                if line.endswith("\n"):
                    yield line
        except EOFError:
            return


def scan(path, expected_meta=None):
    """(meta, {scene: rows} complete, no-partial-scenes flag).

    ``expected_meta`` lets a writer reject a mismatched recording after its
    first line, without scanning a potentially very large ray stream.
    """
    meta, counts, trailers = None, {}, {}
    for i, line in enumerate(_lines(path)):
        if i and line.startswith('{"stage": "'):
            # row fast path: the writer emits "stage" first, names are plain
            end = line.find('"', 11)
            if end > 0:
                scene = line[11:end]
                counts[scene] = counts.get(scene, 0) + 1
                continue
        try:
            row = json.loads(line)
        except ValueError:
            return None, {}, False
        if i == 0:
            if row.get("format") != FORMAT:
                return None, {}, False
            meta = row
            if expected_meta is not None and meta != expected_meta:
                return meta, {}, False
        elif "scene_end" in row:
            trailers[row["scene_end"]] = row["rows"]
        elif "stage" in row:
            counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    done = {s: n for s, n in trailers.items() if counts.get(s, 0) == n}
    clean = set(counts) == set(trailers) and len(done) == len(trailers)
    return meta, done, clean


class RaysReader:
    """Read-only rays file for --replay: stages consume, nothing traces or
    writes. Geometry fingerprint is NOT checked — replaying on a different
    spectrum/material/precision is the point; budgets are (scene_stream)."""

    readonly = True

    def __init__(self, path):
        meta, done, _ = scan(path)
        if meta is None:
            raise ValueError(
                f"{path}: not a rays.jsonl v2 file (v1 records predate the "
                "shared-stream format — re-trace to record a v2 file)")
        self.path, self.meta, self.done = path, meta, done

    def scene_records(self, scene):
        return _file_records(self.path, scene)

    def write(self, scene, rec):
        pass

    def finish_scene(self, scene):
        pass

    def close(self):
        pass


class MultiRaysReader:
    """Union of read-only rays files for --replay: scenes stream file by
    file with mode ids offset, so k recordings act as one. Per scene the
    mode counts add up; n_rays must agree across files."""

    readonly = True

    def __init__(self, paths):
        self.parts = [RaysReader(p) for p in paths]
        self.path = " + ".join(p.path for p in self.parts)
        self.meta = dict(self.parts[0].meta)
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
            for rec in _file_records(part.path, scene):
                yield rec._replace(mode=rec.mode + offset) if offset else rec
            offset += part.meta["budgets"][scene][0]

    def write(self, scene, rec):
        pass

    def finish_scene(self, scene):
        pass

    def close(self):
        pass


class RaysFile:
    """The run's rays file: append to a matching existing file or create one.

    An incompatible, incomplete, or unreadable existing file is never
    replaced unless ``force`` explicitly permits it.  ``done`` scenes are
    complete and readable back mid-run.
    """

    readonly = False

    def __init__(self, path, cfg, quick, force: bool = False):
        self.path = path
        self.sidecar_path = metadata_path(path)
        self.lean = bool(getattr(cfg, "lean_rays", False))
        self.meta = metadata(cfg, quick)
        self.done = {}
        # Exclusive creation closes the check/open race when force is absent.
        mode = "w" if force else "x"
        if os.path.exists(path):
            try:
                meta, done, clean = scan(path, expected_meta=self.meta)
            except (OSError, UnicodeError, ValueError, KeyError, IndexError,
                    AttributeError, TypeError) as exc:
                if not force:
                    raise ValueError(
                        f"{path}: existing rays file is unreadable; refusing "
                        "to overwrite it (pass --force to replace it)"
                    ) from exc
                meta, done, clean = None, {}, False
            compatible = (meta == self.meta and clean)
            if compatible:
                mode, self.done = "a", done
            elif not force:
                raise ValueError(
                    f"{path}: existing rays file is incomplete or its metadata "
                    "does not match this run; refusing to overwrite it "
                    "(pass --force to replace it)"
                )
        if mode in {"w", "x"} or os.path.exists(self.sidecar_path):
            write_metadata(path, sidecar_metadata(cfg, quick), force=force)
        self._fh = _open(path, mode)
        if mode in {"w", "x"}:
            self._fh.write(json.dumps(self.meta) + "\n")
        self.has_sidecar = os.path.exists(self.sidecar_path)
        self._scene, self._count = None, 0

    def write(self, scene: str, rec: RayRecord):
        if scene in self.done:
            return
        if scene != self._scene:
            self._scene, self._count = scene, 0
        row = {"stage": scene, "mode": rec.mode, "ray": rec.ray,
               "fate": rec.fate, "pixel": rec.pixel}
        if self.lean:
            row["opl"] = float(rec.opl)
            row["sins"] = [float(s) for s in rec.sins]
        else:
            row["opl"] = str(rec.opl)
            row["sins"] = [str(s) for s in rec.sins]
        if rec.fate == "screen":
            row["x"], row["y"] = float(rec.point[0]), float(rec.point[1])
            row["dx"] = float(rec.direction[0])
            row["dy"] = float(rec.direction[1])
            if rec.refl and not self.lean:
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
    # rows are written "stage"-first: skip foreign scenes without json.loads
    prefix = '{"stage": ' + json.dumps(scene) + ","
    for line in _lines(path):
        if not line.startswith(prefix):
            continue
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


def _traced_records(sim, scene, src_cfg, scr_cfg, optic, aim_factory,
                    seed_offset, quick):
    """The stage-2/6 rng stream; every record is teed into the run's writer."""
    cfg = sim.cfg
    rng = random.Random(cfg.seed * _SCENE_SEED_STRIDE + seed_offset)
    source = Source(src_cfg, rng)
    screen = ScreenGrid(scr_cfg)
    n_modes, n_rays = src_cfg.budget(quick)
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


def scene_stream(sim, scene, src_cfg, scr_cfg, optic, aim_factory,
                 seed_offset: int, quick: int):
    """(records, "file"|"trace"): the file when the run's rays file already
    holds this scene, complete and at these budgets, else tracing (teeing
    into the file)."""
    n_modes, n_rays = src_cfg.budget(quick)
    w = sim.rays
    if w is not None and w.readonly:   # --replay: the file is the only source
        if scene not in w.done:
            raise ValueError(f"--replay: scene {scene!r} is not in {w.path} "
                             f"(recorded: {sorted(w.done)})")
        if w.meta["budgets"].get(scene) != [n_modes, n_rays]:
            raise ValueError(
                f"--replay: scene {scene!r} budgets "
                f"{w.meta['budgets'].get(scene)} != config {[n_modes, n_rays]}"
                " — match n_modes/n_rays and --quick to the recording")
        if w.done[scene] != n_modes * n_rays:
            raise ValueError(
                f"--replay: scene {scene!r} holds {w.done[scene]} rows, "
                f"budgets promise {n_modes * n_rays} — thinned recording")
        return _counted(w.scene_records(scene), w.done[scene], scene,
                        w.path), "file"
    if (w is not None and scene in w.done
            and w.meta["budgets"].get(scene) == [n_modes, n_rays]
            and w.done[scene] == n_modes * n_rays):
        return _counted(_file_records(w.path, scene), w.done[scene], scene,
                        w.path), "file"
    return _traced_records(sim, scene, src_cfg, scr_cfg, optic, aim_factory,
                           seed_offset, quick), "trace"

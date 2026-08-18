"""rays.jsonl v2: one traced-geometry stream shared by the stages.

The tracer's work (geometry, no physics) is recorded once and reused: a
stage consumes the file instead of re-tracing when the geometry fingerprint
and the per-scene budgets match; otherwise it traces and tees the records
into the run's writer. Rows carry the PRE-threshold
fate — amplitude_min is physics and every consumer applies its own.

Layout: one ignored preamble line (new files write {}), one row per ray, and
a {"scene_end": scene, "rows": n} trailer per completed scene. Metadata is
stored exclusively in the adjacent rays-fingerprint.yaml; the preamble's
contents are never interpreted. Screen-fate
rows add x, y, dx, dy and the refl bounce points in float64 (enough for the
float estimators, stages 7/8/10/11); opl/sins stay full-precision strings for
the Number path (--replay of stages 2/10). The file is gzipped
(rays.jsonl.gz).

trace.lean_rays writes opl/sins as float64 json numbers and drops refl
(meta gains "lean": true): stage 10 and rescreen are bit-identical (they
float() these fields anyway), the file shrinks ~4x on bounce-heavy scenes;
the Number path and the beamlet stage refuse such a file (require_full_rows).

rays_v3 is the per-mode layout (a directory: fingerprint, index, one gzip
section per mode and ray range); RaysReader and Stage 14 accept either.
convert_rays_v3 converts a v2 file, topup_trace adds rays to existing modes.
"""

import enum
import gzip
import json
import math
import os
import random
import zlib

import yaml

from . import rays_v3
from .native import make_tracer
from .screen import ScreenGrid
from .source import Source
from .types import RayRecord, ray_record

FORMAT = 2
METADATA_NAME = "rays-fingerprint.yaml"

# Mixes cfg.seed with a per-scene offset into an independent rng stream.
_SCENE_SEED_STRIDE = 1000003


class SceneSeed(enum.IntEnum):
    """Per-scene rng-stream tag: its lowercase name keys the lattice streams;
    the integer is only used to replay legacy (sequential-v2) recordings,
    added to cfg.seed*_SCENE_SEED_STRIDE."""
    FREE = 2         # no-optics scene (stages 2, 7, 8, 11, 12)
    CAPILLARY = 4    # capillary (stages 6, 7, 8, 10, 11)
    CAPILLARY_TOPUP = 6   # legacy-archive tail substreams of topup_trace
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


def budgets(cfg, quick: int) -> dict:
    """Scene -> [n_modes, n_rays], the same clamps as the stage loops."""
    def per(src):
        return list(src.budget(quick))
    out = {}
    if cfg.free_source is not None:
        out["free"] = per(cfg.free_source)
    if cfg.capillary is not None:
        out["capillary"] = per(cfg.capillary.source)
    return out


def sidecar_metadata(cfg, quick: int) -> dict:
    """Structured metadata for ``rays-fingerprint.yaml``."""
    meta = {"format": FORMAT, "geometry": geometry_metadata(cfg),
            "budgets": budgets(cfg, quick), "rng": RNG_SCHEME}
    if cfg.lean_rays:
        meta["lean"] = True
    return meta


def metadata_equal(left: dict, right: dict) -> bool:
    """Strict, order-independent comparison preserving JSON scalar types."""
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
    return canonical(left) == canonical(right)


def metadata_path(rays_path: str | os.PathLike) -> str:
    """Return the metadata sidecar beside a canonical rays recording
    (inside a v3 archive directory)."""
    path = os.fspath(rays_path)
    if rays_v3.is_v3(path):
        return os.path.join(path, METADATA_NAME)
    return os.path.join(os.path.dirname(path), METADATA_NAME)


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


def write_metadata(rays_path: str | os.PathLike, meta: dict) -> str:
    """Create a sidecar without clobbering any existing metadata."""
    if not isinstance(meta, dict):
        raise TypeError("rays metadata must be a mapping")
    path = metadata_path(rays_path)
    if os.path.lexists(path):
        try:
            existing = read_metadata(rays_path)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{path}: existing metadata is unreadable; remove it "
                "manually or choose another output directory"
            ) from exc
        else:
            if metadata_equal(existing, meta):
                return path
        raise ValueError(
            f"{path}: metadata already exists and differs; remove it "
            "manually or choose another output directory"
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    created = False
    try:
        # Exclusive creation is the no-lock, no-clobber publication rule.
        # A concurrent writer wins or this call fails; neither overwrites the
        # other's metadata.
        with open(path, "x", encoding="utf-8", newline="\n") as fh:
            created = True
            yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=True)
            fh.flush()
            os.fsync(fh.fileno())
    except FileExistsError as exc:
        raise ValueError(
            f"{path}: metadata appeared concurrently; remove it manually "
            "or choose another output directory"
        ) from exc
    except BaseException:
        if created:
            try:
                os.remove(path)
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


def _body(path):
    """Return ``(has_preamble, lines)`` after skipping exactly one line.

    The preamble's contents are intentionally never parsed. New recordings
    write ``{}``. Only a complete line counts as a preamble, because
    :func:`_lines` omits an unterminated live tail.
    """
    lines = _lines(path)
    try:
        next(lines)
    except StopIteration:
        return False, ()
    return True, lines


def _body_lines(path):
    """Iterate archive body lines after the ignored preamble."""
    _, lines = _body(path)
    yield from lines


def _validate_stream_metadata(meta, path):
    """Validate the sidecar fields needed by every rays consumer."""
    sidecar = metadata_path(path)
    if type(meta.get("format")) is not int or meta["format"] != FORMAT:
        raise ValueError(f"{sidecar}: rays metadata format must be {FORMAT}")
    if not isinstance(meta.get("geometry"), dict):
        raise ValueError(f"{sidecar}: rays geometry metadata must be a mapping")
    scene_budgets = meta.get("budgets")
    if not isinstance(scene_budgets, dict):
        raise ValueError(f"{sidecar}: rays budgets must be a mapping")
    for scene, budget in scene_budgets.items():
        if (not isinstance(scene, str) or not isinstance(budget, list)
                or len(budget) != 2
                or any(type(value) is not int or value < 1
                       for value in budget)):
            raise ValueError(f"{sidecar}: invalid rays budget for {scene!r}")
    if "lean" in meta and meta["lean"] is not True:
        raise ValueError(f"{sidecar}: lean must be true when present")


def _scan_rows(path):
    """Strictly scan a closed archive and return ``(done, clean)``.

    Unlike the live-read helpers, this consumes the underlying stream
    directly: an unterminated text line is dirty, and gzip EOF/checksum
    failures propagate to the caller.
    """
    counts, trailers, validated_scenes = {}, {}, set()
    with _open(path, "r") as fh:
        preamble = next(fh, None)
        if preamble is None or not preamble.endswith("\n"):
            return {}, False
        for line in fh:
            if not line.endswith("\n"):
                return {}, False
            if line.startswith('{"stage": "'):
                # Row fast path: the writer emits "stage" first, names are
                # plain strings and the required keys in a fixed order.
                # Rewritten/non-canonical rows fall through to JSON parsing.
                end = line.find('"', 11)
                pos = end
                for marker in (', "mode": ', ', "ray": ', ', "fate": ',
                               ', "pixel": ', ', "opl": ', ', "sins": '):
                    pos = line.find(marker, pos + 1)
                    if pos < 0:
                        break
                if end > 0 and pos >= 0 and line.endswith("}\n"):
                    scene = line[11:end]
                    if scene in validated_scenes:
                        counts[scene] = counts.get(scene, 0) + 1
                        continue
            try:
                row = json.loads(line)
            except ValueError:
                return {}, False
            if not isinstance(row, dict):
                return {}, False
            if "scene_end" in row:
                trailers[row["scene_end"]] = row["rows"]
            elif "stage" in row:
                required = {"stage", "mode", "ray", "fate", "pixel", "opl",
                            "sins"}
                if (not required <= set(row)
                        or not isinstance(row["stage"], str)):
                    return {}, False
                if (row["fate"] == "screen"
                        and not {"x", "y", "dx", "dy"} <= set(row)):
                    return {}, False
                counts[row["stage"]] = counts.get(row["stage"], 0) + 1
                validated_scenes.add(row["stage"])
    done = {scene: rows for scene, rows in trailers.items()
            if counts.get(scene, 0) == rows}
    clean = set(counts) == set(trailers) and len(done) == len(trailers)
    return done, clean


def scan(path, expected_meta=None):
    """(meta, {scene: rows} complete, no-partial-scenes flag).

    Metadata comes exclusively from the adjacent sidecar. ``expected_meta``
    lets a writer reject a mismatch before scanning a potentially very large
    ray stream. The archive's first complete line is an ignored preamble.
    """
    meta = read_metadata(path)
    _validate_stream_metadata(meta, path)
    if expected_meta is not None and not metadata_equal(meta, expected_meta):
        return meta, {}, False
    done, clean = _scan_rows(path)
    # The sidecar defines the active scenes. This lets a structured metadata
    # update retire a scene without rewriting a very large immutable archive;
    # retired rows are still syntax/integrity checked by _scan_rows above.
    done = {scene: rows for scene, rows in done.items()
            if scene in meta["budgets"]}
    return meta, done, clean


class RaysReader:
    """Read-only rays file for --replay: stages consume, nothing traces or
    writes. Geometry fingerprint is NOT checked — replaying on a different
    spectrum/material/precision is the point; budgets are (scene_stream).
    A directory path is a v3 archive (rays_v3)."""

    readonly = True

    def __init__(self, path):
        self._index = None
        if rays_v3.is_v3(path):
            self._index = rays_v3.load_index(path)
            meta = rays_v3.metadata(path)
            done = {scene: math.prod(budget)
                    for scene, budget in meta["budgets"].items()}
        else:
            meta, done, clean = scan(path)
            if not clean:
                raise ValueError(
                    f"{path}: rays archive is incomplete or corrupt; remove it "
                    "manually or choose another recording"
                )
        self.path, self.meta, self.done = path, meta, done

    def scene_records(self, scene):
        if self._index is not None:
            return rays_v3.scene_records(self.path, self._index, scene)
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
    """The v2 row of one record: key order is part of the file contract."""
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


class RaysFile:
    """The run's rays file: append to a matching existing file or create one.

    An incompatible, incomplete, or unreadable existing file is never
    replaced. ``done`` scenes are complete and readable back mid-run. Only
    one writer may use an output directory at a time; concurrent appenders are
    intentionally unsupported without a lock protocol.
    """

    readonly = False

    def __init__(self, path, cfg, quick):
        self.path = path
        self.sidecar_path = metadata_path(path)
        self.lean = bool(getattr(cfg, "lean_rays", False))
        self.meta = sidecar_metadata(cfg, quick)
        self.done = {}
        if os.path.lexists(path):
            try:
                meta, done, clean = scan(path, expected_meta=self.meta)
            except (OSError, UnicodeError, ValueError, KeyError, IndexError,
                    AttributeError, TypeError, EOFError, zlib.error) as exc:
                raise ValueError(
                    f"{path}: existing rays file or metadata sidecar "
                    f"{self.sidecar_path} is missing, corrupt, or unreadable; "
                    "remove the existing result manually or choose another "
                    "output directory"
                ) from exc
            complete_rows_match_budgets = all(
                scene in meta["budgets"]
                and rows == math.prod(meta["budgets"][scene])
                for scene, rows in done.items()
            )
            compatible = (meta is not None
                          and metadata_equal(meta, self.meta) and clean
                          and complete_rows_match_budgets)
            if compatible:
                self.done = done
                self._fh = _open(path, "a")
            else:
                raise ValueError(
                    f"{path}: existing rays file is incomplete or its metadata "
                    "does not match this run; remove the existing result "
                    "manually or choose another output directory"
                )
        else:
            if os.path.lexists(self.sidecar_path):
                raise ValueError(
                    f"{self.sidecar_path}: metadata exists without {path}; "
                    "remove it manually or choose another output directory"
                )
            fh = None
            created = False
            try:
                # Exclusive creation closes the check/open race. Publish the
                # sidecar only after the archive has a complete preamble.
                fh = _open(path, "x")
                created = True
                fh.write("{}\n")
                fh.flush()
                write_metadata(path, self.meta)
            except BaseException:
                if fh is not None:
                    try:
                        fh.close()
                    except BaseException:
                        pass
                if created:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                raise
            self._fh = fh
        self.has_sidecar = True
        self._scene, self._count = None, 0

    def write(self, scene: str, rec: RayRecord):
        if scene in self.done:
            return
        if scene != self._scene:
            self._scene, self._count = scene, 0
        self._fh.write(json.dumps(row_of(scene, rec, self.lean),
                                  ensure_ascii=False) + "\n")
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
    for line in _body_lines(path):
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
    """One rng stream per global mode (lattice-v1); every record is teed
    into the run's writer."""
    cfg = sim.cfg
    source = Source(src_cfg, None)
    screen = ScreenGrid(scr_cfg)
    n_modes, n_rays = src_cfg.budget(quick)
    tracer = make_tracer(optic)
    writer = sim.rays
    for mode in range(n_modes):
        rng = stream_rng(cfg.seed, seed_offset, src_cfg.mode_start + mode)
        source.rng = rng
        origin = source.mode_origin()
        aim = aim_factory(source, screen, rng)
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

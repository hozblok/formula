"""Stage 14: exact disk-backed delete-one-mode coherence jackknife.

The ray archive is decoded once per cache miss.  Per-mode ``W/Ic`` rows are
written by the native store and every later classification reads only that
compact cache.  This module deliberately owns provenance, strict JSONL
validation, aggregate serialization and publication; the dense numerical
loops live in ``bindings_stage14.cpp``.
"""

from __future__ import annotations

from array import array
from collections import Counter
from contextlib import contextmanager
import copy
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

from ... import __version__, _formula
from .. import rays_v3, render
from .altcoh import FloatLineAmplitudes
from ..shared.progress import Progress
from ..rays import (_validate_stream_metadata, geometry_metadata,
                   metadata_equal, read_metadata)
from ..screen import ScatterRaster, ScreenGrid
from .stage14_flags import (FlagThresholds, PixelCounters, PixelStatistics,
                            serialize_pixel, validate_counters, validate_ref,
                            w_signal_status)
from ..shared.units import m_to_um
from ..shared.utils import durable_open


STAGE_ID = 14
CACHE_SCHEMA = 1
ESTIMATOR_VERSION = 1
CACHE_DIR = "stage14-cache"
RESULT_DIR = "stage14"
ROWS_NAME = "mode-rows.f64"
AGGREGATES_NAME = "aggregates.bin"
META_NAME = "meta.json"
AGG_MAGIC = b"CPS14AGG"
AGG_FORMAT = 1
STREAM_COUNTERS = ("emitted", "screen", "absorbed", "lost", "off_window",
                   "reflected_rays", "reflections")
RESULT_PAYLOAD_NAMES = (
    "mu-jack.jsonl",
    "14-capillary-jack-mu.svg",
    "14-capillary-jack-mu-map.svg",
    "14-capillary-jack-mu-err.svg",
    "14-capillary-jack-mu-flags.svg",
    "14a-capillary-jack-slice.svg",
    "14b-capillary-jack-intensity.svg",
    "14b-capillary-jack-intensity-log.svg",
    "14b-capillary-jack-intensity-slice.svg",
    "14b-capillary-jack-intensity-log-slice.svg",
    "14b-capillary-jack-density.svg",
    "14c-capillary-jack-overlay.svg",
    "14d-capillary-ray-scatter.svg",
    "14e-capillary-ref-passport.svg",
    "14f-capillary-jack-ic.svg",
    "14f-capillary-jack-ic-log.svg",
    "14f-capillary-jack-ic-err.svg",
    "14f-capillary-jack-ic-err-log.svg",
    "14f-capillary-jack-ic-slice.svg",
    "14f-capillary-jack-ic-log-slice.svg",
)
def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _identity(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _write_json(path: str, value) -> None:
    with durable_open(path, encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")


def _stat_contract(value) -> dict:
    """Stable identity fields available from both stat() and fstat()."""
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def _publish_directory(partial: str, final: str, names: list[str]) -> None:
    """Publish files or child trees into an exclusively reserved directory.

    The metadata marker is supplied last by callers.  A crash can therefore
    leave an unmistakably incomplete directory, but can never replace a
    previously published result or cache.
    """
    try:
        os.mkdir(final)
    except FileExistsError as exc:
        raise ValueError(
            f"{final}: publication conflict; remove it manually"
        ) from exc
    try:
        for name in names:
            os.rename(os.path.join(partial, name), os.path.join(final, name))
        os.rmdir(partial)
    except BaseException:
        # Keep both paths as loud, fail-closed evidence.  The final directory
        # has no valid metadata until the last move succeeds.
        raise


def _publish_result_tree(partial: str, final: str,
                         fallback_names: list[str]) -> None:
    """Publish a complete result tree atomically where the OS permits it.

    Windows ``rename`` is an atomic no-replace directory move, which is the
    production platform for the large archives.  On platforms where rename
    may replace an empty destination, retain the older exclusive-directory,
    metadata-last fallback rather than weakening no-clobber semantics.
    """
    if os.name == "nt":
        try:
            os.rename(partial, final)
        except FileExistsError as exc:
            raise ValueError(
                f"{final}: publication conflict; remove it manually"
            ) from exc
        return
    _publish_directory(partial, final, fallback_names)


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid Stage-14 metadata") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: Stage-14 metadata must be an object")
    return value


def _typed_array(typecode: str, data: bytes | bytearray | memoryview) -> array:
    out = array(typecode)
    out.frombytes(data)
    if sys.byteorder != "little":
        out.byteswap()
    return out


def _array_bytes(values: array) -> bytes:
    if sys.byteorder == "little":
        return values.tobytes()
    clone = array(values.typecode, values)
    clone.byteswap()
    return clone.tobytes()


def _require_array_bytes(name: str, data, itemsize: int, count: int) -> bytes:
    value = bytes(data)
    expected = itemsize * count
    if len(value) != expected:
        raise ValueError(f"native Stage-14 {name}: {len(value)} bytes, expected {expected}")
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _recorded_screen_z(meta: dict, path: str) -> float:
    try:
        geo = meta["geometry"]
        screen = _deep_merge(geo["screen"], geo["capillary"].get("screen", {}))
        z = float(screen["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: rays metadata has no capillary screen z") from exc
    if not math.isfinite(z):
        raise ValueError(f"{path}: recorded capillary screen z is not finite")
    return z


def _trace_core(geometry: dict) -> dict:
    """Trace physics that must agree; independent seed/budgets/screens do not."""
    try:
        cap = copy.deepcopy(geometry["capillary"])
        source = cap["source"]
    except (KeyError, TypeError) as exc:
        raise ValueError("rays geometry lacks capillary/source") from exc
    cap.pop("screen", None)
    cap.pop("screens", None)
    source.pop("n_modes", None)
    source.pop("n_rays", None)
    return {
        "precision": geometry.get("precision"),
        "max_bounces": geometry.get("max_bounces"),
        "capillary": cap,
    }


def _screen_contract(grid: ScreenGrid, ref: int) -> dict:
    return {
        "z": float(grid.z), "center": [grid.cxf, grid.cyf],
        "edge_x": grid.exf, "edge_y": grid.eyf,
        "nx": grid.nx, "ny": grid.ny, "reference_pixel": ref,
        "reference_xy": list(grid.pixel_xy(ref)),
        "pixel_convention": "half-open-row-major-iy-ix-v1",
    }


def _reference_pixel(screen_cfg, grid: ScreenGrid) -> int:
    xy = screen_cfg.reference
    xf, yf = xy if xy is not None else (grid.cxf, grid.cyf)
    if not (math.isfinite(xf) and math.isfinite(yf)
            and grid.x0f <= xf < grid.x0f + grid.exf
            and grid.y0f <= yf < grid.y0f + grid.eyf):
        raise ValueError(
            "stage 14 reference must lie inside the half-open capillary screen"
        )
    pixel = grid.pixel((xf, yf))
    if pixel is None:
        raise ValueError("stage 14 reference cannot be mapped to a screen pixel")
    return pixel


def _physics_contract(sim) -> dict:
    return {
        "energy_kev": str(sim.cfg.energy_kev),
        "lines": [{"energy_kev": str(line.e_kev), "k": float(line.k),
                   "weight": float(line.weight)} for line in sim.lines],
        "material": sim.cfg.material.name,
        "per_line_fresnel": bool(sim.per_line),
        "amplitude_min": float(sim.cfg.amplitude_min),
        "precision": int(sim.cfg.precision),
        "deposit": "float-line-amplitudes-phase-exp-v1",
    }


def _analysis_signature(sim, screen_contract: dict) -> dict:
    return {
        "cache_schema": CACHE_SCHEMA,
        "estimator_version": ESTIMATOR_VERSION,
        "capsysred_version": __version__,
        "stage_id": STAGE_ID,
        "scene": "capillary",
        "jackknife_unit": "mode",
        "payload": {"dtype": "<f8", "layout": "mode,pixel,(Wre,Wim,Ic)",
                    "cell_bytes": 24},
        "screen": screen_contract,
        "physics": _physics_contract(sim),
    }


@dataclass
class InputPart:
    path: str
    meta: dict
    n_modes: int
    n_rays: int
    source_z: float
    input_id: str
    seed: object
    analysis_id: str
    cache_dir: str
    identity: dict
    bytes: int = 0
    v3_index: object = None


@dataclass
class CachePart:
    input: InputPart
    meta: dict
    rows_path: str
    aggregates_path: str
    cache_hit: bool


@dataclass
class ScreenTarget:
    """One independently fingerprinted Stage-14 target screen."""

    label: str
    index: int
    cfg: object
    grid: ScreenGrid
    ref: int
    contract: dict
    signature: dict
    parts: list[InputPart]
    caches: list[CachePart | None]
    scatter: ScatterRaster

    @property
    def output_subdir(self) -> str:
        return "" if self.index == 0 else f"screen-{self.index}"


@dataclass
class CacheBuild:
    """A missing cache prepared for one screen and one input archive."""

    target: ScreenTarget
    part: InputPart
    partial: str
    rows_path: str
    aggregate_path: str
    store: object
    scatter: ScatterRaster
    started_at: float


def _prepare_inputs(sim, paths, signature: dict) -> list[InputPart]:
    if not paths:
        raise ValueError("stage 14 requires at least one rays archive")
    expected_core = _trace_core(geometry_metadata(sim.cfg))
    parts = []
    seeds, input_ids = set(), set()
    expected_rays = None
    for pathlike in paths:
        path = os.path.abspath(os.fspath(pathlike))
        v3_index = None
        if rays_v3.is_v3(path):
            v3_index = rays_v3.load_index(path)
            meta = rays_v3.metadata(path)
        else:
            meta = read_metadata(path)
            _validate_stream_metadata(meta, path)
        budget = meta["budgets"].get("capillary")
        if not isinstance(budget, list) or len(budget) != 2:
            raise ValueError(f"{path}: no capillary scene in rays metadata")
        n_modes, n_rays = budget
        if n_modes < 2:
            raise ValueError(f"{path}: Stage 14 needs at least two capillary modes")
        if expected_rays is None:
            expected_rays = n_rays
        elif expected_rays != n_rays:
            raise ValueError("stage 14 union requires equal rays-per-mode in all parts")
        actual_core = _trace_core(meta["geometry"])
        if not metadata_equal(actual_core, expected_core):
            raise ValueError(
                f"{path}: capillary trace geometry differs from the Stage-14 config"
            )
        snapshot = _json_copy(meta)
        concrete = {
            "rays_metadata": snapshot,
            "resolved_path": os.path.normcase(os.path.realpath(path)),
        }
        if v3_index is not None:
            # v3 archives are content-addressed: the index lists every
            # section's sha256, and each section is hashed while streamed.
            concrete["index_sha256"] = rays_v3.index_digest(path)
            archive_bytes = v3_index.total_bytes()
        else:
            # Rays archives are immutable under the no-clobber contract.  The
            # descriptor is checked again before and after the one strict
            # cache-building pass, closing path/symlink replacement windows.
            stat = os.stat(path)
            concrete["archive_stat"] = _stat_contract(stat)
            archive_bytes = stat.st_size
        input_id = _identity(concrete)
        seed = meta["geometry"].get("seed")
        if type(seed) is not int:
            raise ValueError(
                f"{path}: rays metadata geometry.seed must be an integer"
            )
        seed_key = _canonical(seed)
        if input_id in input_ids or seed_key in seeds:
            raise ValueError(
                f"stage 14 union repeats a rays input or seed ({path})"
            )
        input_ids.add(input_id)
        seeds.add(seed_key)
        identity = {
            "analysis_signature": signature,
            "input": concrete,
            "source_screen_z": _recorded_screen_z(meta, path),
            "n_modes": n_modes,
            "n_rays": n_rays,
        }
        analysis_id = _identity(identity)[:32]
        # CAPSYSRED_STAGE14_CACHE relocates the cache root away from the archive.
        cache_root = os.environ.get("CAPSYSRED_STAGE14_CACHE")
        if cache_root:
            os.makedirs(cache_root, exist_ok=True)
        else:
            cache_root = os.path.join(os.path.dirname(path), CACHE_DIR)
        cache_dir = os.path.join(cache_root, analysis_id)
        parts.append(InputPart(path, meta, n_modes, n_rays,
                               identity["source_screen_z"], input_id, seed,
                               analysis_id, cache_dir, identity,
                               archive_bytes, v3_index))
    configured_modes, configured_rays = sim.cfg.capillary.source.budget()
    total_modes = sum(part.n_modes for part in parts)
    if (total_modes, expected_rays) != (configured_modes, configured_rays):
        raise ValueError(
            "stage 14 replay budgets differ from config: "
            f"recorded union {[total_modes, expected_rays]} != "
            f"configured {[configured_modes, configured_rays]}"
        )
    return parts


def _screen_targets(sim, paths) -> list[ScreenTarget]:
    """Bind main and configured extra screens to independent cache IDs."""
    cap = sim.cfg.capillary
    configs = [cap.screen, *cap.screens]
    targets = []
    for index, screen_cfg in enumerate(configs):
        label = "capillary" if index == 0 else f"capillary-s{index}"
        grid = ScreenGrid(screen_cfg)
        ref = _reference_pixel(screen_cfg, grid)
        contract = _screen_contract(grid, ref)
        signature = _analysis_signature(sim, contract)
        parts = _prepare_inputs(sim, paths, signature)
        targets.append(ScreenTarget(
            label, index, screen_cfg, grid, ref, contract, signature,
            parts, [None] * len(parts), ScatterRaster(screen_cfg),
        ))
    return targets


def _native_store(path, n_modes, n_pixels, ref, kms, weights):
    cls = getattr(_formula, "Stage14Store", None)
    if cls is None:
        raise RuntimeError(
            "the native extension predates Stage 14; rebuild/install formula"
        )
    return cls(path, n_modes, n_pixels, ref, kms, weights)


@contextmanager
def _archive_lines(part: InputPart):
    """Iterator of archive text lines: preamble, rows, scene trailers.

    v2: the exact archive inode selected during input preparation.  v3: the
    capillary sections mode-major (each hashed against the index) behind a
    synthetic preamble and scene trailer, so the strict loop is shared.
    """
    if part.v3_index is not None:
        expected = part.identity["input"]["index_sha256"]
        if rays_v3.index_digest(part.path) != expected:
            raise ValueError(f"{part.path}: rays index changed before cache build")

        def lines():
            yield "{}\n"
            for raw in rays_v3.scene_lines(part.path, part.v3_index, "capillary"):
                yield raw.decode("utf-8")
            yield json.dumps({"scene_end": "capillary",
                              "rows": part.n_modes * part.n_rays}) + "\n"

        yield lines()
        if rays_v3.index_digest(part.path) != expected:
            raise ValueError(f"{part.path}: rays index changed during cache build")
        return
    expected = part.identity["input"]["archive_stat"]
    with open(part.path, "rb") as raw_fh:
        if not metadata_equal(_stat_contract(os.fstat(raw_fh.fileno())),
                              expected):
            raise ValueError(
                f"{part.path}: rays archive changed before cache build"
            )
        with gzip.GzipFile(fileobj=raw_fh, mode="rb") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8",
                                  newline="") as text_fh:
                yield iter(text_fh)
        if not metadata_equal(_stat_contract(os.fstat(raw_fh.fileno())),
                              expected):
            raise ValueError(
                f"{part.path}: rays archive changed during cache build"
            )


def _amplitudes_factory(sim):
    if sim.per_line or len(sim.lines) == 1:
        return FloatLineAmplitudes(sim.cfg.material, sim.lines, sim.cfg.precision)
    central = FloatLineAmplitudes(
        sim.cfg.material, [SimpleNamespace(e_kev=sim.cfg.energy_kev)],
        sim.cfg.precision,
    )
    n = len(sim.lines)

    def frozen(sins):
        value = central(sins)[0]
        return [value] * n

    return frozen


def _project_coordinates(row: dict, source_z: float, target_z: float):
    try:
        x, y = float(row["x"]), float(row["y"])
        dx, dy = float(row["dx"]), float(row["dy"])
        opl = float(row["opl"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("screen ray lacks finite x/y/direction/opl") from exc
    if not all(map(math.isfinite, (x, y, dx, dy, opl))):
        raise ValueError("screen ray has non-finite x/y/direction/opl")
    dz = math.sqrt(max(1.0 - dx * dx - dy * dy, 0.0))
    if dz <= 0.0:
        return x, y, opl, False
    step = (target_z - source_z) / dz
    x, y, opl = x + dx * step, y + dy * step, opl + step
    if not all(map(math.isfinite, (x, y, opl))):
        raise ValueError("re-projected screen ray is non-finite")
    return x, y, opl, True


def _project_row(row: dict, source_z: float, grid: ScreenGrid):
    x, y, opl, reachable = _project_coordinates(
        row, source_z, float(grid.z))
    return x, y, grid.pixel((x, y)) if reachable else None, opl


def _row_ids(row: dict, path: str):
    mode, ray = row.get("mode"), row.get("ray")
    if (type(mode) is not int or type(ray) is not int
            or mode < 0 or ray < 0):
        raise ValueError(f"{path}: ray mode/ray ids must be nonnegative integers")
    return mode, ray


def _build_stream_fanout(sim, part: InputPart, builds: list[CacheBuild],
                         log) -> tuple[list[dict], list[int]]:
    """Strictly decode one archive into every missing screen cache."""
    if not builds:
        raise ValueError("Stage-14 fan-out needs at least one cache build")
    for build in builds:
        other = build.part
        if (os.path.abspath(other.path) != os.path.abspath(part.path)
                or other.n_modes != part.n_modes
                or other.n_rays != part.n_rays
                or other.source_z != part.source_z
                or other.input_id != part.input_id):
            raise ValueError("Stage-14 fan-out inputs do not describe one archive")
    amps_of = _amplitudes_factory(sim)
    expected = part.n_modes * part.n_rays
    suffix = "s" if len(builds) != 1 else ""
    progress = Progress(
        f"14 cache capillary fan-out ({len(builds)} screen{suffix})", expected)
    counts, trailers, known_seen = {}, {}, set()
    stats_rows = [dict.fromkeys(STREAM_COUNTERS, 0) for _ in builds]
    bounce_hist = [0] * (sim.cfg.max_bounces + 1)
    current_mode = None
    target_rows = 0
    try:
        with _archive_lines(part) as fh:
            preamble = next(fh, "")
            if not preamble or not preamble.endswith("\n"):
                raise ValueError(f"{part.path}: missing complete ignored preamble")
            for line_no, line in enumerate(fh, 2):
                if not line.endswith("\n"):
                    raise ValueError(f"{part.path}:{line_no}: unterminated JSON row")
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{part.path}:{line_no}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{part.path}:{line_no}: row must be an object")
                if "scene_end" in row:
                    scene, rows = row.get("scene_end"), row.get("rows")
                    if not isinstance(scene, str) or type(rows) is not int or rows < 0:
                        raise ValueError(f"{part.path}:{line_no}: invalid scene trailer")
                    if scene in trailers:
                        raise ValueError(f"{part.path}:{line_no}: duplicate scene trailer")
                    trailers[scene] = rows
                    continue
                scene = row.get("stage")
                if not isinstance(scene, str):
                    raise ValueError(f"{part.path}:{line_no}: ray row lacks stage")
                if scene in trailers:
                    raise ValueError(
                        f"{part.path}:{line_no}: {scene} row follows its trailer"
                    )
                mode, ray = _row_ids(row, part.path)
                if "pixel" not in row or not (
                        row["pixel"] is None
                        or (type(row["pixel"]) is int and row["pixel"] >= 0)):
                    raise ValueError(
                        f"{part.path}:{line_no}: invalid saved pixel id"
                    )
                try:
                    opl_value = float(row["opl"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{part.path}:{line_no}: invalid optical path length"
                    ) from exc
                if not math.isfinite(opl_value):
                    raise ValueError(
                        f"{part.path}:{line_no}: non-finite optical path length"
                    )
                sins_raw = row.get("sins")
                if not isinstance(sins_raw, list):
                    raise ValueError(f"{part.path}:{line_no}: sins must be a list")
                try:
                    sins = [float(value) for value in sins_raw]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{part.path}:{line_no}: invalid sins") from exc
                if not all(math.isfinite(value) for value in sins):
                    raise ValueError(f"{part.path}:{line_no}: non-finite sins")
                fate = row.get("fate")
                if fate not in ("screen", "absorbed", "lost"):
                    raise ValueError(
                        f"{part.path}:{line_no}: invalid ray fate {fate!r}"
                    )
                if fate == "screen":
                    try:
                        screen_values = tuple(float(row[name]) for name in
                                              ("x", "y", "dx", "dy"))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{part.path}:{line_no}: screen ray lacks finite "
                            "x/y/direction"
                        ) from exc
                    if not all(math.isfinite(value) for value in screen_values):
                        raise ValueError(
                            f"{part.path}:{line_no}: screen ray has non-finite "
                            "x/y/direction"
                        )
                counts[scene] = counts.get(scene, 0) + 1
                budget = part.meta["budgets"].get(scene)
                if budget:
                    idx = counts[scene] - 1
                    if (mode, ray) != divmod(idx, budget[1]):
                        raise ValueError(
                            f"{part.path}:{line_no}: non-canonical {scene} mode/ray"
                        )
                known_seen.add(scene)
                if scene != "capillary":
                    continue
                if target_rows >= expected or (mode, ray) != divmod(target_rows, part.n_rays):
                    raise ValueError(
                        f"{part.path}:{line_no}: capillary mode/ray gap, duplicate or reorder"
                    )
                if mode != current_mode:
                    if current_mode is not None:
                        for build in builds:
                            build.store.fold_mode()
                    for build in builds:
                        build.store.begin_mode(mode)
                    current_mode = mode
                target_rows += 1
                for stats in stats_rows:
                    stats["emitted"] += 1
                nb = len(sins)
                if nb > sim.cfg.max_bounces:
                    raise ValueError(f"{part.path}:{line_no}: too many reflections")
                bounce_hist[nb] += 1
                if nb:
                    for stats in stats_rows:
                        stats["reflected_rays"] += 1
                        stats["reflections"] += nb
                amps = None
                if fate == "screen":
                    amps = amps_of(sins)
                    if (sim.cfg.amplitude_min > 0.0
                            and max(abs(a) for a in amps) < sim.cfg.amplitude_min):
                        fate = "absorbed"
                if fate == "screen":
                    projected = {}
                    for index, build in enumerate(builds):
                        z = float(build.target.grid.z)
                        coords = projected.get(z)
                        if coords is None:
                            coords = projected[z] = _project_coordinates(
                                row, part.source_z, z)
                        x, y, opl, reachable = coords
                        build.scatter.add((x, y))
                        pixel = (build.target.grid.pixel((x, y))
                                 if reachable else None)
                        if pixel is None:
                            stats_rows[index]["off_window"] += 1
                        else:
                            build.store.add_ray(pixel, opl, amps)
                            stats_rows[index]["screen"] += 1
                else:
                    for stats in stats_rows:
                        stats[fate] += 1
                progress.step()
        if current_mode is not None:
            for build in builds:
                build.store.fold_mode()
    except BaseException:
        progress.finish("failed")
        raise
    progress.finish(
        "on screens " + ", ".join(
            f"{build.target.label}={stats['screen']:,}"
            for build, stats in zip(builds, stats_rows)))
    if target_rows != expected:
        raise ValueError(
            f"{part.path}: capillary holds {target_rows} rows, expected {expected}"
        )
    if trailers.get("capillary") != target_rows:
        raise ValueError(f"{part.path}: missing or inconsistent capillary trailer")
    for scene in known_seen:
        if trailers.get(scene) != counts.get(scene):
            raise ValueError(f"{part.path}: incomplete scene {scene!r}")
        budget = part.meta["budgets"].get(scene)
        if budget and counts[scene] != math.prod(budget):
            raise ValueError(f"{part.path}: scene {scene!r} contradicts its budget")
    if set(trailers) != set(counts):
        raise ValueError(f"{part.path}: orphan scene trailer or partial scene")
    for build, stats in zip(builds, stats_rows):
        if stats["emitted"] != sum(
                stats[name] for name in
                ("screen", "off_window", "absorbed", "lost")):
            raise ValueError(
                f"{part.path}: inconsistent Stage-14 counters for "
                f"{build.target.label}")
    log(f"  stage 14 cache input validated: {part.n_modes} modes × {part.n_rays} rays")
    return stats_rows, bounce_hist


def _expected_aggregate_size(npix: int, max_bounces: int,
                             scatter_cells: int) -> int:
    return 16 + 56 * npix + 8 * (7 + max_bounces + 1 + scatter_cells)


def _estimated_peak_rss(npix: int, n_lines: int, scatter_cells: int) -> int:
    """Conservative planning estimate, not a measured process maximum.

    The final classification intentionally keeps one Python dict per pixel for
    the public result API, so its object overhead dominates the dense C++
    store.  A generous 4 KiB/pixel budget keeps the preflight honest on both
    32- and 64-bit CPython while still remaining far below the legacy
    mode×pixel dictionaries.
    """
    native_build = npix * (24 * n_lines + 128)
    python_finalize = 512 * 1024 * 1024 + npix * 4096 + scatter_cells * 40
    return max(native_build, python_finalize)


def _estimated_fanout_peak_rss(targets: list[ScreenTarget], n_lines: int) -> int:
    """Conservative peak for simultaneous stores and retained screen results."""
    unique_builds = {}
    for target in targets:
        key = _canonical(target.signature)
        unique_builds.setdefault(key, target)
    native_build = 512 * 1024 * 1024 + sum(
        target.grid.nx * target.grid.ny * (24 * n_lines + 128)
        + target.scatter.nx * target.scatter.ny * 40
        for target in unique_builds.values())
    retained_results = 512 * 1024 * 1024 + sum(
        target.grid.nx * target.grid.ny * 4096
        + target.scatter.nx * target.scatter.ny * 40
        for target in targets)
    return max(native_build, retained_results)


def _write_aggregates(path: str, native: dict, npix: int, stats: dict,
                      bounce_hist: list[int], scatter: ScatterRaster) -> tuple[int, str]:
    fields = (
        ("I", 8), ("w_re", 8), ("w_im", 8), ("ic", 8), ("n_rays", 8),
        ("m_realizations", 4), ("m_pair_realizations", 4),
        ("m_ref_realizations", 4), ("max_rays_per_realization", 4),
    )
    chunks = [struct.pack("<8sII", AGG_MAGIC, AGG_FORMAT, 0)]
    for name, width in fields:
        chunks.append(_require_array_bytes(name, native[name], width, npix))
    chunks.append(struct.pack("<7Q", *(stats[name] for name in STREAM_COUNTERS)))
    chunks.append(struct.pack(f"<{len(bounce_hist)}Q", *bounce_hist))
    flat_scatter = [value for row in scatter.counts for value in row]
    chunks.append(struct.pack(f"<{len(flat_scatter)}Q", *flat_scatter))
    expected = _expected_aggregate_size(npix, len(bounce_hist) - 1,
                                        len(flat_scatter))
    size = sum(map(len, chunks))
    if size != expected:
        raise ValueError(f"Stage-14 aggregate size {size}, expected {expected}")
    h = hashlib.sha256()
    with durable_open(path, "xb") as fh:
        for chunk in chunks:
            fh.write(chunk)
            h.update(chunk)
    return size, h.hexdigest()


def _read_aggregates(part: CachePart, npix: int, max_bounces: int,
                     scatter_cells: int) -> dict:
    expected_size = _expected_aggregate_size(npix, max_bounces, scatter_cells)
    meta_file = part.meta["files"][AGGREGATES_NAME]
    if meta_file.get("bytes") != expected_size:
        raise ValueError(f"{part.aggregates_path}: metadata has wrong size")
    try:
        with open(part.aggregates_path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise ValueError(f"{part.aggregates_path}: cannot read aggregates") from exc
    if len(data) != expected_size:
        raise ValueError(f"{part.aggregates_path}: truncated aggregates")
    digest = hashlib.sha256(data).hexdigest()
    if digest != meta_file.get("sha256"):
        raise ValueError(f"{part.aggregates_path}: aggregate SHA-256 mismatch")
    if struct.unpack_from("<8sII", data, 0) != (AGG_MAGIC, AGG_FORMAT, 0):
        raise ValueError(f"{part.aggregates_path}: invalid aggregate header")
    offset = 16

    def take(typecode, width, count):
        nonlocal offset
        value = _typed_array(typecode, data[offset:offset + width * count])
        offset += width * count
        return value

    out = {
        "I": take("d", 8, npix), "w_re": take("d", 8, npix),
        "w_im": take("d", 8, npix), "ic": take("d", 8, npix),
        "n_rays": take("Q", 8, npix),
        "m_realizations": take("I", 4, npix),
        "m_pair_realizations": take("I", 4, npix),
        "m_ref_realizations": take("I", 4, npix),
        "max_rays_per_realization": take("I", 4, npix),
    }
    counters = struct.unpack_from("<7Q", data, offset)
    offset += 56
    out["stats"] = dict(zip(STREAM_COUNTERS, counters))
    out["bounce_hist"] = list(struct.unpack_from(
        f"<{max_bounces + 1}Q", data, offset))
    offset += 8 * (max_bounces + 1)
    out["scatter"] = list(struct.unpack_from(f"<{scatter_cells}Q", data, offset))
    offset += 8 * scatter_cells
    if offset != len(data):
        raise ValueError(f"{part.aggregates_path}: aggregate trailing bytes")
    return out


def _load_cache(part: InputPart, signature: dict, npix: int,
                max_bounces: int, scatter: ScatterRaster) -> CachePart | None:
    final, partial = part.cache_dir, part.cache_dir + ".partial"
    if os.path.lexists(partial):
        raise ValueError(
            f"{partial}: incomplete Stage-14 cache exists; remove it manually"
        )
    if not os.path.lexists(final):
        return None
    meta_path = os.path.join(final, META_NAME)
    meta = _read_json(meta_path)
    if (meta.get("cache_schema") != CACHE_SCHEMA
            or meta.get("analysis_id") != part.analysis_id
            or not metadata_equal(meta.get("identity"), part.identity)
            or not metadata_equal(meta.get("analysis_signature"), signature)):
        raise ValueError(
            f"{final}: incompatible Stage-14 cache; remove it manually"
        )
    rows_path = os.path.join(final, ROWS_NAME)
    aggregates_path = os.path.join(final, AGGREGATES_NAME)
    expected_rows = part.n_modes * npix * 24
    try:
        rows_size = os.path.getsize(rows_path)
        agg_size = os.path.getsize(aggregates_path)
    except OSError as exc:
        raise ValueError(f"{final}: incomplete Stage-14 cache") from exc
    expected_agg = _expected_aggregate_size(
        npix, max_bounces, scatter.nx * scatter.ny)
    if (rows_size != expected_rows
            or meta.get("files", {}).get(ROWS_NAME, {}).get("bytes") != expected_rows
            or agg_size != expected_agg):
        raise ValueError(f"{final}: Stage-14 cache size mismatch")
    cache = CachePart(part, meta, rows_path, aggregates_path, True)
    _read_aggregates(cache, npix, max_bounces, scatter.nx * scatter.ny)
    return cache


def _prepare_cache_build(sim, target: ScreenTarget,
                         part: InputPart) -> CacheBuild:
    """Exclusively reserve one missing cache and construct its native store."""
    cache_root = os.path.dirname(part.cache_dir)
    os.makedirs(cache_root, exist_ok=True)
    partial = part.cache_dir + ".partial"
    try:
        os.mkdir(partial)
    except FileExistsError as exc:
        raise ValueError(
            f"{partial}: cache conflict; remove it manually or choose another input"
        ) from exc
    rows_path = os.path.join(partial, ROWS_NAME)
    aggregate_path = os.path.join(partial, AGGREGATES_NAME)
    kms = [float(line.k) for line in sim.lines]
    weights = [float(line.weight) for line in sim.lines]
    npix = target.grid.nx * target.grid.ny
    store = _native_store(
        rows_path, part.n_modes, npix, target.ref, kms, weights)
    return CacheBuild(
        target, part, partial, rows_path, aggregate_path, store,
        ScatterRaster(target.cfg), time.time(),
    )


def _finish_cache_build(sim, build: CacheBuild, stats: dict,
                        bounce_hist: list[int]) -> CachePart:
    """Validate, serialize and publish one fan-out store after stream EOF."""
    target, part = build.target, build.part
    grid, scatter = target.grid, build.scatter
    npix = grid.nx * grid.ny
    native = build.store.finish()
    rows_size = int(native["payload_bytes"])
    rows_sha = str(native["payload_sha256"])
    expected_rows = part.n_modes * npix * 24
    if (rows_size != expected_rows
            or os.path.getsize(build.rows_path) != expected_rows):
        raise ValueError("native Stage-14 store wrote the wrong payload size")
    if sum(_typed_array("Q", native["n_rays"])) != stats["screen"]:
        raise ValueError(
            f"Stage-14 density differs from {target.label} screen counter")
    agg_size, agg_sha = _write_aggregates(
        build.aggregate_path, native, npix, stats, bounce_hist, scatter)
    for path in (build.rows_path, build.aggregate_path):
        with open(path, "rb+") as fh:
            os.fsync(fh.fileno())
    meta = {
        "cache_schema": CACHE_SCHEMA,
        "estimator_version": ESTIMATOR_VERSION,
        "analysis_id": part.analysis_id,
        "analysis_signature": target.signature,
        "identity": part.identity,
        "stage_id": STAGE_ID,
        "screen": target.label,
        "input_id": part.input_id,
        "input_path": part.path,
        "source_screen_z": part.source_z,
        "n_modes": part.n_modes,
        "n_rays_per_mode": part.n_rays,
        "n_pixels": npix,
        "reference_pixel": target.ref,
        "max_bounces": sim.cfg.max_bounces,
        "scatter": {
            "nx": scatter.nx, "ny": scatter.ny,
            "x0": scatter.x0f, "y0": scatter.y0f,
            "edge_x": scatter.exf, "edge_y": scatter.eyf,
            "convention": "row-major-iy-ix-half-open-v1",
        },
        "files": {
            ROWS_NAME: {"bytes": rows_size, "sha256": rows_sha},
            AGGREGATES_NAME: {"bytes": agg_size, "sha256": agg_sha},
        },
        "build": {
            "seconds": time.time() - build.started_at,
            "ray_archive_bytes": part.bytes,
            "yaml_file": sim.cfg.yaml_file,
        },
        "capsysred_version": __version__,
    }
    _write_json(os.path.join(build.partial, META_NAME), meta)
    _publish_directory(
        build.partial, part.cache_dir,
        [ROWS_NAME, AGGREGATES_NAME, META_NAME],
    )
    return CachePart(
        part, meta, os.path.join(part.cache_dir, ROWS_NAME),
        os.path.join(part.cache_dir, AGGREGATES_NAME), False)


def _missing_cache_specs(targets: list[ScreenTarget]):
    """Unique missing cache directories; identical screens share content."""
    unique = {}
    for target in targets:
        for part, cached in zip(target.parts, target.caches):
            if cached is None:
                unique.setdefault(os.path.normcase(part.cache_dir),
                                  (target, part))
    return list(unique.values())


def _preflight_fanout_builds(missing, max_bounces: int, log) -> None:
    """Check the summed footprint of every screen x input cache miss."""
    groups = {}
    for target, part in missing:
        npix = target.grid.nx * target.grid.ny
        scatter_cells = target.scatter.nx * target.scatter.ny
        required = part.n_modes * npix * 24 + _expected_aggregate_size(
            npix, max_bounces, scatter_cells)
        probe = os.path.dirname(os.path.dirname(part.cache_dir))
        stat = os.stat(probe)
        group = groups.setdefault(
            stat.st_dev, {"probe": probe, "bytes": 0, "caches": 0,
                          "screens": set()})
        group["bytes"] += required
        group["caches"] += 1
        group["screens"].add(target.label)
    for group in groups.values():
        free = shutil.disk_usage(group["probe"]).free
        required = group["bytes"]
        log(f"  Stage 14 fan-out disk preflight: {group['caches']} cache(s), "
            f"{len(group['screens'])} screen(s), "
            f"{required / (1024 ** 3):.3f} GiB required; "
            f"{free / (1024 ** 3):.1f} GiB free")
        if free < required + 64 * 1024 * 1024:
            raise ValueError(
                f"{group['probe']}: insufficient free space for all missing "
                "Stage-14 screen caches")


def _load_or_build_screen_caches(sim, targets: list[ScreenTarget], log) -> set[str]:
    """Resolve every target cache and decode each missing archive only once.

    The return value is the set of input IDs whose gzip streams were opened.
    It is physical I/O provenance for the whole fan-out invocation; individual
    screen results still report their own hit/miss state.
    """
    cache_by_dir = {}
    for target in targets:
        npix = target.grid.nx * target.grid.ny
        for index, part in enumerate(target.parts):
            cache = _load_cache(
                part, target.signature, npix, sim.cfg.max_bounces,
                target.scatter)
            target.caches[index] = cache
            if cache is not None:
                cache_by_dir[os.path.normcase(part.cache_dir)] = cache

    missing = _missing_cache_specs(targets)
    _preflight_fanout_builds(missing, sim.cfg.max_bounces, log)
    by_input = {}
    for target, part in missing:
        by_input.setdefault(part.input_id, []).append((target, part))

    archives_read = set()
    for specs in by_input.values():
        labels = ", ".join(target.label for target, _ in specs)
        part = specs[0][1]
        log(f"  Stage 14 cache miss fan-out: {part.path} -> {labels}")
        builds = [_prepare_cache_build(sim, target, screen_part)
                  for target, screen_part in specs]
        stats_rows, bounce_hist = _build_stream_fanout(
            sim, part, builds, log)
        archives_read.add(part.input_id)
        for build, stats in zip(builds, stats_rows):
            cache = _finish_cache_build(sim, build, stats, bounce_hist)
            cache_by_dir[os.path.normcase(build.part.cache_dir)] = cache

    for target in targets:
        for index, part in enumerate(target.parts):
            key = os.path.normcase(part.cache_dir)
            cache = cache_by_dir.get(key)
            if cache is None:
                raise ValueError(
                    f"{part.cache_dir}: Stage-14 cache was not resolved")
            target.caches[index] = cache
        hits = sum(cache.cache_hit for cache in target.caches)
        log(f"  Stage 14 {target.label}: cache hits {hits} of "
            f"{len(target.caches)}")
    return archives_read


def _combine_aggregates(caches: list[CachePart], npix: int, max_bounces: int,
                        scatter_cells: int) -> dict:
    items = [
        _read_aggregates(cache, npix, max_bounces, scatter_cells)
        for cache in caches
    ]
    if not items:
        raise ValueError("Stage-14 union requires at least one cache part")
    combined = dict(items[0])
    for name in ("I", "w_re", "w_im", "ic"):
        combined[name] = array(
            "d", (math.fsum(item[name][i] for item in items)
                  for i in range(npix))
        )
    combined["n_rays"] = array(
        "Q", (sum(item["n_rays"][i] for item in items)
              for i in range(npix))
    )
    for name in ("m_realizations", "m_pair_realizations",
                 "m_ref_realizations"):
        values = [sum(item[name][i] for item in items)
                  for i in range(npix)]
        if any(value > 0xFFFFFFFF for value in values):
            raise ValueError(f"combined Stage-14 {name} overflows uint32")
        combined[name] = array("I", values)
    combined["max_rays_per_realization"] = array(
        "I", (max(item["max_rays_per_realization"][i] for item in items)
              for i in range(npix))
    )
    combined["stats"] = {
        name: sum(item["stats"][name] for item in items)
        for name in STREAM_COUNTERS
    }
    combined["bounce_hist"] = [
        sum(item["bounce_hist"][i] for item in items)
        for i in range(max_bounces + 1)
    ]
    combined["scatter"] = [
        sum(item["scatter"][i] for item in items)
        for i in range(scatter_cells)
    ]
    if sum(combined["n_rays"]) != combined["stats"]["screen"]:
        raise ValueError("combined Stage-14 density differs from stream counters")
    return combined


def _finalize(caches: list[CachePart], aggregate: dict, npix: int, ref: int):
    fn = getattr(_formula, "stage14_finalize", None)
    if fn is None:
        raise RuntimeError("the native extension has no Stage-14 finalizer")
    expected_hashes = [cache.meta["files"][ROWS_NAME]["sha256"] for cache in caches]
    return fn(
        [cache.rows_path for cache in caches],
        [cache.input.n_modes for cache in caches], npix, ref,
        _array_bytes(aggregate["I"]), _array_bytes(aggregate["w_re"]),
        _array_bytes(aggregate["w_im"]), _array_bytes(aggregate["ic"]),
        expected_hashes,
    )


def _ref_warnings(rows, grid: ScreenGrid, ref: int, n_modes: int):
    warnings, diagnostics = [], {}
    ref_row = rows[ref]
    lam = ref_row["n_rays"] / n_modes
    diagnostics["lambda_ref"] = lam
    diagnostics["n_rays_ref"] = ref_row["n_rays"]
    diagnostics["m_realizations_ref"] = ref_row["m_realizations"]
    diagnostics["m_pair_realizations_ref"] = ref_row["m_pair_realizations"]
    diagnostics["m_ref_realizations_histogram"] = {
        str(value): count for value, count in sorted(Counter(
            row["m_ref_realizations"] for row in rows).items())
    }
    if lam < 8.0:
        warnings.append("low-rays-per-realization-at-ref")
    pair_rows = [row for row in rows if row["m_pair_realizations"] > 0]
    if pair_rows:
        below = sum(row["n_rays"] <= ref_row["n_rays"] for row in pair_rows)
        percentile = 100.0 * below / len(pair_rows)
        diagnostics["ref_density_percentile"] = percentile
        if percentile < 50.0:
            warnings.append("below-median-ref-density")
        covered = sum(row["m_ref_realizations"] > 0 for row in pair_rows)
        diagnostics["w_cover"] = covered / len(pair_rows)
    iy, ix = divmod(ref, grid.nx)
    neighbors = []
    for jy in range(max(0, iy - 1), min(grid.ny, iy + 2)):
        for jx in range(max(0, ix - 1), min(grid.nx, ix + 2)):
            p = jy * grid.nx + jx
            if p != ref:
                neighbors.append(rows[p]["flag"] == "trusted")
    diagnostics["trusted_ref_neighbors"] = sum(neighbors)
    diagnostics["ref_neighbors"] = len(neighbors)
    if neighbors and not all(neighbors):
        warnings.append("untrusted-ref-neighborhood")
    return warnings, diagnostics


def _grid(values, nx, ny):
    return [list(values[i * nx:(i + 1) * nx]) for i in range(ny)]


def _figure_saver(result_dir: str):
    """Single gate for figure writes; the caller checks names against the manifest."""
    written = set()

    def save(name, fig):
        written.add(name)
        render.save(os.path.join(result_dir, name), fig)

    return written, save


def _stage14_figures(save, rows, aggregate, final, grid: ScreenGrid,
                     ref: int, flag_counts: Counter, n_modes: int,
                     screen_label: str = "capillary"):
    nx, ny = grid.nx, grid.ny
    mu = [row["mu_raw"] for row in rows]
    err = [row["mu_raw_err"] for row in rows]
    flags = [row["flag"] for row in rows]
    lit_flag_counts = Counter(
        row["flag"] for row in rows if row["n_rays"] > 0
    )
    density = [float(v) for v in aggregate["n_rays"]]
    intensity = [float(v) for v in aggregate["I"]]
    mu_grid, err_grid = _grid(mu, nx, ny), _grid(err, nx, ny)
    flag_grid = _grid(flags, nx, ny)
    i_grid, d_grid = _grid(intensity, nx, ny), _grid(density, nx, ny)
    extent = (m_to_um(grid.x0f), m_to_um(grid.x0f + grid.exf),
              m_to_um(grid.y0f), m_to_um(grid.y0f + grid.eyf))
    ref_xy = grid.pixel_xy(ref)
    mark = (m_to_um(ref_xy[0]), m_to_um(ref_xy[1]))
    sub = (f"{screen_label}; {n_modes} exact delete-one-mode units; "
           "raw μ, display clipped at 1")
    cell_x, cell_y = m_to_um(grid.exf) / nx, m_to_um(grid.eyf) / ny
    cell = (f"{cell_x:.3g} µm" if abs(cell_x - cell_y) < 5e-4
            else f"{cell_x:.3g} × {cell_y:.3g} µm")
    mu_sub = (f"ref cell at ({round(mark[0], 3) or 0.0:g}, "
              f"{round(mark[1], 3) or 0.0:g}) µm, cell {cell}")
    mu_fig = render.heatmap(mu_grid, extent, "|μ(P,P_ref)|", "x, µm", "y, µm",
                            mu_sub, "|μ|", vmax=1.0, mark=mark, w=448, equal=True)
    err_fig = render.heatmap(err_grid, extent, "σ_jack(μ)", "x, µm", "y, µm",
                             "", "σ", mark=mark, w=448, equal=True)
    flag_fig = render.category_map(flag_grid, extent, "Cell classification", "x, µm",
                                   "y, µm", "", mark=mark,
                                   counts=flag_counts,
                                   lit_counts=lit_flag_counts,
                                   lit_total=sum(row["n_rays"] > 0 for row in rows),
                                   w=448, equal=True)
    save("14-capillary-jack-mu.svg",
         render.hstack([mu_fig, err_fig, flag_fig]))
    save("14-capillary-jack-mu-map.svg", mu_fig)
    save("14-capillary-jack-mu-err.svg", err_fig)
    save("14-capillary-jack-mu-flags.svg", flag_fig)
    iy = ref // nx
    xs = [m_to_um(x) for x in grid.xs()]
    row_ids = range(iy * nx, (iy + 1) * nx)
    xs_row = [xs[p % nx] for p in row_ids]

    def _row(fn):
        # per-cell values along the reference row; None = gap, lines break
        return [fn(rows[p]) for p in row_ids]

    def banded(r):
        return r["mu_raw"] is not None and r["mu_raw_err"] is not None

    mu_band = _row(lambda r: min(r["mu_raw"], 1.0) if banded(r) else None)
    series = []
    if any(v is not None for v in mu_band):
        series.append({
            "xs": xs_row, "ys": mu_band,
            "lo": _row(lambda r: max(r["mu_raw"] - r["mu_raw_err"], 0.0)
                       if banded(r) else None),
            "hi": _row(lambda r: min(r["mu_raw"] + r["mu_raw_err"], 1.0)
                       if banded(r) else None),
            "label": "min(μ,1) ± σ_jack",
        })
    without_err = [p for p in row_ids if rows[p]["mu_raw"] is not None
                   and rows[p]["mu_raw_err"] is None]
    if without_err:
        series.append({"xs": [xs[p % nx] for p in without_err],
                       "ys": [min(rows[p]["mu_raw"], 1.0) for p in without_err],
                       "label": "μ; σ unavailable", "dots": True,
                       "color": "#CC79A7"})
    if not series:
        series.append({"xs": [xs[0], xs[-1]], "ys": [0.0, 0.0],
                       "label": "no defined μ on this slice", "dash": "2,3"})
    for flag in ("null-Ic", "noisy-Ic", "noisy-mu", "over-mu"):
        ids = [p for p in row_ids if rows[p]["flag"] == flag]
        if ids:
            series.append({"xs": [xs[p % nx] for p in ids],
                           "ys": [min(rows[p]["mu_raw"] or 0.0, 1.0) for p in ids],
                           "label": flag, "color": render.FLAG_COLORS[flag],
                           "dots": True})
    slice_label = f"slice y = {round(m_to_um(grid.ys()[iy]), 3) or 0.0:g} µm"
    slice_fig = render.line_chart(
        series, f"|μ(P,P_ref)| — {slice_label}",
        "x, µm", "|μ|", mu_sub, vlines=[(m_to_um(ref_xy[0]), "ref")], w=760)

    def _strip(vals, ylabel):
        return render.line_chart(
            [{"xs": xs_row, "ys": vals,
              "lo": [0.0 if v is not None else None for v in vals], "hi": vals,
              "color": "#ff7f0e", "width": 1.0}], "", "x, µm", ylabel,
            w=760, h=150)

    err_vals = _row(lambda r: r["mu_raw_err"])
    if any(v is not None for v in err_vals):
        slice_fig = render.vstack([slice_fig, _strip(err_vals, "σ_jack")])
    save("14a-capillary-jack-slice.svg", slice_fig)
    ref_line = [(m_to_um(ref_xy[0]), "ref")]

    def _slice_chart(vals, title, ylabel, empty, y_zero=True):
        series = ([{"xs": xs_row, "ys": vals, "label": ylabel}]
                  if any(v is not None for v in vals) else
                  [{"xs": [xs[0], xs[-1]], "ys": [0.0, 0.0],
                    "label": empty, "dash": "2,3"}])
        return render.line_chart(series, f"{title} — {slice_label}", "x, µm",
                                 ylabel, mu_sub, vlines=ref_line,
                                 y_zero=y_zero, w=760)

    row_i = [intensity[p] for p in row_ids]
    save("14b-capillary-jack-intensity-slice.svg",
         _slice_chart(row_i, "intensity", "I", "no rays"))
    save("14b-capillary-jack-intensity-log-slice.svg",
         _slice_chart([math.log10(v) if v > 0 else None for v in row_i],
                             "intensity, log scale", "log10 I",
                             "no lit cells", y_zero=False))
    ic_grid = _grid([row["ic"] for row in rows], nx, ny)
    ic_err_grid = _grid([row["ic_err"] for row in rows], nx, ny)
    for name, grid_, title, cbar, log_ in (
            ("14f-capillary-jack-ic.svg", ic_grid,
             "coherent intensity Ic", "Ic", False),
            ("14f-capillary-jack-ic-log.svg", ic_grid,
             "coherent intensity Ic, log scale", "Ic", True),
            ("14f-capillary-jack-ic-err.svg", ic_err_grid,
             "σ_jack(Ic)", "σ", False),
            ("14f-capillary-jack-ic-err-log.svg", ic_err_grid,
             "σ_jack(Ic), log scale", "σ", True)):
        save(name,
         render.heatmap(grid_, extent, title, "x, µm", "y, µm", "",
                                   cbar, w=518, equal=True, log=log_))
    ic_row = _row(lambda r: r["ic"])
    ic_fig = _slice_chart(ic_row, "coherent intensity Ic", "Ic", "no paired cells")
    ic_errs = _row(lambda r: r["ic_err"])
    if any(v is not None for v in ic_errs):
        ic_fig = render.vstack([ic_fig, _strip(ic_errs, "σ_jack(Ic)")])
    save("14f-capillary-jack-ic-slice.svg", ic_fig)
    ic_log_fig = _slice_chart(
        [math.log10(v) if v is not None and v > 0 else None for v in ic_row],
        "coherent intensity Ic, log scale", "log10 Ic", "no positive Ic",
        y_zero=False)
    rel = _row(lambda r: r["ic_err"] / r["ic"]
               if r["ic_err"] is not None and (r["ic"] or 0) > 0 else None)
    if any(v is not None for v in rel):
        ic_log_fig = render.vstack([ic_log_fig, _strip(rel, "σ_jack(Ic)/Ic")])
    save("14f-capillary-jack-ic-log-slice.svg",
         ic_log_fig)
    if ny > 1:
        intensity_fig = render.heatmap(i_grid, extent, "intensity", "x, µm",
                                       "y, µm", "", "I", w=518, equal=True)
        log_fig = render.heatmap(i_grid, extent, "intensity, log scale", "x, µm",
                                 "y, µm", "", "I", w=518, equal=True, log=True)
        density_fig = render.heatmap(d_grid, extent, "ray density", "x, µm",
                                     "y, µm", "", "rays", w=518, equal=True)
    else:
        imax, dmax = max(intensity) or 1.0, max(density) or 1.0
        intensity_fig = render.line_chart([
            {"xs": xs, "ys": [v / imax for v in intensity], "label": "I/max"}],
            "intensity", "x, µm", "normalized", sub, w=760)
        lit = [(x, v) for x, v in zip(xs, intensity) if v > 0]
        log_fig = render.line_chart(
            [{"xs": [x for x, _ in lit], "ys": [math.log10(v) for _, v in lit],
              "label": "log10 I"}] if lit else
            [{"xs": [xs[0], xs[-1]], "ys": [0.0, 0.0],
              "label": "no lit cells", "dash": "2,3"}],
            "intensity, log scale", "x, µm", "log10 I", "", y_zero=False, w=760)
        density_fig = render.line_chart([
            {"xs": xs, "ys": [v / dmax for v in density], "label": "rays/max",
             "dash": "6,4"}], "ray density", "x, µm", "normalized", "", w=760)
    save("14b-capillary-jack-intensity.svg",
         intensity_fig)
    save("14b-capillary-jack-intensity-log.svg",
         log_fig)
    save("14b-capillary-jack-density.svg",
         density_fig)
    save("14c-capillary-jack-overlay.svg",
         render.overlay_map(mu_grid, flag_grid, extent,
                                   "Stage-14 non-trusted overlay", "x, µm", "y, µm",
                                   sub, mark=mark, equal=True))
    scatter_meta = final["scatter"]
    scatter_grid = _grid(aggregate["scatter"], scatter_meta["nx"], scatter_meta["ny"])
    scatter_extent = (m_to_um(scatter_meta["x0"]),
                      m_to_um(scatter_meta["x0"] + scatter_meta["edge_x"]),
                      m_to_um(scatter_meta["y0"]),
                      m_to_um(scatter_meta["y0"] + scatter_meta["edge_y"]))
    save("14d-capillary-ray-scatter.svg",
         render.ray_scatter(scatter_grid, scatter_extent,
                                   f"{screen_label}: ray locations on target screen",
                                   "x, µm", "y, µm", sub))
    return {
        "mu_raw": mu_grid, "mu_raw_err": err_grid, "flag": flag_grid,
        "intensity": i_grid, "density": d_grid,
    }


def _ref_passport(save, rows, ref: int, ref_status: str, ref_warnings,
                  diagnostics: dict, thresholds: FlagThresholds, n_modes: int,
                  screen_label: str) -> None:
    """14e: the reference pixel's own numbers against the registered and
    guide thresholds; only recorded mu-jack/meta fields."""
    row = rows[ref]
    ic, err = row["ic"], row["ic_err"]
    snr = ic / err if ic is not None and err else None
    z_ref = thresholds.ref_ic_n_sigma
    pairs, lit, n_ref = (row["m_pair_realizations"], row["m_realizations"],
                         row["n_rays"])
    pair_rows = [r for r in rows if r["m_pair_realizations"] > 0]
    covered = sum(r["m_ref_realizations"] > 0 for r in pair_rows)
    others = [r["n_rays"] for r in pair_rows if not r["is_reference"]]
    median = statistics.median(others) if others else None
    pct, w_cover = diagnostics.get("ref_density_percentile"), diagnostics.get("w_cover")
    lam = diagnostics["lambda_ref"]
    checks = [
        {"name": "Ic_ref significance",
         "big": f"{snr:.1f}σ" if snr is not None else "n/a",
         "detail": (f"Ic_ref = {ic:.0f} ± {err:.0f}  ·  {snr / z_ref:.1f}× the weak line"
                    if snr is not None else f"ref_status = {ref_status}"),
         "value": snr, "lo": 0.0, "hi": 20.0, "threshold": z_ref,
         "threshold_label": f"≥ ref_ic_n_sigma = {z_ref:g} (registered)"},
        {"name": "pairness of the reference", "big": f"{pairs / n_modes:.0%}",
         "detail": (f"{pairs} of {n_modes} realizations gave ≥ 2 rays  ·  "
                    f"lit in {lit} ({lit / n_modes:.0%})"),
         "value": pairs / n_modes, "lo": 0.0, "hi": 1.0, "threshold": 0.5,
         "threshold_label": "≥ 50 % (guide)"},
        {"name": "rays per realization at ref", "big": f"{lam:.2f}",
         "detail": (f"n_rays = {n_ref}  ·  mean over lit realizations "
                    f"{n_ref / lit if lit else 0.0:.2f}  ·  max {row['max_rays_per_realization']}"),
         "value": lam, "lo": 0.0, "hi": 20.0, "threshold": 8.0,
         "threshold_label": "≥ 8 (guide 8–16)"},
        {"name": "density rank among paired cells",
         "big": f"{pct:.0f}%" if pct is not None else "n/a",
         "detail": (f"n_rays(ref) = {n_ref}  vs  map median {median:.0f}  ·  "
                    f"{n_ref / median:.1f}× median" if median else
                    f"n_rays(ref) = {n_ref}  ·  no other paired cells"),
         "value": pct, "lo": 0.0, "hi": 100.0, "threshold": 50.0,
         "threshold_label": "≥ 50 % (guide)"},
        {"name": "cross-link with the map  (w_cover)",
         "big": f"{w_cover:.2f}" if w_cover is not None else "n/a",
         "detail": f"paired cells sharing a realization with ref: {covered} of {len(pair_rows)}",
         "value": w_cover, "lo": 0.0, "hi": 1.0, "threshold": 0.9,
         "threshold_label": "≥ 0.9 (guide)"},
    ]
    title = (f"Reference-cell passport — "
             f"({row['x_um']:.1f}, {row['y_um']:.1f}) µm, N_jk = {n_modes}, "
             f"ref_ic_n_sigma = {z_ref:g}")
    save("14e-capillary-ref-passport.svg",
         render.gauge_table(checks, title))


def preflight_stage14_output(out_dir: str) -> None:
    """Refuse an existing/partial result before tracing or cache work."""
    result_dir = os.path.join(out_dir, RESULT_DIR)
    partial = result_dir + ".partial"
    if os.path.lexists(result_dir) or os.path.lexists(partial):
        raise ValueError(
            f"{result_dir}: Stage-14 result already exists; remove it manually "
            "or choose another output directory"
        )


def _finalize_screen(sim, target: ScreenTarget, output_dir: str,
                     fanout_metrics: dict,
                     *, write_meta: bool = True) -> dict:
    """Finalize and serialize one screen from its resolved cache parts."""
    screen_started = time.time()
    grid, ref = target.grid, target.ref
    npix = grid.nx * grid.ny
    parts = target.parts
    cache_parts = list(target.caches)
    if not cache_parts or any(cache is None for cache in cache_parts):
        raise ValueError(f"Stage-14 {target.label} caches are unresolved")
    scatter_meta = cache_parts[0].meta["scatter"]
    for cache in cache_parts[1:]:
        if (not metadata_equal(cache.meta["analysis_signature"], target.signature)
                or not metadata_equal(cache.meta["scatter"], scatter_meta)):
            raise ValueError(
                f"Stage-14 {target.label} union cache parts are incompatible")
    aggregate = _combine_aggregates(
        cache_parts, npix, sim.cfg.max_bounces,
        scatter_meta["nx"] * scatter_meta["ny"])
    native = _finalize(cache_parts, aggregate, npix, ref)
    # Dense native outputs: every undefined value has a separate mask; Python
    # never infers missingness from NaN or a plausible numeric zero.
    ic_err = _typed_array("d", native["ic_err"])
    w_err = _typed_array("d", native["w_err"])
    mu_raw = _typed_array("d", native["mu_raw"])
    mu_err = _typed_array("d", native["mu_raw_err"])
    n_valid = _typed_array("I", native["n_mu_loo_valid"])
    mu_defined = bytes(native["mu_raw_defined"])
    mu_err_defined = bytes(native["mu_raw_err_defined"])
    ref_ic_loo = _typed_array("d", native["ic_ref_loo"])
    n_modes = sum(part.n_modes for part in parts)
    for name, values in (("ic_err", ic_err), ("w_err", w_err),
                         ("mu_raw", mu_raw), ("mu_raw_err", mu_err),
                         ("n_mu_loo_valid", n_valid)):
        if len(values) != npix:
            raise ValueError(f"native Stage-14 {name} has wrong length")
    if len(mu_defined) != npix or len(mu_err_defined) != npix:
        raise ValueError("native Stage-14 validity masks have wrong length")
    if len(ref_ic_loo) != n_modes:
        raise ValueError("native Stage-14 reference LOO row has wrong length")
    hashes = list(native["payload_sha256"])
    expected_hashes = [cache.meta["files"][ROWS_NAME]["sha256"]
                       for cache in cache_parts]
    if hashes != expected_hashes:
        raise ValueError("Stage-14 payload SHA-256 mismatch")
    thresholds = FlagThresholds(**sim.cfg.stage14_flag_thresholds)
    ref_status = validate_ref(
        m_pair_realizations=int(aggregate["m_pair_realizations"][ref]),
        ic_ref=float(aggregate["ic"][ref]),
        ic_ref_err=float(ic_err[ref]),
        ic_ref_loo=ref_ic_loo,
        n_jackknife_units=n_modes,
        thresholds=thresholds,
    )
    rows = []
    for pixel in range(npix):
        counters = PixelCounters(
            int(aggregate["n_rays"][pixel]),
            int(aggregate["m_realizations"][pixel]),
            int(aggregate["m_pair_realizations"][pixel]),
            int(aggregate["m_ref_realizations"][pixel]),
            int(aggregate["max_rays_per_realization"][pixel]),
        )
        validate_counters(counters)
        I = float(aggregate["I"][pixel])
        ic = float(aggregate["ic"][pixel])
        w_abs_value = math.hypot(aggregate["w_re"][pixel], aggregate["w_im"][pixel])
        is_ref = pixel == ref
        # Apply the taxonomy's strict missingness before calling its serializer.
        if counters.n_rays == 0:
            stats = PixelStatistics(I, counters)
        elif counters.m_pair_realizations == 0:
            has_w = counters.m_ref_realizations > 0
            stats = PixelStatistics(
                I, counters, w_abs=w_abs_value if has_w else None,
                w_err=float(w_err[pixel]) if has_w else None)
        else:
            lower = ic - thresholds.ic_n_sigma * float(ic_err[pixel])
            upper = ic + thresholds.ic_n_sigma * float(ic_err[pixel])
            early_ic = upper < 0.0 or lower <= 0.0
            has_w = counters.m_ref_realizations > 0
            allow_mu = (not is_ref and not early_ic and ref_status == "ok" and has_w)
            stats = PixelStatistics(
                I, counters, ic=ic, ic_err=float(ic_err[pixel]),
                w_abs=w_abs_value if has_w else None,
                w_err=float(w_err[pixel]) if has_w else None,
                mu_raw=float(mu_raw[pixel]) if allow_mu and mu_defined[pixel] else None,
                mu_raw_err=(float(mu_err[pixel]) if allow_mu and mu_err_defined[pixel]
                            else None),
                n_mu_loo_valid=(int(n_valid[pixel]) if allow_mu and mu_defined[pixel]
                                else None),
            )
            if allow_mu and not mu_defined[pixel]:
                raise ValueError(
                    f"Stage-14 total μ is unexpectedly undefined at pixel {pixel}"
                )
        x, y = grid.pixel_xy(pixel)
        rows.append(serialize_pixel(
            stats, pixel=pixel, x_um=m_to_um(x), y_um=m_to_um(y),
            is_reference=is_ref, ref_status=ref_status,
            n_jackknife_units=n_modes, thresholds=thresholds,
            screen=target.label))
    flag_counts = Counter(row["flag"] for row in rows)
    ref_warnings, ref_diagnostics = _ref_warnings(rows, grid, ref, n_modes)
    w_census = (Counter(
        w_signal_status(row["w_abs"], row["w_err"], thresholds)
        for row in rows if not row["is_reference"])
        if ref_status == "ok" else
        Counter({"unknown": sum(not row["is_reference"] for row in rows)}))
    over_mu_partial_loo = sum(
        row["flag"] == "over-mu"
        and row["n_mu_loo_valid"] is not None
        and row["n_mu_loo_valid"] < n_modes
        for row in rows
    )
    remediation = {
        "estimator-errors": flag_counts["negative-Ic"],
        "statistics-limited": sum(flag_counts[name] for name in (
            "solo-rays-only", "noisy-Ic", "no-ref-realizations",
            "noisy-mu", "over-mu")),
        "measured-null": flag_counts["null-Ic"],
        "usable": flag_counts["trusted"],
        "background-no-rays": flag_counts["no-rays"],
        "unclassified": flag_counts[None],
    }
    part_provenance = []
    mode_offset = 0
    for part in parts:
        part_provenance.append({
            "input_id": part.input_id,
            "cache_id": part.analysis_id,
            "path": part.path,
            "seed": part.seed,
            "n_modes": part.n_modes,
            "n_rays": part.n_rays,
            "local_mode_start": 0,
            "local_mode_end": part.n_modes,
            "global_mode_start": mode_offset,
            "global_mode_end": mode_offset + part.n_modes,
        })
        mode_offset += part.n_modes
    scatter_cells = scatter_meta["nx"] * scatter_meta["ny"]
    estimated_peak_rss = _estimated_peak_rss(npix, len(sim.lines), scatter_cells)
    cache_bytes_written = sum(
        sum(entry["bytes"] for entry in cache.meta["files"].values())
        for cache in cache_parts if not cache.cache_hit)
    target_miss_ray_bytes = sum(
        part.bytes
        for part, cache in zip(parts, cache_parts) if not cache.cache_hit)
    result_final = {
        "scatter": scatter_meta,
        "pass1_seconds": float(native.get("pass1_seconds", 0.0)),
        "pass2_seconds": float(native.get("pass2_seconds", 0.0)),
        "mode_rows_bytes_read": int(native.get("bytes_read", 0)),
        # Physical gzip I/O is shared by fan-out and belongs to the root run,
        # not independently to every screen report.
        "ray_archive_bytes_read": (
            fanout_metrics["physical_ray_archive_bytes_read"]
            if target.index == 0 else 0),
        "fanout_physical_ray_archive_bytes_read":
            fanout_metrics["physical_ray_archive_bytes_read"],
        "target_cache_miss_ray_archive_bytes": target_miss_ray_bytes,
        "cache_bytes_written": cache_bytes_written,
    }
    if target.output_subdir:
        os.mkdir(output_dir)
    jsonl_path = os.path.join(output_dir, "mu-jack.jsonl")
    with durable_open(jsonl_path, encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    written, save = _figure_saver(output_dir)
    maps = _stage14_figures(
        save, rows, aggregate, result_final, grid, ref,
        flag_counts, n_modes, target.label)
    _ref_passport(save, rows, ref, ref_status, ref_warnings, ref_diagnostics,
                  thresholds, n_modes, target.label)
    expected = {name for name in RESULT_PAYLOAD_NAMES if name.endswith(".svg")}
    if written != expected:
        raise ValueError(
            "stage-14 figures drifted from RESULT_PAYLOAD_NAMES: "
            f"missing {sorted(expected - written)}, "
            f"unexpected {sorted(written - expected)}")
    screen_seconds = time.time() - screen_started
    result_meta = {
        "stage_id": STAGE_ID, "screen": target.label,
        "schema": 1, "capsysred_version": __version__,
        "yaml_file": sim.cfg.yaml_file,
        "analysis_signature": target.signature,
        "input_cache_ids": [part.analysis_id for part in parts],
        "input_ids": [part.input_id for part in parts],
        "input_paths": [part.path for part in parts],
        "part_order": list(range(len(parts))),
        "parts": part_provenance,
        "screen_geometry": target.contract,
        "flag_thresholds": sim.cfg.stage14_flag_thresholds,
        "jackknife_computed": True,
        "n_jackknife_units": n_modes,
        "jackknife_unit": "mode",
        "ref_status": ref_status,
        "ref_warnings": ref_warnings,
        "ref_diagnostics": ref_diagnostics,
        "over_mu_partial_loo": over_mu_partial_loo,
        "flag_counts": {("null" if key is None else key): value
                        for key, value in flag_counts.items()},
        "w_signal_census": dict(w_census),
        "remediation_counts": remediation,
        "stats": aggregate["stats"],
        "bounce_hist": aggregate["bounce_hist"],
        "cache": {
            "hits": sum(cache.cache_hit for cache in cache_parts),
            "misses": sum(not cache.cache_hit for cache in cache_parts),
            "directories": [cache.input.cache_dir for cache in cache_parts],
            "payload_bytes": sum(cache.meta["files"][ROWS_NAME]["bytes"]
                                 for cache in cache_parts),
        },
        "performance": {
            **result_final,
            "total_seconds": screen_seconds,
            "screen_finalize_seconds": screen_seconds,
            "estimated_peak_rss_bytes": estimated_peak_rss,
        },
    }
    if write_meta:
        _write_json(os.path.join(output_dir, META_NAME), result_meta)
    prefix = os.path.join(RESULT_DIR, target.output_subdir)
    return {
        "maps": maps, "rows": rows, "screen": grid,
        "screen_name": target.label,
        "stats": aggregate["stats"], "bounce_hist": aggregate["bounce_hist"],
        "n_modes": n_modes, "n_rays": parts[0].n_rays,
        "ref_pixel": ref, "ref_status": ref_status,
        "ref_warnings": ref_warnings, "ref_diagnostics": ref_diagnostics,
        "flag_counts": flag_counts, "w_signal_census": w_census,
        "remediation_counts": remediation,
        "over_mu_partial_loo": over_mu_partial_loo,
        "cache_parts": cache_parts, "cache_hits": result_meta["cache"]["hits"],
        "seconds": screen_seconds, "result_meta": result_meta,
        "files": [os.path.join(prefix, name)
                  for name in (META_NAME, *RESULT_PAYLOAD_NAMES)],
    }


def run_stage14(sim, out_dir: str, rays_paths, log=print) -> dict:
    """Build all configured screen caches in one pass per input and publish."""
    run_started = time.time()
    result_dir = os.path.join(out_dir, RESULT_DIR)
    partial = result_dir + ".partial"
    preflight_stage14_output(out_dir)
    if sim.cfg.capillary is None:
        raise ValueError("stage 14 requires a configured capillary.source")

    targets = _screen_targets(sim, rays_paths)
    fanout_peak_rss = _estimated_fanout_peak_rss(targets, len(sim.lines))
    log("  Stage 14 screens: " + ", ".join(
        f"{target.label}={target.grid.nx}×{target.grid.ny}"
        for target in targets))
    log(f"  Stage 14 fan-out estimated peak RSS: "
        f"{fanout_peak_rss / (1024 ** 3):.2f} GiB")
    archives_read = _load_or_build_screen_caches(sim, targets, log)
    representative_parts = {
        part.input_id: part for part in targets[0].parts
    }
    physical_ray_bytes = sum(
        representative_parts[input_id].bytes
        for input_id in archives_read)
    written_caches = {}
    for target in targets:
        for cache in target.caches:
            if not cache.cache_hit:
                written_caches.setdefault(
                    os.path.normcase(cache.input.cache_dir), cache)
    physical_cache_bytes = sum(
        sum(entry["bytes"] for entry in cache.meta["files"].values())
        for cache in written_caches.values())
    fanout_metrics = {
        "physical_ray_archive_bytes_read": physical_ray_bytes,
        "physical_cache_bytes_written": physical_cache_bytes,
        "estimated_peak_rss_bytes": fanout_peak_rss,
    }

    # Cache construction may take hours.  Re-check the result namespace before
    # creating the one atomic publication tree; existing output is never
    # augmented or overwritten screen-by-screen.
    preflight_stage14_output(out_dir)
    os.mkdir(partial)
    results = []
    for target in targets:
        output_dir = (partial if not target.output_subdir else
                      os.path.join(partial, target.output_subdir))
        log(f"  Stage 14 finalize: {target.label}")
        results.append(_finalize_screen(
            sim, target, output_dir, fanout_metrics,
            write_meta=target.index != 0))

    global_seconds = time.time() - run_started
    aliases = {}
    for target in targets:
        for part in target.parts:
            aliases.setdefault(os.path.normcase(part.cache_dir), []).append(
                target.label)
    children = [{
        "screen": target.label,
        "output": target.output_subdir or ".",
        "screen_geometry": target.contract,
        "analysis_signature": target.signature,
        "input_cache_ids": [part.analysis_id for part in target.parts],
        "cache_directories": [part.cache_dir for part in target.parts],
    } for target in targets]
    fanout_manifest = {
        "schema": 1,
        "screens": children,
        "cache_aliases": [
            {"cache_directory": directory, "screens": labels}
            for directory, labels in aliases.items() if len(labels) > 1
        ],
        **fanout_metrics,
        "global_seconds": global_seconds,
    }
    primary = results[0]
    primary["seconds"] = global_seconds
    primary["result_meta"]["performance"].update({
        "total_seconds": global_seconds,
        "estimated_peak_rss_bytes": fanout_peak_rss,
    })
    primary["result_meta"]["fanout"] = fanout_manifest
    _write_json(os.path.join(partial, META_NAME), primary["result_meta"])

    # Child screen directories are moved as whole trees.  The main metadata is
    # deliberately last and therefore remains the validity marker for the
    # complete multi-screen result.
    publish_names = list(RESULT_PAYLOAD_NAMES)
    publish_names.extend(target.output_subdir for target in targets[1:])
    publish_names.append(META_NAME)
    _publish_result_tree(partial, result_dir, publish_names)

    primary["extra_results"] = results[1:]
    primary["fanout"] = fanout_manifest
    return primary


__all__ = ["preflight_stage14_output", "run_stage14"]

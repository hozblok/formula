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
import struct
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

from .. import __version__, _formula
from . import render
from .altcoh import FloatLineAmplitudes
from .progress import Progress
from .rays import (_validate_stream_metadata, geometry_metadata,
                   metadata_equal, read_metadata)
from .screen import ScatterRaster, ScreenGrid
from .stage14_flags import (FlagThresholds, PIXEL_FLAGS, PixelCounters,
                            PixelStatistics, serialize_pixel, validate_counters,
                            validate_ref, w_signal_status)
from .units import m_to_um


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
def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _identity(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _write_json(path: str, value) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def _stat_contract(value) -> dict:
    """Stable identity fields available from both stat() and fstat()."""
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def _publish_directory(partial: str, final: str, names: list[str]) -> None:
    """Publish regular files into an exclusively reserved directory.

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


def _screen_contract(cap, grid: ScreenGrid, ref: int) -> dict:
    return {
        "z": float(grid.z), "center": [grid.cxf, grid.cyf],
        "edge_x": grid.exf, "edge_y": grid.eyf,
        "nx": grid.nx, "ny": grid.ny, "reference_pixel": ref,
        "reference_xy": list(grid.pixel_xy(ref)),
        "pixel_convention": "half-open-row-major-iy-ix-v1",
    }


def _reference_pixel(cap, grid: ScreenGrid) -> int:
    xy = cap.screen.reference
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


@dataclass
class CachePart:
    input: InputPart
    meta: dict
    rows_path: str
    aggregates_path: str
    cache_hit: bool


def _prepare_inputs(sim, paths, quick: int, signature: dict) -> list[InputPart]:
    if not paths:
        raise ValueError("stage 14 requires at least one rays archive")
    expected_core = _trace_core(geometry_metadata(sim.cfg))
    parts = []
    seeds, input_ids = set(), set()
    expected_rays = None
    for pathlike in paths:
        path = os.path.abspath(os.fspath(pathlike))
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
        stat = os.stat(path)
        snapshot = _json_copy(meta)
        concrete = {
            "rays_metadata": snapshot,
            "resolved_path": os.path.normcase(os.path.realpath(path)),
            # Rays archives are immutable under the no-clobber contract.  The
            # descriptor is checked again before and after the one strict
            # cache-building pass, closing path/symlink replacement windows.
            "archive_stat": _stat_contract(stat),
        }
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
        cache_dir = os.path.join(os.path.dirname(path), CACHE_DIR, analysis_id)
        parts.append(InputPart(path, meta, n_modes, n_rays,
                               identity["source_screen_z"], input_id, seed,
                               analysis_id, cache_dir, identity))
    configured_modes, configured_rays = sim.cfg.capillary.source.budget(quick)
    total_modes = sum(part.n_modes for part in parts)
    if (total_modes, expected_rays) != (configured_modes, configured_rays):
        raise ValueError(
            "stage 14 replay budgets differ from config/--quick: "
            f"recorded union {[total_modes, expected_rays]} != "
            f"configured {[configured_modes, configured_rays]}"
        )
    return parts


def _native_store(path, n_modes, n_pixels, ref, kms, weights):
    cls = getattr(_formula, "Stage14Store", None)
    if cls is None:
        raise RuntimeError(
            "the native extension predates Stage 14; rebuild/install formula"
        )
    return cls(path, n_modes, n_pixels, ref, kms, weights)


@contextmanager
def _archive_text(part: InputPart):
    """Open the exact archive inode selected during input preparation."""
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
                yield text_fh
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


def _project_row(row: dict, source_z: float, grid: ScreenGrid):
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
        return x, y, None, opl
    step = (float(grid.z) - source_z) / dz
    x, y, opl = x + dx * step, y + dy * step, opl + step
    if not all(map(math.isfinite, (x, y, opl))):
        raise ValueError("re-projected screen ray is non-finite")
    return x, y, grid.pixel((x, y)), opl


def _row_ids(row: dict, path: str):
    mode, ray = row.get("mode"), row.get("ray")
    if (type(mode) is not int or type(ray) is not int
            or mode < 0 or ray < 0):
        raise ValueError(f"{path}: ray mode/ray ids must be nonnegative integers")
    return mode, ray


def _build_stream(sim, part: InputPart, grid: ScreenGrid, store,
                  scatter: ScatterRaster, log) -> tuple[dict, list[int]]:
    """Strict one-pass archive validation and target-scene deposit."""
    amps_of = _amplitudes_factory(sim)
    expected = part.n_modes * part.n_rays
    progress = Progress("14 cache capillary", expected)
    counts, trailers, known_seen = {}, {}, set()
    stats = {name: 0 for name in STREAM_COUNTERS}
    bounce_hist = [0] * (sim.cfg.max_bounces + 1)
    current_mode = None
    target_rows = 0
    try:
        with _archive_text(part) as fh:
            preamble = fh.readline()
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
                        store.fold_mode()
                    store.begin_mode(mode)
                    current_mode = mode
                target_rows += 1
                stats["emitted"] += 1
                nb = len(sins)
                if nb > sim.cfg.max_bounces:
                    raise ValueError(f"{part.path}:{line_no}: too many reflections")
                bounce_hist[nb] += 1
                if nb:
                    stats["reflected_rays"] += 1
                    stats["reflections"] += nb
                amps = None
                if fate == "screen":
                    amps = amps_of(sins)
                    if (sim.cfg.amplitude_min > 0.0
                            and max(abs(a) for a in amps) < sim.cfg.amplitude_min):
                        fate = "absorbed"
                if fate == "screen":
                    x, y, pixel, opl = _project_row(row, part.source_z, grid)
                    scatter.add((x, y))
                    if pixel is None:
                        stats["off_window"] += 1
                    else:
                        store.add_ray(pixel, opl, amps)
                        stats["screen"] += 1
                else:
                    stats[fate] += 1
                progress.step()
        if current_mode is not None:
            store.fold_mode()
    except BaseException:
        progress.finish("failed")
        raise
    progress.finish(f"on screen {stats['screen']:,}")
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
    if stats["emitted"] != sum(stats[name] for name in
                                ("screen", "off_window", "absorbed", "lost")):
        raise ValueError(f"{part.path}: inconsistent Stage-14 stream counters")
    log(f"  stage 14 cache input validated: {part.n_modes} modes × {part.n_rays} rays")
    return stats, bounce_hist


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
    with open(path, "xb") as fh:
        for chunk in chunks:
            fh.write(chunk)
            h.update(chunk)
        fh.flush()
        os.fsync(fh.fileno())
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


def _build_cache(sim, part: InputPart, signature: dict, grid: ScreenGrid,
                 ref: int, log) -> CachePart:
    npix = grid.nx * grid.ny
    scatter = ScatterRaster(sim.cfg.capillary.screen)
    cache_root = os.path.dirname(part.cache_dir)
    os.makedirs(cache_root, exist_ok=True)
    partial = part.cache_dir + ".partial"
    required = part.n_modes * npix * 24 + _expected_aggregate_size(
        npix, sim.cfg.max_bounces, scatter.nx * scatter.ny)
    rss_estimate = _estimated_peak_rss(
        npix, len(sim.lines), scatter.nx * scatter.ny)
    free = shutil.disk_usage(cache_root).free
    log(f"  stage 14 cache: {required / (1024 ** 3):.3f} GiB required; "
        f"{free / (1024 ** 3):.1f} GiB free; "
        f"estimated peak RSS {rss_estimate / (1024 ** 3):.2f} GiB")
    if free < required + 64 * 1024 * 1024:
        raise ValueError(
            f"{cache_root}: insufficient free space for Stage-14 cache"
        )
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
    store = _native_store(rows_path, part.n_modes, npix, ref, kms, weights)
    t0 = time.time()
    try:
        stats, bounce_hist = _build_stream(
            sim, part, grid, store, scatter, log)
        native = store.finish()
        rows_size = int(native["payload_bytes"])
        rows_sha = str(native["payload_sha256"])
        expected_rows = part.n_modes * npix * 24
        if rows_size != expected_rows or os.path.getsize(rows_path) != expected_rows:
            raise ValueError("native Stage-14 store wrote the wrong payload size")
        if sum(_typed_array("Q", native["n_rays"])) != stats["screen"]:
            raise ValueError("Stage-14 density total differs from screen counter")
        agg_size, agg_sha = _write_aggregates(
            aggregate_path, native, npix, stats, bounce_hist, scatter)
        for path in (rows_path, aggregate_path):
            with open(path, "rb+") as fh:
                os.fsync(fh.fileno())
        build_seconds = time.time() - t0
        meta = {
            "cache_schema": CACHE_SCHEMA,
            "estimator_version": ESTIMATOR_VERSION,
            "analysis_id": part.analysis_id,
            "analysis_signature": signature,
            "identity": part.identity,
            "stage_id": STAGE_ID,
            "screen": "capillary",
            "input_id": part.input_id,
            "input_path": part.path,
            "source_screen_z": part.source_z,
            "n_modes": part.n_modes,
            "n_rays_per_mode": part.n_rays,
            "n_pixels": npix,
            "reference_pixel": ref,
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
            "build": {"seconds": build_seconds,
                      "ray_archive_bytes": os.path.getsize(part.path),
                      "yaml_file": sim.cfg.yaml_file},
            "capsysred_version": __version__,
        }
        _write_json(os.path.join(partial, META_NAME), meta)
        _publish_directory(
            partial, part.cache_dir,
            [ROWS_NAME, AGGREGATES_NAME, META_NAME],
        )
    except BaseException:
        # The exclusive .partial directory is deliberately retained as an
        # unmistakable failed build; no valid-looking meta exists at final.
        raise
    return CachePart(part, meta, os.path.join(part.cache_dir, ROWS_NAME),
                     os.path.join(part.cache_dir, AGGREGATES_NAME), False)


def _preflight_cache_builds(parts: list[InputPart], cached_parts,
                            npix: int, max_bounces: int,
                            scatter_cells: int, log) -> None:
    """Check the total missing-cache footprint per physical filesystem."""
    aggregate_bytes = _expected_aggregate_size(
        npix, max_bounces, scatter_cells)
    groups = {}
    for part, cached in zip(parts, cached_parts):
        if cached is not None:
            continue
        probe = os.path.dirname(os.path.dirname(part.cache_dir))
        stat = os.stat(probe)
        group = groups.setdefault(
            stat.st_dev, {"probe": probe, "bytes": 0, "parts": 0})
        group["bytes"] += part.n_modes * npix * 24 + aggregate_bytes
        group["parts"] += 1
    for group in groups.values():
        free = shutil.disk_usage(group["probe"]).free
        required = group["bytes"]
        log(f"  Stage 14 disk preflight: {group['parts']} cache part(s), "
            f"{required / (1024 ** 3):.3f} GiB required; "
            f"{free / (1024 ** 3):.1f} GiB free")
        if free < required + 64 * 1024 * 1024:
            raise ValueError(
                f"{group['probe']}: insufficient free space for all missing "
                "Stage-14 cache parts"
            )


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


def _stage14_figures(result_dir: str, rows, aggregate, final, grid: ScreenGrid,
                     ref: int, flag_counts: Counter, n_modes: int):
    nx, ny, npix = grid.nx, grid.ny, grid.nx * grid.ny
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
    sub = f"{n_modes} exact delete-one-mode units; raw μ, display clipped at 1"
    main = render.hstack([
        render.heatmap(mu_grid, extent, "|μ_raw(P,P_ref)|", "x, µm", "y, µm",
                       sub, "|μ|", vmax=1.0, mark=mark, w=430, equal=True),
        render.heatmap(err_grid, extent, "σ_jack(μ_raw)", "x, µm", "y, µm",
                       "null is masked", "σ", mark=mark, w=430, equal=True),
        render.category_map(flag_grid, extent, "Stage-14 pixel flags", "x, µm",
                            "y, µm", "first-match normative taxonomy", mark=mark,
                            counts=flag_counts,
                            lit_counts=lit_flag_counts,
                            lit_total=sum(row["n_rays"] > 0 for row in rows),
                            w=600, equal=True),
    ])
    render.save(os.path.join(result_dir, "14-capillary-jack-mu.svg"), main)
    iy = ref // nx
    xs = [m_to_um(x) for x in grid.xs()]
    row_ids = range(iy * nx, (iy + 1) * nx)
    good = [p for p in row_ids if rows[p]["mu_raw"] is not None]
    with_err = [p for p in good if rows[p]["mu_raw_err"] is not None]
    series = []
    if with_err:
        series.append({
            "xs": [xs[p % nx] for p in with_err],
            "ys": [min(rows[p]["mu_raw"], 1.0) for p in with_err],
            "lo": [max(rows[p]["mu_raw"] - rows[p]["mu_raw_err"], 0.0)
                   for p in with_err],
            "hi": [min(rows[p]["mu_raw"] + rows[p]["mu_raw_err"], 1.0)
                   for p in with_err],
            "label": "min(μ_raw,1) ± σ_jack",
        })
    without_err = [p for p in good if rows[p]["mu_raw_err"] is None]
    if without_err:
        series.append({"xs": [xs[p % nx] for p in without_err],
                       "ys": [min(rows[p]["mu_raw"], 1.0) for p in without_err],
                       "label": "μ_raw; σ unavailable", "dots": True,
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
    render.save(os.path.join(result_dir, "14a-capillary-jack-slice.svg"),
                render.line_chart(series, "Stage-14 reference-row slice", "x, µm",
                                  "|μ|", f"y={m_to_um(grid.ys()[iy]):.3g} µm",
                                  vlines=[(m_to_um(ref_xy[0]), "ref")], w=760))
    if ny > 1:
        intensity_fig = render.hstack([
            render.heatmap(i_grid, extent, "intensity", "x, µm", "y, µm", sub,
                           "I", w=500, equal=True),
            render.heatmap(d_grid, extent, "ray density", "x, µm", "y, µm", "",
                           "rays", w=500, equal=True),
        ])
    else:
        imax, dmax = max(intensity) or 1.0, max(density) or 1.0
        intensity_fig = render.line_chart([
            {"xs": xs, "ys": [v / imax for v in intensity], "label": "I/max"},
            {"xs": xs, "ys": [v / dmax for v in density], "label": "rays/max",
             "dash": "6,4"}], "intensity and density", "x, µm", "normalized",
            sub, w=760)
    render.save(os.path.join(result_dir, "14b-capillary-jack-intensity.svg"),
                intensity_fig)
    render.save(os.path.join(result_dir, "14c-capillary-jack-overlay.svg"),
                render.overlay_map(mu_grid, flag_grid, extent,
                                   "Stage-14 non-trusted overlay", "x, µm", "y, µm",
                                   sub, mark=mark, equal=True))
    scatter_meta = final["scatter"]
    scatter_grid = _grid(aggregate["scatter"], scatter_meta["nx"], scatter_meta["ny"])
    scatter_extent = (m_to_um(scatter_meta["x0"]),
                      m_to_um(scatter_meta["x0"] + scatter_meta["edge_x"]),
                      m_to_um(scatter_meta["y0"]),
                      m_to_um(scatter_meta["y0"] + scatter_meta["edge_y"]))
    render.save(os.path.join(result_dir, "14d-capillary-ray-scatter.svg"),
                render.ray_scatter(scatter_grid, scatter_extent,
                                   "capillary: ray locations on target screen",
                                   "x, µm", "y, µm", sub))
    return {
        "mu_raw": mu_grid, "mu_raw_err": err_grid, "flag": flag_grid,
        "intensity": i_grid, "density": d_grid,
    }


def preflight_stage14_output(out_dir: str) -> None:
    """Refuse an existing/partial result before tracing or cache work."""
    result_dir = os.path.join(out_dir, RESULT_DIR)
    partial = result_dir + ".partial"
    if os.path.lexists(result_dir) or os.path.lexists(partial):
        raise ValueError(
            f"{result_dir}: Stage-14 result already exists; remove it manually "
            "or choose another output directory"
        )


def run_stage14(sim, out_dir: str, rays_paths, quick: int, log=print) -> dict:
    """Build/reuse per-input caches, finalize their logical union and publish."""
    t0 = time.time()
    result_dir = os.path.join(out_dir, RESULT_DIR)
    partial = result_dir + ".partial"
    preflight_stage14_output(out_dir)
    cap = sim.cfg.capillary
    if cap is None:
        raise ValueError("stage 14 requires a configured capillary.source")
    grid = ScreenGrid(cap.screen)
    ref = _reference_pixel(cap, grid)
    npix = grid.nx * grid.ny
    screen_contract = _screen_contract(cap, grid, ref)
    signature = _analysis_signature(sim, screen_contract)
    parts = _prepare_inputs(sim, rays_paths, quick, signature)
    # Preflight every part before starting a potentially multi-hour cache
    # build.  A conflict in part 5 must not be discovered only after parts
    # 1--4 have already consumed their archives.
    cache_parts = []
    scatter_contract = ScatterRaster(cap.screen)
    for part in parts:
        cache_parts.append(_load_cache(
            part, signature, npix, sim.cfg.max_bounces,
            scatter_contract))
    _preflight_cache_builds(
        parts, cache_parts, npix, sim.cfg.max_bounces,
        scatter_contract.nx * scatter_contract.ny, log)
    for index, (part, cached) in enumerate(zip(parts, cache_parts)):
        if cached is None:
            log(f"  Stage 14 cache miss: {part.path}")
            cached = _build_cache(sim, part, signature, grid, ref, log)
        else:
            log(f"  Stage 14 cache hit: {cached.input.cache_dir}")
        cache_parts[index] = cached
    scatter_meta = cache_parts[0].meta["scatter"]
    for cache in cache_parts[1:]:
        if (not metadata_equal(cache.meta["analysis_signature"],
                               cache_parts[0].meta["analysis_signature"])
                or not metadata_equal(cache.meta["scatter"], scatter_meta)):
            raise ValueError("Stage-14 union cache parts are incompatible")
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
            lower = ic - thresholds.z * float(ic_err[pixel])
            upper = ic + thresholds.z * float(ic_err[pixel])
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
            n_jackknife_units=n_modes, thresholds=thresholds))
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
    ray_bytes_read = sum(
        os.path.getsize(cache.input.path)
        for cache in cache_parts if not cache.cache_hit)
    result_final = {
        "scatter": scatter_meta,
        "pass1_seconds": float(native.get("pass1_seconds", 0.0)),
        "pass2_seconds": float(native.get("pass2_seconds", 0.0)),
        "mode_rows_bytes_read": int(native.get("bytes_read", 0)),
        "ray_archive_bytes_read": ray_bytes_read,
        "cache_bytes_written": cache_bytes_written,
    }
    # Repeat the exclusive preflight after the potentially long finalize to
    # catch a concurrent publisher without overwriting it.
    preflight_stage14_output(out_dir)
    os.mkdir(partial)
    jsonl_path = os.path.join(partial, "mu-jack.jsonl")
    with open(jsonl_path, "x", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    maps = _stage14_figures(partial, rows, aggregate, result_final, grid, ref,
                            flag_counts, n_modes)
    total_seconds = time.time() - t0
    result_meta = {
        "stage_id": STAGE_ID, "screen": "capillary",
        "schema": 1, "capsysred_version": __version__,
        "yaml_file": sim.cfg.yaml_file,
        "analysis_signature": signature,
        "input_cache_ids": [part.analysis_id for part in parts],
        "input_ids": [part.input_id for part in parts],
        "input_paths": [part.path for part in parts],
        "part_order": list(range(len(parts))),
        "parts": part_provenance,
        "screen_geometry": screen_contract,
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
            "total_seconds": total_seconds,
            "estimated_peak_rss_bytes": estimated_peak_rss,
        },
    }
    _write_json(os.path.join(partial, META_NAME), result_meta)
    _publish_directory(
        partial, result_dir,
        ["mu-jack.jsonl", "14-capillary-jack-mu.svg",
         "14a-capillary-jack-slice.svg", "14b-capillary-jack-intensity.svg",
         "14c-capillary-jack-overlay.svg", "14d-capillary-ray-scatter.svg",
         META_NAME],
    )
    return {
        "maps": maps, "rows": rows, "screen": grid,
        "stats": aggregate["stats"], "bounce_hist": aggregate["bounce_hist"],
        "n_modes": n_modes, "n_rays": parts[0].n_rays,
        "ref_pixel": ref, "ref_status": ref_status,
        "ref_warnings": ref_warnings, "ref_diagnostics": ref_diagnostics,
        "flag_counts": flag_counts, "w_signal_census": w_census,
        "remediation_counts": remediation,
        "over_mu_partial_loo": over_mu_partial_loo,
        "cache_parts": cache_parts, "cache_hits": result_meta["cache"]["hits"],
        "seconds": total_seconds, "result_meta": result_meta,
        "files": [os.path.join(RESULT_DIR, name) for name in (
            META_NAME, "mu-jack.jsonl", "14-capillary-jack-mu.svg",
            "14a-capillary-jack-slice.svg", "14b-capillary-jack-intensity.svg",
            "14c-capillary-jack-overlay.svg", "14d-capillary-ray-scatter.svg")],
    }


__all__ = ["preflight_stage14_output", "run_stage14"]

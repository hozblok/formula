"""Stage 14 archive contracts, cache construction and parallel workers."""

from __future__ import annotations

from array import array
from collections.abc import Iterator
import concurrent.futures
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import operator
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

from .... import __version__, _formula
from ... import rays_v3
from ...fresnel import FloatLineAmplitudes
from ...env import Env
from .._contracts import _physics_contract
from .checkpoints import (
    RANGE_SLOTS,
    mode_ranges,
    range_agg_name,
    range_done,
    range_manifest_name,
    range_rows_name,
    read_range_manifest,
)
from ...shared.progress import Progress
from ...rays import RNG_SCHEME, geometry_metadata, metadata_equal, recorded_screen_z
from ...screen import ScatterRaster, ScreenGrid
from ...shared.utils import durable_open


STAGE_ID = 14
CACHE_SCHEMA = 1
ESTIMATOR_VERSION = 1
CACHE_DIR = "stage14-cache"
ROWS_NAME = "mode-rows.f64"
AGGREGATES_NAME = "aggregates.bin"
META_NAME = "meta.json"
AGG_MAGIC = b"CPS14AGG"
AGG_FORMAT = 1
STREAM_COUNTERS = ("emitted", "screen", "absorbed", "lost", "off_window",
                   "reflected_rays", "reflections")
_KNOWN_RNG_SCHEMES = (RNG_SCHEME,)


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


def _publish_cache_tree(ready: str, final: str) -> None:
    """One-rename cache publication on every platform (single-writer).

    POSIX rename may replace an *empty* destination, so probe first; the
    accepted single-writer scope makes the probe race-free enough."""
    if os.path.lexists(final):
        raise ValueError(f"{final}: publication conflict; remove it manually")
    try:
        os.rename(ready, final)
    except OSError as exc:
        raise ValueError(
            f"{final}: publication conflict; remove it manually"
        ) from exc


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


def _recorded_screen_z(meta: dict, path: str) -> float:
    try:
        z = recorded_screen_z(meta["geometry"], "capillary")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: rays metadata has no capillary screen z") from exc
    if z is None:
        raise ValueError(f"{path}: rays metadata has no capillary screen z")
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
    bytes: int
    index: object


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
        if not rays_v3.is_v3(path):
            raise ValueError(f"{path}: not a v3 rays archive directory")
        index = rays_v3.load_index(path)
        meta = rays_v3.metadata(path)
        # Jackknife counts modes as independent realizations: that is proven
        # only for recorded lattice-v1 provenance (stream keys
        # "{seed}/{scene}/{mode}" are disjoint for distinct seeds).
        rng = meta.get("rng")
        scheme = rng.get("scheme") if isinstance(rng, dict) else rng
        if scheme not in _KNOWN_RNG_SCHEMES:
            raise ValueError(
                f"{path}: rays metadata lacks a known rng scheme "
                f"(got {scheme!r}); realization independence is unproven — "
                "re-record with provenance")
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
        # v3 archives are content-addressed: the index lists every section's
        # sha256, and each section is hashed while streamed.
        concrete = {
            "rays_metadata": snapshot,
            "resolved_path": os.path.normcase(os.path.realpath(path)),
            "index_sha256": rays_v3.index_digest(path),
        }
        archive_bytes = index.total_bytes()
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
        # Env.stage14_cache relocates the cache root away from the archive.
        cache_root = Env.stage14_cache()
        if cache_root:
            os.makedirs(cache_root, exist_ok=True)
        else:
            cache_root = os.path.join(os.path.dirname(path), CACHE_DIR)
        cache_dir = os.path.join(cache_root, analysis_id)
        parts.append(InputPart(path, meta, n_modes, n_rays,
                               identity["source_screen_z"], input_id, seed,
                               analysis_id, cache_dir, identity,
                               archive_bytes, index))
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
    configs = cap.all_screens
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
def _archive_lines(part: InputPart,
                   mode_range: tuple[int, int] | None = None,
                   ) -> Iterator[Iterator[str]]:
    """Iterator of archive text lines: the capillary sections mode-major
    (each hashed against the index) behind a synthetic preamble and scene
    trailer, so the strict row loop sees one stream.  ``mode_range``
    restricts it to modes [m0, m1).
    """
    m0, m1 = mode_range if mode_range else (0, part.n_modes)
    expected = part.identity["input"]["index_sha256"]
    if rays_v3.index_digest(part.path) != expected:
        raise ValueError(f"{part.path}: rays index changed before cache build")

    def lines():
        yield "{}\n"
        for sections in part.index.modes("capillary")[m0:m1]:
            for entry in sections:
                for raw in rays_v3.iter_section_lines(part.path, entry):
                    yield raw.decode("utf-8")
        yield json.dumps({"scene_end": "capillary",
                          "rows": (m1 - m0) * part.n_rays}) + "\n"

    yield lines()
    if rays_v3.index_digest(part.path) != expected:
        raise ValueError(f"{part.path}: rays index changed during cache build")


def _amplitudes_factory(sim):
    if sim.per_line or len(sim.lines) == 1:
        return FloatLineAmplitudes(sim.cfg.material, sim.lines, sim.numbers)
    central = FloatLineAmplitudes(
        sim.cfg.material, [SimpleNamespace(e_kev=sim.cfg.energy_kev)],
        sim.numbers,
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


def _row_ids(row: dict, path: str):
    mode, ray = row.get("mode"), row.get("ray")
    if (type(mode) is not int or type(ray) is not int
            or mode < 0 or ray < 0):
        raise ValueError(f"{path}: ray mode/ray ids must be nonnegative integers")
    return mode, ray


def _validate_row(row: dict, line_no: int, path: str, trailers: dict):
    """Shape checks of one ray row (``trailers`` is read only); returns
    the parsed ``(scene, mode, ray, sins, fate)``."""
    scene = row.get("stage")
    if not isinstance(scene, str):
        raise ValueError(f"{path}:{line_no}: ray row lacks stage")
    if scene in trailers:
        raise ValueError(
            f"{path}:{line_no}: {scene} row follows its trailer"
        )
    mode, ray = _row_ids(row, path)
    if "pixel" not in row or not (
            row["pixel"] is None
            or (type(row["pixel"]) is int and row["pixel"] >= 0)):
        raise ValueError(
            f"{path}:{line_no}: invalid saved pixel id"
        )
    try:
        opl_value = float(row["opl"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_no}: invalid optical path length"
        ) from exc
    if not math.isfinite(opl_value):
        raise ValueError(
            f"{path}:{line_no}: non-finite optical path length"
        )
    sins_raw = row.get("sins")
    if not isinstance(sins_raw, list):
        raise ValueError(f"{path}:{line_no}: sins must be a list")
    try:
        sins = [float(value) for value in sins_raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no}: invalid sins") from exc
    if not all(math.isfinite(value) for value in sins):
        raise ValueError(f"{path}:{line_no}: non-finite sins")
    fate = row.get("fate")
    if fate not in ("screen", "absorbed", "lost"):
        raise ValueError(
            f"{path}:{line_no}: invalid ray fate {fate!r}"
        )
    if fate == "screen":
        try:
            screen_values = tuple(float(row[name]) for name in
                                  ("x", "y", "dx", "dy"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_no}: screen ray lacks finite "
                "x/y/direction"
            ) from exc
        if not all(math.isfinite(value) for value in screen_values):
            raise ValueError(
                f"{path}:{line_no}: screen ray has non-finite "
                "x/y/direction"
            )
    return scene, mode, ray, sins, fate


def _build_stream_fanout(sim, part: InputPart, builds: list[CacheBuild],
                         log, mode_range=None,
                         quiet=False) -> tuple[list[dict], list[int]]:
    """Strictly decode one archive (or a v3 mode range) into every missing
    screen cache."""
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
    m0, m1 = mode_range if mode_range else (0, part.n_modes)
    expected = (m1 - m0) * part.n_rays
    suffix = "s" if len(builds) != 1 else ""
    progress = Progress(
        f"14 cache capillary fan-out ({len(builds)} screen{suffix})", expected,
        silent=quiet)
    counts, trailers, known_seen = {}, {}, set()
    stats_rows = [dict.fromkeys(STREAM_COUNTERS, 0) for _ in builds]
    bounce_hist = [0] * (sim.cfg.max_bounces + 1)
    current_mode = None
    target_rows = 0
    try:
        archive = (_archive_lines(part, mode_range) if mode_range is not None
                   else _archive_lines(part))
        with archive as fh:
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
                scene, mode, ray, sins, fate = _validate_row(
                    row, line_no, part.path, trailers)
                counts[scene] = counts.get(scene, 0) + 1
                budget = part.meta["budgets"].get(scene)
                if budget:
                    idx = counts[scene] - 1
                    if scene == "capillary":
                        idx += m0 * budget[1]
                    if (mode, ray) != divmod(idx, budget[1]):
                        raise ValueError(
                            f"{part.path}:{line_no}: non-canonical {scene} mode/ray"
                        )
                known_seen.add(scene)
                if scene != "capillary":
                    continue
                if target_rows >= expected or (mode, ray) != divmod(
                        m0 * part.n_rays + target_rows, part.n_rays):
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
        expected_scene = ((m1 - m0) * budget[1] if scene == "capillary"
                          else math.prod(budget)) if budget else None
        if budget and counts[scene] != expected_scene:
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
    log(f"  stage 14 cache input validated: modes {m0}..{m1} of {part.n_modes} "
        f"× {part.n_rays} rays")
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


def _estimated_fanout_peak_rss(targets: list[ScreenTarget], n_lines: int,
                               jobs: int = 1) -> int:
    """Conservative peak for simultaneous stores and retained screen results."""
    unique_builds = {}
    for target in targets:
        key = _canonical(target.signature)
        unique_builds.setdefault(key, target)
    workers = _effective_workers(jobs, RANGE_SLOTS)
    native_build = 512 * 1024 * 1024 + workers * sum(
        target.grid.nx * target.grid.ny * (24 * n_lines + 128)
        + target.scatter.nx * target.scatter.ny * 40
        for target in unique_builds.values())
    # Parent-side assembly: the double-field tile budget plus one running
    # total and one parsed item (56 B/pixel each).
    fold = (_TILE_BUDGET + 2 * 56 * max(
        target.grid.nx * target.grid.ny
        for target in unique_builds.values()) if jobs > 1 else 0)
    retained_results = 512 * 1024 * 1024 + fold + sum(
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
    return _parse_aggregates(data, part.aggregates_path, npix, max_bounces,
                             scatter_cells)


def _parse_aggregates(data: bytes, label: str, npix: int, max_bounces: int,
                      scatter_cells: int) -> dict:
    expected_size = _expected_aggregate_size(npix, max_bounces, scatter_cells)
    if len(data) != expected_size:
        raise ValueError(f"{label}: truncated aggregates")
    if struct.unpack_from("<8sII", data, 0) != (AGG_MAGIC, AGG_FORMAT, 0):
        raise ValueError(f"{label}: invalid aggregate header")
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
        raise ValueError(f"{label}: aggregate trailing bytes")
    return out


def _load_cache(part: InputPart, signature: dict, npix: int,
                max_bounces: int, scatter: ScatterRaster,
                allow_partial: bool = False) -> CachePart | None:
    """A valid final always wins; ``allow_partial`` (parallel builds) makes a
    leftover .partial resume state instead of an error when no final exists."""
    final, partial = part.cache_dir, part.cache_dir + ".partial"
    if not os.path.lexists(final):
        if os.path.lexists(partial) and not allow_partial:
            raise ValueError(
                f"{partial}: incomplete Stage-14 cache exists; remove it "
                "manually or resume with CAPSYSRED_STAGE14_JOBS > 1"
            )
        return None
    meta_path = os.path.join(final, META_NAME)
    if not os.path.lexists(meta_path):
        # The cache is published as one directory rename, so a meta-less
        # final can never be ours: fail closed, never auto-delete.
        raise ValueError(
            f"{final}: cache directory without {META_NAME}; remove it manually"
        )
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
    if os.path.lexists(partial):
        # Orphan of a crash between publication and cleanup (single-writer):
        # a validated final wins over any leftover partial, whatever jobs is.
        shutil.rmtree(partial)
    return cache


def _prepare_cache_build(sim, target: ScreenTarget, part: InputPart,
                         with_store: bool = True) -> CacheBuild:
    """Exclusively reserve one missing cache and construct its native store.

    ``with_store=False`` (parallel range build): the partial directory is
    reused if present — completed range files inside it resume the build —
    and no whole-archive store is opened."""
    cache_root = os.path.dirname(part.cache_dir)
    os.makedirs(cache_root, exist_ok=True)
    partial = part.cache_dir + ".partial"
    if with_store:
        try:
            os.mkdir(partial)
        except FileExistsError as exc:
            raise ValueError(
                f"{partial}: cache conflict; remove it manually or choose another input"
            ) from exc
    else:
        os.makedirs(partial, exist_ok=True)
    rows_path = os.path.join(partial, ROWS_NAME)
    kms = [float(line.k) for line in sim.lines]
    weights = [float(line.weight) for line in sim.lines]
    npix = target.grid.nx * target.grid.ny
    store = (_native_store(rows_path, part.n_modes, npix, target.ref,
                           kms, weights) if with_store else None)
    return CacheBuild(
        target, part, partial, rows_path, store,
        ScatterRaster(target.cfg), time.time(),
    )


def _finish_cache_build(sim, build: CacheBuild, stats: dict,
                        bounce_hist: list[int], native: dict | None = None) -> CachePart:
    """Validate, serialize and publish one fan-out store after stream EOF.

    ``native`` overrides the live store (parallel path: summed range
    aggregates plus the concatenated payload sha/bytes)."""
    target, part = build.target, build.part
    grid, scatter = target.grid, build.scatter
    npix = grid.nx * grid.ny
    if native is None:
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
    # The publishable directory; a stale one was dropped before assembly.
    ready = os.path.join(build.partial, "ready.tmp")
    os.mkdir(ready)
    os.rename(build.rows_path, os.path.join(ready, ROWS_NAME))
    agg_size, agg_sha = _write_aggregates(
        os.path.join(ready, AGGREGATES_NAME), native, npix, stats,
        bounce_hist, scatter)
    for name in (ROWS_NAME, AGGREGATES_NAME):
        with open(os.path.join(ready, name), "rb+") as fh:
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
    _write_json(os.path.join(ready, META_NAME), meta)
    # One atomic directory rename: a meta-less final cannot exist, and the
    # range checkpoints in the partial survive until the cache is published.
    _publish_cache_tree(ready, part.cache_dir)
    try:
        shutil.rmtree(build.partial)
    except OSError:
        pass    # published; an orphan partial is removed on the next hit
    return CachePart(
        part, meta, os.path.join(part.cache_dir, ROWS_NAME),
        os.path.join(part.cache_dir, AGGREGATES_NAME), False)


# ProcessPoolExecutor on Windows is limited by WaitForMultipleObjects.
_MAX_NT_WORKERS = 61

# Bounded-memory reducer: per double field, at most this many bytes of range
# tiles coexist; every pixel still gets ONE global math.fsum over all ranges.
_TILE_BUDGET = 256 * 1024 * 1024
_DOUBLE_FIELDS = ("I", "w_re", "w_im", "ic")


def _effective_workers(jobs: int, tasks: int) -> int:
    workers = max(1, min(jobs, tasks))
    if os.name == "nt":
        workers = min(workers, _MAX_NT_WORKERS)
    return workers


def _fold_int_aggregates(total: dict, item: dict) -> None:
    """In-place fold of the integer aggregate fields; exact in any order.

    m_* counters widen to "Q" while folding; _seal_int_aggregates narrows
    them back after the uint32 overflow check."""
    for name in ("n_rays", "m_realizations", "m_pair_realizations",
                 "m_ref_realizations"):
        total[name] = array("Q", map(operator.add, total[name], item[name]))
    total["max_rays_per_realization"] = array(
        "I", map(max, total["max_rays_per_realization"],
                 item["max_rays_per_realization"]))
    total["stats"] = {name: total["stats"][name] + item["stats"][name]
                      for name in STREAM_COUNTERS}
    total["bounce_hist"] = [a + b for a, b in
                            zip(total["bounce_hist"], item["bounce_hist"])]
    total["scatter"] = [a + b for a, b in
                        zip(total["scatter"], item["scatter"])]


def _seal_int_aggregates(total: dict) -> None:
    for name in ("m_realizations", "m_pair_realizations",
                 "m_ref_realizations"):
        values = total[name]
        if len(values) and max(values) > 0xFFFFFFFF:
            raise ValueError(f"summed Stage-14 {name} overflows uint32")
        if values.typecode != "I":
            total[name] = array("I", values)


def _fsum_double_fields(paths: list[str], npix: int,
                        tile_budget: int = _TILE_BUDGET) -> dict:
    """One global math.fsum per pixel across every range file, bounded RAM.

    The double fields sit first in the fixed aggregate layout; each is read
    in pixel tiles from all files — no windowed folding ever happens."""
    tile = max(1, tile_budget // (8 * max(1, len(paths))))
    out = {}
    for index, name in enumerate(_DOUBLE_FIELDS):
        base = 16 + index * 8 * npix
        total = array("d")
        handles = [open(path, "rb") for path in paths]
        try:
            for p0 in range(0, npix, tile):
                count = min(tile, npix - p0)
                tiles = []
                for fh in handles:
                    fh.seek(base + 8 * p0)
                    chunk = fh.read(8 * count)
                    if len(chunk) != 8 * count:
                        raise ValueError(f"{fh.name}: truncated aggregates")
                    tiles.append(_typed_array("d", chunk))
                total.extend(math.fsum(part[i] for part in tiles)
                             for i in range(count))
        finally:
            for fh in handles:
                fh.close()
        out[name] = total
    return out


def _range_worker(raw_cfg: dict, archive_path: str, contract: dict,
                  m0: int, m1: int,
                  screens: list[tuple[int, str, str, dict]]) -> tuple[int, int]:
    """One parallel builder process: decode modes [m0, m1) of a v3 archive
    into per-range rows/aggregate files inside each screen's partial dir.

    ``raw_cfg`` is the parent's merged config dict and ``contract`` the
    parent-fixed input identity — the worker re-reads no mutable config
    and refuses an archive that drifted since input preparation."""
    from ...simulation import Simulation
    sim = Simulation.from_dict(raw_cfg)
    # First check: config indirections (a spectrum table file) re-resolve
    # here, so the reconstructed physics must equal the parent's exactly.
    if not metadata_equal(_physics_contract(sim), contract["physics"]):
        raise ValueError(
            "stage-14 worker physics differs from the parent's contract "
            "(spectrum table or config drifted under the build)")
    expected_sha = contract["index_sha256"]
    if rays_v3.index_digest(archive_path) != expected_sha:
        raise ValueError(
            f"{archive_path}: rays index changed since input preparation")
    index = rays_v3.load_index(archive_path)
    meta = rays_v3.metadata(archive_path)
    n_modes, n_rays = contract["n_modes"], contract["n_rays"]
    source_z = contract["source_z"]
    if (meta["budgets"].get("capillary") != [n_modes, n_rays]
            or _recorded_screen_z(meta, archive_path) != source_z):
        raise ValueError(
            f"{archive_path}: rays metadata differs from the prepared input")
    part = InputPart(
        archive_path, meta, n_modes, n_rays, source_z, "", None, "", "",
        {"input": {"index_sha256": expected_sha}},
        0, index)
    cap = sim.cfg.capillary
    configs = cap.all_screens
    kms = [float(line.k) for line in sim.lines]
    weights = [float(line.weight) for line in sim.lines]
    builds = []
    for screen_index, label, partial, expected_contract in screens:
        cfg = configs[screen_index]
        grid = ScreenGrid(cfg)
        ref = _reference_pixel(cfg, grid)
        # Typed screen config is rebuilt from raw here; the parent's targets
        # came from its live typed config — any post-construction mutation
        # of the latter must fail loudly, not deposit under a foreign meta.
        if not metadata_equal(_screen_contract(grid, ref), expected_contract):
            raise ValueError(
                f"{label}: worker screen contract differs from the parent's "
                "(config mutated after construction?)")
        target = SimpleNamespace(grid=grid, label=label, cfg=cfg,
                                 index=screen_index)
        rows_tmp = os.path.join(partial, range_rows_name(m0, m1) + ".tmp")
        if os.path.lexists(rows_tmp):
            os.remove(rows_tmp)
        store = _native_store(rows_tmp, m1 - m0, grid.nx * grid.ny,
                              ref, kms, weights)
        builds.append(CacheBuild(target, part, partial, rows_tmp,
                                 store, ScatterRaster(cfg), time.time()))
    stats_rows, bounce_hist = _build_stream_fanout(
        sim, part, builds, lambda *_: None, mode_range=(m0, m1), quiet=True)
    for build, stats in zip(builds, stats_rows):
        npix = build.target.grid.nx * build.target.grid.ny
        native = build.store.finish()
        expected_rows = (m1 - m0) * npix * 24
        if (int(native["payload_bytes"]) != expected_rows
                or os.path.getsize(build.rows_path) != expected_rows):
            raise ValueError("Stage-14 range store wrote the wrong payload size")
        if sum(_typed_array("Q", native["n_rays"])) != stats["screen"]:
            raise ValueError(
                f"Stage-14 range density differs from {build.target.label} "
                "screen counter")
        agg_tmp = os.path.join(build.partial, range_agg_name(m0, m1) + ".tmp")
        if os.path.lexists(agg_tmp):
            os.remove(agg_tmp)
        agg_size, agg_sha = _write_aggregates(agg_tmp, native, npix, stats,
                                              bounce_hist, build.scatter)
        for tmp in (build.rows_path, agg_tmp):
            with open(tmp, "rb+") as fh:
                os.fsync(fh.fileno())
        # The manifest is the pair's commit record; drop a stale one first so
        # the range reads as not-done while its files are being swapped.
        manifest_path = os.path.join(build.partial,
                                     range_manifest_name(m0, m1))
        if os.path.lexists(manifest_path):
            os.remove(manifest_path)
        os.replace(build.rows_path,
                   os.path.join(build.partial, range_rows_name(m0, m1)))
        os.replace(agg_tmp,
                   os.path.join(build.partial, range_agg_name(m0, m1)))
        manifest = {"rows": {"bytes": expected_rows,
                             "sha256": str(native["payload_sha256"])},
                    "aggregates": {"bytes": agg_size, "sha256": agg_sha}}
        with open(manifest_path + ".tmp", "w", encoding="utf-8",
                  newline="\n") as fh:
            json.dump(manifest, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(manifest_path + ".tmp", manifest_path)
    return m0, m1


def _build_fanout_parallel(sim, part: InputPart, builds: list[CacheBuild],
                           jobs: int, log):
    """Decode one v3 archive in parallel mode ranges (published atomically,
    so a rerun reuses completed ranges), then assemble each screen's
    canonical rows file and summed aggregates."""
    max_bounces = sim.cfg.max_bounces
    ranges = mode_ranges(part.n_modes)
    screens = [(b.target.index, b.target.label, b.partial, b.target.contract)
               for b in builds]
    sizes = [(b.partial, b.target.grid.nx * b.target.grid.ny,
              _expected_aggregate_size(
                  b.target.grid.nx * b.target.grid.ny, max_bounces,
                  b.scatter.nx * b.scatter.ny)) for b in builds]
    todo = []
    for m0, m1 in ranges:
        if all(range_done(partial, m0, m1, (m1 - m0) * npix * 24, agg)
               for partial, npix, agg in sizes):
            log(f"  stage 14 range {m0}..{m1}: reused")
        else:
            todo.append((m0, m1))
    if todo:
        # Workers get the config by value and the parent-fixed input identity:
        # the yaml and the archive may not drift under a running build.
        contract = {"index_sha256": part.identity["input"]["index_sha256"],
                    "n_modes": part.n_modes, "n_rays": part.n_rays,
                    "source_z": part.source_z,
                    "physics": _physics_contract(sim)}
        workers = _effective_workers(jobs, len(todo))
        log(f"  stage 14 workers: {workers} over {len(todo)} ranges")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers) as pool:
            futures = {pool.submit(_range_worker, sim.cfg.raw,
                                   os.path.abspath(part.path), contract,
                                   m0, m1, screens): (m0, m1)
                       for m0, m1 in todo}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                m0, m1 = futures[fut]
                done += 1
                log(f"  stage 14 range {m0}..{m1} built ({done}/{len(todo)})")
        if rays_v3.index_digest(part.path) != contract["index_sha256"]:
            raise ValueError(f"{part.path}: rays index changed during cache build")
    stats_rows, natives = [], []
    bounce_hist = None
    for build in builds:
        npix = build.target.grid.nx * build.target.grid.ny
        # A stale ready.tmp is a crashed publication attempt; drop it before
        # assembly so at most two rows copies (ranges + canonical) coexist.
        stale_ready = os.path.join(build.partial, "ready.tmp")
        if os.path.lexists(stale_ready):
            shutil.rmtree(stale_ready)
        total = None
        manifests = {}
        agg_paths = []
        for m0, m1 in ranges:
            manifests[m0, m1] = read_range_manifest(build.partial, m0, m1)
            path = os.path.join(build.partial, range_agg_name(m0, m1))
            with open(path, "rb") as fh:
                data = fh.read()
            if (hashlib.sha256(data).hexdigest()
                    != manifests[m0, m1]["aggregates"]["sha256"]):
                raise ValueError(
                    f"{path}: range checkpoint corrupted (aggregates differ "
                    "from the manifest sha256)")
            item = _parse_aggregates(data, path, npix, max_bounces,
                                     build.scatter.nx * build.scatter.ny)
            # Integer fields fold exactly in any order; the double fields
            # get one global per-pixel fsum from the tile pass below.
            if total is None:
                total = item
            else:
                _fold_int_aggregates(total, item)
            agg_paths.append(path)
        _seal_int_aggregates(total)
        total.update(_fsum_double_fields(agg_paths, npix))
        if os.path.lexists(build.rows_path):
            os.remove(build.rows_path)
        digest = hashlib.sha256()
        size = 0
        with open(build.rows_path, "wb") as out:
            for m0, m1 in ranges:
                range_digest = hashlib.sha256()
                range_path = os.path.join(build.partial,
                                          range_rows_name(m0, m1))
                with open(range_path, "rb") as fh:
                    while True:
                        chunk = fh.read(1 << 22)
                        if not chunk:
                            break
                        out.write(chunk)
                        digest.update(chunk)
                        range_digest.update(chunk)
                        size += len(chunk)
                if (range_digest.hexdigest()
                        != manifests[m0, m1]["rows"]["sha256"]):
                    raise ValueError(
                        f"{range_path}: range checkpoint corrupted (rows "
                        "differ from the manifest sha256)")
            out.flush()
            os.fsync(out.fileno())
        # Range checkpoints stay on disk until this cache's publication:
        # _finish_cache_build removes the whole partial after the rename.
        for iy in range(build.scatter.ny):
            row = build.scatter.counts[iy]
            base = iy * build.scatter.nx
            for ix in range(build.scatter.nx):
                row[ix] = total["scatter"][base + ix]
        native = {name: _array_bytes(total[name]) for name in
                  ("I", "w_re", "w_im", "ic", "n_rays", "m_realizations",
                   "m_pair_realizations", "m_ref_realizations",
                   "max_rays_per_realization")}
        native["payload_bytes"] = size
        native["payload_sha256"] = digest.hexdigest()
        natives.append(native)
        stats_rows.append(total["stats"])
        bounce_hist = total["bounce_hist"]
    return stats_rows, bounce_hist, natives


def _missing_cache_specs(targets: list[ScreenTarget]):
    """Unique missing cache directories; identical screens share content."""
    unique = {}
    for target in targets:
        for part, cached in zip(target.parts, target.caches):
            if cached is None:
                unique.setdefault(os.path.normcase(part.cache_dir),
                                  (target, part))
    return list(unique.values())


def _preflight_fanout_builds(missing, max_bounces: int, log, jobs: int = 1) -> None:
    """Check the summed footprint of every screen x input cache miss."""
    groups = {}
    for target, part in missing:
        npix = target.grid.nx * target.grid.ny
        scatter_cells = target.scatter.nx * target.scatter.ny
        agg = _expected_aggregate_size(npix, max_bounces, scatter_cells)
        rows = part.n_modes * npix * 24
        # jobs > 1: range files coexist with the concatenated canonical rows,
        # and one range aggregate exists per range of the actual plan.
        ranges_n = len(mode_ranges(part.n_modes)) if jobs > 1 else 0
        required = rows * (2 if jobs > 1 else 1) + agg * (1 + ranges_n)
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


def _load_or_build_screen_caches(sim, targets: list[ScreenTarget], log,
                                 jobs: int = 1) -> set[str]:
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
                target.scatter, allow_partial=jobs > 1)
            target.caches[index] = cache
            if cache is not None:
                cache_by_dir[os.path.normcase(part.cache_dir)] = cache

    missing = _missing_cache_specs(targets)
    _preflight_fanout_builds(missing, sim.cfg.max_bounces, log, jobs)
    by_input = {}
    for target, part in missing:
        by_input.setdefault(part.input_id, []).append((target, part))

    archives_read = set()
    for specs in by_input.values():
        labels = ", ".join(target.label for target, _ in specs)
        part = specs[0][1]
        log(f"  Stage 14 cache miss fan-out: {part.path} -> {labels}")
        if jobs > 1:
            builds = [_prepare_cache_build(sim, target, screen_part,
                                           with_store=False)
                      for target, screen_part in specs]
            stats_rows, bounce_hist, natives = _build_fanout_parallel(
                sim, part, builds, jobs, log)
        else:
            builds = [_prepare_cache_build(sim, target, screen_part)
                      for target, screen_part in specs]
            stats_rows, bounce_hist = _build_stream_fanout(
                sim, part, builds, log)
            natives = [None] * len(builds)
        archives_read.add(part.input_id)
        for build, stats, native in zip(builds, stats_rows, natives):
            cache = _finish_cache_build(sim, build, stats, bounce_hist, native)
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

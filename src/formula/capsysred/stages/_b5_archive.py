"""Validated ray prefixes with an unwrapped, carrier-subtracted action phase."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .. import rays_v3


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _line(stream, path):
    line = stream.readline()
    if not line.endswith(b"\n"):
        raise ValueError(f"{path}: missing or unterminated archive line")
    try:
        row = json.loads(line)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid archive JSON") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{path}: archive row must be a mapping")
    return row


def _finite(values, size, what):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{what}: invalid coordinates") from exc
    if arr.shape != (size,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: expected {size} finite coordinates")
    return arr


def _decimal(value, what):
    try:
        number = Decimal(str(value))
    except (ValueError, InvalidOperation) as exc:
        raise ValueError(f"{what}: invalid decimal") from exc
    if not number.is_finite():
        raise ValueError(f"{what}: non-finite decimal")
    return number


def _budget(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def read_sample(archive, *, scene="capillary", max_modes=4,
                max_rays_per_mode=20000, target_z, k, mode_start=0):
    """Read the first ray IDs of selected v3 modes, retaining screen-fate rays.

    Each selected compressed section is SHA-256 checked in full. Only its
    requested prefix is decoded; the unselected suffix is not schema-checked.
    """
    max_modes = _budget(max_modes, "max_modes")
    if isinstance(mode_start, bool) or not isinstance(mode_start, int) or mode_start < 0:
        raise ValueError("mode_start must be a nonnegative integer")
    max_rays_per_mode = _budget(max_rays_per_mode, "max_rays_per_mode")
    target_z, k = float(target_z), float(k)
    if not math.isfinite(target_z) or not math.isfinite(k) or k <= 0:
        raise ValueError("target_z and k must be finite, with k positive")
    archive = Path(archive).resolve()
    if not archive.is_dir():
        raise ValueError("stage 17 action diagnostic requires a v3 archive directory")
    fingerprint = Path(rays_v3.fingerprint_path(archive))
    index_path = Path(rays_v3.index_path(archive))
    stamps = {p: _stamp(p) for p in (fingerprint, index_path)}
    hashes = {p: _digest(p) for p in stamps}
    meta = rays_v3.read_fingerprint(archive)
    index = rays_v3.load_index(archive)
    if meta.get("lean"):
        raise ValueError("stage 17 action diagnostic requires non-lean rays with reflection points")
    sections_by_mode = index.modes(scene)
    if not sections_by_mode:
        raise ValueError(f"archive contains no {scene!r} scene")
    geometry = meta["geometry"]
    screen = dict(geometry.get("screen") or {})
    scene_geometry = geometry.get(scene)
    if isinstance(scene_geometry, dict):
        screen.update(scene_geometry.get("screen") or {})
    if "z" not in screen:
        raise ValueError("archive fingerprint does not specify the recorded screen z")
    recorded_z_decimal = _decimal(screen["z"], "recorded screen z")
    recorded_z = float(recorded_z_decimal)
    if not math.isfinite(recorded_z) or target_z < recorded_z:
        raise ValueError("target_z must be at or downstream of the recorded screen")
    extension = target_z - recorded_z
    modes, files, mode_stats = [], [], []
    checked_stamps = {}
    for sections in sections_by_mode[mode_start:mode_start+max_modes]:
        mode_id = sections[0].mode
        stop = min(max_rays_per_mode, sections[-1].r1)
        origins = None
        points, directions, phases, reflections, sines, ray_ids = [], [], [], [], [], []
        fates = Counter()
        for section in sections:
            if section.r0 >= stop:
                break
            path = Path(rays_v3.section_path(archive, section))
            before = _stamp(path)
            digest = _digest(path)
            if _stamp(path) != before:
                raise ValueError(f"{path}: section changed during hash verification")
            if before[2] != section.bytes or digest.lower() != section.sha256.lower():
                raise ValueError(f"{path}: section size or SHA-256 does not match the index")
            checked_stamps[path] = before
            prefix_stop = min(stop, section.r1)
            with gzip.open(path, "rb") as stream:
                header = _line(stream, path)
                rays_v3._check_header(header, section, str(path))
                origin_float = _finite(header.get("origin"), 3, "source origin")
                origin_decimal = tuple(_decimal(v, "source origin") for v in header["origin"])
                if origins is not None and origin_decimal != origins:
                    raise ValueError(f"{path}: source origin differs between sections of mode {mode_id}")
                origins = origin_decimal
                for ray_id in range(section.r0, prefix_stop):
                    row = _line(stream, path)
                    if (row.get("stage") != scene or row.get("mode") != mode_id
                            or row.get("ray") != ray_id):
                        raise ValueError(f"{path}: ray identity contradicts the index at ray {ray_id}")
                    fate = row.get("fate")
                    if fate not in ("screen", "absorbed", "lost"):
                        raise ValueError(f"{path}: invalid ray fate at ray {ray_id}")
                    fates[fate] += 1
                    if fate != "screen":
                        continue
                    point = _finite([row.get("x"), row.get("y")], 2, "screen point")
                    direction = _finite([row.get("dx"), row.get("dy")], 2, "direction")
                    r2 = float(direction @ direction)
                    if r2 >= 1:
                        raise ValueError(f"{path}: transverse direction is not part of a forward unit vector")
                    dz = math.sqrt(1 - r2)
                    if not isinstance(row.get("sins"), list) or not isinstance(row.get("refl", []), list):
                        raise ValueError(f"{path}: invalid reflection history at ray {ray_id}")
                    sins = _finite(row["sins"], len(row["sins"]), "grazing sines")
                    refl = [_finite(p, 3, "reflection point").tolist() for p in row.get("refl", [])]
                    if len(refl) != len(sins):
                        raise ValueError(f"{path}: missing or inconsistent reflection points at ray {ray_id}")
                    if np.any((sins < 0) | (sins > 1)):
                        raise ValueError(f"{path}: grazing sine outside [0,1]")
                    if not isinstance(row.get("opl"), str):
                        raise ValueError(f"{path}: non-lean OPL must be a decimal string")
                    opl = _decimal(row["opl"], "optical path")
                    with localcontext() as context:
                        context.prec = max(80, len(row["opl"]) + 20,
                                           len(str(origins[2])) + 20,
                                           len(str(recorded_z_decimal)) + 20)
                        axial = recorded_z_decimal - origins[2]
                        excess = float(opl - axial)
                    # The rationalized increment retains sub-ulp paraxial path excess.
                    excess += extension * r2 / (dz * (1 + dz))
                    phase = k * excess
                    point = point + direction * (extension / dz)
                    if not math.isfinite(phase) or not np.all(np.isfinite(point)):
                        raise ValueError(f"{path}: non-finite reprojected ray")
                    points.append(point)
                    directions.append(direction)
                    phases.append(phase)
                    reflections.append(refl)
                    sines.append(sins.tolist())
                    ray_ids.append(ray_id)
                if prefix_stop == section.r1:
                    trailer = _line(stream, path)
                    expected = dict(scene_end=scene, mode=mode_id, r0=section.r0,
                                    r1=section.r1, rows=section.rows)
                    if any(trailer.get(key) != value for key, value in expected.items()) or stream.read(1):
                        raise ValueError(f"{path}: section trailer contradicts the index")
            if _stamp(path) != before:
                raise ValueError(f"{path}: section changed while its prefix was read")
            files.append(dict(file=section.file, mode=mode_id, bytes=section.bytes,
                              sha256=digest, sampled_ray_start=section.r0,
                              sampled_ray_stop=prefix_stop))
        origin_strings = [str(v) for v in origins]
        modes.append(dict(mode=mode_id, origin=origin_float,
                          source_origin=origin_float, source_origin_decimal=origin_strings,
                          points=np.asarray(points).reshape(-1, 2),
                          directions=np.asarray(directions).reshape(-1, 2),
                          phase_opl=np.asarray(phases), refl=reflections, sins=sines,
                          ray_ids=np.asarray(ray_ids, dtype=np.int64), recorded_z=recorded_z))
        mode_stats.append(dict(mode=mode_id, prefix_rows=stop,
                               screen_rows=len(ray_ids), fate_counts=dict(fates)))
    for path, before in {**stamps, **checked_stamps}.items():
        if _stamp(path) != before:
            raise ValueError(f"{path}: archive changed during sampling")
    for path, digest in hashes.items():
        if _digest(path) != digest:
            raise ValueError(f"{path}: archive metadata changed during sampling")
    return modes, dict(archive=str(archive), format=3, scene=scene,
                      fingerprint_sha256=hashes[fingerprint], index_sha256=hashes[index_path],
                      archive_budget=index.budgets[scene], max_modes=max_modes,
                      mode_start=mode_start,
                      max_rays_per_mode=max_rays_per_mode, target_z=target_z,
                      recorded_z=recorded_z, k=k, files=files, modes=mode_stats,
                      sampling="first-ray-id-prefix; retain screen fate",
                      verification="full selected-section SHA-256; decoded prefix schema",
                      phase_convention="k*(OPL-(target_z-source_z)); no Fresnel phase")

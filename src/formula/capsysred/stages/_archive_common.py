"""Archive validation and Fresnel amplitudes shared by field estimators."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .. import rays_v3
from ..rays import geometry_core, geometry_metadata, metadata_equal


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _fresnel_product(sins, delta, beta):
    out = np.ones(len(sins), dtype=complex)
    for i, values in enumerate(sins):
        s = np.asarray(values, dtype=float)
        root = np.sqrt(s*s - 2*delta + 2j*beta)
        out[i] = np.prod((s-root)/(s+root))
    return out


def _archive_contract(sim, archive):
    fingerprint = Path(rays_v3.fingerprint_path(archive))
    try:
        digest = hashlib.sha256(fingerprint.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"{archive}: unreadable v3 rays fingerprint") from exc
    meta = rays_v3.read_fingerprint(archive)
    expected = geometry_core(geometry_metadata(sim.cfg)).get("capillary")
    actual = geometry_core(meta["geometry"]).get("capillary")
    if expected is not None:
        expected.pop("screen", None)
    if actual is not None:
        actual.pop("screen", None)
    if not metadata_equal(expected, actual):
        raise ValueError(f"{archive}: capillary geometry/source does not match the stage 17 configuration")
    if meta.get("lean"):
        raise ValueError("stage 17 archive_phase requires complete reflection points and unrounded OPL")
    if hashlib.sha256(fingerprint.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{archive}: fingerprint changed during geometry validation")
    return digest

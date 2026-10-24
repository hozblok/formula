"""Stage 19 configuration, isolated from optional array-based estimators."""

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import math


DEFAULTS = {
    "source_archive": None,
    "max_modes": None,
    "workers": 1,
    "observables": ["point", "coherent_cell"],
    "screen_indices": None,
    "resume": True,
    "archive_nodes": "none",
    "archive_compression": "gzip",
    "intensity_floor": 1e-6,
    "minimum_accepted_fraction": 0.95,
    "preview_modes": [],
    "mesh": {},
    "kernel": {},
}


def validate(cfg):
    raw = cfg.raw.get("stage19", {})
    if not isinstance(raw, dict):
        raise ValueError("stage19 must be a mapping")
    unknown = set(raw) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"stage19 has unknown keys {sorted(unknown)}")
    result = deepcopy(DEFAULTS)
    result.update(deepcopy(raw))
    for key in ("max_modes", "workers"):
        value = result[key]
        if key == "max_modes" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"stage19.{key} must be a positive integer")
    if result["source_archive"] is not None and not isinstance(result["source_archive"], str):
        raise ValueError("stage19.source_archive must be a path string or null")
    if not isinstance(result["resume"], bool):
        raise ValueError("stage19.resume must be boolean")
    if not isinstance(result["archive_nodes"], str) or result["archive_nodes"] not in {"none", "accepted", "all"}:
        raise ValueError("stage19.archive_nodes must be none, accepted or all")
    if (not isinstance(result["archive_compression"], str)
            or result["archive_compression"] not in {"gzip", "xz"}):
        raise ValueError("stage19.archive_compression must be gzip or xz")
    value = result["intensity_floor"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("stage19.intensity_floor must be finite and in [0, 1)")
    value = result["minimum_accepted_fraction"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("stage19.minimum_accepted_fraction must be finite and in [0, 1]")
    observables = result["observables"]
    if (not isinstance(observables, list) or not observables
            or any(v not in {"point", "coherent_cell"} for v in observables)
            or len(observables) != len(set(observables))):
        raise ValueError("stage19.observables must contain unique point/coherent_cell entries")
    indices = result["screen_indices"]
    if indices is not None and (not isinstance(indices, list) or not indices
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in indices)
            or len(indices) != len(set(indices))):
        raise ValueError("stage19.screen_indices must contain unique nonnegative integers")
    previews = result["preview_modes"]
    if (not isinstance(previews, list)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in previews)
            or len(previews) != len(set(previews))):
        raise ValueError("stage19.preview_modes must contain unique positive integers")
    for key in ("mesh", "kernel"):
        if not isinstance(result[key], dict):
            raise ValueError(f"stage19.{key} must be a mapping")
    kernel = result["kernel"]
    backend = kernel.get("backend", "contour")
    common = {"backend", "nthreads", "pixel_order"}
    keys = {"contour": common | {"order", "phase_step", "max_subdivisions"},
            "native_nufft": common | {"source_order", "source_safety", "max_source_order",
                "max_source_points", "nufft_eps", "contour_check", "integration_tolerance"}}
    if backend not in keys or set(kernel) - keys[backend]:
        raise ValueError("stage19.kernel has an unsupported backend or option")
    for key in ("nthreads", "pixel_order", "order", "max_subdivisions", "source_order",
                "max_source_order", "max_source_points"):
        if key in kernel and (isinstance(kernel[key], bool) or not isinstance(kernel[key], int) or kernel[key] < 1):
            raise ValueError(f"stage19.kernel.{key} must be a positive integer")
    if not 1 <= kernel.get("pixel_order", 4) <= 16:
        raise ValueError("stage19.kernel.pixel_order must be in [1, 16]")
    if backend == "native_nufft":
        if not 2 <= kernel.get("source_order", 6) <= kernel.get("max_source_order", 512) <= 4096:
            raise ValueError("stage19.kernel requires 2 <= source_order <= max_source_order <= 4096")
        if not isinstance(kernel.get("contour_check", True), bool):
            raise ValueError("stage19.kernel.contour_check must be boolean")
    for key in ("phase_step", "source_safety", "nufft_eps", "integration_tolerance"):
        if key in kernel and (isinstance(kernel[key], bool) or not isinstance(kernel[key], (int, float))
                or not math.isfinite(kernel[key]) or kernel[key] <= 0):
            raise ValueError(f"stage19.kernel.{key} must be positive and finite")
    if "nufft_eps" in kernel and kernel["nufft_eps"] >= 1:
        raise ValueError("stage19.kernel.nufft_eps must be less than one")
    from ._s19_mesh import DEFAULTS as mesh_defaults
    unknown = set(result["mesh"]) - mesh_defaults.keys()
    if unknown:
        raise ValueError(f"stage19.mesh has unknown keys {sorted(unknown)}")
    mesh = result["mesh"]
    cache_directory = mesh.get("record_cache_directory")
    if cache_directory is not None and (not isinstance(cache_directory, str) or not cache_directory.strip()):
        raise ValueError("stage19.mesh.record_cache_directory must be a nonempty path string or null")
    cache_capacity = mesh.get("record_cache_capacity", 4096)
    if isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int) or cache_capacity < 1:
        raise ValueError("stage19.mesh.record_cache_capacity must be a positive integer")
    phase_coordinates = mesh.get("phase_coordinates", "pullback")
    if not isinstance(phase_coordinates, str) or phase_coordinates not in {"pullback", "physical_exit"}:
        raise ValueError("stage19.mesh.phase_coordinates must be pullback or physical_exit")
    target = mesh.get("target_accepted_fraction")
    if target is not None:
        try:
            value = Decimal(str(target))
        except (ValueError, InvalidOperation) as exc:
            raise ValueError("stage19.mesh.target_accepted_fraction must be finite or null") from exc
        if (isinstance(target, bool) or not value.is_finite()
                or not Decimal(str(result["minimum_accepted_fraction"])) <= value <= 1):
            raise ValueError("stage19.mesh.target_accepted_fraction must be between minimum_accepted_fraction and 1")
    return result

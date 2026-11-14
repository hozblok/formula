"""Stage 14 orchestration across input archives and configured screens."""

from __future__ import annotations

import os
import time

from ...env import Env
from .cache import (
    META_NAME,
    _estimated_fanout_peak_rss,
    _load_or_build_screen_caches,
    _screen_targets,
    _write_json,
)
from .output import RESULT_DIR, RESULT_PAYLOAD_NAMES, _finalize_screen, _publish_result_tree


def preflight_stage14_output(out_dir: str) -> None:
    """Refuse an existing/partial result before tracing or cache work."""
    result_dir = os.path.join(out_dir, RESULT_DIR)
    partial = result_dir + ".partial"
    if os.path.lexists(result_dir) or os.path.lexists(partial):
        raise ValueError(
            f"{result_dir}: Stage-14 result already exists; remove it manually "
            "or choose another output directory"
        )


def run_stage14(sim, out_dir: str, rays_paths, log=print) -> dict:
    """Build all configured screen caches in one pass per input and publish."""
    jobs = Env.stage14_jobs()
    run_started = time.time()
    result_dir = os.path.join(out_dir, RESULT_DIR)
    partial = result_dir + ".partial"
    preflight_stage14_output(out_dir)
    if sim.cfg.capillary is None:
        raise ValueError("stage 14 requires a configured capillary.source")

    targets = _screen_targets(sim, rays_paths)
    fanout_peak_rss = _estimated_fanout_peak_rss(targets, len(sim.lines), jobs)
    log("  Stage 14 screens: " + ", ".join(
        f"{target.label}={target.grid.nx}×{target.grid.ny}"
        for target in targets))
    if jobs > 1:
        log(f"  Stage 14 cache builder jobs: {jobs} ({Env.STAGE14_JOBS})")
    log(f"  Stage 14 fan-out estimated peak RSS: "
        f"{fanout_peak_rss / (1024 ** 3):.2f} GiB")
    archives_read = _load_or_build_screen_caches(sim, targets, log, jobs)
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

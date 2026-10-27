"""Stage 18 preflight, ordered mode execution and ensemble accumulation."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import importlib.util
import os
import time

import numpy as np

from ... import rays_v3
from .._map_common import _hash, _map_grid
from .._archive_common import _archive_contract
from . import publication, workers


def preflight_b9_output(out_dir):
    for name in ("stage18", "stage18.partial"):
        if os.path.lexists(Path(out_dir)/name):
            raise ValueError(f"{name} already exists; choose another output directory")


def preflight_b9_inputs(sim, options):
    cap = sim.cfg.capillary
    if cap is None:
        raise ValueError("stage 18 requires a capillary scene")
    if len(sim.lines) != 1:
        raise ValueError("stage 18 requires a monochromatic spectrum")
    screens = cap.all_screens
    if options["screen_index"] >= len(screens):
        raise ValueError("b9_estimator.screen_index is outside configured screens")
    if float(screens[options["screen_index"]].z) <= float(cap.z1):
        raise ValueError("stage 18 contour target must strictly follow the exit")
    for bore in cap.bores:
        if bore.get("kind", "cylinder") not in ("cylinder", "torus") or "sides" in bore:
            raise ValueError("stage 18 exit reconstruction supports circular bores")
    retrace = options.get("cylinder_retrace")
    if retrace:
        for i in retrace["bores"]:
            if i >= len(cap.bores):
                raise ValueError("stage 18 cylinder_retrace bore index is outside configured bores")
            bore = cap.bores[i]
            if bore.get("kind", "cylinder") != "cylinder" or bore.get("bend") or "sides" in bore:
                raise ValueError("stage 18 cylinder_retrace requires straight circular cylinders")
    adaptive = options.get("adaptive_retrace")
    if adaptive and any(i >= len(cap.bores) for i in adaptive["bores"]):
        raise ValueError("stage 18 adaptive_retrace bore index is outside configured bores")
    curved = options.get("curved_retrace")
    if curved and any(i >= len(cap.bores) for i in curved["bores"]):
        raise ValueError("stage 18 curved_retrace bore index is outside configured bores")
    if retrace and options["amplitude_mode"] == "shared_flux":
        raise ValueError("stage 18 shared_flux requires adaptive_retrace or an archive mesh, not cylinder_retrace")
    if importlib.util.find_spec("finufft") is None:
        raise ValueError("stage 18 requires the optional finufft package")


def run_b9_stage(sim, out_dir, options, *, rays_paths=None, log=None):
    preflight_b9_inputs(sim, options)
    preflight_b9_output(out_dir)
    if not rays_paths or len(rays_paths) != 1:
        raise ValueError("stage 18 needs exactly one complete v3 ray archive via --replay")
    archive = str(Path(rays_paths[0]).resolve())
    fingerprint_hash = _archive_contract(sim, archive)
    index_path = Path(rays_v3.index_path(archive))
    index_hash = _hash(index_path)
    index = rays_v3.load_index(archive)
    if _hash(index_path) != index_hash:
        raise ValueError("archive index changed in preflight")
    archive_modes, archive_rays = index.budgets.get("capillary", (0, 0))
    count = min(options["max_modes"], archive_modes-options["mode_start"])
    budgets = sorted(options["map_ray_budgets"] or [options["rays_per_mode"]])
    if count < 1 or min(budgets) < 3 or max(budgets) > min(archive_rays, options["rays_per_mode"]):
        raise ValueError("stage 18 modes/ray budgets do not fit the archive")
    cap = sim.cfg.capillary
    screen = cap.all_screens[options["screen_index"]]
    grid = _map_grid(screen, options["map_stride"])
    base = dict(archive=archive, options=options, budgets=budgets, grid=grid,
        cap=dict(z0=float(cap.z0), z1=float(cap.z1), bores=sim.cfg.raw["capillary"]["bores"]),
        target_z=float(screen.z), k=float(sim.lines[0].k), delta=sim.delta_f, beta=sim.beta_f,
        fingerprint_sha256=fingerprint_hash, index_sha256=index_hash)
    jobs = [dict(base, mode=i) for i in range(options["mode_start"], options["mode_start"]+count)]
    groups = options.get("carrier_groups", 0)
    shape = (len(grid["y"]), len(grid["x"]))
    states = {(b, s): dict(rowsI=[], rowsW=[], sumI=np.zeros(shape), sumW=np.zeros(shape, complex))
        for b in budgets for s in options["phase_subdivisions"]}
    matched = {b: dict(I=np.zeros(shape), W=np.zeros(shape, complex), ray_count=np.zeros(shape, np.int64)) for b in budgets}
    partial, final = publication.prepare_output(out_dir)
    for job in jobs:
        job["output_folder"] = str(partial)
        job["save_retraced_meshes"] = job["mode"] == options["mode_start"]
    result = publication.metadata(sim, options, base, archive_modes, archive_rays)
    started, executor = time.perf_counter(), None

    def outcomes():
        # a mode whose prescribed geometry violates the model is skipped and recorded, not fatal
        if executor is not None:
            futures = [executor.submit(workers._mode_job, job) for job in jobs]
            for job, future in zip(jobs, futures):
                try:
                    yield job, future.result(), None
                except ValueError as exc:
                    yield job, None, exc
        else:
            for job in jobs:
                try:
                    yield job, workers._mode_job(job), None
                except ValueError as exc:
                    yield job, None, exc

    try:
        if options["map_jobs"] > 1:
            executor = ProcessPoolExecutor(max_workers=options["map_jobs"])
        completed_count = 0
        for processed, (job, completed, failure) in enumerate(outcomes(), 1):
            if failure is not None:
                fatal = ("archive metadata", "archive has", "max_missing_area_fraction")
                if any(word in str(failure) for word in fatal):
                    raise failure
                result["failed_modes"].append(dict(mode=job["mode"], error=str(failure)))
                if log:
                    log(f"  Stage18 mode {job['mode']} skipped: {failure}")
                if processed == len(jobs) and completed_count:
                    result["outputs"].extend(publication._snapshot(partial, states, matched, grid, completed_count,
                                                      carrier_groups=groups, options=options))
                    result["seconds"] = time.perf_counter()-started
                    publication.write_metadata(partial, result)
                continue
            completed_count += 1
            record = {key: completed[key] for key in ("mode", "origin", "origin_decimal", "archive", "meshes", "seconds")}
            record["variants"] = []
            for key, item in completed["variants"].items():
                field = item["field"]
                intensity = abs(field)**2
                cross = field*field[grid["ref_index"]].conjugate()
                if item.get("residual") is not None:
                    intensity = intensity-item["residual"]["variance"]
                    cross = cross-item["residual"]["reference_covariance"]
                state = states[key]
                state["rowsI"].append(intensity)
                state["rowsW"].append(cross)
                state["sumI"] += intensity
                state["sumW"] += cross
                record["variants"].append({**item["diagnostics"], "rays": key[0], "subdivisions": key[1]})
                if completed_count == 1:
                    result["field_outputs"].append(
                        publication.save_field(partial, completed, key, item, grid, groups, options))
            for budget, item in completed["matched"].items():
                for name in ("I", "W", "ray_count"):
                    matched[budget][name] += item[name]
            result["modes"].append(record)
            result["validation"] = publication._validation_status(result["modes"])
            result["completed_source_modes"] = completed_count
            if log:
                coverage = min(m["accepted_entrance_area_fraction"] for m in completed["meshes"].values())
                log(f"  Stage18 mode {completed['mode']}: minimum entrance coverage {coverage:.4f}; {completed['seconds']:.1f}s")
                if any(m["final_amplitude_probes_passed"] is False for m in completed["meshes"].values()):
                    log("  Stage18 warning: final amplitude correction failed some adaptive probes; this is a diagnostic map")
            if completed_count in options["map_snapshots"] or processed == len(jobs):
                result["outputs"].extend(publication._snapshot(partial, states, matched, grid, completed_count,
                                                  carrier_groups=groups, options=options))
                result["seconds"] = time.perf_counter()-started
                publication.write_metadata(partial, result)
        if not completed_count:
            raise ValueError("stage 18: every requested source mode failed; see failed_modes in meta.json")
        if executor is not None:
            executor.shutdown(wait=True)
            executor = None
        if (_hash(index_path) != index_hash
                or _hash(rays_v3.fingerprint_path(archive)) != fingerprint_hash):
            raise ValueError("stage 18 archive metadata changed during run")
        result["seconds"] = time.perf_counter()-started
        result["full_coherence_computed"] = True
        publication.publish(partial, final, result)
    except BaseException as exc:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        result.update(status="failed-partial", full_coherence_computed=False,
                      failure=repr(exc), seconds=time.perf_counter()-started)
        publication.write_metadata(partial, result)
        raise
    return dict(results=result, files=["stage18/meta.json", *("stage18/"+row["file"] for row in result["outputs"]+result["field_outputs"])],
        report=["## Stage 18 — experimental B9 coherence", "",
            f"- {count} empirical source modes; exit GO reconstruction and Fresnel propagation",
            f"- field_representation={options.get('field_representation', 'contour_p1')}",
            "- accuracy_validated=false; ray mesh, field representation and source controls required", ""])

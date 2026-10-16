"""Experimental archive-based canonical field and coherence maps."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

from .. import rays_v3
from ..screen import ScreenGrid


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _implementation_hashes():
    base = Path(__file__).parent
    return {name: _hash(base/name) for name in
            ("_b5_map.py", "_b5_field.py", "_b5_transport.py", "_b5_archive.py", "stage17.py")}


def _map_grid(screen, stride):
    grid = ScreenGrid(screen)
    hx, hy = grid.exf/grid.nx, grid.eyf/grid.ny
    if not math.isclose(hx, hy, rel_tol=1e-12, abs_tol=0):
        raise ValueError("stage17 archive_canonical currently requires square receiver cells")
    ref = grid.ref_pixel(screen.reference)
    ref_y, ref_x = divmod(ref, grid.nx)
    ix, iy = np.arange(ref_x % stride, grid.nx, stride), np.arange(ref_y % stride, grid.ny, stride)
    if len(ix) < 2 or len(iy) < 2:
        raise ValueError("stage17 map_stride leaves fewer than two cells on an axis")
    x = grid.x0f+(ix+.5)*hx
    y = grid.y0f+(iy+.5)*hy
    return dict(x=x, y=y, native_ix=ix, native_iy=iy, cell_width=hx,
                nx=grid.nx, ny=grid.ny, x0=grid.x0f, y0=grid.y0f,
                edge_x=grid.exf, edge_y=grid.eyf,
                ref_index=(int(np.flatnonzero(iy == ref_y)[0]), int(np.flatnonzero(ix == ref_x)[0])),
                reference=grid.pixel_xy(ref))


def matched_ray_field(points, phase, fresnel, grid):
    """Native stage14 binning on exactly the selected receiver cells."""
    shape = (len(grid["y"]), len(grid["x"]))
    field = np.zeros(shape, complex)
    self_i = np.zeros(shape)
    count = np.zeros(shape, np.int64)
    lookup_x, lookup_y = np.full(grid["nx"], -1), np.full(grid["ny"], -1)
    lookup_x[grid["native_ix"]] = np.arange(shape[1])
    lookup_y[grid["native_iy"]] = np.arange(shape[0])
    fx = (points[:, 0]-grid["x0"])/grid["edge_x"]
    fy = (points[:, 1]-grid["y0"])/grid["edge_y"]
    inside = (fx >= 0) & (fx < 1) & (fy >= 0) & (fy < 1)
    rays = np.flatnonzero(inside)
    ix = lookup_x[(fx[inside]*grid["nx"]).astype(int)]
    iy = lookup_y[(fy[inside]*grid["ny"]).astype(int)]
    selected = (ix >= 0) & (iy >= 0)
    rays, ix, iy = rays[selected], ix[selected], iy[selected]
    values = fresnel[rays]*np.exp(1j*phase[rays])
    np.add.at(field, (iy, ix), values)
    np.add.at(self_i, (iy, ix), abs(values)**2)
    np.add.at(count, (iy, ix), 1)
    intensity = abs(field)**2-self_i
    cross = field*field[grid["ref_index"]].conjugate()
    cross[grid["ref_index"]] -= self_i[grid["ref_index"]]
    return dict(I=intensity, W=cross, ray_count=count)


def debiased_observables(field, self_intensity, self_cross, prefix_n, reference):
    if prefix_n < 2:
        raise ValueError("canonical map needs at least two emitted rays per source mode")
    plugin_i = abs(field)**2
    plugin_w = field*field[reference].conjugate()
    factor = prefix_n/(prefix_n-1)
    return dict(I=factor*(plugin_i-self_intensity), W=factor*(plugin_w-self_cross),
                pluginI=plugin_i, pluginW=plugin_w, selfI=self_intensity)


def _mode_job(job):
    from ._b5_archive import read_sample
    from ._b5_field import reconstruct_field
    from ._b5_transport import transport_mode
    from .stage17 import _fresnel_product

    start = time.perf_counter()
    if _implementation_hashes() != job["implementation_hashes"]:
        raise ValueError("stage17 implementation changed after the map run started")
    samples, archive_meta = read_sample(job["archive"], max_modes=1, mode_start=job["mode"],
                                        max_rays_per_mode=job["max_rays"], target_z=job["target_z"], k=job["k"])
    for name in ("fingerprint_sha256", "index_sha256"):
        if archive_meta[name] != job[name]:
            raise ValueError(f"archive {name} changed after stage17 preflight")
    if len(samples) != 1:
        raise ValueError("archive mode disappeared while stage17 was running")
    mode = samples[0]
    prefix_rows = archive_meta["modes"][0]["prefix_rows"]
    if prefix_rows < job["max_rays"]:
        raise ValueError("archive ray budget is smaller than the requested canonical map budget")
    cap = SimpleNamespace(**job["cap"])
    transport = transport_mode(mode, cap, job["target_z"]) if len(mode["points"]) else None
    fresnel = _fresnel_product(mode["sins"], job["delta"], job["beta"])
    origin = np.asarray(mode["origin"])
    source_distance = float(cap.z0)-origin[2]
    if source_distance <= 0:
        raise ValueError("canonical map source must precede the entrance plane")
    if transport is not None:
        entrance_distance = np.sqrt(np.sum((transport["entrance"]-origin[:2])**2, axis=1)+source_distance**2)
        uz0 = source_distance/entrance_distance
        uzout = np.sqrt(1-np.sum(mode["directions"]**2, axis=1))
        source_amplitude = np.sqrt(uz0/uzout)/entrance_distance
    grid = job["grid"]
    shape = (len(grid["y"]), len(grid["x"]))
    variants, matched = {}, {}
    for budget in job["budgets"]:
        prefix = mode["ray_ids"] < budget
        matched[budget] = matched_ray_field(mode["points"][prefix], mode["phase_opl"][prefix], fresnel[prefix], grid)
        valid = prefix & transport["valid"] if transport is not None else prefix
        valid_ids = np.flatnonzero(valid)
        for width in job["widths"]:
            if len(valid_ids):
                result = reconstruct_field(mode["points"][valid], mode["directions"][valid], mode["phase_opl"][valid],
                    fresnel[valid], transport["Q"][valid], transport["P"][valid], transport["maslov"][valid],
                    area_weights=job["entrance_area"]/budget, source_amplitude=source_amplitude[valid],
                    k=job["k"], width=width, x=grid["x"], y=grid["y"], cell_width=grid["cell_width"],
                    reference=grid["reference"])
            else:
                result = dict(field=np.zeros(shape, complex), self_intensity=np.zeros(shape), self_cross=np.zeros(shape, complex),
                              ref_index=grid["ref_index"], metadata={"rays": 0, "contributing_rays": 0})
            if tuple(result["ref_index"]) != tuple(grid["ref_index"]):
                raise ValueError("canonical field reference differs from the native stage14 reference")
            result["metadata"].update(emitted_prefix_rays=budget, screen_prefix_rays=int(prefix.sum()),
                                       transport_valid_prefix_rays=int(valid.sum()),
                                       excluded_transport_prefix_rays=int(prefix.sum()-valid.sum()),
                                       emitted_non_screen_prefix_rays=int(budget-prefix.sum()))
            variants[(budget, width)] = result
    if _implementation_hashes() != job["implementation_hashes"]:
        raise ValueError("stage17 implementation changed during a map worker")
    return dict(mode=int(mode["mode"]), origin=origin.tolist(), origin_decimal=mode["source_origin_decimal"],
                variants=variants, matched=matched, archive=archive_meta,
                transport=transport["diagnostics"] if transport is not None else {"ray_count": 0, "valid_count": 0},
                seconds=time.perf_counter()-start)


def normalized_coherence(intensity, cross, reference):
    mu = np.full(intensity.shape, np.nan+1j*np.nan)
    ref_i = intensity[reference]
    valid = np.isfinite(intensity) & np.isfinite(cross) & (intensity > 0)
    if np.isfinite(ref_i) and ref_i > 0:
        mu[valid] = cross[valid]/np.sqrt(intensity[valid]*ref_i)
    return mu


def _jackknife(rows_i, rows_w, reference):
    count = len(rows_i)
    shape = rows_i[0].shape
    if count < 3:
        return np.full(shape, np.nan), np.zeros(shape, np.int32)
    total_i, total_w = np.sum(rows_i, axis=0), np.sum(rows_w, axis=0)
    mean, m2, valid_count = np.zeros(shape), np.zeros(shape), np.zeros(shape, np.int32)
    for intensity, cross in zip(rows_i, rows_w):
        mu = abs(normalized_coherence(total_i-intensity, total_w-cross, reference))
        valid = np.isfinite(mu)
        valid_count[valid] += 1
        difference = mu[valid]-mean[valid]
        mean[valid] += difference/valid_count[valid]
        m2[valid] += difference*(mu[valid]-mean[valid])
    error = np.full(shape, np.nan)
    complete = valid_count == count
    error[complete] = np.sqrt((count-1)/count*np.maximum(m2[complete], 0))
    return error, valid_count


def _array_counts(intensity, mu):
    return {"pixels": int(intensity.size), "positive_intensity": int(np.sum(np.isfinite(intensity) & (intensity > 0))),
            "nonpositive_intensity": int(np.sum(np.isfinite(intensity) & (intensity <= 0))),
            "nonfinite_intensity": int(np.sum(~np.isfinite(intensity))),
            "finite_mu": int(np.sum(np.isfinite(mu))),
            "over_one_mu": int(np.sum(np.isfinite(mu) & (abs(mu) > 1+1e-10))),
            "max_abs_mu": float(np.nanmax(abs(mu))) if np.any(np.isfinite(mu)) else None}


def _atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix+".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _snapshot(partial, accumulators, matched_accumulators, grid, count):
    output = []
    for (budget, width), state in accumulators.items():
        intensity = state["sumI"]/count
        cross = state["sumW"]/count
        plugin_i, plugin_w = state["pluginI"]/count, state["pluginW"]/count
        mu = normalized_coherence(intensity, cross, grid["ref_index"])
        plugin_mu = normalized_coherence(plugin_i, plugin_w, grid["ref_index"])
        error, valid_loo = _jackknife(state["rowsI"], state["rowsW"], grid["ref_index"])
        matched = matched_accumulators[budget]
        matched_i, matched_w = matched["I"]/count, matched["W"]/count
        matched_mu = normalized_coherence(matched_i, matched_w, grid["ref_index"])
        filename = f"map-b{budget}-w{width*1e6:g}um-m{count}.npz"
        _atomic_npz(partial/filename, x=grid["x"], y=grid["y"], I=intensity, W=cross, mu=mu,
                    mu_err=error, jackknife_valid_modes=valid_loo, pluginI=plugin_i, pluginW=plugin_w, mu_plugin=plugin_mu,
                    selfI=state["selfI"]/count, matched_stage14_I=matched_i, matched_stage14_W=matched_w,
                    matched_stage14_mu=matched_mu, matched_stage14_rays=matched["ray_count"],
                    n_modes=np.array(count), emitted_rays_per_mode=np.array(budget), width_m=np.array(width),
                    receiver_width_m=np.array(grid["cell_width"]), ref_index=np.array(grid["ref_index"]))
        output.append(dict(file=filename, source_modes=count, rays_per_mode=budget, width_m=width,
                           counts=_array_counts(intensity, mu), plugin_counts=_array_counts(plugin_i, plugin_mu),
                           matched_stage14_counts=_array_counts(matched_i, matched_mu),
                           reference_intensity=float(intensity[grid["ref_index"]]),
                           reference_plugin_intensity=float(plugin_i[grid["ref_index"]]),
                           reference_matched_stage14_intensity=float(matched_i[grid["ref_index"]]),
                           sha256=_hash(partial/filename)))
    return output


def run_canonical_map(sim, out_dir, options, *, rays_paths, log=None):
    from .stage17 import _archive_contract, _dump, preflight_b5_inputs, preflight_b5_output

    preflight_b5_inputs(sim, options)
    preflight_b5_output(out_dir)
    if not rays_paths or len(rays_paths) != 1:
        raise ValueError("stage17 archive_canonical needs exactly one complete v3 rays archive")
    archive = str(Path(rays_paths[0]).resolve())
    fingerprint_hash = _archive_contract(sim, archive)
    index_path = Path(rays_v3.index_path(archive))
    index_hash = _hash(index_path)
    index = rays_v3.load_index(archive)
    if _hash(index_path) != index_hash:
        raise ValueError("archive index changed during canonical map preflight")
    archive_modes, archive_rays = index.budgets.get("capillary", (0, 0))
    modes = min(options["max_modes"], archive_modes)
    if modes < 1:
        raise ValueError("stage17 archive contains no capillary source modes")
    budgets = sorted(set(options.get("map_ray_budgets") or [options["rays_per_mode"]]))
    if min(budgets) < 2 or max(budgets) > min(options["rays_per_mode"], archive_rays):
        raise ValueError("canonical map ray budgets must be >=2 and fit the configured/archive ray prefix")
    widths = sorted(set(options["widths_m"]))
    cap = sim.cfg.capillary
    screen = [cap.screen, *cap.screens][options["screen_index"]]
    grid = _map_grid(screen, options["map_stride"])
    cap_data = {"z0": float(cap.z0), "bores": sim.cfg.raw["capillary"]["bores"]}
    area = sum(math.pi*float(bore["radius"])**2 for bore in cap.bores)
    implementation = _implementation_hashes()
    base_job = dict(archive=archive, cap=cap_data, max_rays=max(budgets), target_z=float(screen.z),
                    k=float(sim.lines[0].k), delta=sim.delta_f, beta=sim.beta_f,
                    fingerprint_sha256=fingerprint_hash, index_sha256=index_hash,
                    implementation_hashes=implementation, grid=grid, widths=widths, budgets=budgets, entrance_area=area)
    jobs = [dict(base_job, mode=mode) for mode in range(modes)]
    shape = (len(grid["y"]), len(grid["x"]))
    accumulators = {(budget, width): dict(rowsI=[], rowsW=[], sumI=np.zeros(shape), sumW=np.zeros(shape, complex),
                       pluginI=np.zeros(shape), pluginW=np.zeros(shape, complex), selfI=np.zeros(shape))
                    for budget in budgets for width in widths}
    matched_accumulators = {budget: dict(I=np.zeros(shape), W=np.zeros(shape, complex), ray_count=np.zeros(shape, np.int64))
                            for budget in budgets}
    partial, final = Path(out_dir)/"stage17.partial", Path(out_dir)/"stage17"
    partial.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = dict(provider="archive_canonical", status="experimental-semiclassical-coherence-map",
                  full_coherence_computed=True, accuracy_validated=False, options=options,
                  source_mode_count=modes, archive_mode_count=archive_modes, archive_rays_per_mode=archive_rays,
                  screen_z_m=float(screen.z), distance_after_exit_m=float(screen.z)-float(cap.z1),
                  k_per_m=base_job["k"], delta=sim.delta_f, beta=sim.beta_f, entrance_area_m2=area,
                  screen_grid={key: value for key, value in grid.items() if key not in ("x", "y")},
                  source_sampling="first archived source modes, equal empirical weights; not exact Gaussian source quadrature",
                  source_amplitude="(1 / geometric source-to-entrance distance) * sqrt(uz_entrance / uz_target)",
                  ray_quadrature="uniform entrance-area sampling: each emitted ray has weight total entrance area / full prefix N; lost and invalid transport rays contribute zero",
                  estimator="per-mode U-statistic N/(N-1) times field outer product minus identical-ray diagonal, then equal source-mode mean",
                  matched_stage14="same archived ray prefix and native screen cells; raw Fresnel-ray units, no geometry correction, self-pairs removed; arithmetic mean over modes",
                  plugin_diagnostic="positive field outer-product estimator without self-pair subtraction; finite-ray quadrature bias remains",
                  jackknife="delete-one-source-mode standard error of |mu|; available only with >=3 modes and all positive leave-one intensities; does not estimate model bias",
                  limitations=["Leading semiclassical canonical Gaussian reconstruction; no verified full-wave error bound.",
                               "Finite width regularizes caustics and smooths sharp aperture boundaries; width and ray-density convergence must be tested separately.",
                               "Invalid transport rays are excluded with zero contribution, not redistributed; recorded diagnostics bound only their frequency, not field error.",
                               "Unbiased diagonal subtraction can produce nonpositive intensity or |mu|>1 at insufficient ray/source counts; these are retained and flagged, never clipped.",
                               "Gaussian source averaging uses the finite archived empirical ensemble; comparing to 5000 modes mixes estimator and source-sampling effects."],
                  implementation_sha256=implementation, archive=archive, fingerprint_sha256=fingerprint_hash,
                  index_sha256=index_hash, modes=[], outputs=[])
    executor = None
    try:
        if options["map_jobs"] > 1:
            executor = ProcessPoolExecutor(max_workers=options["map_jobs"])
            stream = executor.map(_mode_job, jobs, chunksize=1)
        else:
            stream = map(_mode_job, jobs)
        for count, completed in enumerate(stream, 1):
            mode_record = {name: completed[name] for name in ("mode", "origin", "origin_decimal", "archive", "transport", "seconds")}
            mode_record["variants"] = []
            for key, field_result in completed["variants"].items():
                budget, width = key
                observations = debiased_observables(field_result["field"], field_result["self_intensity"],
                                                   field_result["self_cross"], budget, grid["ref_index"])
                state = accumulators[key]
                state["rowsI"].append(observations["I"])
                state["rowsW"].append(observations["W"])
                state["sumI"] += observations["I"]
                state["sumW"] += observations["W"]
                for name in ("pluginI", "pluginW", "selfI"):
                    state[name] += observations[name]
                mode_record["variants"].append(dict(budget=budget, width_m=width, field=field_result["metadata"]))
            for budget, data in completed["matched"].items():
                for name in ("I", "W", "ray_count"):
                    matched_accumulators[budget][name] += data[name]
            result["modes"].append(mode_record)
            result["completed_source_modes"] = count
            if log:
                tr = completed["transport"]
                log(f"  Stage17 mode {completed['mode']}: {tr['valid_count']}/{tr['ray_count']} valid transport rays; "
                    f"{len(completed['variants'])} map variants, worker {completed['seconds']:.1f}s")
            if count in options.get("map_snapshots", []) or count == modes:
                result["outputs"].extend(_snapshot(partial, accumulators, matched_accumulators, grid, count))
                result["seconds"] = time.perf_counter()-start
                _dump(partial/"meta.json", result)
                if log:
                    log(f"  Stage17 saved {len(accumulators)} map snapshots at {count} source modes")
        if executor is not None:
            executor.shutdown(wait=True)
            executor = None
        if (_hash(index_path) != index_hash or _hash(rays_v3.fingerprint_path(archive)) != fingerprint_hash
                or _implementation_hashes() != implementation):
            raise ValueError("archive metadata or stage17 implementation changed during the map run")
        result["seconds"] = time.perf_counter()-start
        _dump(partial/"meta.json", result)
        if final.exists():
            raise ValueError("stage17 appeared concurrently; refusing to replace it")
        os.rename(partial, final)
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(partial)
        raise
    files = ["stage17/meta.json", *(f"stage17/{row['file']}" for row in result["outputs"])]
    return {"results": result, "files": files,
            "report": ["## Stage 17 — experimental B5 canonical coherence map", "",
                       f"- screen z={float(screen.z):g} m; {modes} empirical source modes; {len(accumulators)} width/ray-budget variants",
                       "- full_coherence_computed=true; accuracy_validated=false; semiclassical Gaussian reconstruction",
                       "- same-ray stage14 baseline, unbiased and positive plugin maps, source-mode jackknife and transport diagnostics saved", ""]}

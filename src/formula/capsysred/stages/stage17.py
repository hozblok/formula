"""Stage 17: archive phase checks and experimental canonical coherence maps.

The archive_phase provider audits local action charts; archive_canonical
adds variational amplitudes and a finite-width semiclassical reconstruction.
"""

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree

from .. import rays_v3
from ..rays import geometry_core, geometry_metadata, metadata_equal
from ._b5_phase import fit_phase_gradient

RESULT_DIR = "stage17"


def preflight_b5_output(out_dir):
    for name in (RESULT_DIR, RESULT_DIR + ".partial"):
        if os.path.lexists(os.path.join(out_dir, name)):
            raise ValueError(f"{name} already exists; choose another output directory")


def preflight_b5_inputs(sim, options):
    cap = sim.cfg.capillary
    if cap is None:
        raise ValueError("stage 17 archive_phase requires a capillary scene")
    if len(sim.lines) != 1:
        raise ValueError("stage 17 archive_phase currently requires a monochromatic spectrum")
    screens = [cap.screen, *cap.screens]
    if options["screen_index"] >= len(screens):
        raise ValueError("b5_estimator.screen_index is outside the configured screen list")
    if float(screens[options["screen_index"]].z) < float(cap.z1):
        raise ValueError("stage 17 target screen must follow the capillary exit")
    for bore in cap.bores:
        if bore.get("kind", "cylinder") not in ("cylinder", "torus") or "sides" in bore:
            raise ValueError("stage 17 archive_phase currently supports circular bores only")


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


def _reflection_factor(sins, delta, beta, reflection="fresnel"):
    """Per-ray reflection factor: Fresnel product, or ideal (-1)^n with n reflections."""
    if reflection == "fresnel":
        return _fresnel_product(sins, delta, beta)
    if reflection == "ideal_minus_one":
        return np.asarray([(-1.0)**len(s) for s in sins], dtype=complex)
    raise ValueError(f"unknown reflection model {reflection!r}")


def entrance_families(mode, cap, target_z):
    """Coarse labels (entrance bore, reflection count), not certified branches."""
    source = np.asarray(mode["origin"], dtype=float)
    centers = np.asarray([[float(v) for v in b["center"]] for b in cap.bores])
    radii = np.asarray([float(b["radius"]) for b in cap.bores])
    groups = defaultdict(list)
    unassigned = 0
    entrance = np.full_like(mode["points"], np.nan)
    for i, points in enumerate(mode["refl"]):
        if points:
            first = np.asarray(points[0], dtype=float)
        else:
            first = np.r_[mode["points"][i], target_z]
        denominator = first[2]-source[2]
        if denominator <= 0:
            unassigned += 1
            continue
        q = source[:2] + (first[:2]-source[:2])*(float(cap.z0)-source[2])/denominator
        entrance[i] = q
        distance = np.linalg.norm(centers-q, axis=1)
        matches = np.flatnonzero(distance <= radii*(1+1e-7))
        if len(matches) != 1:
            unassigned += 1
            continue
        groups[(int(matches[0]), len(mode["sins"][i]))].append(i)
    return {key: np.asarray(ids, dtype=int) for key, ids in groups.items()}, unassigned, entrance


def ray_chart(points, directions, phases, k, distance, chart):
    """A partial Legendre chart on one fixed-source ray manifold."""
    coordinates = np.asarray(points, dtype=float).copy()
    gradients = k*np.asarray(directions, dtype=float).copy()
    action = np.asarray(phases, dtype=float).copy()
    for axis, variable in enumerate(chart):
        if variable == "p":
            coordinates[:, axis] = distance*directions[:, axis]
            gradients[:, axis] = -k*points[:, axis]/distance
            action -= k*directions[:, axis]*points[:, axis]
    return coordinates, gradients, action


def inspect_patch(points, gradients, phases, amplitudes, train, test, options):
    """Fit gradients only; validate action differences on untouched path phases."""
    fit = fit_phase_gradient(points[train], gradients[train], degree=options["degree"],
                             anchor_point=points[train[0]], anchor_phase=phases[train[0]])
    error = fit.phase(points[test])-phases[test]
    gerror = fit.gradient(points[test])-gradients[test]
    p = points[test]
    a = amplitudes[test]
    exact_field = a*np.exp(1j*phases[test])
    exact = exact_field[:, None]*exact_field.conj()[None, :]
    fitted = fit.pair_multiplier(p)*a[:, None]*a.conj()[None, :]
    chord = p[:, None, :]-p[None, :, :]
    midpoint = (p[:, None, :]+p[None, :, :])/2
    ray = np.exp(1j*np.sum(chord*fit.gradient(midpoint), axis=-1))*a[:, None]*a.conj()[None, :]
    corrected = ray*fit.residual_multiplier(p, baseline_gradient=fit.gradient)
    norm = np.linalg.norm(exact)
    trace = np.trace(exact).real
    accepted = (fit.diagnostics["full_rank"] and fit.diagnostics["condition"] <= options["max_condition"]
                and np.max(np.abs(error)) <= options["phase_tolerance"])
    record = {
        "accepted_phase_test": bool(accepted), "train_count": len(train), "test_count": len(test),
        "phase_error_rms_rad": float(np.sqrt(np.mean(error**2))),
        "phase_error_max_rad": float(np.max(np.abs(error))),
        "gradient_error_rms_per_m": float(np.sqrt(np.mean(gerror**2))),
        "fit_diagnostics": fit.diagnostics,
        "unit_geometric_amplitude_diagnostic": {
            "ray_kernel_rel_hs": float(np.linalg.norm(ray-exact)/norm) if norm else None,
            "fitted_kernel_rel_hs": float(np.linalg.norm(fitted-exact)/norm) if norm else None,
            "residual_identity_rel_hs": float(np.linalg.norm(corrected-fitted)/norm) if norm else None,
            "ray_min_eigenvalue_over_trace": float(np.linalg.eigvalsh(ray)[0]/trace) if trace else None,
            "fitted_min_eigenvalue_over_trace": float(np.linalg.eigvalsh(fitted)[0]/trace) if trace else None,
        },
    }
    model = {"degree": fit.degree, "center_m": fit.center, "scale_m": fit.scale,
             "coefficients_rad": fit.coefficients, "anchor_point_m": fit.anchor_point,
             "anchor_phase_rad": fit.anchor_phase}
    return record, model


def audit_mode(mode, sim, options, target_z, archive_number=0):
    points = mode["points"]
    k = float(sim.lines[0].k)
    distance = target_z-float(mode["origin"][2])
    charts = ["xx"] if options["charts"] == "screen" else ["xx", "xp", "px", "pp"]
    representations = {chart: ray_chart(points, mode["directions"], mode["phase_opl"], k, distance, chart)
                       for chart in charts}
    amplitudes = _reflection_factor(mode["sins"], sim.delta_f, sim.beta_f, options["reflection"])
    groups, unassigned, entrance = entrance_families(mode, sim.cfg.capillary, target_z)
    neighborhood = entrance if options["patch_space"] == "entrance" else points
    patches, models, families = [], [], []
    for (bore, reflections), ids in sorted(groups.items()):
        count = len(ids)
        families.append({"bore": bore, "reflections": reflections, "sample_rays": count,
                         "enough_samples": count >= options["min_neighbors"]})
        if count < options["min_neighbors"]:
            continue
        tree = cKDTree(neighborhood[ids])
        rng = np.random.default_rng(np.random.SeedSequence([options["seed"], archive_number, mode["mode"], bore, reflections]))
        pool = options["min_neighbors"]
        centers = rng.choice(pool, min(pool, options["patches_per_family"]), replace=False)
        for center in centers:
            n = min(count, options["neighbors"])
            _, nearest = tree.query(neighborhood[ids[center]], k=n)
            chosen = ids[np.atleast_1d(nearest)]
            attempts = []
            while True:
                subset = chosen[:n]
                # Validation chooses chart/size; final test rays are never used for that choice.
                index = np.arange(n)
                train, validation, test = index[index % 5 < 3], index[index % 5 == 3], index[index % 5 == 4]
                candidates = []
                for chart, (coordinates, gradients, phases) in representations.items():
                    candidate, _ = inspect_patch(coordinates[subset], gradients[subset], phases[subset],
                                                 amplitudes[subset], train, validation, options)
                    score = candidate["phase_error_max_rad"]
                    if not candidate["fit_diagnostics"]["full_rank"] or candidate["fit_diagnostics"]["condition"] > options["max_condition"]:
                        score = math.inf
                    candidates.append((score, chart, candidate))
                _, selected_chart, validation_record = min(candidates, key=lambda item: item[0])
                attempts.append({"neighbors": n, "chart": selected_chart,
                                 "validation_phase_error_max_rad": validation_record["phase_error_max_rad"],
                                 "validation_accepted": validation_record["accepted_phase_test"]})
                if validation_record["accepted_phase_test"] or n == options["min_neighbors"]:
                    break
                n = max(options["min_neighbors"], n//2)
            coordinates, gradients, phases = representations[selected_chart]
            record, model = inspect_patch(coordinates[subset], gradients[subset], phases[subset],
                                          amplitudes[subset], train, test, options)
            record["accepted_phase_test"] &= validation_record["accepted_phase_test"]
            record.update({"chart": selected_chart, "momentum_coordinate_scale_m": distance,
                           "validation_phase_error_max_rad": validation_record["phase_error_max_rad"],
                           "validation_ray_ids": mode["ray_ids"][subset[validation]]})
            patch_id = len(patches)
            record.update({"patch": patch_id, "bore": bore, "reflections": reflections,
                           "center_ray_id": int(mode["ray_ids"][ids[center]]),
                           "center_m": points[ids[center]], "bounds_m": [points[subset].min(axis=0), points[subset].max(axis=0)],
                           "train_ray_ids": mode["ray_ids"][subset[train]],
                           "test_ray_ids": mode["ray_ids"][subset[test]], "attempts": attempts})
            patches.append(record)
            models.append({"patch": patch_id, "chart": selected_chart, "momentum_coordinate_scale_m": distance, **model})
    return {"archive_number": archive_number, "mode": mode["mode"], "origin_m": mode["origin"],
            "screen_rays": len(points), "unassigned_entrance_rays": unassigned,
            "coarse_families": families, "patches": patches}, models


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


def run_b5_stage(sim, out_dir, options, *, rays_paths, log=None):
    from ._b5_archive import read_sample

    if options["provider"] == "archive_canonical":
        from ._b5_map import run_canonical_map
        return run_canonical_map(sim, out_dir, options, rays_paths=rays_paths, log=log)

    preflight_b5_inputs(sim, options)
    preflight_b5_output(out_dir)
    if not rays_paths:
        raise ValueError("stage 17 archive_phase needs a full v3 rays archive via --replay")
    paths = [os.path.abspath(os.fspath(p)) for p in rays_paths]
    fingerprints = [_archive_contract(sim, path) for path in paths]
    cap = sim.cfg.capillary
    screen = [cap.screen, *cap.screens][options["screen_index"]]
    target_z = float(screen.z)
    k = float(sim.lines[0].k)
    partial = Path(out_dir)/f"{RESULT_DIR}.partial"
    final = Path(out_dir)/RESULT_DIR
    partial.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        records, models, archives = [], [], []
        for archive_number, path in enumerate(paths):
            if log:
                log(f"  Stage 17: verifying/sampling up to {options['max_modes']} modes from {path}")
            samples, metadata = read_sample(path, max_modes=options["max_modes"],
                                             max_rays_per_mode=options["rays_per_mode"], target_z=target_z, k=k)
            if metadata["fingerprint_sha256"] != fingerprints[archive_number]:
                raise ValueError(f"{path}: fingerprint changed after geometry validation")
            archives.append(metadata)
            for mode in samples:
                record, fitted = audit_mode(mode, sim, options, target_z, archive_number)
                records.append(record)
                models.append({"archive_number": archive_number, "mode": mode["mode"], "models": fitted})
                if log:
                    accepted = sum(p["accepted_phase_test"] for p in record["patches"])
                    log(f"  mode {mode['mode']}: {len(mode['points'])} screen rays, "
                        f"{accepted}/{len(record['patches'])} local phase tests accepted")
        patches = [p for record in records for p in record["patches"]]
        accepted = [p for p in patches if p["accepted_phase_test"]]
        result = {
            "provider": "archive_phase", "status": "experimental-phase-only", "full_coherence_computed": False,
            "options": options, "screen_z_m": target_z, "distance_after_exit_m": target_z-float(cap.z1),
            "k_per_m": k, "delta": sim.delta_f, "beta": sim.beta_f,
            "n_modes": len(records), "n_patches": len(patches), "accepted_phase_patches": len(accepted),
            "center_selection": "seeded per family among its first min_neighbors rays; stable as sample prefixes grow",
            "phase_error_max_rad": max((p["phase_error_max_rad"] for p in patches), default=None),
            "accepted_phase_error_max_rad": max((p["phase_error_max_rad"] for p in accepted), default=None),
            "archives": archives, "seconds": time.perf_counter()-start,
            "limitations": [
                "Fits endpoint eikonal charts; not a derived reflection propagator for the full capillary.",
                "Bore/reflection-count groups are coarse labels, not certified single branches.",
                "Patch acceptance is a held-out sample test, not a uniform error certificate or area coverage.",
                "Validation rays select neighborhood/chart; acceptance also requires disjoint final test rays, but this is not a uniform certificate.",
                "Phase-space charts are local Legendre representations of a fixed-source ray manifold, not independently validated full boundary operators.",
                "Geometric amplitudes, Maslov history, chart gluing, detector quadrature and inter-channel W are not reconstructed.",
                "Kernel HS and PSD diagnostics use unit geometric amplitude and the recorded Fresnel product; they are not errors of physical coherence.",
                "No full C-2 I or mu map is produced, and no comparison with the 5000-mode stage-14 map is claimed.",
            ],
            "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__), Path(__file__).with_name("_b5_phase.py"), Path(__file__).with_name("_b5_archive.py"))},
        }
        _dump(partial/"meta.json", result)
        _dump(partial/"phase-patches.json", records)
        _dump(partial/"phase-models.json", models)
        if final.exists():
            raise ValueError("stage17 appeared concurrently; refusing to replace it")
        os.rename(partial, final)
    except BaseException:
        shutil.rmtree(partial)
        raise
    return {"results": result,
            "files": [f"{RESULT_DIR}/{name}" for name in ("meta.json", "phase-patches.json", "phase-models.json")],
            "report": ["## Stage 17 — experimental B5 archive phase validation", "",
                       f"- screen z={target_z:g} m; {len(records)} source modes; {len(accepted)}/{len(patches)} local tests accepted",
                       "- full_coherence_computed=false; local endpoint phases only, no capillary intensity/coherence map",
                       "- detailed phase errors, recorded Fresnel pair diagnostics and fitted models: stage17/*.json", ""]}

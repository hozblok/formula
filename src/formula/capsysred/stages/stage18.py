"""Stage 18: experimental exit-ray meshes with B9 contour diffraction."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import time

import numpy as np

from .. import rays_v3
from ._b5_map import (_array_counts, _atomic_npz, _hash, _jackknife, _map_grid,
                     matched_ray_field, normalized_coherence)
from .stage17 import _archive_contract, _dump, _fresnel_product


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
    screens = [cap.screen, *cap.screens]
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


def implementation_hashes():
    from ... import _formula

    base = Path(__file__).parent
    hashes = {name: _hash(base/name) for name in
            ("stage18.py", "_b9_archive.py", "_b9_contour.py", "_b9_carrier.py", "_b9_phase.py",
             "_b9_cylinder_mesh.py", "_b9_adaptive_mesh.py", "_b9_amplitude.py", "_b9_regular.py", "_b9_quadrature.py",
             "_b9_curved_mesh.py", "_b9_curved.py",
             "_b5_archive.py", "_b5_transport.py", "_b5_map.py", "stage17.py",
             "../trace.py", "../native.py", "../surfaces.py", "../shared/nums.py", "../config.py", "../rays_v3.py")}
    hashes["compiled_formula_extension"] = _hash(Path(_formula.__file__))
    return hashes


def refine_chirped_mesh(mesh, subdivisions, k, distance):
    """Sample affine unwrapped phase/amplitude on conforming subtriangles."""
    n = int(subdivisions)
    if n != subdivisions or n < 1 or distance <= 0:
        raise ValueError("positive subdivision count and propagation distance required")
    barycentric = []
    for i in range(n):
        for j in range(n-i):
            a = np.array([i, j], float)/n
            b = np.array([i+1, j], float)/n
            c = np.array([i, j+1], float)/n
            barycentric.append(np.array([a, b, c]))
            if i+j < n-1:
                d = np.array([i+1, j+1], float)/n
                barycentric.append(np.array([b, d, c]))
    uv = np.asarray(barycentric)
    bary = np.concatenate([1-uv.sum(axis=-1, keepdims=True), uv], axis=-1)
    xy = np.einsum("svj,tjk->tsvk", bary, mesh["triangles"]).reshape(-1, 3, 2)
    phase = np.einsum("svj,tj->tsv", bary, mesh["vertex_phase"]).reshape(-1, 3)
    amplitude = np.einsum("svj,tj->tsv", bary, mesh["vertex_amplitude"]).reshape(-1, 3)
    phase += k/(2*distance)*np.sum(xy**2, axis=-1)
    values = amplitude*np.exp(1j*phase)
    span = np.ptp(phase, axis=1)
    return xy, values, dict(subtriangles=len(xy), subdivisions=n,
        chirped_vertex_phase_span_p95_rad=float(np.quantile(span, .95)) if len(span) else None,
        chirped_vertex_phase_span_max_rad=float(span.max()) if len(span) else None,
        interpretation="P1 complex chirped field sampled from affine unwrapped phase and complex amplitude; no new ray information")


def _replace_cylinders(mesh, nodes, replacements, cap):
    if not replacements:
        return mesh
    bore_ids = nodes["bore"][mesh["ray_indices"][:, 0]]
    keep = ~np.isin(bore_ids, list(replacements))
    names = ("triangles", "entrance_triangles", "vertex_phase", "vertex_amplitude",
             "vertex_directions", "entrance_area", "exit_area")
    parts = [{key: mesh[key][keep] for key in names}, *[item[0] for item in replacements.values()]]
    result = {key: np.concatenate([part[key] for part in parts]) for key in names}
    areas = np.asarray(mesh["metadata"]["accepted_entrance_area_by_bore_m2"]).copy()
    for i, (replacement, _) in replacements.items():
        areas[i] = replacement["entrance_area"].sum()
    total = sum(np.pi*float(b["radius"])**2 for b in cap.bores)
    result["metadata"] = dict(
        triangles=len(result["triangles"]), total_entrance_area_m2=total,
        accepted_entrance_area_m2=float(areas.sum()),
        accepted_entrance_area_fraction=float(areas.sum()/total),
        accepted_entrance_area_by_bore_m2=areas.tolist(),
        archived_triangles_retained=int(keep.sum()),
        cylinder_retrace={str(i): stats for i, (_, stats) in replacements.items()
                          if "adaptive_controls" not in replacements[i][0]["metadata"]},
        adaptive_retrace={str(i): stats for i, (_, stats) in replacements.items()
                          if "adaptive_controls" in replacements[i][0]["metadata"]},
        archive_before_replacement=mesh["metadata"],
        scope="Selected bores use prescribed or adaptively checked entrance partitions; other bores retain their archive reconstruction and missing areas.")
    return result


def _amplitude(mesh, nodes, mode):
    if mode == "tube_flux":
        from ._b9_archive import apply_tube_flux
        return apply_tube_flux(mesh, nodes)
    if mode == "shared_flux":
        from ._b9_amplitude import apply_shared_flux
        return apply_shared_flux(mesh, nodes)
    return mesh


def _flat_to_curved(mesh, k):
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    x, p, a, u = (mesh[key] for key in ("triangles", "vertex_phase", "vertex_amplitude", "vertex_directions"))
    midx = x[:, pairs].mean(axis=2)
    midp = p[:, pairs].mean(axis=2)-k/8*np.sum(
        (u[:, pairs[:, 1]]-u[:, pairs[:, 0]])*(x[:, pairs[:, 1]]-x[:, pairs[:, 0]]), axis=-1)
    density = a*(mesh["exit_area"]/mesh["entrance_area"])[:, None]
    return dict(entrance_triangles=mesh["entrance_triangles"], entrance_area=mesh["entrance_area"],
        position_nodes=np.concatenate((x, midx), axis=1), phase_nodes=np.concatenate((p, midp), axis=1),
        weight_nodes=np.concatenate((density, density[:, pairs].mean(axis=2)), axis=1))


def _replace_curved(mesh, nodes, flat_replacements, curved_replacements, cap, k):
    excluded = set(flat_replacements) | set(curved_replacements)
    bore_ids = nodes["bore"][mesh["ray_indices"][:, 0]]
    keep = ~np.isin(bore_ids, list(excluded))
    flat_names = ("triangles", "entrance_triangles", "entrance_area", "exit_area",
                  "vertex_phase", "vertex_amplitude", "vertex_directions")
    parts = [_flat_to_curved({key: mesh[key][keep] for key in flat_names}, k)]
    parts += [_flat_to_curved(item[0], k) for item in flat_replacements.values()]
    parts += [item[0] for item in curved_replacements.values()]
    names = ("entrance_triangles", "entrance_area", "position_nodes", "phase_nodes", "weight_nodes")
    result = {key: np.concatenate([part[key] for part in parts]) for key in names}
    areas = np.asarray(mesh["metadata"]["accepted_entrance_area_by_bore_m2"]).copy()
    for i, (replacement, _) in {**flat_replacements, **curved_replacements}.items():
        areas[i] = replacement["entrance_area"].sum()
    total = sum(np.pi*float(b["radius"])**2 for b in cap.bores)
    result["metadata"] = dict(triangles=len(result["entrance_triangles"]),
        total_entrance_area_m2=total, accepted_entrance_area_m2=float(areas.sum()),
        accepted_entrance_area_fraction=float(areas.sum()/total), accepted_entrance_area_by_bore_m2=areas.tolist(),
        archived_triangles_retained=int(keep.sum()), archive_before_replacement=mesh["metadata"],
        cylinder_retrace={str(i): stats for i, (_, stats) in flat_replacements.items()},
        curved_retrace={str(i): stats for i, (_, stats) in curved_replacements.items()},
        scope="P2 entrance ray maps on selected curved bores; other retained flat exit models are converted by an exact change of integration coordinates, without altering their prescribed fields")
    residual = [item[0] for item in curved_replacements.values() if item[0].get("residual_batches")]
    if residual:
        batches = {int(m["residual_batches"]) for m in residual}
        if len(batches) != 1:
            raise ValueError("stage 18 curved bores must share residual_batches")
        for key in ("residual_entrance_triangles", "residual_area"):
            result[key] = np.concatenate([m[key] for m in residual])
        for key in ("residual_points", "residual_phase", "residual_density", "residual_valid"):
            result[key] = np.concatenate([m[key] for m in residual], axis=1)
        result["residual_batches"] = batches.pop()
        result["metadata"]["residual_sampled_area_m2"] = float(result["residual_area"].sum())
        result["metadata"]["residual_sampled_area_fraction"] = float(result["residual_area"].sum()/total)
    return result


def _residual_correction(batch_fields, reference):
    """Batch mean, the empirical variance of that mean and its complex covariance with the
    reference cell (unbiased over the batches); subtracting them from |E|^2 and E conj E_ref
    removes the Monte-Carlo bias of the sampled residual."""
    count = len(batch_fields)
    mean = batch_fields.mean(axis=0)
    deviation = batch_fields-mean
    variance = np.sum(abs(deviation)**2, axis=0)/(count*(count-1))
    covariance = np.sum(deviation*deviation[(slice(None), *reference)].conjugate()[:, None, None], axis=0)/(count*(count-1))
    return dict(mean=mean, variance=variance, reference_covariance=covariance)


def _save_adaptive_mesh(folder, mode, bore, mesh):
    name = f"mesh-mode{mode}-bore{bore}.npz"
    arrays = {key: value for key, value in mesh.items() if isinstance(value, np.ndarray)}
    arrays.update({f"nodes_{key}": value for key, value in mesh["trace_nodes"].items()
                   if isinstance(value, np.ndarray)})
    arrays.update({f"probe_{key}": value for key, value in mesh["partition_probe_metrics"].items()})
    _atomic_npz(Path(folder)/name, **arrays)
    return dict(file=name, sha256=_hash(Path(folder)/name),
                scope="Full entrance partition, accepted exit mesh and cached vertex/probe GO data; relative optical phase, not a raw decimal ray archive")


def _coverage_check(mesh, cap, limit):
    missing = 1-np.asarray(mesh["metadata"]["accepted_entrance_area_by_bore_m2"])/np.array(
        [np.pi*float(b["radius"])**2 for b in cap.bores])
    return dict(missing_area_fraction_by_bore=missing.tolist(), maximum_missing_area_fraction=float(missing.max()),
                configured_limit=limit, passed=None if limit is None else bool(np.all(missing <= limit)),
                interpretation="Entrance-area gate only; neither a missing-field bound nor a physical accuracy certificate")


def _validation_status(records):
    meshes = [mesh for row in records for mesh in row["meshes"].values()]
    def aggregate(values):
        known = [value for value in values if value is not None]
        return all(known) if known else None
    return dict(
        coverage_gate_passed=aggregate([mesh["coverage_gate"]["passed"] for mesh in meshes]),
        final_amplitude_probes_passed=aggregate([mesh["final_amplitude_probes_passed"] for mesh in meshes]),
        quadrature_convergence_checked=False, source_convergence_checked=False,
        physical_accuracy_validated=False,
        interpretation="Local gates and measured samples are separate from field, source, and physical accuracy; comparisons require separate controlled runs")


def _variant_name(budget, subdivisions, options):
    if options.get("field_representation") in ("phase_quadrature", "curved_tubes"):
        if options.get("phase_backend") == "regular_mixed":
            safety = f"{options['quadrature_safety']:g}".replace(".", "p")
            representation = "-curved" if options.get("field_representation") == "curved_tubes" else ""
            return f"b{budget}{representation}-qm{options['triangle_quadrature_order']}-f{safety}-p{options['phase_degree']}"
        return f"b{budget}-q{options['triangle_quadrature_order']}-p{options['phase_degree']}"
    return f"b{budget}-s{subdivisions}"


def _representation_metadata(options):
    representation = options.get("field_representation", "contour_p1")
    exact_phase = representation in ("phase_quadrature", "curved_tubes")
    mixed = options.get("phase_backend") == "regular_mixed"
    return dict(field_representation=np.array(representation),
                phase_backend=np.array(options.get("phase_backend", "type3")),
                amplitude_mode=np.array(options.get("amplitude_mode", "point_jacobian")),
                phase_degree=np.array(options["phase_degree"] if exact_phase else 1),
                triangle_quadrature_order=np.array(options["triangle_quadrature_order"] if exact_phase and not mixed else 0),
                triangle_min_quadrature_order=np.array(options["triangle_quadrature_order"] if mixed else 0),
                quadrature_safety=np.array(options["quadrature_safety"] if mixed else 0.))


def _mode_job(job):
    from ._b5_archive import read_sample
    from ._b9_archive import exit_nodes, prepare_exit_mesh
    from ._b9_contour import fresnel_from_chirped

    started = time.perf_counter()
    if implementation_hashes() != job["implementation"]:
        raise ValueError("stage 18 implementation changed before worker")
    samples, archive_meta = read_sample(job["archive"], max_modes=1, mode_start=job["mode"],
        max_rays_per_mode=max(job["budgets"]), target_z=job["target_z"], k=job["k"])
    if (archive_meta["fingerprint_sha256"] != job["fingerprint_sha256"]
            or archive_meta["index_sha256"] != job["index_sha256"]):
        raise ValueError("stage 18 archive metadata changed")
    if len(samples) != 1 or archive_meta["modes"][0]["prefix_rows"] < max(job["budgets"]):
        raise ValueError("stage 18 archive has insufficient rays/modes")
    mode = samples[0]
    cap = SimpleNamespace(**job["cap"])
    opts, grid = job["options"], job["grid"]
    fresnel = _fresnel_product(mode["sins"], job["delta"], job["beta"])
    nodes = exit_nodes(mode, cap, mode_z=job["target_z"], exit_z=cap.z1, k=job["k"],
        fresnel=fresnel, determinant_floor=opts["determinant_floor"])
    replacements = {}
    if opts.get("cylinder_retrace"):
        from ._b9_cylinder_mesh import cylinder_exit_mesh

        settings = opts["cylinder_retrace"]
        for i in settings["bores"]:
            replacements[i] = cylinder_exit_mesh(mode["source_origin_decimal"], cap.bores[i],
                cap.z0, cap.z1, job["k"], job["delta"], job["beta"],
                **{key: value for key, value in settings.items() if key != "bores"},
                determinant_floor=opts["determinant_floor"], amplitude_mode=opts["amplitude_mode"])
    if opts.get("adaptive_retrace"):
        from ._b9_adaptive_mesh import adaptive_exit_mesh, audit_amplitude_probes

        settings = opts["adaptive_retrace"]
        for i in settings["bores"]:
            replacement, stats = adaptive_exit_mesh(mode["source_origin_decimal"], cap.bores[i],
                cap.z0, cap.z1, job["k"], job["delta"], job["beta"],
                **{key: value for key, value in settings.items() if key != "bores"},
                determinant_floor=opts["determinant_floor"],
                amplitude_mode="point_jacobian" if opts["amplitude_mode"] == "shared_flux" else opts["amplitude_mode"])
            if opts["amplitude_mode"] == "shared_flux" and len(replacement["triangles"]):
                replacement = _amplitude(replacement, replacement["trace_nodes"], "shared_flux")
            stats["final_amplitude_probe_audit"] = audit_amplitude_probes(replacement)
            if job.get("save_retraced_meshes"):
                stats["saved_mesh"] = _save_adaptive_mesh(job["output_folder"], mode["mode"], i, replacement)
            replacements[i] = replacement, stats
    curved_replacements = {}
    if opts.get("curved_retrace"):
        from ._b9_curved_mesh import curved_exit_mesh

        settings = opts["curved_retrace"]
        for i in settings["bores"]:
            replacement, stats = curved_exit_mesh(mode["source_origin_decimal"], cap.bores[i],
                cap.z0, cap.z1, job["k"], job["delta"], job["beta"],
                x_bounds=[float(grid["x"].min()), float(grid["x"].max())],
                y_bounds=[float(grid["y"].min()), float(grid["y"].max())],
                cell_width=grid["cell_width"], distance=job["target_z"]-cap.z1,
                **{key: value for key, value in settings.items() if key != "bores"},
                determinant_floor=opts["determinant_floor"])
            if job.get("save_retraced_meshes"):
                stats["saved_mesh"] = _save_adaptive_mesh(job["output_folder"], mode["mode"], i, replacement)
            curved_replacements[i] = replacement, stats
    variants, matched, meshes = {}, {}, {}
    for budget in job["budgets"]:
        prefix = mode["ray_ids"] < budget
        matched[budget] = matched_ray_field(mode["points"][prefix], mode["phase_opl"][prefix], fresnel[prefix], grid)
        mesh = prepare_exit_mesh(nodes, cap, budget=budget, holdout_stride=opts["holdout_stride"])
        if len(mesh["triangles"]):
            mesh = _amplitude(mesh, nodes, opts["amplitude_mode"])
        if opts.get("field_representation") == "curved_tubes":
            mesh = _replace_curved(mesh, nodes, replacements, curved_replacements, cap, job["k"])
        else:
            mesh = _replace_cylinders(mesh, nodes, replacements, cap)
        mesh["metadata"]["coverage_gate"] = _coverage_check(mesh, cap, opts.get("max_missing_area_fraction"))
        audits = [stats["final_amplitude_probe_audit"] for _, stats in replacements.values()
                  if "final_amplitude_probe_audit" in stats]
        mesh["metadata"]["final_amplitude_probes_passed"] = (
            all(row["failing_triangles"] == 0 for row in audits) if audits else None)
        meshes[budget] = mesh["metadata"]
        if mesh["metadata"]["coverage_gate"]["passed"] is False:
            if job.get("output_folder"):
                _dump(Path(job["output_folder"])/f"rejected-mesh-mode{mode['mode']}-b{budget}.json", mesh["metadata"])
            raise ValueError(f"stage 18 mode {mode['mode']} exceeds max_missing_area_fraction; mesh diagnostics saved")
        if not len(mesh["entrance_triangles"]):
            raise ValueError(f"stage 18 mode {mode['mode']} has no regular exit triangles")
        for subdivisions in opts["phase_subdivisions"]:
            groups = opts.get("carrier_groups", 0)
            residual = None
            if opts.get("field_representation") == "curved_tubes":
                from ._b9_curved import curved_phase_field

                field_start = time.perf_counter()
                field, diagnostics = curved_phase_field(mesh,
                    k=job["k"], distance=job["target_z"]-cap.z1, x=grid["x"], y=grid["y"],
                    cell_width=grid["cell_width"], pixel_order=opts["pixel_order"],
                    min_order=opts["triangle_quadrature_order"], safety=opts["quadrature_safety"],
                    max_order=opts["quadrature_max_order"], eps=opts["nufft_eps"], nthreads=opts["nufft_threads"],
                    max_nodes_per_batch=opts["max_quadrature_nodes_per_batch"],
                    receiver_channels_per_batch=opts["receiver_channels_per_batch"], return_stats=True)
                diagnostics["representation"] = "P2 entrance ray map, action and pullback density; positive Duffy quadrature"
                diagnostics["timing_scope"] = "phase-order selection, curved source quadrature, Fourier propagation and coherent receiver integration"
                if mesh.get("residual_batches"):
                    from ._b9_curved import curved_residual_fields

                    batch_fields, residual_stats = curved_residual_fields(mesh,
                        k=job["k"], distance=job["target_z"]-cap.z1, x=grid["x"], y=grid["y"],
                        cell_width=grid["cell_width"], pixel_order=opts["pixel_order"], eps=opts["nufft_eps"],
                        nthreads=opts["nufft_threads"], max_nodes_per_batch=opts["max_quadrature_nodes_per_batch"],
                        receiver_channels_per_batch=opts["receiver_channels_per_batch"], return_stats=True)
                    residual = _residual_correction(batch_fields, grid["ref_index"])
                    norm = float(np.linalg.norm(field))
                    residual_stats.update(mean_norm_over_mesh_field=float(np.linalg.norm(residual["mean"])/norm) if norm else None,
                        standard_error_norm_over_mesh_field=float(np.sqrt(residual["variance"].sum())/norm) if norm else None,
                        sampled_area_fraction=mesh["metadata"].get("residual_sampled_area_fraction"),
                        estimator="field += batch mean; I -= variance of the mean; W -= covariance with the reference cell")
                    diagnostics["residual"] = residual_stats
                    field = field+residual["mean"]
            elif opts.get("field_representation") == "phase_quadrature":
                from ._b9_phase import phase_field

                field_start = time.perf_counter()
                propagate, extra = phase_field, dict(backend="finufft", quadrature_order=opts["triangle_quadrature_order"])
                if opts.get("phase_backend") == "regular":
                    from ._b9_regular import regular_phase_field
                    propagate = regular_phase_field
                    extra = dict(receiver_channels_per_batch=opts["receiver_channels_per_batch"],
                                 quadrature_order=opts["triangle_quadrature_order"])
                elif opts.get("phase_backend") == "regular_mixed":
                    from ._b9_quadrature import mixed_phase_field
                    propagate = mixed_phase_field
                    extra = dict(receiver_channels_per_batch=opts["receiver_channels_per_batch"],
                                 min_order=opts["triangle_quadrature_order"], safety=opts["quadrature_safety"],
                                 max_order=opts["quadrature_max_order"])
                field, diagnostics = propagate(mesh,
                    k=job["k"], distance=job["target_z"]-cap.z1,
                    x=grid["x"], y=grid["y"], cell_width=grid["cell_width"],
                    pixel_order=opts["pixel_order"],
                    phase_degree=opts["phase_degree"], eps=opts["nufft_eps"], **extra,
                    nthreads=opts["nufft_threads"], max_nodes_per_batch=opts["max_quadrature_nodes_per_batch"],
                    return_stats=True)
                diagnostics["representation"] = "polynomial optical phase with positive triangle quadrature"
                diagnostics["timing_scope"] = "phase reconstruction, triangle quadrature, point Fourier propagation and coherent receiver integration"
            elif groups:
                from ._b9_carrier import carrier_field

                field_start = time.perf_counter()
                field, diagnostics = carrier_field(mesh, subdivisions, groups,
                    k=job["k"], distance=job["target_z"]-cap.z1,
                    x=grid["x"], y=grid["y"], cell_width=grid["cell_width"], pixel_order=opts["pixel_order"],
                    edge_order=opts["edge_order"], eps=opts["nufft_eps"], backend="finufft",
                    nthreads=opts["nufft_threads"], max_triangles_per_batch=opts["max_triangles_per_batch"])
                diagnostics["subtriangles"] = len(mesh["triangles"])*subdivisions**2
                diagnostics["representation"] = "carrier-demodulated P1 contour field"
                diagnostics["timing_scope"] = "carrier selection, demodulated refinement, contour propagation and coherent receiver integration"
            else:
                tri, values, diagnostics = refine_chirped_mesh(mesh, subdivisions, job["k"], job["target_z"]-cap.z1)
                field_start = time.perf_counter()
                field = np.zeros((len(grid["y"]), len(grid["x"])), complex)
                contour_stats = []
                step = opts["max_triangles_per_batch"]
                for start in range(0, len(tri), step):
                    contribution, stats = fresnel_from_chirped(tri[start:start+step], values[start:start+step],
                        k=job["k"], distance=job["target_z"]-cap.z1,
                        x=grid["x"], y=grid["y"], cell_width=grid["cell_width"], pixel_order=opts["pixel_order"],
                        edge_order=opts["edge_order"], eps=opts["nufft_eps"], backend="finufft",
                        nthreads=opts["nufft_threads"], return_stats=True)
                    field += contribution
                    contour_stats.append(stats)
                diagnostics["contour_batches"] = contour_stats
                diagnostics["representation"] = "P1 chirped contour field"
                diagnostics["timing_scope"] = "contour propagation and coherent batch assembly; excludes chirped mesh refinement"
            if not np.isfinite(field).all():
                raise ValueError("stage 18 produced a non-finite complex field")
            diagnostics["seconds"] = time.perf_counter()-field_start
            diagnostics["carrier_groups"] = groups
            variants[(budget, subdivisions)] = dict(field=field, diagnostics=diagnostics,
                residual=residual if "residual" in diagnostics else None)
            if not groups and opts.get("field_representation", "contour_p1") == "contour_p1":
                del tri, values
    if implementation_hashes() != job["implementation"]:
        raise ValueError("stage 18 implementation changed during worker")
    return dict(mode=int(mode["mode"]), origin=mode["origin"].tolist(),
        origin_decimal=mode["source_origin_decimal"], archive=archive_meta, meshes=meshes,
        variants=variants, matched=matched, seconds=time.perf_counter()-started)


def _snapshot(folder, states, matched_states, grid, count, *, carrier_groups=0, options=None):
    options = options or {}
    rows = []
    for (budget, subdivisions), state in states.items():
        intensity, cross = state["sumI"]/count, state["sumW"]/count
        mu = normalized_coherence(intensity, cross, grid["ref_index"])
        error, valid = _jackknife(state["rowsI"], state["rowsW"], grid["ref_index"])
        matched = matched_states[budget]
        mi, mw = matched["I"]/count, matched["W"]/count
        filename = f"map-{_variant_name(budget, subdivisions, options)}-m{count}.npz"
        _atomic_npz(folder/filename, x=grid["x"], y=grid["y"], I=intensity, W=cross, mu=mu,
            mu_err=error, jackknife_valid_modes=valid, ref_index=np.array(grid["ref_index"]),
            receiver_width_m=np.array(grid["cell_width"]), n_modes=np.array(count),
            accuracy_validated=np.array(False),
            residual_variance_corrected=np.array(bool((options.get("curved_retrace") or {}).get("residual_batches"))),
            emitted_rays_per_mode=np.array(budget), phase_subdivisions=np.array(subdivisions),
            carrier_groups=np.array(carrier_groups),
            **_representation_metadata(options),
            archive_prefix_ray_budget=np.array(budget),
            retraced_bores=np.array(sorted((options.get("cylinder_retrace") or {}).get("bores", [])
                +(options.get("adaptive_retrace") or {}).get("bores", [])
                +(options.get("curved_retrace") or {}).get("bores", [])), dtype=np.int64),
            matched_stage14_I=mi, matched_stage14_W=mw,
            matched_stage14_mu=normalized_coherence(mi, mw, grid["ref_index"]),
            matched_stage14_rays=matched["ray_count"])
        rows.append(dict(file=filename, modes=count, rays=budget, subdivisions=subdivisions,
            counts=_array_counts(intensity, mu), reference_intensity=float(intensity[grid["ref_index"]]),
            sha256=_hash(folder/filename)))
    return rows


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
    screen = [cap.screen, *cap.screens][options["screen_index"]]
    grid = _map_grid(screen, options["map_stride"])
    implementation = implementation_hashes()
    base = dict(archive=archive, options=options, budgets=budgets, grid=grid,
        cap=dict(z0=float(cap.z0), z1=float(cap.z1), bores=sim.cfg.raw["capillary"]["bores"]),
        target_z=float(screen.z), k=float(sim.lines[0].k), delta=sim.delta_f, beta=sim.beta_f,
        fingerprint_sha256=fingerprint_hash, index_sha256=index_hash, implementation=implementation)
    jobs = [dict(base, mode=i) for i in range(options["mode_start"], options["mode_start"]+count)]
    groups = options.get("carrier_groups", 0)
    phase_quadrature = options.get("field_representation") in ("phase_quadrature", "curved_tubes")
    shape = (len(grid["y"]), len(grid["x"]))
    states = {(b, s): dict(rowsI=[], rowsW=[], sumI=np.zeros(shape), sumW=np.zeros(shape, complex))
        for b in budgets for s in options["phase_subdivisions"]}
    matched = {b: dict(I=np.zeros(shape), W=np.zeros(shape, complex), ray_count=np.zeros(shape, np.int64)) for b in budgets}
    partial, final = Path(out_dir)/"stage18.partial", Path(out_dir)/"stage18"
    partial.mkdir(parents=True, exist_ok=False)
    for job in jobs:
        job["output_folder"] = str(partial)
        job["save_retraced_meshes"] = job["mode"] == options["mode_start"]
    result = dict(provider="archive_contour",
        status="experimental-GO-exit-phase-diffraction" if phase_quadrature else "experimental-GO-exit-contour-diffraction",
        full_coherence_computed=False, accuracy_validated=False, options=options,
        archive=archive, archive_mode_count=archive_modes, archive_rays_per_mode=archive_rays,
        fingerprint_sha256=fingerprint_hash, index_sha256=index_hash, implementation_sha256=implementation,
        screen_z_m=float(screen.z), exit_z_m=float(cap.z1), distance_after_exit_m=float(screen.z)-float(cap.z1),
        k_per_m=base["k"], delta=sim.delta_f, beta=sim.beta_f,
        screen_grid={key: value for key, value in grid.items() if key not in ("x", "y")},
        source_sampling="equal empirical weights on selected archived source modes",
        estimator="positive ensemble of complete reconstructed fields, I=mean|E|², W=mean E(x)conj(E(ref)); no ray self-pair subtraction",
        field_model=("regular GO exit branches; selected bores optionally replaced by prescribed or four-probe adaptive entrance meshes; "
            + (f"degree-{options['phase_degree']} optical phase, affine complex amplitude, triangle quadrature and point Fourier sums"
               if phase_quadrature else "affine optical phase and complex amplitude, "
               + ("carrier-demodulated P1" if groups else "refined P1 chirped field") + ", edge Fourier integrals")
            + ", free-space Fresnel diffraction"),
        amplitude_mode=options["amplitude_mode"],
        receiver="coherent mean over native receiver cell by tensor Gauss quadrature; map_stride samples native cells",
        jackknife="delete-one-source standard error of |mu|; excludes mesh, model and propagation error",
        limitations=["A contour quadrature does not validate the GO field supplied at the capillary exit.",
            "Exit caustics, invalid transport and mixed-branch triangles are excluded; missing area has no field-error certificate.",
            "Finite ray geometry and field representation have systematic error; quadrature refinement adds no ray information.",
            "Carrier grouping is not an error bound: different groups can introduce edge-trace jumps, and additional groups do not guarantee monotone improvement.",
            "Fresnel products model specular scalar reflection; curvature/boundary-wave corrections inside the capillary are absent.",
            "Bore/reflection-count/Maslov/orientation checks do not certify branch regularity inside every triangle.",
            "Finite empirical source ensemble differs from a converged Gaussian source integral.",
            "Cylinder retracing addresses only selected straight 0/1-reflection bores; residual boundary bands and curved edges have measured nonzero area.",
            "Polynomial phase quadrature requires independent order and mesh convergence; it is not a caustic regularization."],
        modes=[], outputs=[], field_outputs=[])
    if (options.get("curved_retrace") or {}).get("residual_batches"):
        result["residual_quadrature"] = dict(batches=options["curved_retrace"]["residual_batches"],
            seed=options["curved_retrace"]["residual_seed"],
            estimator="unresolved curved faces enter as the mean of stratified Monte-Carlo batches; the variance of that mean is subtracted from I and its covariance with the reference cell from W",
            limitation="removes the sampling bias of |E|^2 and E conj E_ref, not the GO-model error of the sampled faces; negative corrected intensities in dark cells are reported as nonpositive")
    if options.get("field_representation") == "curved_tubes":
        result["status"] = "experimental-curved-GO-ray-tube-diffraction"
        result["field_model"] = "P2 entrance-to-exit ray map, P2 optical action and P2 complex pullback density on selected bores; retained flat models converted exactly; Fresnel diffraction of the resulting GO branches"
        result["limitations"].append("Curved pullback removes inverse-Jacobian evaluation in the integral, but does not supply a uniform wave solution at a caustic or restore omitted entrance faces.")
    started, executor = time.perf_counter(), None
    try:
        if options["map_jobs"] > 1:
            executor = ProcessPoolExecutor(max_workers=options["map_jobs"])
            stream = executor.map(_mode_job, jobs, chunksize=1)
        else:
            stream = map(_mode_job, jobs)
        for completed_count, completed in enumerate(stream, 1):
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
                    name = f"field-{_variant_name(key[0], key[1], options)}-mode{completed['mode']}.npz"
                    extra = {}
                    if item.get("residual") is not None:
                        extra = dict(residual_variance=item["residual"]["variance"],
                                     residual_reference_covariance=item["residual"]["reference_covariance"],
                                     residual_batches=np.array(len(item["residual"]["variance"]) and
                                                               (options.get("curved_retrace") or {}).get("residual_batches", 0)))
                    _atomic_npz(partial/name, field=field, x=grid["x"], y=grid["y"],
                        receiver_width_m=np.array(grid["cell_width"]), ref_index=np.array(grid["ref_index"]),
                        carrier_groups=np.array(groups), phase_subdivisions=np.array(key[1]),
                        **_representation_metadata(options), **extra)
                    result["field_outputs"].append(dict(file=name, mode=completed["mode"], sha256=_hash(partial/name)))
            for budget, item in completed["matched"].items():
                for name in ("I", "W", "ray_count"):
                    matched[budget][name] += item[name]
            result["modes"].append(record)
            result["validation"] = _validation_status(result["modes"])
            result["completed_source_modes"] = completed_count
            if log:
                coverage = min(m["accepted_entrance_area_fraction"] for m in completed["meshes"].values())
                log(f"  Stage18 mode {completed['mode']}: minimum entrance coverage {coverage:.4f}; {completed['seconds']:.1f}s")
                if any(m["final_amplitude_probes_passed"] is False for m in completed["meshes"].values()):
                    log("  Stage18 warning: final amplitude correction failed some adaptive probes; this is a diagnostic map")
            if completed_count in options["map_snapshots"] or completed_count == count:
                result["outputs"].extend(_snapshot(partial, states, matched, grid, completed_count,
                                                  carrier_groups=groups, options=options))
                result["seconds"] = time.perf_counter()-started
                _dump(partial/"meta.json", result)
        if executor is not None:
            executor.shutdown(wait=True)
            executor = None
        if (implementation_hashes() != implementation or _hash(index_path) != index_hash
                or _hash(rays_v3.fingerprint_path(archive)) != fingerprint_hash):
            raise ValueError("stage 18 implementation or archive metadata changed during run")
        result["seconds"] = time.perf_counter()-started
        result["full_coherence_computed"] = True
        _dump(partial/"meta.json", result)
        if final.exists():
            raise ValueError("stage18 appeared concurrently; refusing replacement")
        os.rename(partial, final)
    except BaseException as exc:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        result.update(status="failed-partial", full_coherence_computed=False,
                      failure=repr(exc), seconds=time.perf_counter()-started)
        _dump(partial/"meta.json", result)
        raise
    return dict(results=result, files=["stage18/meta.json", *("stage18/"+row["file"] for row in result["outputs"]+result["field_outputs"])],
        report=["## Stage 18 — experimental B9 coherence", "",
            f"- {count} empirical source modes; exit GO reconstruction and Fresnel propagation",
            f"- field_representation={options.get('field_representation', 'contour_p1')}",
            "- accuracy_validated=false; ray mesh, field representation and source controls required", ""])

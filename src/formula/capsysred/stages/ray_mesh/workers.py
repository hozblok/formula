"""One-mode exit reconstruction and diffraction workers."""

from types import SimpleNamespace
import time

import numpy as np

from .._map_common import matched_ray_field
from .._archive_common import _fresnel_product
from .publication import _save_adaptive_mesh, _save_rejected_mesh


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
        from .archive import apply_tube_flux
        return apply_tube_flux(mesh, nodes)
    if mode == "shared_flux":
        from .amplitude import apply_shared_flux
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


def _coverage_check(mesh, cap, limit):
    missing = 1-np.asarray(mesh["metadata"]["accepted_entrance_area_by_bore_m2"])/np.array(
        [np.pi*float(b["radius"])**2 for b in cap.bores])
    return dict(missing_area_fraction_by_bore=missing.tolist(), maximum_missing_area_fraction=float(missing.max()),
                configured_limit=limit, passed=None if limit is None else bool(np.all(missing <= limit)),
                interpretation="Entrance-area gate only; neither a missing-field bound nor a physical accuracy certificate")


def _mode_job(job):
    from ..archive_samples import read_sample
    from .archive import exit_nodes, prepare_exit_mesh
    from .contour import fresnel_from_chirped

    started = time.perf_counter()
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
        from .cylinder_mesh import cylinder_exit_mesh

        settings = opts["cylinder_retrace"]
        for i in settings["bores"]:
            replacements[i] = cylinder_exit_mesh(mode["source_origin_decimal"], cap.bores[i],
                cap.z0, cap.z1, job["k"], job["delta"], job["beta"],
                **{key: value for key, value in settings.items() if key != "bores"},
                determinant_floor=opts["determinant_floor"], amplitude_mode=opts["amplitude_mode"])
    if opts.get("adaptive_retrace"):
        from .adaptive_mesh import adaptive_exit_mesh, audit_amplitude_probes

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
        from .curved_mesh import curved_exit_mesh

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
                _save_rejected_mesh(job["output_folder"], mode["mode"], budget, mesh["metadata"])
            raise ValueError(f"stage 18 mode {mode['mode']} exceeds max_missing_area_fraction; mesh diagnostics saved")
        if not len(mesh["entrance_triangles"]):
            raise ValueError(f"stage 18 mode {mode['mode']} has no regular exit triangles")
        for subdivisions in opts["phase_subdivisions"]:
            groups = opts.get("carrier_groups", 0)
            residual = None
            if opts.get("field_representation") == "curved_tubes":
                from .curved import curved_phase_field

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
                    from .curved import curved_residual_fields

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
                from .phase import phase_field

                field_start = time.perf_counter()
                propagate, extra = phase_field, dict(backend="finufft", quadrature_order=opts["triangle_quadrature_order"])
                if opts.get("phase_backend") == "regular":
                    from .regular import regular_phase_field
                    propagate = regular_phase_field
                    extra = dict(receiver_channels_per_batch=opts["receiver_channels_per_batch"],
                                 quadrature_order=opts["triangle_quadrature_order"])
                elif opts.get("phase_backend") == "regular_mixed":
                    from .quadrature import mixed_phase_field
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
                from .carrier import carrier_field

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
    return dict(mode=int(mode["mode"]), origin=mode["origin"].tolist(),
        origin_decimal=mode["source_origin_decimal"], archive=archive_meta, meshes=meshes,
        variants=variants, matched=matched, seconds=time.perf_counter()-started)

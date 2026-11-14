"""Experimental branch-preserving exit meshes from full ray archives."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from ._b5_transport import transport_mode


def _det(a):
    return a[..., 0, 0]*a[..., 1, 1]-a[..., 0, 1]*a[..., 1, 0]


def _stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "rms": None, "median": None, "p95": None, "max": None}
    return {"count": len(values), "rms": float(np.sqrt(np.mean(values**2))),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(np.max(values))}


def exit_nodes(mode, cap, *, mode_z, exit_z, k, fresnel, determinant_floor=1e-12):
    """Backproject the final free leg and reconstruct regular GO exit branches."""
    mode_z, exit_z, k = float(mode_z), float(exit_z), float(k)
    if not np.isfinite([mode_z, exit_z, k, determinant_floor]).all() or k <= 0 or determinant_floor < 0:
        raise ValueError("invalid exit reconstruction parameters")
    if not float(cap.z0) < exit_z <= mode_z:
        raise ValueError("exit must follow entrance and not follow supplied mode plane")
    points = np.asarray(mode["points"], float)
    directions = np.asarray(mode["directions"], float)
    phases = np.asarray(mode["phase_opl"], float)
    fresnel = np.broadcast_to(np.asarray(fresnel, complex), phases.shape)
    if points.shape != (len(phases), 2) or directions.shape != points.shape:
        raise ValueError("ray coordinates and phases have inconsistent shapes")
    if not all(np.isfinite(v).all() for v in (points, directions, phases, fresnel)):
        raise ValueError("exit reconstruction requires finite ray data")
    if any(any(float(hit[2]) > exit_z+1e-12 for hit in path) for path in mode["refl"]):
        raise ValueError("cannot backproject across an archived reflection")
    r2 = np.sum(directions**2, axis=1)
    if np.any(r2 >= 1):
        raise ValueError("ray directions must be forward unit-vector components")
    uz = np.sqrt(1-r2)
    extension = mode_z-exit_z
    exit_points = points-extension*directions/uz[:, None]
    exit_phase = phases-k*extension*r2/(uz*(1+uz))
    exit_mode = dict(mode, points=exit_points, phase_opl=exit_phase)
    transport = transport_mode(exit_mode, cap, exit_z)
    q = transport["Q"]
    determinant = _det(q)
    source = np.asarray(mode["origin"], float)
    source_dz = float(cap.z0)-source[2]
    if source_dz <= 0:
        raise ValueError("source must precede entrance")
    distance = np.sqrt(np.sum((transport["entrance"]-source[:2])**2, axis=1)+source_dz**2)
    uz0 = source_dz/distance
    valid = transport["valid"] & (np.abs(determinant) > determinant_floor)
    amplitude = np.zeros(len(points), complex)
    amplitude[valid] = (np.sqrt(uz0[valid]/uz[valid]/np.abs(determinant[valid]))/distance[valid]
                        * fresnel[valid]*np.exp(-.5j*np.pi*transport["maslov"][valid]))
    ray_ids = np.asarray(mode.get("ray_ids", np.arange(len(points))), np.int64)
    return dict(points=exit_points, phase=exit_phase, amplitude=amplitude, directions=directions,
                entrance=transport["entrance"], bore=transport["bore"], Q=q, P=transport["P"],
                determinant=determinant, maslov=transport["maslov"], valid=valid,
                reflections=np.asarray([len(path) for path in mode["refl"]], np.int64),
                ray_ids=ray_ids, fresnel=fresnel, source_distance=distance, uz0=uz0, uzexit=uz,
                transport=transport["diagnostics"], k=k, exit_z=exit_z,
                determinant_floor=float(determinant_floor))


def _triangulate(nodes, bores, selected):
    all_triangles, triangulations, rejection = [], [], {}
    hull_area = np.zeros(len(bores))
    accepted_area = np.zeros(len(bores))
    counters = {key: 0 for key in ("mixed_reflections", "mixed_maslov", "mixed_orientation",
                                  "invalid_vertex", "folded_exit", "degenerate")}
    for bore_id, bore in enumerate(bores):
        ids = np.flatnonzero(selected & (nodes["bore"] == bore_id))
        if len(ids) < 3:
            continue
        center = np.asarray(bore["center"], float)
        radius = float(bore["radius"])
        try:
            mesh = Delaunay((nodes["entrance"][ids]-center)/radius)
        except QhullError:
            counters["degenerate"] += 1
            continue
        triangles = ids[mesh.simplices]
        q = nodes["entrance"][triangles]
        x = nodes["points"][triangles]
        qdet = _det(np.stack([q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]], axis=-1))
        xdet = _det(np.stack([x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]], axis=-1))
        area = .5*np.abs(qdet)
        masks = {}
        for name, field in (("mixed_reflections", "reflections"), ("mixed_maslov", "maslov")):
            values = nodes[field][triangles]
            masks[name] = np.any(values != values[:, :1], axis=1)
        signs = np.sign(nodes["determinant"][triangles])
        masks["mixed_orientation"] = np.any(signs != signs[:, :1], axis=1)
        masks["invalid_vertex"] = ~np.all(nodes["valid"][triangles], axis=1)
        masks["folded_exit"] = np.sign(xdet*qdet) != signs[:, 0]
        masks["degenerate"] = ((np.abs(qdet) <= 128*np.finfo(float).eps*radius**2)
                                | (np.abs(xdet) <= 128*np.finfo(float).eps*radius**2)
                                | ~np.isfinite(xdet))
        keep = np.ones(len(triangles), bool)
        for name, mask in masks.items():
            counters[name] += int(mask.sum())
            keep &= ~mask
        hull_area[bore_id] = area.sum()
        accepted_area[bore_id] = area[keep].sum()
        all_triangles.append(triangles[keep])
        triangulations.append((bore_id, ids, mesh, keep))
        rejection[str(bore_id)] = dict(total_triangles=len(triangles), kept_triangles=int(keep.sum()))
    triangles = np.concatenate(all_triangles) if all_triangles else np.empty((0, 3), np.int64)
    return triangles, triangulations, dict(hull_area_by_bore_m2=hull_area.tolist(),
        accepted_entrance_area_by_bore_m2=accepted_area.tolist(), rejection_counts=counters,
        bore_triangles=rejection)


def _holdout(nodes, bores, selected, stride):
    held = selected & (nodes["ray_ids"] % stride == stride-1)
    _, triangulations, _ = _triangulate(nodes, bores, selected & ~held)
    residuals = {name: [] for name in ("phase_rad", "geometry_m", "relative_amplitude", "complex_error", "truth_power",
                                      "weighted_error", "weighted_power")}
    checked = eligible = extrapolated = 0
    for bore_id, ids, mesh, keep in triangulations:
        tests = np.flatnonzero(held & (nodes["bore"] == bore_id) & nodes["valid"])
        eligible += len(tests)
        if not len(tests):
            continue
        bore = bores[bore_id]
        coordinates = (nodes["entrance"][tests]-bore["center"])/float(bore["radius"])
        simplexes = mesh.find_simplex(coordinates)
        inside = simplexes >= 0
        inside[inside] &= keep[simplexes[inside]]
        tests, coordinates, simplexes = tests[inside], coordinates[inside], simplexes[inside]
        if not len(tests):
            continue
        triangles = ids[mesh.simplices[simplexes]]
        same = ((nodes["reflections"][tests] == nodes["reflections"][triangles[:, 0]])
                & (nodes["maslov"][tests] == nodes["maslov"][triangles[:, 0]])
                & (np.sign(nodes["determinant"][tests]) == np.sign(nodes["determinant"][triangles[:, 0]])))
        tests, coordinates, simplexes, triangles = tests[same], coordinates[same], simplexes[same], triangles[same]
        if not len(tests):
            continue
        bary2 = np.einsum("nij,nj->ni", mesh.transform[simplexes, :2], coordinates-mesh.transform[simplexes, 2])
        bary = np.column_stack([bary2, 1-bary2.sum(axis=1)])
        mapped = np.einsum("ni,nij->nj", bary, nodes["points"][triangles])
        residuals["geometry_m"].extend(np.linalg.norm(mapped-nodes["points"][tests], axis=1))
        x = nodes["points"][triangles]
        matrix = np.stack([x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]], axis=-1)
        uv = np.linalg.solve(matrix, (nodes["points"][tests]-x[:, 0])[..., None])[..., 0]
        xbary = np.column_stack([1-uv.sum(axis=1), uv])
        extrapolated += int(np.sum(np.any(xbary < -1e-10, axis=1)))
        phase = np.sum(xbary*nodes["phase"][triangles], axis=1)
        amplitude = np.sum(xbary*nodes["amplitude"][triangles], axis=1)
        truth_a = nodes["amplitude"][tests]
        dphase = phase-nodes["phase"][tests]
        residuals["phase_rad"].extend(np.abs(dphase))
        residuals["relative_amplitude"].extend(np.abs(amplitude-truth_a)/np.maximum(np.abs(truth_a), 1e-300))
        error = np.abs(amplitude*np.exp(1j*dphase)-truth_a)**2
        power = np.abs(truth_a)**2
        measure = np.abs(nodes["determinant"][tests])
        residuals["complex_error"].extend(error)
        residuals["truth_power"].extend(power)
        residuals["weighted_error"].extend(error*measure)
        residuals["weighted_power"].extend(power*measure)
        checked += len(tests)
    total_power = sum(residuals.pop("truth_power"))
    square_error = sum(residuals.pop("complex_error"))
    weighted_power = sum(residuals.pop("weighted_power"))
    weighted_error = sum(residuals.pop("weighted_error"))
    return {"method": "withhold ray_id modulo stride; train branch mesh on other rays; evaluate or extrapolate exit affine field at held-out actual positions",
            "stride": stride, "held_out_count": int(held.sum()), "eligible_valid_count": eligible,
            "checked_count": checked, "unchecked_count": int(held.sum())-checked,
            "outside_projected_training_triangle_count": extrapolated,
            "relative_complex_field_rms": float(np.sqrt(square_error/total_power)) if total_power else None,
            "exit_area_weighted_relative_complex_field_rms": float(np.sqrt(weighted_error/weighted_power)) if weighted_power else None,
            "weighting": "abs(det(Q_exit)) converts uniform entrance sampling to exit-area norm on each branch; diagnostic, not an error bound",
            **{key: _stats(value) for key, value in residuals.items()}}


def prepare_exit_mesh(nodes, cap, *, budget=None, holdout_stride=5):
    """Construct affine phase and complex-amplitude triangles on regular branches.

    Missing entrance area is recorded without redistribution. Local interpolation
    tests diagnose approximation error but do not bound the omitted field.
    """
    if budget is not None and (isinstance(budget, bool) or int(budget) != budget or budget < 3):
        raise ValueError("mesh budget must be an integer >= 3")
    if (isinstance(holdout_stride, bool) or int(holdout_stride) != holdout_stride
            or holdout_stride not in (0,) and holdout_stride < 2):
        raise ValueError("holdout_stride must be zero or an integer >= 2")
    selected = np.ones(len(nodes["points"]), bool) if budget is None else nodes["ray_ids"] < budget
    triangles, _, metadata = _triangulate(nodes, cap.bores, selected)
    result = _assemble_exit_mesh(nodes, cap, triangles, metadata, selected)
    if holdout_stride:
        result["metadata"]["holdout"] = _holdout(nodes, cap.bores, selected, int(holdout_stride))
    return result


def _assemble_exit_mesh(nodes, cap, triangles, metadata, selected):
    """Shared field assembly; caller controls and validates entrance topology."""
    xy = nodes["points"][triangles]
    q = nodes["entrance"][triangles]
    vertex_phase = nodes["phase"][triangles]
    vertex_amplitude = nodes["amplitude"][triangles]
    matrix = np.stack([xy[:, 1]-xy[:, 0], xy[:, 2]-xy[:, 0]], axis=1)
    gradient = np.linalg.solve(matrix, (vertex_phase[:, 1:]-vertex_phase[:, :1])[..., None])[..., 0]
    center = xy.mean(axis=1)
    amplitude = vertex_amplitude.mean(axis=1)
    phase = vertex_phase.mean(axis=1)
    entrance_area = .5*np.abs(_det(np.stack([q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]], axis=-1)))
    exit_area = .5*np.abs(_det(matrix))
    total_area = sum(np.pi*float(bore["radius"])**2 for bore in cap.bores)
    amplitude_variation = np.max(np.abs(vertex_amplitude-amplitude[:, None]), axis=1)/np.maximum(np.abs(amplitude), 1e-300)
    gradient_defect = gradient[:, None]-nodes["k"]*nodes["directions"][triangles]
    phase_indicator = np.max(np.abs(np.einsum("nvi,nvi->nv", gradient_defect, xy-center[:, None])), axis=1)
    slow_p1_power = exit_area/6*(np.sum(np.abs(vertex_amplitude)**2, axis=1)
        + np.real(vertex_amplitude[:, 0]*vertex_amplitude[:, 1].conj()
                  + vertex_amplitude[:, 1]*vertex_amplitude[:, 2].conj()
                  + vertex_amplitude[:, 2]*vertex_amplitude[:, 0].conj()))
    metadata.update(selected_screen_rays=int(selected.sum()), valid_exit_rays=int((selected & nodes["valid"]).sum()),
        triangles=len(triangles), total_entrance_area_m2=total_area,
        accepted_entrance_area_m2=float(entrance_area.sum()),
        accepted_entrance_area_fraction=float(entrance_area.sum()/total_area),
        convex_hull_area_fraction=float(sum(metadata["hull_area_by_bore_m2"])/total_area),
        exit_area_counting_branches_m2=float(exit_area.sum()),
        phase_gradient_indicator_rad=_stats(phase_indicator), amplitude_variation=_stats(amplitude_variation),
        exit_power_constant_patch=float(np.sum(exit_area*np.abs(amplitude)**2)),
        exit_power_affine_amplitude=float(np.sum(slow_p1_power)),
        exit_power_vertex_mean=float(np.sum(exit_area*np.mean(np.abs(vertex_amplitude)**2, axis=1))),
        determinant_floor=nodes["determinant_floor"], transport=nodes["transport"],
        amplitude_model="point_jacobian_p1; amplitude mean retained as a constant-patch diagnostic",
        limitations=["Exit GO is singular at exit-plane caustics; invalid, mixed-branch and folded triangles are excluded.",
                     "Missing convex-hull and branch-transition area contributes zero, without a field-error bound.",
                     "Affine phase and affine complex amplitude are approximations requiring mesh convergence.",
                     "Overlapping projected triangles from distinct regular branches add coherently.",
                     "Reflection count, Maslov and determinant sign are necessary branch checks, not proof of interior regularity."])
    return dict(triangles=xy, amplitude=amplitude, phase=phase, gradient=gradient, center=center,
                vertex_phase=vertex_phase, vertex_amplitude=vertex_amplitude,
                vertex_directions=nodes["directions"][triangles],
                triangle_reflections=nodes["reflections"][triangles[:, 0]],
                entrance_triangles=q, ray_indices=triangles,
                entrance_area=entrance_area, exit_area=exit_area, metadata=metadata)


def mesh_from_triangles(nodes, cap, triangles):
    """Assemble a supplied regular branch mesh without changing its triangles.

    Invalid or mixed-branch triangles raise instead of being silently removed.
    The caller is responsible for nonoverlap and coverage of the entrance mesh.
    """
    triangles = np.asarray(triangles)
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)
            or np.any(triangles < 0) or np.any(triangles >= len(nodes["points"]))):
        raise ValueError("supplied triangles must be a nonempty (N,3) array of valid integer node indices")
    triangles = triangles.astype(np.int64, copy=False)
    if (np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0)
            or len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles)):
        raise ValueError("supplied triangles contain repeated vertices or duplicate faces")
    q, x = nodes["entrance"][triangles], nodes["points"][triangles]
    bores = nodes["bore"][triangles]
    if np.any(bores < 0) or np.any(bores >= len(cap.bores)) or np.any(bores != bores[:, :1]):
        raise ValueError("supplied triangles must lie within one assigned bore each")
    for field in ("reflections", "maslov"):
        values = nodes[field][triangles]
        if np.any(values != values[:, :1]):
            raise ValueError(f"supplied triangles mix {field}")
    signs = np.sign(nodes["determinant"][triangles])
    if np.any(signs != signs[:, :1]) or not np.all(nodes["valid"][triangles]):
        raise ValueError("supplied triangles have invalid vertices or mixed Jacobian orientation")
    qdet = _det(np.stack([q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]], axis=-1))
    xdet = _det(np.stack([x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]], axis=-1))
    radii = np.array([float(cap.bores[b]["radius"]) for b in bores[:, 0]])
    threshold = 128*np.finfo(float).eps*radii**2
    if (np.any(abs(qdet) <= threshold) or np.any(abs(xdet) <= threshold)
            or not np.isfinite(xdet).all() or np.any(np.sign(xdet*qdet) != signs[:, 0])):
        raise ValueError("supplied triangles have a folded or degenerate exit projection")
    selected = np.zeros(len(nodes["points"]), bool)
    selected[np.unique(triangles)] = True
    hull_area, accepted_area, per_bore = [], [], {}
    for bore_id, bore in enumerate(cap.bores):
        ids = np.flatnonzero(selected & (nodes["bore"] == bore_id))
        local = bores[:, 0] == bore_id
        accepted_area.append(float(.5*np.sum(abs(qdet[local]))))
        if len(ids) >= 3:
            radius = float(bore["radius"])
            hull = ConvexHull((nodes["entrance"][ids]-bore["center"])/radius)
            hull_area.append(float(hull.volume*radius**2))
            per_bore[str(bore_id)] = dict(total_triangles=int(local.sum()), kept_triangles=int(local.sum()))
        else:
            hull_area.append(0.)
    metadata = dict(hull_area_by_bore_m2=hull_area, accepted_entrance_area_by_bore_m2=accepted_area,
        rejection_counts={name: 0 for name in ("mixed_reflections", "mixed_maslov", "mixed_orientation",
                                               "invalid_vertex", "folded_exit", "degenerate")},
        bore_triangles=per_bore, mesh_topology="Supplied branch triangles preserved; no Delaunay or hidden removal.",
        topology_contract="Caller supplies a nonoverlapping entrance partition and measures omitted areas.")
    return _assemble_exit_mesh(nodes, cap, triangles, metadata, selected)


def _p1_triangle_power(values, area):
    return area/12*(np.sum(np.abs(values)**2, axis=1)+np.abs(np.sum(values, axis=1))**2)


def apply_tube_flux(mesh, nodes):
    """Copy a regular exit mesh and match each patch's finite entrance-tube flux.

    Both fluxes use the exact P1 complex-amplitude mass matrix. Optical phases,
    geometry and omitted areas remain unchanged; this does not bound field error.
    """
    ids = mesh["ray_indices"]
    amplitude = np.asarray(mesh["vertex_amplitude"], complex)
    uz0, uzexit = nodes["uz0"][ids], nodes["uzexit"][ids]
    distance = nodes["source_distance"][ids]
    if (np.any(uz0 <= 0) or np.any(uzexit <= 0) or np.any(distance <= 0)
            or not all(np.isfinite(v).all() for v in (uz0, uzexit, distance, amplitude, nodes["fresnel"][ids]))):
        raise ValueError("finite-tube flux needs finite forward directions, positive source distances and amplitudes")
    input_flux_amplitude = nodes["fresnel"][ids]*np.sqrt(uz0)/distance
    output_flux_amplitude = amplitude*np.sqrt(uzexit)
    pin = _p1_triangle_power(input_flux_amplitude, mesh["entrance_area"])
    pout = _p1_triangle_power(output_flux_amplitude, mesh["exit_area"])
    if np.any((pout <= 0) & (pin > 0)):
        raise ValueError("positive entrance flux cannot be recovered by rescaling a zero exit field")
    scale = np.divide(np.sqrt(pin), np.sqrt(pout), out=np.ones(len(pin)), where=pout > 0)
    corrected = amplitude*scale[:, None]
    if not np.isfinite(corrected).all():
        raise ValueError("finite-tube amplitude rescaling overflowed")
    after = _p1_triangle_power(corrected*np.sqrt(uzexit), mesh["exit_area"])
    metadata = dict(mesh["metadata"])
    metadata["amplitude_model"] = "finite_ray_tube_flux"
    metadata["tube_flux"] = dict(
        input_p1_branch_flux=float(pin.sum()), point_jacobian_p1_exit_branch_flux=float(pout.sum()),
        corrected_p1_exit_branch_flux=float(after.sum()), scale_statistics=_stats(scale),
        scale_minimum=float(np.min(scale)) if len(scale) else None,
        maximum_relative_patch_flux_defect=float(np.max(np.divide(np.abs(after-pin), pin,
                    out=np.zeros_like(pin), where=pin > 0), initial=0)),
        branch_flux_norm_of_correction=float(np.linalg.norm(np.sqrt(pin)-np.sqrt(pout))),
        input_flux="P1 triangle mass of Fresnel*sqrt(uz_entrance)/source_distance",
        output_flux="P1 triangle mass of slow exit amplitude*sqrt(uz_exit)",
        rule="positive per-triangle sqrt(input_flux/output_flux); unwrapped optical phase unchanged",
        assumptions=["Regular single-branch geometric-optics flux transport and affine projected triangle geometry.",
                     "P1 complex Fresnel and source-amplitude quadrature approximates the finite entrance flux.",
                     "Separate-branch flux sums omit interference cross terms and are not total coherent-field power.",
                     "Flux conservation alone gives no small bound on phase, field or coherence error.",
                     "Exit caustics, discarded triangles and inaccurate geometry remain unresolved."])
    average = corrected.mean(axis=1)
    metadata["exit_power_constant_patch"] = float(np.sum(mesh["exit_area"]*np.abs(average)**2))
    metadata["exit_power_affine_amplitude"] = float(np.sum(_p1_triangle_power(corrected, mesh["exit_area"])))
    metadata["exit_power_vertex_mean"] = float(np.sum(mesh["exit_area"]*np.mean(np.abs(corrected)**2, axis=1)))
    metadata["power_convention"] = "Separate patch/branch squared norms; omit cross terms between overlapping branches."
    if "holdout" in metadata:
        metadata["holdout"] = dict(metadata["holdout"], amplitude_model_tested="point_jacobian_before_tube_flux")
    return dict(mesh, vertex_amplitude=corrected, amplitude=average, metadata=metadata)

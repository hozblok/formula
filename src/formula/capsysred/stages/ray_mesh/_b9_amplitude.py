"""Shared positive nodal corrections for regular GO exit amplitudes.

This is an optional numerical representation, not a caustic uniformization.
Separate-branch flux targets must never be used to normalize the coherent sum.
"""
from __future__ import annotations

import numpy as np

from ._b9_archive import _p1_triangle_power, _stats


def apply_shared_flux(mesh, nodes):
    """Return a C0, phase-preserving nodal fit to finite-tube flux scales.

    For each regular face T, s_T = sqrt(Pin_T/Pout_T) uses the same P1
    flux-amplitude mass as ``apply_tube_flux``. Shared real nodal scales minimize

        sum_T sum_(i in T) w_Ti (s_i - s_T)**2,
        w_Ti = area_exit_T * uz_exit_i * abs(A_GO_i)**2 / 3.

    Thus s_i is a positive weighted mean of incident face scales. This diagonal
    objective is a lumped flux-amplitude norm; it is not a constrained exact
    flux-conservation solve. Patch and total branch-flux residuals are reported.

    The input must contain the uncorrected point-Jacobian amplitude. Sharing is
    by ray_indices only, never by coincident exit coordinates across branches.
    A node of zero flux weight keeps scale=1 and its existing zero amplitude.
    Invalid/overflowing data and positive-input/zero-output faces raise; no
    clipping, hidden smoothing thresholds or fitted global normalization occur.
    """
    ids = np.asarray(mesh["ray_indices"])
    if (ids.ndim != 2 or ids.shape[1] != 3 or not len(ids)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= len(nodes["amplitude"]))):
        raise ValueError("shared flux requires nonempty triangles of valid node indices")
    reference = np.asarray(nodes["amplitude"], complex)[ids]
    amplitude = np.asarray(mesh["vertex_amplitude"], complex)
    if amplitude.shape != ids.shape or not np.array_equal(amplitude, reference):
        raise ValueError("shared flux requires the uncorrected point-Jacobian nodal amplitude")
    entrance_area = np.asarray(mesh["entrance_area"], float)
    exit_area = np.asarray(mesh["exit_area"], float)
    uz0 = np.asarray(nodes["uz0"], float)[ids]
    uzexit = np.asarray(nodes["uzexit"], float)[ids]
    distance = np.asarray(nodes["source_distance"], float)[ids]
    fresnel = np.asarray(nodes["fresnel"], complex)[ids]
    if (entrance_area.shape != (len(ids),) or exit_area.shape != (len(ids),)
            or any(not np.isfinite(v).all() for v in
                   (amplitude, entrance_area, exit_area, uz0, uzexit, distance, fresnel))
            or np.any(entrance_area <= 0) or np.any(exit_area <= 0)
            or np.any(uz0 <= 0) or np.any(uzexit <= 0) or np.any(distance <= 0)):
        raise ValueError("shared flux requires finite amplitudes, positive areas, distances and forward directions")
    # Supplied branch topology remains the caller's responsibility. When branch
    # labels are present, reject accidental connections across those labels.
    for key in ("bore", "reflections", "maslov"):
        if key in nodes:
            values = np.asarray(nodes[key])[ids]
            if np.any(values != values[:, :1]):
                raise ValueError(f"shared flux triangles mix {key}")
    input_amplitude = fresnel*np.sqrt(uz0)/distance
    output_amplitude = amplitude*np.sqrt(uzexit)
    pin = _p1_triangle_power(input_amplitude, entrance_area)
    pout = _p1_triangle_power(output_amplitude, exit_area)
    if not np.isfinite(pin).all() or not np.isfinite(pout).all():
        raise ValueError("shared flux mass overflowed")
    if np.any((pout <= 0) & (pin > 0)):
        raise ValueError("positive entrance flux cannot be recovered from zero exit amplitude")
    face_scale = np.divide(np.sqrt(pin), np.sqrt(pout), out=np.ones(len(pin)), where=pout > 0)
    used, inverse = np.unique(ids, return_inverse=True)
    local_ids = inverse.reshape(ids.shape)
    weights = exit_area[:, None]*uzexit*np.abs(amplitude)**2/3
    if not np.isfinite(weights).all() or not np.isfinite(face_scale).all():
        raise ValueError("shared flux weights or scales overflowed")
    weight_sum = np.bincount(local_ids.ravel(), weights=weights.ravel(), minlength=len(used))
    weighted_scales = np.bincount(local_ids.ravel(), weights=(weights*face_scale[:, None]).ravel(),
                                  minlength=len(used))
    nodal_scale = np.divide(weighted_scales, weight_sum, out=np.ones(len(used)), where=weight_sum > 0)
    corrected = amplitude*nodal_scale[local_ids]
    if not np.isfinite(corrected).all() or np.any(nodal_scale < 0):
        raise ValueError("shared positive amplitude correction overflowed")
    after = _p1_triangle_power(corrected*np.sqrt(uzexit), exit_area)
    positive_pin = pin > 0
    relative_defect = np.divide(after-pin, pin, out=np.zeros_like(pin), where=positive_pin)
    residual = nodal_scale[local_ids]-face_scale[:, None]
    normal_residual = np.bincount(local_ids.ravel(), weights=(weights*residual).ravel(), minlength=len(used))
    zero_weight = weight_sum == 0
    norm_target = float(np.sum(weights*face_scale[:, None]**2))
    objective_before = float(np.sum(weights*(1-face_scale[:, None])**2))
    objective_after = float(np.sum(weights*residual**2))
    metadata = dict(mesh["metadata"])
    metadata["amplitude_model"] = "shared_positive_nodal_flux_fit"
    metadata["shared_flux"] = dict(
        unique_used_nodes=len(used), zero_weight_nodes=int(zero_weight.sum()),
        nodal_scale_statistics=_stats(nodal_scale), nodal_scale_minimum=float(nodal_scale.min()),
        face_target_scale_statistics=_stats(face_scale), face_target_scale_minimum=float(face_scale.min()),
        input_p1_branch_flux=float(pin.sum()), point_jacobian_p1_exit_branch_flux=float(pout.sum()),
        corrected_p1_exit_branch_flux=float(after.sum()),
        relative_total_branch_flux_defect=float((after.sum()-pin.sum())/pin.sum()) if pin.sum() > 0 else None,
        absolute_patch_relative_flux_defect=_stats(abs(relative_defect[positive_pin])),
        positive_exit_flux_on_zero_input_faces=float(after[~positive_pin].sum()),
        lumped_objective_before=objective_before, lumped_objective_after=objective_after,
        relative_lumped_target_residual=float(np.sqrt(objective_after/norm_target)) if norm_target > 0 else 0.,
        normal_equation_max_residual=float(np.max(abs(normal_residual), initial=0)),
        objective="min_s>=0 sum_Ti (Aexit_T*uz_i*abs(Apoint_i)^2/3)*(s_i-s_T)^2; s_T=sqrt(Pin_T/Pout_T)",
        rule="Positive shared nodal weighted mean; no clipping and no global renormalization.",
        continuity="One scale per original ray index. Distinct branches at coincident exit coordinates are not merged.",
        phase="Optical phase arrays unchanged; nonzero Fresnel/Maslov nodal phase preserved by nonnegative real scaling.",
        assumptions=[
            "The caller provides a regular branch mesh and physical point-Jacobian nodal amplitudes.",
            "Tube fluxes use consistent P1 flux-amplitude quadrature; they are discrete targets, not exact finite-tube integrals.",
            "The nodal fit sacrifices exact per-face flux matching to remove artificial shared-edge amplitude jumps.",
            "Separate-branch flux sums omit interference cross terms and are not total coherent-field power.",
            "Corrected amplitudes require their own probe and mesh-convergence checks; point-amplitude probes do not validate this fit.",
            "The correction does not repair missing support, exit caustics, phase errors, or unresolved interior folds."])
    if "holdout" in metadata:
        metadata["holdout"] = dict(metadata["holdout"], amplitude_model_tested="point_jacobian_before_shared_flux")
    average = corrected.mean(axis=1)
    metadata["exit_power_constant_patch"] = float(np.sum(exit_area*abs(average)**2))
    metadata["exit_power_affine_amplitude"] = float(np.sum(_p1_triangle_power(corrected, exit_area)))
    metadata["exit_power_vertex_mean"] = float(np.sum(exit_area*np.mean(abs(corrected)**2, axis=1)))
    metadata["power_convention"] = "Separate patch/branch squared norms; omit cross terms between overlapping branches."
    return dict(mesh, vertex_amplitude=corrected, amplitude=average, metadata=metadata,
                shared_flux_node_ids=used, shared_flux_node_scales=nodal_scale)

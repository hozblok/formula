"""Stage 18 result metadata, snapshots and output publication."""

from pathlib import Path
import os

import numpy as np

from .._map_common import _array_counts, _atomic_npz, _hash, _jackknife, normalized_coherence
from .._archive_common import _dump


def _save_adaptive_mesh(folder, mode, bore, mesh):
    name = f"mesh-mode{mode}-bore{bore}.npz"
    arrays = {key: value for key, value in mesh.items() if isinstance(value, np.ndarray)}
    arrays.update({f"nodes_{key}": value for key, value in mesh["trace_nodes"].items()
                   if isinstance(value, np.ndarray)})
    arrays.update({f"probe_{key}": value for key, value in mesh["partition_probe_metrics"].items()})
    _atomic_npz(Path(folder)/name, **arrays)
    return dict(file=name, sha256=_hash(Path(folder)/name),
                scope="Full entrance partition, accepted exit mesh and cached vertex/probe GO data; relative optical phase, not a raw decimal ray archive")


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


def metadata(sim, options, base, archive_modes, archive_rays):
    cap = sim.cfg.capillary
    screen = cap.all_screens[options["screen_index"]]
    archive, grid = base["archive"], base["grid"]
    fingerprint_hash, index_hash = base["fingerprint_sha256"], base["index_sha256"]
    groups = options.get("carrier_groups", 0)
    phase_quadrature = options.get("field_representation") in ("phase_quadrature", "curved_tubes")
    result = dict(provider="archive_contour",
        status="experimental-GO-exit-phase-diffraction" if phase_quadrature else "experimental-GO-exit-contour-diffraction",
        full_coherence_computed=False, accuracy_validated=False, options=options,
        archive=archive, archive_mode_count=archive_modes, archive_rays_per_mode=archive_rays,
        fingerprint_sha256=fingerprint_hash, index_sha256=index_hash,
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
    result["failed_modes"] = []
    return result


def save_field(folder, completed, key, item, grid, groups, options):
    field = item["field"]
    name = f"field-{_variant_name(key[0], key[1], options)}-mode{completed['mode']}.npz"
    extra = {}
    if item.get("residual") is not None:
        extra = dict(residual_variance=item["residual"]["variance"],
                     residual_reference_covariance=item["residual"]["reference_covariance"],
                     residual_batches=np.array(len(item["residual"]["variance"]) and
                                               (options.get("curved_retrace") or {}).get("residual_batches", 0)))
    _atomic_npz(folder/name, field=field, x=grid["x"], y=grid["y"],
        receiver_width_m=np.array(grid["cell_width"]), ref_index=np.array(grid["ref_index"]),
        carrier_groups=np.array(groups), phase_subdivisions=np.array(key[1]),
        **_representation_metadata(options), **extra)
    return dict(file=name, mode=completed["mode"], sha256=_hash(folder/name))


def _save_rejected_mesh(folder, mode, budget, metadata):
    _dump(Path(folder)/f"rejected-mesh-mode{mode}-b{budget}.json", metadata)


def prepare_output(out_dir):
    partial, final = Path(out_dir)/"stage18.partial", Path(out_dir)/"stage18"
    partial.mkdir(parents=True, exist_ok=False)
    return partial, final


def write_metadata(folder, result):
    _dump(folder/"meta.json", result)


def publish(partial, final, result):
    _dump(partial/"meta.json", result)
    if final.exists():
        raise ValueError("stage18 appeared concurrently; refusing replacement")
    os.rename(partial, final)

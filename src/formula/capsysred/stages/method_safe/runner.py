"""Stage 17 preflight and provider dispatch."""

import os

from .phase_audit import RESULT_DIR, run_phase_audit


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
    screens = cap.all_screens
    if options["screen_index"] >= len(screens):
        raise ValueError("b5_estimator.screen_index is outside the configured screen list")
    if float(screens[options["screen_index"]].z) < float(cap.z1):
        raise ValueError("stage 17 target screen must follow the capillary exit")
    for bore in cap.bores:
        if bore.get("kind", "cylinder") not in ("cylinder", "torus", "polygon"):
            raise ValueError("stage 17 archive_phase supports circular, torus and regular-polygon bores only")


def run_b5_stage(sim, out_dir, options, *, rays_paths, log=None):
    if options["provider"] == "archive_canonical":
        from .map import run_canonical_map
        return run_canonical_map(sim, out_dir, options, rays_paths=rays_paths, log=log)

    preflight_b5_inputs(sim, options)
    preflight_b5_output(out_dir)
    return run_phase_audit(sim, out_dir, options, rays_paths=rays_paths, log=log)

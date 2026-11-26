"""Stage 16: wave-optics degree of coherence (YAML section `wave_estimator`).

Baseline estimator: one positive source quadrature; complex W(P, P_ref) and
I(P) accumulate from the same fields (PSD by construction). Providers:
`free` (source -> screen Fresnel), `uisk` (uisk.py: unfolded-image spectral
Kirchhoff: straight regular-polygon bores, plane-wave r(kappa) per bounce,
Kirchhoff entrance/exit masks), `images` (images.py: the same physics for
axis-aligned rectangles as a product of two line sums over every image up to
the guided horizon), `march` (march.py: split-step through the wall as a
medium, cross section free to change along z) and `fb` (fb.py: Fourier-Bessel
modes of circular / torus bores, DIR-ell wall). Paraxial scalar optics, carrier e^{+ikz}
dropped, phasor e^{-i omega t}. Rows: stage16/[screen-N/|free/]mu-wave.jsonl.
"""

import math
import os
import shutil
import time

import numpy as np

from ...screen import ScreenGrid
from ...shared.units import m_to_um
from .fb_scene import fb_capillary_scene
from .funnel_scene import funnel_scene
from .images import ImageSolver, axis_rectangle, far_field
from .march import MarchSolver
from .output import RESULT_DIR, _emit, _emit_far_field, _report_lines
from .polygon import Polygon
from .screen_sampler import Accumulator, ScreenSampler, _finish
from .source import _rule
from .uisk import BoreSolver

FIELD_DUMP = "field-capillary.npz"
MODEL_CLASS = {"free": "free-space", "uisk": "kirchhoff-rims", "images": "kirchhoff-rims",
               "march": "paraxial-medium", "fb": "dir-ell"}


def preflight_wave_output(out_dir):
    """Fail before any work when the result directory already exists (no clobber)."""
    final = os.path.join(out_dir, RESULT_DIR)
    if os.path.lexists(final):
        raise ValueError(f"{final}: publication conflict; remove it manually")


def _raw_spectral_weights(cfg):
    """Weights as configured, before spectral_lines() normalizes them."""
    spec = cfg.spectrum
    mode = spec.get("mode", "monochromatic")
    if mode == "lines":
        return [float(l.get("weight", 1.0)) for l in spec["lines"]]
    if mode == "table":
        from ...spectrum import _table_pairs
        return [w for _, w in _table_pairs(spec["file"], cfg.precision)]
    return [float(ln.weight) for ln in spectral_lines_of(cfg)]


def spectral_lines_of(cfg):
    from ...spectrum import spectral_lines
    return spectral_lines(cfg.spectrum, cfg.energy_kev)


def preflight_wave_inputs(sim, wave):
    """Stage-16 input contract, cheap and before any heavy work: raw spectral weights,
    screens on or past the exit, coherent-cell references inside their windows."""
    weights = _raw_spectral_weights(sim.cfg)
    energies = [float(ln.e_kev) for ln in sim.lines]
    if (not all(math.isfinite(w) and w >= 0.0 for w in weights) or sum(weights) <= 0.0
            or not all(math.isfinite(e) and e > 0.0 for e in energies)):
        raise ValueError("wave_estimator: spectral lines need finite non-negative weights "
                         "with a positive sum and finite positive energies")
    cfg = sim.cfg
    scenes = []
    if cfg.free_source is not None and wave["provider"] in ("auto", "free"):
        scenes.append([cfg.free_screen])
    cap = cfg.capillary
    if wave["provider"] == "free" and cfg.free_source is None:
        raise ValueError("wave_estimator: provider free needs a configured free.source")
    if wave["provider"] in ("uisk", "images", "march", "fb") and cap is None:
        raise ValueError(f"wave_estimator: provider {wave['provider']} needs a configured capillary scene")
    if cap is not None and wave["provider"] in ("auto", "uisk", "images", "march", "fb"):
        prov = capillary_provider(cap, wave)
        if prov in ("uisk", "images", "march"):
            for i, scr in enumerate(cap.all_screens):
                if float(scr.z) < float(cap.z1):
                    raise ValueError(f"wave_estimator: capillary screen {i} at z = {float(scr.z)} "
                                     f"lies before the exit z1 = {float(cap.z1)}; {prov} screens sit "
                                     "on the exit plane or behind it")
        ff = wave["far_field"]
        if ff is not None and len(cap.bores) != 1:
            raise ValueError("wave_estimator: far_field needs a single bore")
        if ff is not None and prov != "fb" and any(ff.get(key) is not None for key in ("n_theta", "n_phi", "ring")):
            raise ValueError("wave_estimator.far_field: n_theta, n_phi and ring belong to provider fb (pad to the others)")
        if wave.get("hard_wall") and prov != "images":
            raise ValueError("wave_estimator: hard_wall is implemented for provider images only")
        if prov != "fb" and (wave.get("fb_figure") or wave.get("fb_figure_noise")):
            raise ValueError("wave_estimator: fb_figure and fb_figure_noise belong to provider fb")
        scenes.append(cap.all_screens)
    for screens in scenes:
        for scr in screens:
            g = ScreenGrid(scr)
            ScreenSampler(g, scr.reference or (g.cxf, g.cyf), wave["observable"], 1)


def _polygon_circumradius(bore):
    if bore.get("sides") is None:
        return float(bore["radius"])
    poly = Polygon(bore)
    return float(np.max(np.linalg.norm(poly.verts - poly.c, axis=1)))


def capillary_provider(cap, wave):
    """Provider for the capillary scene; `auto` picks by geometry: images for rectangles with
    faces along the lattice axes, march for other polygons, fb for circular / torus bores and for
    conformal funnels. An explicit choice must match the geometry (no fallback)."""
    kinds = {bore.get("kind", "cylinder") for bore in cap.bores}
    polygonal = kinds <= {"polygon"}
    circular = kinds <= {"cylinder", "torus"} or kinds == {"funnel"}
    prov = wave["provider"]
    if prov == "auto":
        if polygonal:
            prov = "images" if all(axis_rectangle(b) is not None for b in cap.bores) else "march"
        else:
            prov = "fb" if circular else None
    if prov == "march" and wave.get("march") is None:
        raise ValueError("wave_estimator: provider march (auto for non-rectangular polygons) needs the "
                         "march section {dz, box, sponge, defects}")
    if prov == "uisk" and not polygonal:
        raise ValueError("wave_estimator: provider uisk supports polygon bores "
                         f"only (sides: n, optional apothems); got kinds {sorted(kinds)}")
    if prov == "images":
        if not polygonal or any(axis_rectangle(b) is None for b in cap.bores):
            raise ValueError("wave_estimator: provider images supports rectangles with faces along "
                             f"the lattice axes only (sides: 4, rotation 0 or 90 degrees); got kinds {sorted(kinds)}")
    if prov == "march" and not kinds <= {"polygon", "cylinder", "funnel"}:
        raise ValueError("wave_estimator: provider march supports straight polygon, circular and funnel bores "
                         f"only; got kinds {sorted(kinds)}")
    if prov == "fb" and not circular:
        raise ValueError("wave_estimator: provider fb supports circular bores (radius, optional "
                         f"bend) or conformal funnels only; got kinds {sorted(kinds)}")
    if prov == "fb" and kinds == {"funnel"} and wave.get("fb_wall") not in ("exact-roots", "effective-radius"):
        raise ValueError("wave_estimator: provider fb on funnel bores needs fb_wall: exact-roots or effective-radius")
    if prov == "fb" and kinds != {"funnel"} and wave.get("fb_wall") == "effective-radius":
        raise ValueError("wave_estimator: fb_wall effective-radius is for funnel bores only")
    if prov is None:
        raise ValueError(f"wave_estimator: no provider for bore kinds {sorted(kinds)} "
                         "(uisk: polygons; fb: circular / torus)")
    return prov


def _theta_max(cap, src_nodes, grids, refs=()):
    """Largest paraxial angle the lattice must resolve: entrance cone, screen chirps
    and the chirps of point references (which may lie outside the windows); a screen
    on the exit plane adds nothing."""
    z_in = float(cap.z0) - float(cap.source.position[2])
    theta = 0.0
    for bore in cap.bores:
        c = np.array([float(bore["center"][0]), float(bore["center"][1])])
        rc = _polygon_circumradius(bore)
        far = max(float(np.hypot(*(n - c))) for n in src_nodes) + rc
        theta = max(theta, far / z_in)
        for i, g in enumerate(grids):
            d = float(g.z) - float(cap.z1)
            if d <= 0.0:                # exit plane: the entrance cone alone bounds the field
                continue
            corner = max(abs(g.x0f - c[0]), abs(g.x0f + g.exf - c[0]),
                         abs(g.y0f - c[1]), abs(g.y0f + g.eyf - c[1]))
            if i < len(refs) and refs[i] is not None:
                corner = max(corner, abs(refs[i][0] - c[0]), abs(refs[i][1] - c[1]))
            theta = max(theta, (corner + rc) / d)
    return theta


def run_wave_stage(sim, out_dir, wave, rays_paths=None, log=print):
    """Stage 16 on every configured scene; publishes out_dir/stage16 atomically."""
    preflight_wave_output(out_dir)
    preflight_wave_inputs(sim, wave)
    partial = os.path.join(out_dir, RESULT_DIR + ".partial")
    if os.path.lexists(partial):
        raise ValueError(f"{partial}: stale partial output; remove it manually")
    os.makedirs(partial)
    try:
        return _run(sim, out_dir, partial, wave, rays_paths, log)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)     # never leave a stale partial tree
        raise


def _run(sim, out_dir, partial, wave, rays_paths, log):
    cfg = sim.cfg
    t0 = time.time()
    files, report, results = [], [], {}
    workers = wave["workers"] or -1
    p = cfg.precision
    lines = []
    for ln in sim.lines:
        if float(ln.weight) > 0.0:
            delta, beta = (cfg.material.optical(ln.e_kev)
                           if sim.per_line else (sim.delta_f, sim.beta_f))
            lines.append((float(ln.k), float(ln.weight), delta, beta))
    if cfg.free_source is not None and wave["provider"] in ("auto", "free"):
        res = _free_scene(sim, wave, lines, rays_paths)
        results["free"] = res
        files += _emit(partial, "free", res, wave, log)
        report += res["report"]
    cap = cfg.capillary
    if cap is not None and wave["provider"] in ("auto", "uisk", "images", "march", "fb"):
        prov = capillary_provider(cap, wave)
        if wave["entrance_spot"] and prov not in ("fb", "images", "march"):
            raise ValueError("wave_estimator: entrance_spot is implemented for provider fb, images or march only")
        if prov in ("uisk", "images", "march"):
            scenes = _capillary_scene(sim, wave, lines, rays_paths, workers, log,
                                      dump_dir=partial if wave["dump_field"] else None,
                                      provider=prov, far_dir=partial)
            if wave["dump_field"]:
                files.append(FIELD_DUMP)
            if wave["far_field"] is not None:
                files += scenes[0][1]["far_field_files"]
        elif all(b.get("kind") == "funnel" for b in cap.bores):
            scenes = funnel_scene(sim, wave, lines, log, out_dir=partial)
            files += scenes[0][1].get("fb_files", [])
        else:
            scenes = fb_capillary_scene(sim, wave, lines, rays_paths, log, out_dir=partial)
            files += scenes[0][1].get("fb_files", [])
        for label, res in scenes:
            results[label] = res
            sub = "" if label == "capillary" else "screen-" + label.rsplit("-s", 1)[1]
            files += _emit(partial, sub, res, wave, log)
            report += res["report"]
    if not results:
        raise ValueError(f"wave_estimator: no scene matches provider {wave['provider']!r} "
                         "(free scene: auto|free; capillary bores: auto|uisk|fb)")
    os.rename(partial, os.path.join(out_dir, RESULT_DIR))
    return {"files": [RESULT_DIR + "/" + f for f in files], "report": report,
            "results": results, "seconds": time.time() - t0}


def _free_scene(sim, wave, lines, rays_paths):
    cfg = sim.cfg
    src, scr = cfg.free_source, cfg.free_screen
    nodes, weights, rule = _rule(wave, "free", src, rays_paths)
    grid = ScreenGrid(scr)
    sampler = ScreenSampler(grid, scr.reference or (grid.cxf, grid.cyf),
                            wave["observable"], wave["pixel_subsamples"])
    acc = Accumulator(sampler, 1)
    d = float(scr.z) - float(src.position[2])
    t0 = time.time()
    for k, wl, _, _ in lines:
        for xi, w in zip(nodes, weights):
            e, ref, es = sampler.free_fields(xi, k, d)
            acc.add(w * wl, [e], [ref], [es])
    lam = float(sim.lam)
    meta = {"scene": "free", "provider": "free", "model_class": MODEL_CLASS["free"],
            "source_rule": rule, "n_nodes": int(len(nodes)), "distance_m": d,
            "fresnel_numbers": {"source": float(src.size) ** 2 / (lam * d)}}
    res = _finish(acc, sampler, meta, wave, time.time() - t0)
    res["report"] = _report_lines("free", res, wave, [])
    return res


def _capillary_scene(sim, wave, lines, rays_paths, workers, log, dump_dir=None, provider="uisk", far_dir=None):
    cfg = sim.cfg
    cap = cfg.capillary
    src = cap.source
    spot = wave["entrance_spot"] if provider in ("images", "march") else None
    if spot:
        nodes, weights = np.array([spot["center"]]), np.array([1.0])
        rule = (f"entrance spot: gaussian FWHM {spot['fwhm'] * 1e9:.0f} nm at ({spot['center'][0] * 1e6:.3f}, "
                f"{spot['center'][1] * 1e6:.3f}) um, tilt ({spot['tilt'][0] * 1e3:.3f}, {spot['tilt'][1] * 1e3:.3f}) mrad")
    else:
        nodes, weights, rule = _rule(wave, "capillary", src, rays_paths)
    screens = cap.all_screens
    labels = ["capillary"] + [f"capillary-s{i}" for i in range(1, len(cap.screens) + 1)]
    z_in = float(cap.z0) - float(src.position[2])
    length = float(cap.z1) - float(cap.z0)
    grids = [ScreenGrid(s) for s in screens]
    # samplers first: the cheap reference checks precede the heavy solver grids
    samplers = [ScreenSampler(g, s.reference or (g.cxf, g.cyf), wave["observable"],
                              wave["pixel_subsamples"]) for s, g in zip(screens, grids)]
    refs = [smp.ref_xy for smp in samplers] if wave["observable"] == "point" else ()
    theta_max = _theta_max(cap, nodes, grids, refs)
    lam_h = 2.0 * np.pi / max(k for k, *_ in lines)        # shortest wavelength sets h
    lam_reach = 2.0 * np.pi / min(k for k, *_ in lines)    # longest sets the Fresnel margin
    if spot:
        theta_max = max(theta_max, math.hypot(*spot["tilt"]) + lam_h / spot["fwhm"])
    if provider == "images":
        theta_c = max(math.sqrt(2.0 * delta) for _, _, delta, _ in lines)
        solvers = [ImageSolver(b, z_in, length, theta_max, lam_h, wave, lam_reach, theta_c, spot) for b in cap.bores]
    elif provider == "march":
        solvers = [MarchSolver(b, z_in, length, theta_max, lam_h, wave, lam_reach, spot) for b in cap.bores]
    else:
        solvers = [BoreSolver(b, z_in, length, theta_max, lam_h, wave, lam_reach,
                              cache_masks=len(nodes) * len(lines) > 1) for b in cap.bores]
    accs = [Accumulator(smp, len(solvers)) for smp in samplers]
    # sampling contract: Nyquist angle lam_h/(2h) against theta_max (ratio > 1 = aliasing)
    h_auto = lam_h / (2.0 * theta_max * wave["angle_margin"])
    sampling = {"h_auto_m": h_auto, "sampling_ratio": 2.0 * solvers[0].h * theta_max / lam_h,
                "angle_margin_effective": lam_h / (2.0 * solvers[0].h * theta_max)}
    sampling = {k: float(v) for k, v in sampling.items()}
    sampling["violation"] = bool(sampling["sampling_ratio"] > 1.0)
    if sampling["violation"]:
        log(f"  16 [capillary]: WARNING explicit grid_dx {m_to_um(solvers[0].h):.4f} um under-samples "
            f"theta_max {theta_max * 1e3:.3f} mrad at lambda_min {lam_h * 1e10:.4f} A "
            f"(Nyquist ratio {sampling['sampling_ratio']:.2f} > 1; auto h = {m_to_um(h_auto):.4f} um)")
    elif solvers[0].h > h_auto * (1.0 + 1e-12):
        log(f"  16 [capillary]: note explicit grid_dx {m_to_um(solvers[0].h):.4f} um above auto "
            f"{m_to_um(h_auto):.4f} um: angle margin {sampling['angle_margin_effective']:.2f} "
            f"instead of {wave['angle_margin']:g}")
    if provider == "images":
        log(f"  16 [capillary/images]: {rule}; lattice h = {m_to_um(solvers[0].h):.4f} um, "
            f"theta_max = {theta_max * 1e3:.3f} mrad, images per axis {solvers[0].n_images} "
            f"(horizon {solvers[0].theta_horizon * 1e3:.3f} mrad), line points {solvers[0].line_points}")
    elif provider == "march":
        s0 = solvers[0]
        log(f"  16 [capillary/march]: {rule}; lattice h = {m_to_um(s0.h):.4f} um, {s0.lx.size} x {s0.ly.size} cells, "
            f"{s0.n_steps} steps of {m_to_um(s0.dz):.3f} um, {len(wave['march']['defects'])} defect(s)"
            + (", separable walls" if s0.separable else "") + (", z-dependent" if s0.z_dependent else ""))
    else:
        n_fam = sum(len(b.families) for b in solvers)
        n_cells = max((f.grid[1] - f.grid[0]) * (f.grid[3] - f.grid[2])
                      for b in solvers for f in b.families)
        log(f"  16 [capillary]: {rule}; lattice h = {m_to_um(solvers[0].h):.4f} um, "
            f"theta_max = {theta_max * 1e3:.3f} mrad, {n_fam} families "
            f"({sum(b.pruned for b in solvers)} pruned), largest grid {n_cells:,} cells")
    ff = wave["far_field"]
    far = [None] * len(lines)
    t0 = time.time()
    budget = int(wave["cache_gb"] * 1024 ** 3 / max(len(lines), 1))
    dists = [float(s.z) - float(cap.z1) for s in screens]
    dump = {}
    for li, (k, wl, delta, beta) in enumerate(lines):
        for b in solvers:
            b.cache_transfer(k, delta, beta, budget)
        kerns = [[g.kernels(b.xs, b.ys, k, d, b.h) for b in solvers] for g, d in zip(samplers, dists)]
        every = max(1, len(nodes) // 10)
        for j, (xi, w) in enumerate(zip(nodes, weights)):
            exits = [b.exit_field(xi, k, delta, beta, workers) for b in solvers]
            if dump_dir is not None:
                for bi, u in enumerate(exits):
                    dump[f"field_line{li}_node{j}_bore{bi}"] = np.asarray(u, dtype=np.complex128)
            if ff is not None:
                t, px, py, total, inside = far_field(exits[0], solvers[0].h, k, ff["theta_max"], ff["pad"])
                acc_f = far[li] or [t, 0.0, 0.0, 0.0, 0.0]
                far[li] = [t, acc_f[1] + w * px, acc_f[2] + w * py, acc_f[3] + w * total, acc_f[4] + w * inside]
            for g, acc, kern_b, d in zip(samplers, accs, kerns, dists):
                fields, refs, subs = [], [], []
                for b, u, kern in zip(solvers, exits, kern_b):
                    e, ref, es = g.fields(u, kern, k, d, b.h)
                    fields.append(e)
                    refs.append(ref)
                    subs.append(es)
                acc.add(w * wl, fields, refs, subs)
            if (j + 1) % every == 0 or j + 1 == len(nodes):
                log(f"    nodes {j + 1}/{len(nodes)}, {time.time() - t0:.0f} s")
    if dump_dir is not None:
        for bi, b in enumerate(solvers):
            dump[f"xs_bore{bi}"], dump[f"ys_bore{bi}"] = b.xs, b.ys
        np.savez(os.path.join(dump_dir, FIELD_DUMP), k=np.array([k for k, *_ in lines]),
                 line_weights=np.array([wl for _, wl, *_ in lines]), nodes=np.asarray(nodes, dtype=float),
                 node_weights=np.asarray(weights, dtype=float), h=np.array(solvers[0].h),
                 z_exit=np.array(float(cap.z1)), **dump)
    out = []
    far_files = _emit_far_field(far_dir, far, lines, ff, wave) if ff is not None else []
    a = float(cap.bores[0]["radius"])
    lam0 = float(sim.lam)
    model = MODEL_CLASS[provider] + ("" if provider == "images" or all(b.get("sides") is None or int(b["sides"]) == 4
                                                                       for b in cap.bores) else ",first-order-polygon")
    for label, g, acc, d in zip(labels, samplers, accs, dists):
        nf = {"entrance": a * a / (lam0 * length), "source": a * a / (lam0 * z_in),
              "exit": a * a / (lam0 * d) if d > 0.0 else None}   # None: screen on the exit plane
        meta = {"scene": label, "provider": provider, "model_class": model,
                "source_rule": rule, "n_nodes": int(len(nodes)),
                "lattice_h_m": solvers[0].h, "lattice_wavelength_m": lam_h,
                "theta_max_rad": theta_max,
                "box_margin_m": solvers[0].reach, "theta_cut_rad": solvers[0].theta_cut,
                "sampling": sampling, "fresnel_numbers": nf,
                "distance_m": d, "apothem_m": a,
                "apothems_m": [float(v) for v in cap.bores[0].get("apothems") or [a]]}
        if provider == "images":
            meta.update({"images_per_axis": [list(b.n_images) for b in solvers],
                         "line_points": [list(b.line_points) for b in solvers],
                         "theta_horizon_rad": solvers[0].theta_horizon,
                         "entrance_spot": spot})
            detail = (f"- lattice h = {m_to_um(solvers[0].h):.4f} µm ({wave['mask_supersample']}× mask "
                      f"supersampling), θ_max = {theta_max * 1e3:.3f} mrad, images per axis "
                      f"{[list(b.n_images) for b in solvers]} up to the guided horizon "
                      f"{solvers[0].theta_horizon * 1e3:.3f} mrad; no band window")
        elif provider == "march":
            s0 = solvers[0]
            meta.update({"march": {"dx_m": s0.h, "dz_m": s0.dz, "n_steps": s0.n_steps,
                                   "lattice": [int(s0.lx.size), int(s0.ly.size)], "box_m": wave["march"]["box"],
                                   "sponge_m": wave["march"]["sponge"], "defects": wave["march"]["defects"],
                                   "separable_walls": bool(s0.separable), "z_dependent": bool(s0.z_dependent)},
                         "entrance_spot": spot})
            detail = (f"- lattice h = {m_to_um(s0.h):.4f} µm, {s0.lx.size} × {s0.ly.size} cells, {s0.n_steps} steps "
                      f"of {m_to_um(s0.dz):.3f} µm, box margin {m_to_um(wave['march']['box']):.3f} µm, "
                      f"defects {len(wave['march']['defects'])}; wall as a medium n = 1 − δ + iβ, Kirchhoff entrance")
        else:
            meta.update({"families": [{"bore": i, "count": len(b.families), "pruned": b.pruned,
                                       "sequences": [list(f.seq) for f in b.families],
                                       "grids": [[int(v) for v in f.grid] for f in b.families]}
                                      for i, b in enumerate(solvers)],
                         "max_bounces": int(wave["max_bounces"])})
            detail = (f"- lattice h = {m_to_um(solvers[0].h):.4f} µm ({wave['mask_supersample']}× mask "
                      f"supersampling), θ_max = {theta_max * 1e3:.3f} mrad, families "
                      f"{[len(b.families) for b in solvers]} (pruned {[b.pruned for b in solvers]}), "
                      f"max_bounces = {wave['max_bounces']}")
        res = _finish(acc, g, meta, wave, time.time() - t0)
        res["far_field_files"] = far_files
        res["report"] = _report_lines(label, res, wave, [
            detail,
            f"- N_F: entrance {nf['entrance']:.2f}, source {nf['source']:.2f}, exit "
            + (f"{nf['exit']:.3f}" if nf["exit"] is not None else "— (screen on the exit plane)")
            + "; ray optics is not used anywhere in this stage",
            f"- sampling: h/h_auto = {solvers[0].h / h_auto:.3f}, Nyquist ratio 2hθ_max/λ_min = "
            f"{sampling['sampling_ratio']:.3f}"
            + (" — ALIASING: the lattice under-samples the scene" if sampling["violation"] else ""),
        ])
        out.append((label, res))
    return out

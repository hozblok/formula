"""Scene of the Stage 16 provider fb for conformal funnel bores (the basis and the channel are in funnel.py).
The bores are propagated one after another with all source nodes as one block; W and I accumulate bore by bore (the exit
cores do not overlap), and the fields at the bore centres give the matrix of mu between the centres. With dump_field
the exit coefficients of every bore (all nodes) are written to funnel-exit/ for post-processing without propagation."""

import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np

from ...screen import ScreenGrid
from .funnel import FunnelBasis, FunnelChannel
from .output import _report_lines
from .screen_sampler import ScreenSampler, _finish
from .source import _rule


def funnel_scene(sim, wave, lines, log, out_dir=None):
    """Stage 16 on a scene of conformal funnel bores: exit-plane screens (point observable), the matrix of mu between
    the bore centres. Returns [(label, res), ...] like fb_capillary_scene."""
    cfg = sim.cfg
    cap = cfg.capillary
    if wave["observable"] != "point":
        raise ValueError("wave_estimator: funnel bores support observable point only")
    if wave["entrance_spot"]:
        raise ValueError("wave_estimator: funnel bores take the source of the scene, not entrance_spot")
    bores = cap.bores
    a = float(bores[0]["radius"])
    g = tuple(float(v) for v in bores[0]["g"])
    for b in bores:
        if tuple(float(v) for v in b["g"]) != g or tuple(float(v) for v in b["f"]) != g or float(b["radius"]) != a:
            raise ValueError("wave_estimator: funnel bores need one radius and one profile g = f for all bores")
    L = float(cap.z1) - float(cap.z0)
    D = float(cap.z0) - float(cap.source.position[2])
    screens = cap.all_screens
    for s in screens:
        if abs(float(s.z) - float(cap.z1)) > 1e-12:
            raise ValueError("wave_estimator: funnel bores support screens on the exit plane only")
    nodes, weights, rule = _rule(wave, "capillary", cap.source, None)
    centres = [np.array([float(b["center"][0]), float(b["center"][1])]) for b in bores]
    G_L = 1.0 + g[0] * L + g[1] * L * L
    if not wave["fb_jmax"]:
        raise ValueError("wave_estimator: funnel bores need fb_jmax (every zero below it; the guided modes reach k theta_c a)")
    jmax, layer = int(wave["fb_jmax"]), 0.1
    dz = float(wave["fb_dz"])
    tol = float(wave["fb_chebyshev_tol"])
    wall, integrator = wave["fb_wall"], wave["fb_integrator"]
    split = integrator == "split-step"
    t0 = time.time()
    samplers = [ScreenSampler(ScreenGrid(s), s.reference or (ScreenGrid(s).cxf, ScreenGrid(s).cyf), "point", 1) for s in screens]
    I = [np.zeros((sm.nx, sm.ny)) for sm in samplers]
    W = [np.zeros((sm.nx, sm.ny), complex) for sm in samplers]
    Iref = [0.0] * len(samplers)
    nb, nn = len(bores), len(nodes)
    centre_amp = np.zeros((len(lines), nb, nn), complex)
    info = []
    dump_dir = None
    if out_dir is not None and wave["dump_field"]:
        dump_dir = os.path.join(out_dir, "funnel-exit")
        os.makedirs(dump_dir, exist_ok=True)
    basis = dr = None
    for li, (k, wl, delta, beta) in enumerate(lines):
        basis = FunnelBasis(a=a, g=g, k=k, delta=delta, beta=beta, jmax=jmax, layer=layer, blocks=not split, wall=wall)
        dr = min(float(wave["fb_dr"]), basis.disc.aw / 1500.0)
        chans = [FunnelChannel(basis=basis, c0=c, integrator=integrator) for c in centres]
        exits = {}

        def exit_coeffs(bi):
            if bi not in exits:
                t1 = time.time()
                cin = chans[bi].entrance_modes(nodes, D)
                p_in = (np.abs(cin) ** 2 * basis.disc.mnorm[:, :, None]).sum((0, 1))
                cL, n_steps, deg = chans[bi].propagate(cin, L, dz, tol)
                st = chans[bi].stepper
                p_out = (np.abs(cL) ** 2 * basis.disc.mnorm[:, :, None]).sum((0, 1))
                exits[bi] = cL
                if dump_dir is not None:
                    np.save(os.path.join(dump_dir, f"line{li}-bore{bi}.npy"), cL)
                info.append({"line": li, "bore": bi, "steps": n_steps, "chebyshev_degree": deg, "seconds": time.time() - t1,
                             "phase_gradient_rad": st.phase_gradient_rad if split else None,
                             "projection_loss": st.projection_loss.tolist() if split else None,
                             "transmission": (p_out / p_in).tolist(), "layer_power": chans[bi].layer_power.tolist()})
                how = (f"phase gradient {st.phase_gradient_rad:.2f} rad per step, projection loss {st.projection_loss.max():.1e}"
                       if split else f"Chebyshev degree {deg}")
                log(f"  16 [funnel]: line {li} bore {bi}: {n_steps} steps, {how}, {time.time() - t1:.0f} s")
            return exits[bi]

        # reference fields of every node at every screen reference
        refs = []
        for sm in samplers:
            rx, ry = sm.ref_xy
            hit = [bi for bi, c in enumerate(centres) if math.hypot(rx - G_L * c[0], ry - G_L * c[1]) < a * G_L]
            if not hit:
                raise ValueError(f"wave_estimator: reference ({rx * 1e6:g}, {ry * 1e6:g}) um outside every exit core")
            e, _ = chans[hit[0]].points(exit_coeffs(hit[0]), L, np.array([rx]), np.array([ry]), dr)
            refs.append(e[0] if e.shape[0] else np.zeros(nn, complex))
        wn = wl * weights
        for bi in range(nb):
            cL = exit_coeffs(bi)
            centre_amp[li, bi] = chans[bi].centre_values(cL, L)
            cx, cy = G_L * centres[bi]
            parts = []
            for si, sm in enumerate(samplers):                  # the pixels of every screen inside this exit core, one call
                ix = np.nonzero(np.abs(sm.px - cx) < a * G_L)[0]
                iy = np.nonzero(np.abs(sm.py - cy) < a * G_L)[0]
                if ix.size and iy.size:
                    X, Y = np.meshgrid(sm.px[ix], sm.py[iy], indexing="ij")
                    parts.append((si, ix, iy, X.ravel(), Y.ravel()))
            if parts:
                e, inside = chans[bi].points(cL, L, np.concatenate([p[3] for p in parts]), np.concatenate([p[4] for p in parts]), dr)
                full = np.zeros((inside.size, nn), complex)
                full[inside] = e
                o = 0
                for si, ix, iy, X, Y in parts:
                    blk, ins = full[o:o + X.size], inside[o:o + X.size]
                    o += X.size
                    shape = (ix.size, iy.size)
                    Ib = (np.abs(blk) ** 2 * wn[None, :]).sum(axis=1).reshape(shape)
                    Wb = (blk * np.conj(refs[si])[None, :] * wn[None, :]).sum(axis=1).reshape(shape)
                    I[si][np.ix_(ix, iy)] += np.where(ins.reshape(shape), Ib, 0.0)
                    W[si][np.ix_(ix, iy)] += np.where(ins.reshape(shape), Wb, 0.0)
            exits.pop(bi, None)                                 # free the block after its last use
        for si in range(len(samplers)):
            Iref[si] += float(np.sum(wl * weights * np.abs(refs[si]) ** 2))
    seconds = time.time() - t0
    # centre matrix: W_bc = sum_lines,nodes w E_b conj E_c
    wn = np.array([wl for _, wl, _, _ in lines])[:, None, None] * weights[None, None, :]
    Wc = np.einsum("lbn,lcn->bc", wn * centre_amp, np.conj(centre_amp))
    Ic = np.real(np.diag(Wc))
    mu_c = Wc / np.sqrt(np.outer(Ic, Ic))
    files, out = [], []
    meta_base = {"provider": "fb", "source_rule": rule, "n_nodes": int(nn),
                 "model_class": {"exact-roots": "exact-jk", "effective-radius": "effective-radius"}[wall] + ",funnel-xi",
                 "funnel": {"g": list(g), "G_exit": G_L, "radius_m": a, "jmax": jmax, "layer": layer,
                            "wall": wall, "integrator": integrator, "n_modes": int(basis.disc.nmodes),
                            "A0_m": basis.A0, "dz_m": dz, "chebyshev_tol": tol, "dr_m": dr,
                            "exact_fallback_modes": sum(basis.fallback.values())},
                 "propagation": info, "distance_m": 0.0}
    if out_dir is not None:
        json.dump({"centres_in_m": [c.tolist() for c in centres], "centres_out_m": [(G_L * c).tolist() for c in centres],
                   "I": Ic.tolist(), "mu_re": mu_c.real.tolist(), "mu_im": mu_c.imag.tolist(),
                   "amplitudes_re": centre_amp.real.tolist(), "amplitudes_im": centre_amp.imag.tolist(),
                   "weights": weights.tolist(), "nodes_m": np.asarray(nodes).tolist(),
                   "basis": {"a_m": a, "g": list(g), "jmax": jmax, "layer": layer, "A0_m": basis.A0,
                             "lines": [[k, wl, delta, beta] for k, wl, delta, beta in lines], "D_m": D, "L_m": L,
                             "dump": None if dump_dir is None else "funnel-exit/line{l}-bore{b}.npy, (2M+1, nmax, nodes)"}},
                  open(os.path.join(out_dir, "funnel-centres.json"), "w", encoding="utf-8"))
        files.append("funnel-centres.json")
        if dump_dir is not None:
            files += [f"funnel-exit/{f}" for f in sorted(os.listdir(dump_dir))]
    labels = ["capillary"] + [f"capillary-s{i}" for i in range(1, len(screens))]
    for si, (label, sm) in enumerate(zip(labels, samplers)):
        acc = SimpleNamespace(I=I[si], W=W[si], I_ref=Iref[si], I_pixel=None, I_bore=None, G12=None, ref_index=sm.ref_index)
        meta = dict(meta_base, scene=label)
        res = _finish(acc, sm, meta, wave, seconds)
        res["report"] = _report_lines(label, res, wave, [f"- funnel: G(L) = {G_L:.6f}, jmax {jmax}, {basis.disc.nmodes} modes, "
                                                          f"wall {wall}, integrator {integrator}, "
                                                          f"sub-step {dz * 1e3:g} mm, centres matrix in funnel-centres.json"])
        res["fb_files"] = files
        out.append((label, res))
    return out

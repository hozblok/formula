"""Scene of the Stage 16 provider `fb`: the picklable spec of the scene, the workers over the source nodes and the
files of the far field and of the mode coefficients."""

import math
import os
import time

import numpy as np

from ......physical_constants import HC_KEV_ANGSTROM
from .....screen import ScreenGrid
from .....shared.units import m_to_um
from ...output import _report_lines
from ...screen_sampler import Accumulator, ScreenSampler, _finish
from ...source import _rule
from .basis import Disc, exact_kappa, fb_jmax_rule, mode_damping, wall_offset
from .channel import FBChannel, WallFigure
from .far_field import far_field_amplitude, far_field_profiles, ring_summary
from .lattice import Lattice


# ----------------------------------------------------------------- scene spec (picklable)

def build_spec(sim, wave, lines):
    """Plain floats only: bores, source distance, length, screens, lattice and numerics."""
    cfg = sim.cfg
    cap = cfg.capillary
    z_in = float(cap.z0) - float(cap.source.position[2])
    length = float(cap.z1) - float(cap.z0)
    bores = []
    for b in cap.bores:
        if b.get("kind") not in ("cylinder", "torus"):
            raise ValueError("wave_estimator: provider fb supports circular bores (radius, optional bend); "
                             f"got kind {b.get('kind')!r}")
        bend = b.get("bend")
        bores.append({"center": (float(b["center"][0]), float(b["center"][1])), "radius": float(b["radius"]),
                      "rbend": None if bend is None else float(bend["radius"]),
                      "toward": None if bend is None else (float(bend["toward"][0]), float(bend["toward"][1]))})
    screens = cap.all_screens
    scr = [{"z": float(s.z), "nx": int(s.nx), "ny": int(s.ny), "center": (float(s.center[0]), float(s.center[1])),
            "edge_x": float(s.edge_x), "edge_y": float(s.edge_y),
            "reference": None if s.reference is None else (float(s.reference[0]), float(s.reference[1]))}
           for s in screens]
    k_max = max(k for k, *_ in lines)
    lam_min = 2.0 * math.pi / k_max
    a_max = max(b["radius"] for b in bores)
    ells = [wall_offset(k, delta, beta) for k, _, delta, beta in lines]
    aw = a_max + max(e.real for e in ells)
    theta_cut = float(wave["fb_theta_cut"])
    jmax = int(wave["fb_jmax"]) if wave["fb_jmax"] else fb_jmax_rule(k=k_max, aw=aw, theta_cut=theta_cut)
    theta_modal = jmax / (k_max * aw)
    wall_cut = max(k * aw * math.sqrt(2.0 * delta) for k, _, delta, _ in lines) if wave["fb_wall_cut"] else None
    tilt = max((length / b["rbend"]) if b["rbend"] else 0.0 for b in bores)
    h_auto = lam_min / (2.0 * (theta_modal + tilt) * float(wave["fb_angle_margin"]))
    px = scr[0]["edge_x"] / scr[0]["nx"]
    if wave["grid_dx"]:
        h = float(wave["grid_dx"])
    else:
        h = px / max(1, int(math.ceil(px / h_auto)))
    # lattice half-width: windows, spread of the modal band over the drift, Fresnel tails, bore exits
    half = 0.0
    for s in scr:
        d = s["z"] - float(cap.z1)
        corner = max(abs(s["center"][0]) + s["edge_x"] / 2, abs(s["center"][1]) + s["edge_y"] / 2)
        half = max(half, corner + float(wave["pad"]) * ((theta_modal + tilt) * d + 4.0 * math.sqrt(lam_min * d)))
    for b in bores:
        c0 = np.array(b["center"]); t = np.array(b["toward"]) if b["toward"] else np.zeros(2)
        cl = c0 + (t * length ** 2 / (2.0 * b["rbend"]) if b["rbend"] else 0.0)
        half = max(half, float(np.max(np.abs(cl))) + aw + 2.0 * h)
    if wave["fb_lattice_half"]:
        half = float(wave["fb_lattice_half"])
    figure = {"terms": wave["fb_figure"] or [], "noise": wave["fb_figure_noise"]}
    scale = WallFigure(figure["terms"], figure["noise"], length).min_scale()
    dz = min(float(wave["fb_dz"]), scale / 16.0) if scale else float(wave["fb_dz"])
    ff = wave["far_field"]
    far = None
    if ff is not None:
        n_theta = ff.get("n_theta") or int(math.ceil(ff["theta_max"] / (lam_min / (8.0 * aw)))) + 1
        far = {"theta_max": ff["theta_max"], "n_theta": n_theta, "n_phi": ff.get("n_phi"),
               "ring": ff.get("ring") or [0.0, ff["theta_max"]]}
    return {"z_in": z_in, "length": length, "z_exit": float(cap.z1), "bores": bores, "screens": scr, "aw": aw, "jmax": jmax, "wall_cut": wall_cut,
            "theta_modal": theta_modal, "tilt": tilt, "h": h, "h_auto": h_auto, "half": half,
            "lines": [(float(k), float(wl), float(dl), float(bt)) for k, wl, dl, bt in lines],
            "ells": [(e.real, e.imag) for e in ells], "cheb_tol": float(wave["fb_chebyshev_tol"]),
            "dr": float(wave["fb_dr"]), "wall": wave["fb_wall"], "dtype": wave["fb_grid_dtype"],
            "observable": wave["observable"], "per_bore": wave["fb_per_bore_maps"], "spot": wave["entrance_spot"],
            "workers": wave["workers"] or -1, "cache_gb": float(wave["cache_gb"]),
            "figure": figure, "dz": dz, "far": far, "dump": bool(wave["dump_field"])}


class _PixelCfg:
    def __init__(self, s):
        self.z, self.nx, self.ny = s["z"], s["nx"], s["ny"]
        self.center, self.edge_x, self.edge_y = s["center"], s["edge_x"], s["edge_y"]
        self.reference = s["reference"]


# ----------------------------------------------------------------- worker

_W = {}


def _init_worker(spec):
    """Per-process state: basis, channels per line, lattice, samplers and accumulators."""
    t0 = time.time()
    dtype = np.complex64 if spec["dtype"] == "complex64" else np.complex128
    disc = Disc(aw=spec["aw"], jmax=spec["jmax"], wall_cut=spec["wall_cut"])
    disc.fine_table(spec["dr"], dtype=np.float32)
    figure = WallFigure(spec["figure"]["terms"], spec["figure"]["noise"], spec["length"])
    chans, fallback = {}, {}
    for k, wl, delta, beta in spec["lines"]:
        if spec["wall"] == "exact-roots":
            row = []
            for b in spec["bores"]:
                kap, ok = exact_kappa(disc, a=b["radius"], k=k, delta=delta, beta=beta)
                dirl = disc.kap2 / (2.0 * k) - 1j * mode_damping(disc, k, delta, beta)
                h0 = np.where(disc.valid, np.where(ok[disc.am], (kap * kap)[disc.am] / (2.0 * k), dirl), 0.0)
                fallback[k] = fallback.get(k, 0) + int(np.sum(disc.validp & ~ok))
                row.append(FBChannel(c0=b["center"], t=b["toward"], rbend=b["rbend"], disc=disc, k=k, h0=h0, figure=figure, dz=spec["dz"]))
            chans[k] = row
        else:
            gamma = mode_damping(disc, k, delta, beta) if spec["wall"] == "dir-ell" else None
            chans[k] = [FBChannel(c0=b["center"], t=b["toward"], rbend=b["rbend"], disc=disc, k=k, gamma=gamma, figure=figure, dz=spec["dz"])
                        for b in spec["bores"]]
    lat = Lattice(spec["half"], spec["h"], dtype=dtype, workers=spec["workers"], edge0=spec["screens"][0]["center"][0] - spec["screens"][0]["edge_x"] / 2,
                  cache_bytes=spec["cache_gb"] * 2 ** 30)
    grids = [ScreenGrid(_PixelCfg(s)) for s in spec["screens"]]
    b = [lat.cell_plan(g)[0] for g in grids]
    samplers = [ScreenSampler(g, s["reference"] or (g.cxf, g.cyf), spec["observable"], bb)
                for g, s, bb in zip(grids, spec["screens"], b)]
    nb = len(spec["bores"])
    per_bore = spec["per_bore"] if spec["per_bore"] is not None else nb <= 2
    far = None
    if spec["far"] is not None:
        f = spec["far"]
        nphi = f["n_phi"] or 1 << int(math.ceil(math.log2(max(4 * disc.M + 4, 256))))
        far = {"thetas": np.linspace(0.0, f["theta_max"], f["n_theta"]), "n_phi": int(nphi), "ring": f["ring"]}
    _W.update(spec=spec, disc=disc, chans=chans, lat=lat, grids=grids, samplers=samplers,
              nb=nb, per_bore=per_bore, dtype=dtype, t_init=time.time() - t0, degrees=[],
              fallback=fallback, far=far)


def _node_fields(spec, k, chans, lat, xi, per_bore=True):
    """Exit-plane lattice fields for one source node and one line: one array per bore, or
    their sum in a single array (one lattice-sized buffer instead of one per bore); with the
    entrance and exit mode coefficients of every bore."""
    out, coeffs = [], []
    for ch in chans:
        c0 = ch.spot_modes(spec["spot"]) if spec["spot"] else ch.entrance_modes(xi, spec["z_in"])
        cL, deg = ch.propagate(c0, spec["length"], spec["cheb_tol"])
        coeffs.append((c0, cL))
        _W["degrees"].append(deg)
        cz = ch.axis(spec["length"])
        lo = np.searchsorted(lat.x, cz - ch.disc.aw - 2 * lat.h)
        hi = np.searchsorted(lat.x, cz + ch.disc.aw + 2 * lat.h)
        xg, yg = lat.x[lo[0]:hi[0]], lat.x[lo[1]:hi[1]]
        block = ch.lab_field(c=cL, z=spec["length"], xg=xg, yg=yg, dr=spec["dr"]).astype(lat.dtype)
        if per_bore or not out:
            out.append(np.zeros((lat.n, lat.n), lat.dtype))
        out[-1][lo[1]:hi[1], lo[0]:hi[0]] += block
    return out, coeffs


def _run_nodes(args):
    """Accumulate W/I over a chunk of (node, weight) pairs; returns the partial sums."""
    nodes, weights, idx = args
    spec, lat, per_bore = _W["spec"], _W["lat"], _W["per_bore"]
    # fresh sums per task: a pool worker may serve several chunks
    accs = [Accumulator(smp, _W["nb"] if per_bore else 1) for smp in _W["samplers"]]
    _W["degrees"] = []
    z1 = spec["z_exit"]                                     # exit plane (lab z), not z_in + L
    disc, far = _W["disc"], _W["far"]
    nl = len(spec["lines"])
    extra = {"p_in": np.zeros(nl), "p_out": np.zeros(nl), "dump": {}, "fallback": _W["fallback"], "far_meta": far,
             "basis": {"aw": disc.aw, "zeros": disc.zp, "valid": disc.validp, "m": np.array(disc.ms)}}
    if far is not None:
        nt = far["thetas"].size
        extra.update(ring=np.zeros((nl, far["n_phi"])), radial=np.zeros((nl, nt)), map=np.zeros((nt, far["n_phi"])))
    for j, xi, w in zip(idx, nodes, weights):
        for li, (k, wl, delta, beta) in enumerate(spec["lines"]):
            chans = _W["chans"][k]
            fields, coeffs = _node_fields(spec, k, chans, lat, xi, per_bore)
            extra["p_in"][li] += w * sum(disc.norm(c0) for c0, _ in coeffs)
            extra["p_out"][li] += w * sum(disc.norm(cL) for _, cL in coeffs)
            if spec["dump"]:
                for bi, (c0, cL) in enumerate(coeffs):
                    extra["dump"][f"c_in_line{li}_node{j}_bore{bi}"] = c0
                    extra["dump"][f"c_out_line{li}_node{j}_bore{bi}"] = cL
            if far is not None:
                P = np.abs(far_field_amplitude(disc, coeffs[0][1], k, far["thetas"], far["n_phi"])) ** 2
                ring, radial = far_field_profiles(P, far["thetas"], k, far["ring"])
                extra["ring"][li] += w * ring
                extra["radial"][li] += w * radial
                extra["map"] += w * wl * P
            for s, g, smp, acc in zip(spec["screens"], _W["grids"], _W["samplers"], accs):
                d = s["z"] - z1
                cells, subs = [], []
                for E in fields:
                    Ed = lat.drift(E, k, d)
                    cells.append(lat.cells(Ed, g, lat.cell_filter(g)))
                    subs.append(lat.subsamples(Ed, g) if spec["observable"] == "coherent_cell" else None)
                acc.add(w * wl, cells, [None] * len(cells), subs)
    return [(a.I, a.W, a.I_ref, a.I_pixel, a.I_bore, a.G12) for a in accs], len(nodes), _W["t_init"], \
        (int(np.median(_W["degrees"])) if _W["degrees"] else None), int(_W["disc"].nmodes), extra


# ----------------------------------------------------------------- scene driver

def fb_capillary_scene(sim, wave, lines, rays_paths, log, out_dir=None):
    cfg = sim.cfg
    cap = cfg.capillary
    if wave["entrance_spot"]:
        sp = wave["entrance_spot"]
        nodes, weights = np.array([sp["center"]]), np.array([1.0])
        rule = (f"entrance spot: gaussian FWHM {sp['fwhm'] * 1e9:.0f} nm at ({sp['center'][0] * 1e6:.3f}, "
                f"{sp['center'][1] * 1e6:.3f}) um, tilt ({sp['tilt'][0] * 1e3:.3f}, {sp['tilt'][1] * 1e3:.3f}) mrad")
    else:
        nodes, weights, rule = _rule(wave, "capillary", cap.source, rays_paths)
    spec = build_spec(sim, wave, lines)
    labels = ["capillary"] + [f"capillary-s{i}" for i in range(1, len(cap.screens) + 1)]
    lam0 = float(sim.lam)
    nyq = spec["h"] <= spec["h_auto"] * (1.0 + 1e-12)
    log(f"  16 [capillary/fb]: {rule}; j_max {spec['jmax']} (theta_modal {spec['theta_modal']*1e3:.3f} mrad, "
        f"tilt {spec['tilt']*1e3:.3f} mrad), lattice h = {m_to_um(spec['h']):.4f} um (auto {m_to_um(spec['h_auto']):.4f}), "
        f"half {m_to_um(spec['half']):.1f} um, chebyshev"
        + (f", guided cut j^2 - m^2 < {spec['wall_cut']:.1f}^2" if spec["wall_cut"] else "")
        + f", wall {spec['wall']} ell = {spec['ells'][0][0]*1e9:.3f} + {spec['ells'][0][1]*1e9:.3f}i nm"
        + ("" if nyq else " — note: lattice coarser than the auto step"))
    t0 = time.time()
    jobs = int(wave["fb_jobs"] or 1)
    order = np.arange(len(nodes))
    chunks = [(nodes[i::jobs], weights[i::jobs], order[i::jobs]) for i in range(jobs)]
    chunks = [c for c in chunks if len(c[0])]
    if len(chunks) == 1:
        _init_worker(spec)
        parts = [_run_nodes(chunks[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(chunks), initializer=_init_worker, initargs=(spec,)) as pool:
            parts = pool.map(_run_nodes, chunks)
    grids = [ScreenGrid(_PixelCfg(s)) for s in spec["screens"]]
    lat = Lattice(spec["half"], spec["h"], edge0=spec["screens"][0]["center"][0] - spec["screens"][0]["edge_x"] / 2)
    bs = [lat.cell_plan(g)[0] for g in grids]
    samplers = [ScreenSampler(g, s["reference"] or (g.cxf, g.cyf), wave["observable"], bb)
                for g, s, bb in zip(grids, spec["screens"], bs)]
    nb = len(spec["bores"])
    per_bore = spec["per_bore"] if spec["per_bore"] is not None else nb <= 2
    accs = [Accumulator(smp, nb if per_bore else 1) for smp in samplers]
    if sum(p[1] for p in parts) != len(nodes):
        raise RuntimeError(f"wave_estimator (fb): {sum(p[1] for p in parts)} nodes accumulated, {len(nodes)} expected")
    for part, n_nodes, t_init, deg, _, _ in parts:
        for acc, (I, W, I_ref, I_pix, I_bore, G12) in zip(accs, part):
            acc.I += I; acc.W += W; acc.I_ref += I_ref
            if acc.I_pixel is not None and I_pix is not None:
                acc.I_pixel += I_pix
            if acc.I_bore is not None and I_bore is not None:
                acc.I_bore += I_bore
            if acc.G12 is not None and G12 is not None:
                acc.G12 += G12
    degrees = [p[3] for p in parts if p[3] is not None]
    extra = parts[0][5]
    for p in parts[1:]:
        for key, v in p[5].items():
            if key == "dump":
                extra["dump"].update(v)
            elif key not in ("fallback", "far_meta", "basis"):
                extra[key] = extra[key] + v
    fig = WallFigure(spec["figure"]["terms"], spec["figure"]["noise"], spec["length"])
    fallback = ({f"{float(HC_KEV_ANGSTROM) * k / (2.0e10 * math.pi):.6f}": n for k, n in sorted(extra["fallback"].items())}
                if spec["wall"] == "exact-roots" else None)
    files = []
    if out_dir is not None and spec["far"] is not None:
        files += _emit_far_field(out_dir, spec, lines, extra, wave)
    if out_dir is not None and spec["dump"]:
        files += _emit_dump(out_dir, spec, lines, extra, nodes, weights)
    transmission = [float(po / pi) if pi > 0.0 else None for pi, po in zip(extra["p_in"], extra["p_out"])]
    wall_label = {"dir-ell": "dir-ell", "dir-ell-real": "dir-ell,no-absorption", "exact-roots": "exact-jk"}[spec["wall"]]
    out = []
    a = spec["bores"][0]["radius"]
    for label, s, smp, acc in zip(labels, spec["screens"], samplers, accs):
        d = s["z"] - float(cap.z1)
        nf = {"entrance": a * a / (lam0 * spec["length"]),
              "source": None if spec["spot"] else a * a / (lam0 * spec["z_in"]),
              "exit": (a * a / (lam0 * d)) if d > 0 else None}
        bent = any(b["rbend"] for b in spec["bores"])
        meta = {"scene": label, "provider": "fb",
                "model_class": wall_label + (",parabolic-axis" if bent else "") + (",hadamard-figure" if fig.has_wall else "")
                + (",axis-path" if fig.has_axis else ""),
                "figure": spec["figure"], "figure_dz_m": spec["dz"] if (fig.z_dependent or fig.has_axis) else None,
                "exact_fallback_modes": fallback, "transmission": transmission, "far_field": spec["far"],
                "source_rule": rule, "n_nodes": int(len(nodes)),
                "lattice_h_m": spec["h"], "lattice_h_auto_m": spec["h_auto"], "lattice_half_m": spec["half"],
                "lattice_n": lat.n, "theta_modal_rad": spec["theta_modal"], "tilt_rad": spec["tilt"],
                "sampling": {"nyquist_ratio": 2.0 * spec["h"] * (spec["theta_modal"] + spec["tilt"]) / (2.0 * math.pi / max(k for k, *_ in lines))},
                "jmax": spec["jmax"], "wall_cut": spec["wall_cut"], "n_modes": parts[0][4],
                "propagator": "chebyshev" if bent else "diagonal",
                "chebyshev_degree_median": (int(np.median(degrees)) if degrees and bent else None),
                "wall": spec["wall"], "wall_offset_nm": [[e[0] * 1e9, e[1] * 1e9] for e in spec["ells"]],
                "aw_m": spec["aw"], "fresnel_numbers": nf, "distance_m": d, "bores": spec["bores"],
                "per_bore_maps": per_bore, "jobs": jobs, "worker_init_s": max(p[2] for p in parts)}
        res = _finish(acc, smp, meta, wave, time.time() - t0)
        wall_text = {"dir-ell": f"wall DIR-ℓ, ℓ = {spec['ells'][0][0]*1e9:.3f} + {spec['ells'][0][1]*1e9:.3f}i nm, damping × Fresnel loss at θ_w",
                     "dir-ell-real": f"wall DIR-ℓ, ℓ = {spec['ells'][0][0]*1e9:.3f} nm (absorption off)",
                     "exact-roots": "wall exact roots: J_m / K_m characteristic equation per mode and line"
                              + (f" (DIR-ℓ fallback for {sum(fallback.values())} mode(s))" if fallback and sum(fallback.values()) else "")}
        lines_r = [
            f"- provider fb: j_max = {spec['jmax']} (θ_modal = {spec['theta_modal']*1e3:.3f} mrad, tilt {spec['tilt']*1e3:.3f} mrad), "
            + (f"chebyshev degree ≈ {int(np.median(degrees)) if degrees else '?'}" if (bent or fig.has_wall or fig.has_axis)
               else "diagonal propagator (straight bore)")
            + (f", guided-mode cut j² − m² < {spec['wall_cut']:.1f}²" if spec["wall_cut"] else "")
            + f"; {wall_text[spec['wall']]}",
            f"- lattice h = {m_to_um(spec['h']):.4f} µm (auto {m_to_um(spec['h_auto']):.4f}), half-width {m_to_um(spec['half']):.1f} µm, "
            f"{lat.n}² points; cells = exact Fourier box of the band-limited field; screens by angular-spectrum drift",
            f"- N_F: entrance {nf['entrance']:.2f}, source " + (f"{nf['source']:.2f}" if nf["source"] else "— (entrance spot)") + ", exit "
            + (f"{nf['exit']:.3f}" if nf["exit"] else "— (screen on the exit plane)") + "; ray optics is not used anywhere in this stage"]
        if fig.has_wall or fig.has_axis:
            lines_r.append(f"- figure: {len(spec['figure']['terms'])} term(s)" + (", random field" if spec["figure"]["noise"] else "")
                           + (" (Hadamard wall, first order)" if fig.has_wall else "") + (" (axis path, co-moving frame)" if fig.has_axis else "")
                           + (f", z step {spec['dz'] * 1e6:.1f} µm" if (fig.z_dependent or fig.has_axis) else ""))
        if label == "capillary":
            lines_r.append("- modal transmission P_exit / P_entrance: " + ", ".join(f"{t:.4f}" for t in transmission if t is not None))
        res["report"] = _report_lines(label, res, wave, lines_r)
        out.append((label, res))
    out[0][1]["fb_files"] = files
    return out


def _emit_far_field(out_dir, spec, lines, extra, wave):
    """far-field.json (per line and weighted ring statistics, modal transmission), far-field-fb.npz (ring and
    radial profiles per line, weighted map) and the weighted ring profile as SVG."""
    import json
    from ..... import render
    far = extra["far_meta"]
    th, nphi = far["thetas"], far["n_phi"]
    phi = 2.0 * math.pi * np.arange(nphi) / nphi
    rows = []
    wsum = np.zeros(nphi)
    for li, (k, wl, *_) in enumerate(lines):
        r = ring_summary(extra["ring"][li])
        r.update(e_kev=float(HC_KEV_ANGSTROM) * k / (2.0e10 * math.pi), weight=float(wl),
                 transmission=float(extra["p_out"][li] / extra["p_in"][li]) if extra["p_in"][li] > 0.0 else None)
        rows.append(r)
        wsum += wl * extra["ring"][li]
    summary = {"theta_max_rad": far["thetas"][-1], "n_theta": int(th.size), "n_phi": nphi, "ring_rad": far["ring"],
               "lines": rows, "weighted": ring_summary(wsum),
               "convention": "A(theta, phi) = int E exp(-i k theta.x) d^2x from the exit modes (Lommel form); ring = "
                             "dP/dphi = (k/2pi)^2 int_ring |A|^2 theta dtheta; phi measured in the bore frame; "
                             "lines and nodes summed by intensity"}
    with open(os.path.join(out_dir, "far-field.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    np.savez(os.path.join(out_dir, "far-field-fb.npz"), theta=th, phi=phi, ring=extra["ring"], radial=extra["radial"],
             map=extra["map"], k=np.array([k for k, *_ in lines]), weight=np.array([wl for _, wl, *_ in lines]),
             p_in=extra["p_in"], p_out=extra["p_out"])
    mean = float(wsum.mean()) or 1.0
    name = "16-far-field.svg"
    render.save(os.path.join(out_dir, name), render.line_chart(
        [{"xs": [float(v) for v in np.degrees(phi)], "ys": [float(v) / mean for v in wsum], "label": "dP/dφ / mean"}],
        "far-field ring profile (weighted over lines)", "φ, deg", "dP/dφ / mean",
        f"ring {far['ring'][0] * 1e3:.3f}–{far['ring'][1] * 1e3:.3f} mrad, {nphi} azimuths", w=760))
    return ["far-field.json", "far-field-fb.npz", name]


def _emit_dump(out_dir, spec, lines, extra, nodes, weights):
    """field-capillary.npz: entrance and exit mode coefficients per line / node / bore and the basis
    (padded zeros j_mn over |m|, validity, a_w, signed m order); fields follow from sum c J_|m|(j r / a_w) e^{i m phi}."""
    b = extra["basis"]
    np.savez(os.path.join(out_dir, "field-capillary.npz"), k=np.array([k for k, *_ in lines]),
             line_weights=np.array([wl for _, wl, *_ in lines]), nodes=np.asarray(nodes, float),
             node_weights=np.asarray(weights, float), aw=np.array(b["aw"]), zeros=b["zeros"], valid=b["valid"],
             m=b["m"], length=np.array(spec["length"]), z_exit=np.array(spec["z_exit"]), **extra["dump"])
    return ["field-capillary.npz"]

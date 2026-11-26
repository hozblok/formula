"""Stage 20: one coherent wave packet along one circular bore, straight or bent (`wave_packet`).

The entrance spot of `wave_estimator.entrance_spot` is projected on the Fourier-Bessel basis of provider fb
(wave.fb: Dirichlet disc a + Re ell, DIR-ell or exact roots, a bend as the potential k xi / R in the frame
co-moving with the axis, Chebyshev propagator) and carried to a ladder of z inside the bore. At every z: the
power P(z) in the basis, the transmission T = P(z) / P(0), the return fidelity F(z) = |<E(0), E(z)>|^2 /
(P(0) P(z)) against the entrance field, the centroid and the rms width of the intensity along xi (toward the
bend centre) and eta; the intensity along the diameter through the spot and the azimuthal power density as
carpets over z; cross-section maps with the mode coefficients at the requested slices; the far field of the
modes at the requested z, in the frame of that cross-section. Intensities of several spectral lines add with
their weights; P, T, F, the moments, the slices and the coefficients are per line. No rays, no screens.
Output: stage20/packet.npz, meta.json, 20-return.svg."""

import json
import math
import os
import shutil
import time

import numpy as np

from ... import __version__
from ...physical_constants import HC_KEV_ANGSTROM
from .. import render
from ..shared.units import m_to_um
from .wave.output import _json_default
from .wave.fb import (Disc, FBChannel, WallFigure, exact_kappa, far_field_amplitude, far_field_profiles,
                      fb_jmax_rule, mode_damping, ring_summary, wall_offset)

RESULT_DIR = "stage20"
NPZ = "packet.npz"
SVG = "20-return.svg"


# ----------------------------------------------------------------- contract

def preflight_packet_output(out_dir):
    """Fail before any work when the result directory already exists (no clobber)."""
    final = os.path.join(out_dir, RESULT_DIR)
    if os.path.lexists(final):
        raise ValueError(f"{final}: publication conflict; remove it manually")


def preflight_packet_inputs(sim, wave, packet):
    """Stage-20 input contract: one circular bore, a coherent entrance spot, provider fb, a ladder inside the bore."""
    cap = sim.cfg.capillary
    if cap is None:
        raise ValueError("wave_packet: stage 20 needs a configured capillary scene")
    if len(cap.bores) != 1:
        raise ValueError(f"wave_packet: stage 20 takes one bore, the scene has {len(cap.bores)}")
    if cap.bores[0].get("kind") not in ("cylinder", "torus"):
        raise ValueError("wave_packet: stage 20 takes a circular bore (radius, optional bend); "
                         f"got kind {cap.bores[0].get('kind')!r}")
    if wave["provider"] not in ("auto", "fb"):
        raise ValueError(f"wave_packet: stage 20 runs on provider fb, wave_estimator.provider is {wave['provider']!r}")
    if wave["fb_wall"] == "effective-radius":
        raise ValueError("wave_packet: fb_wall effective-radius is for funnel bores only")
    if wave["entrance_spot"] is None:
        raise ValueError("wave_packet: stage 20 needs wave_estimator.entrance_spot (a coherent spot on the entrance face)")
    if packet["far_field_at"] and wave["far_field"] is None:
        raise ValueError("wave_packet.far_field_at needs wave_estimator.far_field {theta_max, n_theta, n_phi, ring}")
    if not any(float(ln.weight) > 0.0 for ln in sim.lines):
        raise ValueError("wave_packet: spectral lines need a positive weight")
    length = float(cap.z1) - float(cap.z0)
    z_max = packet["z_max"] if packet["z_max"] is not None else length
    if z_max > length * (1.0 + 1e-12):
        raise ValueError(f"wave_packet.z_max {z_max} lies beyond the bore length {length}")
    for key in ("slices", "far_field_at"):
        bad = [z for z in packet[key] if z < 0.0 or z > z_max * (1.0 + 1e-12)]
        if bad:
            raise ValueError(f"wave_packet.{key}: {bad} outside [0, z_max = {z_max}]")


def build_packet_spec(sim, wave, packet, lines):
    """Plain floats only: the bore, the basis, the ladder, the carpets, the slices and the far field."""
    cfg = sim.cfg
    cap = cfg.capillary
    b = cap.bores[0]
    length = float(cap.z1) - float(cap.z0)
    bend = b.get("bend")
    bore = {"center": (float(b["center"][0]), float(b["center"][1])), "radius": float(b["radius"]),
            "rbend": None if bend is None else float(bend["radius"]),
            "toward": None if bend is None else (float(bend["toward"][0]), float(bend["toward"][1]))}
    k_max = max(k for k, *_ in lines)
    lam_min = 2.0 * math.pi / k_max
    ells = [wall_offset(k, delta, beta) for k, _, delta, beta in lines]
    aw = bore["radius"] + max(e.real for e in ells)
    theta_cut = float(wave["fb_theta_cut"])
    jmax = int(wave["fb_jmax"]) if wave["fb_jmax"] else fb_jmax_rule(k=k_max, aw=aw, theta_cut=theta_cut)
    wall_cut = max(k * aw * math.sqrt(2.0 * delta) for k, _, delta, _ in lines) if wave["fb_wall_cut"] else None
    figure = {"terms": wave["fb_figure"] or [], "noise": wave["fb_figure_noise"]}
    scale = WallFigure(figure["terms"], figure["noise"], length).min_scale()
    dz = min(float(wave["fb_dz"]), scale / 16.0) if scale else float(wave["fb_dz"])
    z_max = float(packet["z_max"]) if packet["z_max"] is not None else length
    z_step = float(packet["z_step"]) if packet["z_step"] is not None else z_max / 200.0
    n = max(1, int(math.ceil(z_max / z_step - 1e-9)))
    ladder = list(np.linspace(0.0, z_max, n + 1))
    for z in list(packet["slices"]) + list(packet["far_field_at"]):
        if min(abs(z - v) for v in ladder) > 1e-12 * max(z_max, 1e-9):
            ladder.append(float(z))
    ladder = sorted(ladder)
    far = None
    ff = wave["far_field"]
    if ff is not None and packet["far_field_at"]:
        n_theta = ff.get("n_theta") or int(math.ceil(ff["theta_max"] / (lam_min / (8.0 * aw)))) + 1
        far = {"theta_max": ff["theta_max"], "n_theta": n_theta, "n_phi": ff.get("n_phi"),
               "ring": ff.get("ring") or [0.0, ff["theta_max"]], "z": sorted(float(z) for z in packet["far_field_at"])}
    return {"length": length, "bore": bore, "aw": aw, "jmax": jmax, "wall_cut": wall_cut, "theta_cut": theta_cut,
            "theta_modal": jmax / (k_max * aw), "lines": [(float(k), float(wl), float(dl), float(bt)) for k, wl, dl, bt in lines],
            "ells": [(e.real, e.imag) for e in ells], "cheb_tol": float(wave["fb_chebyshev_tol"]), "wall": wave["fb_wall"],
            "spot": wave["entrance_spot"], "figure": figure, "dz": dz, "ladder": ladder, "z_step": z_max / n, "z_max": z_max,
            "slices": sorted(float(z) for z in packet["slices"]), "slice_pixel": float(packet["slice_pixel"]),
            "carpet_pixel": float(packet["carpet_pixel"]), "far": far, "dephasing_from": float(packet["dephasing_from"]),
            "loss": packet["loss"]}


# ----------------------------------------------------------------- synthesis in the bore frame

def frame_field(disc, c, xi, eta, dr):
    """Field sum_mn c_mn J_|m|(j_mn r / a_w) e^{i m phi} at the bore-frame points (xi, eta) [m], zero outside the disc;
    c: mode coefficients (2M+1, nmax); dr: step of the fine radial table [m]."""
    rf, tab = disc.radial_table(c, dr)
    r = np.hypot(xi, eta)
    inside = r < disc.aw
    ri, phi = r[inside], np.arctan2(eta[inside], xi[inside])
    E = np.zeros(ri.size, complex)
    for i, m in enumerate(disc.ms):
        E += (np.interp(ri, rf, tab[i].real) + 1j * np.interp(ri, rf, tab[i].imag)) * np.exp(1j * m * phi)
    out = np.zeros(np.shape(xi), complex)
    out[inside] = E
    return out


def diameter_field(disc, c, phi_s, dr):
    """Field on the diameter through the azimuth phi_s: (u, E) with u the signed distance from the centre, positive
    toward phi_s, on the fine radial table of step dr [m]."""
    rf, tab = disc.radial_table(c, dr)
    ms = np.array(disc.ms)
    plus = np.exp(1j * ms * phi_s) @ tab
    minus = np.exp(1j * ms * (phi_s + math.pi)) @ tab
    return np.concatenate([-rf[:0:-1], rf]), np.concatenate([minus[:0:-1], plus])


def moments(disc, G):
    """Power and the intensity centroid / rms along xi and eta from the field G on the polar quadrature grid."""
    I = np.abs(G) ** 2
    w = (disc.wr * disc.r)[:, None] * (2.0 * math.pi / disc.nphi)
    P = float(np.sum(w * I))
    if P <= 0.0:
        return P, (math.nan,) * 4
    xi = disc.r[:, None] * np.cos(disc.phi)[None, :]
    eta = disc.r[:, None] * np.sin(disc.phi)[None, :]
    mx, me = float(np.sum(w * I * xi)) / P, float(np.sum(w * I * eta)) / P
    sx = math.sqrt(max(float(np.sum(w * I * (xi - mx) ** 2)) / P, 0.0))
    se = math.sqrt(max(float(np.sum(w * I * (eta - me) ** 2)) / P, 0.0))
    return P, (mx, me, sx, se)


def return_peaks(z, F, n=3):
    """Local maxima of F(z) after its first local minimum, strongest first: [(z, F, index)]."""
    F = np.asarray(F, float)
    if F.size < 3:
        return []
    mins = [i for i in range(1, F.size - 1) if F[i] <= F[i - 1] and F[i] < F[i + 1]]
    if not mins:
        return []
    i0 = mins[0]
    peaks = [i for i in range(i0 + 1, F.size - 1) if F[i] >= F[i - 1] and F[i] > F[i + 1]]
    if F.size - 1 > i0 and F[-1] > F[-2]:
        peaks.append(F.size - 1)
    peaks.sort(key=lambda i: -F[i])
    return [(float(z[i]), float(F[i]), int(i)) for i in peaks[:n]]


# ----------------------------------------------------------------- driver

def _channel(spec, disc, figure, k, delta, beta, fallback):
    """Channel of one line: wall dir-ell with the loss as the boundary operator (wall_loss = i Im ell) or as the diagonal
    damping of stage 16; dir-ell-real without loss; exact roots as the complex diagonal."""
    b = spec["bore"]
    if spec["wall"] == "dir-ell" and spec["loss"] == "boundary":
        ell = wall_offset(k, delta, beta)
        return FBChannel(c0=b["center"], t=b["toward"], rbend=b["rbend"], disc=disc, k=k, figure=figure, dz=spec["dz"],
                         wall_loss=1j * ell.imag)
    if spec["wall"] == "exact-roots":
        kap, ok = exact_kappa(disc, a=b["radius"], k=k, delta=delta, beta=beta)
        dirl = disc.kap2 / (2.0 * k) - 1j * mode_damping(disc, k, delta, beta)
        h0 = np.where(disc.valid, np.where(ok[disc.am], (kap * kap)[disc.am] / (2.0 * k), dirl), 0.0)
        fallback[k] = int(np.sum(disc.validp & ~ok))
        return FBChannel(c0=b["center"], t=b["toward"], rbend=b["rbend"], disc=disc, k=k, h0=h0, figure=figure, dz=spec["dz"])
    gamma = mode_damping(disc, k, delta, beta) if spec["wall"] == "dir-ell" else None
    return FBChannel(c0=b["center"], t=b["toward"], rbend=b["rbend"], disc=disc, k=k, gamma=gamma, figure=figure, dz=spec["dz"])


def run_packet_stage(sim, out_dir, wave, packet, log=print):
    """Stage 20 on the capillary scene; publishes out_dir/stage20 atomically."""
    preflight_packet_output(out_dir)
    preflight_packet_inputs(sim, wave, packet)
    partial = os.path.join(out_dir, RESULT_DIR + ".partial")
    if os.path.lexists(partial):
        raise ValueError(f"{partial}: stale partial output; remove it manually")
    os.makedirs(partial)
    try:
        res = _run(sim, partial, wave, packet, log)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    os.rename(partial, os.path.join(out_dir, RESULT_DIR))
    return {"files": [RESULT_DIR + "/" + f for f in res["files"]], "report": res["report"],
            "results": {"capillary": res["results"]}, "seconds": res["seconds"]}


def _run(sim, partial, wave, packet, log):
    cfg = sim.cfg
    t0 = time.time()
    lines = []
    for ln in sim.lines:
        if float(ln.weight) > 0.0:
            delta, beta = cfg.material.optical(ln.e_kev) if sim.per_line else (sim.delta_f, sim.beta_f)
            lines.append((float(ln.k), float(ln.weight), delta, beta))
    spec = build_packet_spec(sim, wave, packet, lines)
    disc = Disc(aw=spec["aw"], jmax=spec["jmax"], wall_cut=spec["wall_cut"])
    disc.fine_table(spec["carpet_pixel"], dtype=np.float32)
    figure = WallFigure(spec["figure"]["terms"], spec["figure"]["noise"], spec["length"])
    bent = spec["bore"]["rbend"] is not None
    sp = spec["spot"]
    b = spec["bore"]
    # the azimuth of the spot centre in the bore frame (xi toward the bend centre; x when straight)
    t = np.array(b["toward"]) / np.linalg.norm(b["toward"]) if b["toward"] else np.array([1.0, 0.0])
    d = np.array(sp["center"]) - np.array(b["center"])
    xi_s, eta_s = float(d @ t), float(d @ np.array([-t[1], t[0]]))
    phi_s = math.atan2(eta_s, xi_s) if math.hypot(xi_s, eta_s) > 0.0 else 0.0
    log(f"  20 [capillary/fb]: spot FWHM {sp['fwhm'] * 1e9:.0f} nm at ({sp['center'][0] * 1e6:.3f}, {sp['center'][1] * 1e6:.3f}) um "
        f"(bore frame xi {xi_s * 1e6:.3f}, eta {eta_s * 1e6:.3f} um), tilt ({sp['tilt'][0] * 1e3:.3f}, {sp['tilt'][1] * 1e3:.3f}) mrad; "
        f"bore a = {m_to_um(b['radius']):.3f} um" + (f", bend R = {b['rbend']:g} m toward ({t[0]:.3f}, {t[1]:.3f})" if bent else ", straight")
        + f"; j_max {spec['jmax']} ({disc.nmodes} modes, |m| <= {disc.M}, theta_cut {spec['theta_cut'] * 1e3:.3f} mrad), wall {spec['wall']}"
        + (f", guided cut j^2 - m^2 < {spec['wall_cut']:.1f}^2" if spec["wall_cut"] else "")
        + f"; ladder {len(spec['ladder'])} points to {spec['z_max'] * 1e3:.3f} mm, step {spec['z_step'] * 1e6:.1f} um")
    ladder = np.array(spec["ladder"])
    nz, nl = ladder.size, len(lines)
    slices, far = spec["slices"], spec["far"]
    is_slice = [any(abs(z - s) <= 1e-12 * max(spec["z_max"], 1e-9) for s in slices) for z in ladder]
    is_far = [far is not None and any(abs(z - s) <= 1e-12 * max(spec["z_max"], 1e-9) for s in far["z"]) for z in ladder]
    u_line, _ = diameter_field(disc, disc.empty(), phi_s, spec["carpet_pixel"])
    P = np.zeros((nl, nz)); F = np.zeros((nl, nz)); th2 = np.zeros((nl, nz)); mom = np.full((nl, nz, 4), np.nan)
    carpet = np.zeros((nz, u_line.size)); azimuth = np.zeros((nz, disc.nphi))
    xs = np.arange(-disc.aw, disc.aw + spec["slice_pixel"] / 2, spec["slice_pixel"])
    XI, ETA = np.meshgrid(xs, xs)
    slice_E = np.zeros((nl, len(slices), xs.size, xs.size), complex) if slices else None
    coef = np.zeros((nl, len(slices), 2 * disc.M + 1, disc.nmax), complex) if slices else None
    c_in = np.zeros((nl, 2 * disc.M + 1, disc.nmax), complex)
    if far is not None:
        thetas = np.linspace(0.0, far["theta_max"], far["n_theta"])
        nphi_f = far["n_phi"] or 1 << int(math.ceil(math.log2(max(4 * disc.M + 4, 256))))
        ff_map = np.zeros((len(far["z"]), thetas.size, nphi_f)); ff_ring = np.zeros((len(far["z"]), nphi_f))
        ff_radial = np.zeros((len(far["z"]), thetas.size))
    fallback, degrees = {}, []
    for li, (k, wl, delta, beta) in enumerate(lines):
        ch = _channel(spec, disc, figure, k, delta, beta, fallback)
        c0 = ch.spot_modes(sp)
        c_in[li] = c0
        P0 = disc.norm(c0)
        c, z_prev, si, fi = c0, 0.0, 0, 0
        for iz, z in enumerate(ladder):
            if z > z_prev:
                c, deg = ch.propagate(c, z - z_prev, spec["cheb_tol"], z0=z_prev)
                degrees.append(deg)
                z_prev = z
            Pz = disc.norm(c)
            P[li, iz] = Pz
            F[li, iz] = abs(np.sum(np.conj(c0) * c * disc.mnorm)) ** 2 / (P0 * Pz) if Pz > 0.0 else 0.0
            th2[li, iz] = float(np.sum(np.abs(c) ** 2 * disc.kap2 * disc.mnorm)) / (Pz * k * k) if Pz > 0.0 else 0.0
            G = disc.to_grid(c)
            _, mom[li, iz] = moments(disc, G)
            azimuth[iz] += wl * np.sum((disc.wr * disc.r)[:, None] * np.abs(G) ** 2, axis=0)
            _, E_line = diameter_field(disc, c, phi_s, spec["carpet_pixel"])
            carpet[iz] += wl * np.abs(E_line) ** 2
            if is_slice[iz]:
                slice_E[li, si] = frame_field(disc, c, XI, ETA, spec["carpet_pixel"])
                coef[li, si] = c
                si += 1
            if is_far[iz]:
                A2 = np.abs(far_field_amplitude(disc, c, k, thetas, nphi_f)) ** 2
                ring, radial = far_field_profiles(A2, thetas, k, far["ring"])
                ff_map[fi] += wl * A2; ff_ring[fi] += wl * ring; ff_radial[fi] += wl * radial
                fi += 1
    T = P / P[:, :1]
    i_from = int(np.searchsorted(ladder, spec["dephasing_from"] * spec["z_max"] * (1.0 - 1e-12)))
    summary = []
    for li, (k, wl, *_) in enumerate(lines):
        peaks = return_peaks(ladder, F[li])
        mins = [i for i in range(1, nz - 1) if F[li, i] <= F[li, i - 1] and F[li, i] < F[li, i + 1]]
        summary.append({"e_kev": float(HC_KEV_ANGSTROM) * k / (2.0e10 * math.pi), "weight": wl, "power_in_basis": float(P[li, 0]),
                        "transmission_end": float(T[li, -1]),
                        "first_minimum": None if not mins else {"z_m": float(ladder[mins[0]]), "F": float(F[li, mins[0]])},
                        "return_peaks": [{"z_m": z, "F": f, "T": float(T[li, i])} for z, f, i in peaks],
                        "dephased_mean_F": float(np.mean(F[li, i_from:])) if i_from < nz else None,
                        "dephased_from_m": float(ladder[i_from]) if i_from < nz else None})
    arrays = {"z": ladder, "P": P, "T": T, "F": F, "theta_rms": np.sqrt(th2), "xi_mean": mom[:, :, 0], "eta_mean": mom[:, :, 1],
              "xi_rms": mom[:, :, 2], "eta_rms": mom[:, :, 3], "carpet_u": u_line, "carpet_I": carpet, "phi": disc.phi, "azimuth_I": azimuth,
              "k": np.array([k for k, *_ in lines]), "line_weights": np.array([wl for _, wl, *_ in lines]), "c_in": c_in,
              "aw": np.array(disc.aw), "zeros": disc.zp, "valid": disc.validp, "m": np.array(disc.ms), "phi_spot": np.array(phi_s),
              "slice_z": np.array(slices), "slice_x": xs, "slice_y": xs}
    if slices:
        arrays.update(slice_E=slice_E, coef=coef)
    if far is not None:
        arrays.update(ff_z=np.array(far["z"]), ff_theta=thetas, ff_phi=2.0 * math.pi * np.arange(nphi_f) / nphi_f, ff_map=ff_map,
                      ff_ring=ff_ring, ff_radial=ff_radial)
    np.savez(os.path.join(partial, NPZ), **arrays)
    files = [NPZ]
    wall_label = {"dir-ell": "dir-ell," + ("boundary-loss" if spec["loss"] == "boundary" else "diagonal-loss"),
                  "dir-ell-real": "dir-ell,no-absorption", "exact-roots": "exact-jk"}[spec["wall"]]
    seconds = time.time() - t0
    meta = {"scene": "capillary", "stage": 20, "provider": "fb",
            "model_class": wall_label + (",parabolic-axis" if bent else "") + (",hadamard-figure" if figure.has_wall else "")
            + (",axis-path" if figure.has_axis else ""),
            "bore": b, "length_m": spec["length"], "entrance_spot": sp, "spot_frame": {"xi_m": xi_s, "eta_m": eta_s, "phi_rad": phi_s},
            "basis": {"aw_m": disc.aw, "jmax": spec["jmax"], "wall_cut": spec["wall_cut"], "n_modes": int(disc.nmodes), "m_max": int(disc.M),
                      "theta_cut_rad": spec["theta_cut"], "theta_modal_rad": spec["theta_modal"], "nr": int(disc.nr), "nphi": int(disc.nphi)},
            "wall": spec["wall"], "loss": spec["loss"] if spec["wall"] == "dir-ell" else ("none" if spec["wall"] == "dir-ell-real" else "diagonal"),
            "wall_offset_nm": [[e[0] * 1e9, e[1] * 1e9] for e in spec["ells"]],
            "exact_fallback_modes": ({f"{float(HC_KEV_ANGSTROM) * k / (2.0e10 * math.pi):.6f}": n for k, n in sorted(fallback.items())}
                                     if spec["wall"] == "exact-roots" else None),
            "propagator": "chebyshev" if degrees else "diagonal",
            "chebyshev_degree_median": int(np.median(degrees)) if degrees else None,
            "chebyshev_tol": spec["cheb_tol"], "figure": spec["figure"],
            "figure_dz_m": spec["dz"] if (figure.z_dependent or figure.has_axis) else None,
            "ladder": {"n": int(nz), "z_step_m": spec["z_step"], "z_max_m": spec["z_max"]},
            "carpets": {"diameter_pixel_m": spec["carpet_pixel"], "diameter_points": int(u_line.size), "azimuths": int(disc.nphi)},
            "slices_m": slices, "slice_pixel_m": spec["slice_pixel"], "far_field": far,
            "far_field_rings": None if far is None else [dict(ring_summary(ff_ring[i]), z_m=far["z"][i]) for i in range(len(far["z"]))],
            "lines": summary, "wave_estimator": dict(wave), "wave_packet": dict(packet),
            "conventions": {"frame": "bore frame (xi, eta): xi along the bend `toward` (toward the centre of curvature), eta = z x xi; "
                                     "a straight bore uses the lab x, y; the frame co-moves with the parabolic axis c0 + t z^2 / (2R)",
                            "F": "|<E(0), E(z)>|^2 / (P(0) P(z)) over the mode coefficients (orthogonal basis)",
                            "T": "P(z) / P(0), P in the guided basis",
                            "loss": "boundary: the DIR-ell loss as the first-order boundary operator of the complex wall offset "
                                    "(non-diagonal in the disc modes); diagonal: the per-mode damping of stage 16",
                            "carpet_I": "intensity on the diameter through the spot centre; carpet_u signed from the bore centre, "
                                        "positive toward the spot; lines summed with their weights",
                            "azimuth_I": "dP/dphi = int |E|^2 r dr over the whole radius at the azimuths phi; phi counted from xi",
                            "moments": "intensity centroid and rms along xi and eta, per line",
                            "theta_rms": "rms far-field angle sqrt(<kappa^2>) / k from the mode coefficients, per line",
                            "slice_E": "complex field on (slice_x, slice_y) = (xi, eta) at slice_z, per line; "
                                       "exp(-i omega t), carrier e^{+ikz} dropped, no lab tilt phase",
                            "coef": "mode coefficients at slice_z per line; field = sum c J_|m|(j r / a_w) e^{i m phi}",
                            "far_field": "A(theta, phi) = int E exp(-i k theta.x) d^2x of the modes at ff_z (Lommel form), in the frame "
                                         "of that cross-section; ring = dP/dphi over the band; lines summed by intensity"},
            "status": "unverified", "seconds": seconds, "capsysred_version": __version__}
    with open(os.path.join(partial, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    files.append("meta.json")
    series = []
    for li, s in enumerate(summary):
        lab = f"{s['e_kev']:.4g} keV" if nl > 1 else ""
        series.append({"xs": [float(v) * 1e3 for v in ladder], "ys": [float(v) for v in F[li]], "label": ("F " + lab).strip()})
        series.append({"xs": [float(v) * 1e3 for v in ladder], "ys": [float(v) for v in T[li]], "label": ("T " + lab).strip(), "dash": "5,3"})
    render.save(os.path.join(partial, SVG), render.line_chart(
        series, "return fidelity and transmission along the bore", "z, mm", "F, T",
        f"spot {sp['fwhm'] * 1e9:.0f} nm, {'bend R = %g m' % b['rbend'] if bent else 'straight bore'}, {disc.nmodes} modes", w=760))
    files.append(SVG)
    for f in files:
        log(f"  → {RESULT_DIR}/{f}")
    report = [f"## Stage 20 — wave packet along the bore [capillary]",
              f"- provider fb ({meta['model_class']}); bore a = {m_to_um(b['radius']):.3f} µm"
              + (f", bend R = {b['rbend']:g} m toward ({t[0]:.3f}, {t[1]:.3f})" if bent else ", straight")
              + f"; spot FWHM {sp['fwhm'] * 1e9:.0f} nm at xi {xi_s * 1e6:.3f}, eta {eta_s * 1e6:.3f} µm, "
              f"tilt ({sp['tilt'][0] * 1e3:.3f}, {sp['tilt'][1] * 1e3:.3f}) mrad",
              f"- basis j_max = {spec['jmax']} ({disc.nmodes} modes, |m| ≤ {disc.M}, θ_cut {spec['theta_cut'] * 1e3:.3f} mrad)"
              + (f", guided cut j² − m² < {spec['wall_cut']:.1f}²" if spec["wall_cut"] else "")
              + f"; wall {spec['wall']} (loss {meta['loss']}), ℓ = {spec['ells'][0][0] * 1e9:.3f} + {spec['ells'][0][1] * 1e9:.3f}i nm; "
              + (f"chebyshev propagator, degree ≈ {meta['chebyshev_degree_median']}" if meta["propagator"] == "chebyshev" else "diagonal propagator"),
              f"- ladder {nz} points to {spec['z_max'] * 1e3:.3f} mm (step {spec['z_step'] * 1e6:.1f} µm); diameter carpet "
              f"{u_line.size} points of {spec['carpet_pixel'] * 1e9:.1f} nm, {disc.nphi} azimuths"
              + (f"; slices at {', '.join(f'{z * 1e3:.3f}' for z in slices)} mm ({xs.size}² pixels of {spec['slice_pixel'] * 1e9:.1f} nm)" if slices else "")
              + (f"; far field at {', '.join(f'{z * 1e3:.3f}' for z in far['z'])} mm" if far is not None else "")]
    for s in summary:
        pk = "; ".join(f"F = {p['F']:.4f} at {p['z_m'] * 1e3:.3f} mm (T {p['T']:.4f})" for p in s["return_peaks"]) or "none found"
        fm = f"first minimum F = {s['first_minimum']['F']:.4f} at {s['first_minimum']['z_m'] * 1e3:.3f} mm; " if s["first_minimum"] else ""
        report.append(f"- {s['e_kev']:.4f} keV: power in the basis {s['power_in_basis']:.4f}; {fm}return peaks {pk}; "
                      f"T(z_max) = {s['transmission_end']:.4f}"
                      + (f"; dephased mean F = {s['dephased_mean_F']:.4f} over {s['dephased_from_m'] * 1e3:.2f}–{spec['z_max'] * 1e3:.2f} mm"
                         if s["dephased_mean_F"] is not None else ""))
    report.append("- status: unverified (basis and step convergence are not run inside this stage); no rays are used anywhere in this stage")
    report.append(f"- time: {seconds:.1f} s")
    return {"files": files, "report": report, "results": {"meta": meta, "F": F, "T": T, "z": ladder, "summary": summary}, "seconds": seconds}

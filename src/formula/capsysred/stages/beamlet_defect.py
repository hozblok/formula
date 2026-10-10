"""Stage 11 defect indicator: how far the beamlet sum of a scene is from a solution
of its paraxial Dirichlet problem, per source mode, without a wave reference.

Every beamlet is an exact free-space paraxial solution, so the sum fails only in
three computable places:
  D0  entrance representation. A uniform fan of beams through the entrance disc
      equals the point-source field convolved with the launch kernel
      K(d) = exp(i k d.(G0 - I/z_in) d / 2), G0 the beam curvature tensor at the
      entrance; D0 = ||1_disc - 1_disc * K / int K|| / ||1_disc|| (same for every mode).
  G   wall trace. Every ray segment continued as a full beam (the incident beam past
      its hit, the reflected beam before it) is summed on the bore wall r = a + Re(ell)
      at sections in z; G is the RMS amplitude of that trace over wall x z relative to
      the RMS amplitude of the entrance field (the exact field has zero trace).
  T   exit tails. The deposit keeps the final beam of a ray only; the beams of the
      earlier segments still reach the exit plane inside the bore.
indicator = D0 + G + T per mode; a launch scan (w0, waist_z pairs) picks the launch
with the smallest RMS indicator over modes.  No certified bound is claimed: lifting the
wall trace into a boundary layer gives a residual above the field norm even for the
exact Dirichlet field at grazing incidence, so the defects are reported as indicators.

Usage: python -m formula.capsysred.stages.beamlet_defect SCENE.yaml ARCHIVE -o OUT
         [--scan w0_um:waist_z,...] [--rays N] [--sections S] [--modes M] [--jobs J]
Rows: OUT/stage11-defect/modes.jsonl (one per launch and mode), defect.json (summary).
"""

import argparse
import cmath
import concurrent.futures
import json
import math
import os
import time

import numpy as np

from .. import rays_v3
from ..gamma import bounce_lenses, reflect
from ..surfaces import CapillaryBundle
from .altcoh import FloatLineAmplitudes
from .beamlet import _launch

RESULT_DIR = "stage11-defect"
DEFAULTS = {"rays": 50000, "sections": 60, "ring_step": 0.5e-6, "window": 4.0,
            "modes": None, "jobs": 1, "scan": None}


def options(cfg):
    raw = dict(cfg.raw.get("beamlet_defect") or {})
    unknown = raw.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"beamlet_defect has unknown keys {sorted(unknown)}")
    opt = {**DEFAULTS, **raw}
    for key in ("rays", "sections", "jobs"):
        if not isinstance(opt[key], int) or isinstance(opt[key], bool) or opt[key] < 1:
            raise ValueError(f"beamlet_defect.{key}: a positive integer")
    if opt["modes"] is not None and (not isinstance(opt["modes"], int) or opt["modes"] < 1):
        raise ValueError("beamlet_defect.modes: null or a positive integer")
    for key in ("ring_step", "window"):
        if not isinstance(opt[key], (int, float)) or opt[key] <= 0:
            raise ValueError(f"beamlet_defect.{key}: a positive number")
    if opt["scan"] is not None:
        if not isinstance(opt["scan"], list) or not all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in opt["scan"]):
            raise ValueError("beamlet_defect.scan: a list of [w0, waist_z] pairs")
    return opt


# ------------------------------------------------------------------ beams
def drift_amp(qxx, qxy, qyy, s):
    """On-axis amplitude factor of a drift s: product of the principal roots
    1/sqrt(1 - s/s_i) over the roots of det(Q + s) (as gamma.propagate)."""
    tr = qxx + qyy
    det = qxx * qyy - qxy * qxy
    disc = np.sqrt(tr * tr - 4.0 * det)
    amp = np.ones_like(tr)
    for root in (0.5 * (disc - tr), -0.5 * (disc + tr)):
        safe = np.where(root == 0, 1.0, root)
        amp = amp * np.where(root == 0, 0.0, 1.0 / np.sqrt(1.0 - s / safe))
    return amp


def launch_q(zr_t, zr_s, waist_dz, drift):
    """Scalar q per axis after the virtual drift -waist_dz and a drift from the source, in
    the exponent convention exp(ik d^2 / 2q): q = (z - z_w) - i z_R (the chain stores the
    conjugate, Q = i z_R + drift, and the deposit conjugates it back)."""
    return (complex(drift - waist_dz, -zr_t), complex(drift - waist_dz, -zr_s))


def entrance_defect(k, a_w, z_in, q_t, q_s):
    """D0 of a uniform beam fan on a disc of radius a_w: relative L2 distance between
    the disc indicator and its convolution with the launch kernel (FFT)."""
    w_e = max(math.sqrt(2.0 / (k * (1.0 / q).imag)) for q in (q_t, q_s))
    h = min(0.05e-6, w_e / 10.0)
    half = a_w + 5.0 * w_e
    n = int(math.ceil(2.0 * half / h))
    if n > 6000:
        n = 6000
    h = 2.0 * half / n
    x = (np.arange(n) - n // 2) * h                      # d = 0 exactly on a node: ifftshift aligns the kernel
    X, Y = np.meshgrid(x, x)
    disc = (np.hypot(X, Y) <= a_w).astype(complex)
    kernel = np.exp(0.5j * k * (X * X * (1.0 / q_t - 1.0 / z_in) + Y * Y * (1.0 / q_s - 1.0 / z_in)))
    kernel *= (np.abs(X) <= 5.0 * w_e) & (np.abs(Y) <= 5.0 * w_e)
    conv = np.fft.ifft2(np.fft.fft2(disc) * np.fft.fft2(np.fft.ifftshift(kernel))) / kernel.sum()
    inside = disc.real > 0.5
    return float(np.sqrt(np.sum(np.abs(1.0 - conv[inside]) ** 2) / inside.sum())), float(w_e)


def kernel_integral(k, z_in, q_t, q_s):
    return cmath.sqrt(2j * math.pi / (k * (1.0 / q_t - 1.0 / z_in))) * cmath.sqrt(2j * math.pi / (k * (1.0 / q_s - 1.0 / z_in)))


# ------------------------------------------------------------------ archive
def read_mode(archive, sections, n_rays):
    """Screen-fate rays of the first n_rays ray ids of a mode (floats), the mode origin
    and the counts of every fate."""
    origin = [float(v) for v in rays_v3.section_header(archive, sections[0])["origin"]]
    rows, fates, total = [], {"screen": 0, "absorbed": 0, "lost": 0}, 0
    for entry in sections:
        if entry.r0 >= n_rays:
            break
        for line in rays_v3.iter_section_lines(archive, entry):
            if total >= n_rays:
                break
            total += 1
            row = json.loads(line)
            fates[row["fate"]] = fates.get(row["fate"], 0) + 1
            if row["fate"] == "screen":
                rows.append((float(row["x"]), float(row["y"]), float(row["dx"]), float(row["dy"]),
                             float(row["opl"]), [float(s) for s in row["sins"]],
                             [[float(c) for c in p] for p in row.get("refl", ())]))
        if total >= n_rays:
            break
    return origin, rows, fates, total


# ------------------------------------------------------------------ segments
class Segments:
    """Flat arrays of every ray segment of a mode: beam state at the segment start."""

    def __init__(self, n):
        self.qxx = np.zeros(n, complex); self.qxy = np.zeros(n, complex); self.qyy = np.zeros(n, complex)
        self.a = np.zeros(n, complex); self.fres = np.zeros(n, complex)
        self.opl = np.zeros(n); self.x0 = np.zeros(n); self.y0 = np.zeros(n); self.z0 = np.zeros(n)
        self.ux = np.zeros(n); self.uy = np.zeros(n); self.z1 = np.zeros(n)
        self.final = np.zeros(n, bool); self.bore = np.zeros(n, int); self.ray = np.zeros(n, int)


def build_segments(rows, origin, z_rec, bundle, zr_t, zr_s, waist_dz, fresnel, k, centers, radii, z0_cap):
    """Chain every ray through its hits: per segment Q, amplitude, path phase, Fresnel
    product of the bounces before it, start point, slope, z range, final flag, bore."""
    count = sum(len(r[6]) + 1 for r in rows)
    seg = Segments(count)
    d2, b2 = fresnel
    i = 0
    for ray, (x, y, dx, dy, opl, sins, refl) in enumerate(rows):
        dz = math.sqrt(max(1.0 - dx * dx - dy * dy, 0.0))
        pts = [tuple(p) for p in refl]
        points = [tuple(origin)] + pts + [(x, y, z_rec)]
        if pts:
            outs = [tuple(q - p for p, q in zip(a_, b_)) for a_, b_ in zip(pts, pts[1:])] + [(dx, dy, dz)]
            lenses = bounce_lenses(bundle, pts, outs)
        else:
            lenses = []
        # entrance point of the first segment -> bore
        p0, p1 = points[0], points[1]
        t = (z0_cap - p0[2]) / (p1[2] - p0[2])
        ex, ey = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
        bore = int(np.argmin((centers[:, 0] - ex) ** 2 + (centers[:, 1] - ey) ** 2))
        q = (complex(-waist_dz, zr_t), 0j, complex(-waist_dz, zr_s))
        amp, fr, path = 1.0 + 0j, 1.0 + 0j, 0.0
        n_seg = len(points) - 1
        for j in range(n_seg):
            pa, pb = points[j], points[j + 1]
            seg.qxx[i], seg.qxy[i], seg.qyy[i] = q
            seg.a[i], seg.fres[i], seg.opl[i] = amp, fr, path
            seg.x0[i], seg.y0[i], seg.z0[i] = pa
            dzj = pb[2] - pa[2]
            seg.ux[i], seg.uy[i] = (pb[0] - pa[0]) / dzj, (pb[1] - pa[1]) / dzj
            seg.z1[i] = pb[2]
            seg.final[i] = j == n_seg - 1
            seg.bore[i] = bore
            seg.ray[i] = ray
            length = math.dist(pa, pb)
            tr = q[0] + q[2]
            disc = cmath.sqrt(tr * tr - 4.0 * (q[0] * q[2] - q[1] * q[1]))
            for root in (0.5 * (disc - tr), -0.5 * (disc + tr)):
                amp *= 0.0 if root == 0 else 1.0 / cmath.sqrt(1.0 - length / root)
            q = (q[0] + length, q[1], q[2] + length)
            path += length
            if j < len(lenses):
                q = reflect(q, *lenses[j])
                s = sins[j]
                root = cmath.sqrt(s * s - d2 + 1j * b2)
                fr *= (s - root) / (s + root)
            i += 1
    return seg


def beam_values(seg, idx, z, px, py, k, z_src):
    """Values of the selected segment beams at points (px, py) [n_sel, n_pts] on the plane z."""
    ux, uy = seg.ux[idx][:, None], seg.uy[idx][:, None]
    dz = (z - seg.z0[idx])[:, None]
    stretch = np.sqrt(1.0 + ux * ux + uy * uy)
    s = dz * stretch
    qxx, qxy, qyy = seg.qxx[idx][:, None] + s, seg.qxy[idx][:, None], seg.qyy[idx][:, None] + s
    amp = seg.a[idx][:, None] * drift_amp(seg.qxx[idx][:, None], seg.qxy[idx][:, None], seg.qyy[idx][:, None], s)
    det = qxx * qyy - qxy * qxy
    # the deposit convention: exp(i (k/2) conj(G) d.d) with G = Q^-1, amplitude conj(a)
    gxx, gxy, gyy = np.conj(qyy / det), np.conj(-qxy / det), np.conj(qxx / det)
    dx = px - (seg.x0[idx][:, None] + ux * dz)
    dy = py - (seg.y0[idx][:, None] + uy * dz)
    phase = k * (seg.opl[idx][:, None] + s - (z - z_src))
    expo = 1j * phase + 1j * k * (ux * dx + uy * dy) + 0.5j * k * (gxx * dx * dx + 2.0 * gxy * dx * dy + gyy * dy * dy)
    return np.conj(amp) * seg.fres[idx][:, None] * np.exp(expo)


def beam_width(seg, idx, z):
    """Widest 1/e axis of the selected beams on the plane z, and their centres."""
    dz = z - seg.z0[idx]
    s = dz * np.sqrt(1.0 + seg.ux[idx] ** 2 + seg.uy[idx] ** 2)
    qxx, qxy, qyy = seg.qxx[idx] + s, seg.qxy[idx], seg.qyy[idx] + s
    det = qxx * qyy - qxy * qxy
    gi = ((qyy / det).imag, (-qxy / det).imag, (qxx / det).imag)
    mean = 0.5 * (gi[0] + gi[2])
    dev = np.hypot(0.5 * (gi[0] - gi[2]), gi[1])
    hi = mean + dev
    w = np.where(hi < 0, np.sqrt(-2.0 / np.where(hi < 0, hi, -1.0)), np.inf)
    return w, seg.x0[idx] + seg.ux[idx] * dz, seg.y0[idx] + seg.uy[idx] * dz


def bore_axis(bore, z):
    c = np.array([float(bore["center"][0]), float(bore["center"][1])])
    bend = bore.get("bend")
    if bend:
        t = np.array([float(bend["toward"][0]), float(bend["toward"][1])])
        t /= np.linalg.norm(t)
        c = c + t * z * z / (2.0 * float(bend["radius"]))
    return c


def _shape(bore):
    """(sides, rotation) of a polygon bore, None for a circle."""
    sides = bore.get("sides")
    if not sides:
        return None
    rot = bore.get("rotation")
    rot = float(rot) if rot is not None else math.radians(float(bore.get("rotation_deg", 0.0)))
    return int(sides), rot


def _stretch(shape, phis):
    """Radius factor of the bore outline at the angles phis: a polygon of apothem a has
    r = a / cos(angle to the nearest face normal); a circle 1."""
    if shape is None:
        return np.ones(phis.size)
    n, rot = shape
    rel = np.mod(phis - rot + math.pi / n, 2.0 * math.pi / n) - math.pi / n
    return 1.0 / np.cos(rel)


def _ring_sum(seg, idx, z, cx, cy, radii, phis, k, z_src, window, stretch):
    """Sum of the selected beams on the points (radii * stretch(phi), phis) around (cx, cy),
    evaluating each beam only within its angular window; returns [n_r, n_phi]."""
    n_r, n_phi = radii.size, phis.size
    out = np.zeros((n_r, n_phi), complex)
    if idx.size == 0:
        return out
    w, bx, by = beam_width(seg, idx, z)
    w = np.where(np.isfinite(w), w, radii.max())
    phi_b = np.arctan2(by - cy, bx - cx)
    dphi = phis[1] - phis[0]
    half = np.minimum(np.ceil((window * w / k_safe(radii.min())) / dphi).astype(int) + 1, n_phi // 2)
    order = np.argsort(half)
    idx, half, phi_b = idx[order], half[order], phi_b[order]
    start = 0
    while start < idx.size:
        h = half[start]
        stop = start
        while stop < idx.size and half[stop] == h:
            stop += 1
        stop = min(stop, start + 2048)
        sel = idx[start:stop]
        m0 = np.rint((phi_b[start:stop] - phis[0]) / dphi).astype(int)
        cols = (m0[:, None] + np.arange(-h, h + 1)[None, :]) % n_phi      # [n_sel, 2h+1]
        rr = radii[None, :, None] * stretch[cols][:, None, :]              # [n_sel, n_r, 2h+1]
        px = cx + rr * np.cos(phis[cols])[:, None, :]
        py = cy + rr * np.sin(phis[cols])[:, None, :]
        vals = beam_values(seg, sel, z, px.reshape(len(sel), -1), py.reshape(len(sel), -1), k, z_src)
        vals = vals.reshape(len(sel), n_r, -1)
        np.add.at(out, (slice(None), cols), np.transpose(vals, (1, 0, 2)))
        start = stop
    return out


def _halves(seg, idx, z, cx, cy, radii, phis, k, z_src, window, stretch):
    """Beam sums of the even and odd ray halves: |a + b|^2 is the full sum, |a - b|^2 its
    ray-quadrature noise."""
    even = seg.ray[idx] % 2 == 0
    return (_ring_sum(seg, idx[even], z, cx, cy, radii, phis, k, z_src, window, stretch),
            _ring_sum(seg, idx[~even], z, cx, cy, radii, phis, k, z_src, window, stretch))


def k_safe(r):
    return max(r, 1e-12)


def wall_trace(seg, bores, a_w, zs, k, z_src, ring_step, window):
    """Per section: the wall integral of |beam sum|^2 and of its ray-noise part, summed
    over the bores (polygon faces through the apothem stretch)."""
    n_phi = int(math.ceil(2.0 * math.pi * a_w / ring_step))
    phis = np.arange(n_phi) * 2.0 * math.pi / n_phi
    radii = np.array([a_w])
    full, shot = np.zeros(zs.size), np.zeros(zs.size)
    for jz, z in enumerate(zs):
        active = (seg.z0 <= z + 0.1) & (seg.z1 >= z - 0.1)                   # every segment is a full beam; cheap pre-cut
        for b, bore in enumerate(bores):
            stretch = _stretch(_shape(bore), phis)
            cx, cy = bore_axis(bore, z)
            cand = np.flatnonzero(active & (seg.bore == b))
            if cand.size == 0:
                continue
            w, bx, by = beam_width(seg, cand, z)
            near = np.hypot(bx - cx, by - cy) <= a_w * stretch.max() + window * np.where(np.isfinite(w), w, 0.0)
            ra, rb = _halves(seg, cand[near], z, cx, cy, radii, phis, k, z_src, window, stretch)
            measure = a_w * stretch ** 2 * (2.0 * math.pi / n_phi)
            full[jz] += float(np.sum(np.abs(ra[0] + rb[0]) ** 2 * measure))
            shot[jz] += float(np.sum(np.abs(ra[0] - rb[0]) ** 2 * measure))
    return full, shot


def exit_tails(seg, bores, a_w, z_exit, k, z_src, ring_step, window):
    """T^2 and its ray-noise part: beams of non-final segments on the exit plane inside
    the bores (polar grid, polygon outline through the apothem stretch)."""
    n_phi = int(math.ceil(2.0 * math.pi * a_w / ring_step))
    phis = np.arange(n_phi) * 2.0 * math.pi / n_phi
    full = shot = 0.0
    for b, bore in enumerate(bores):
        stretch = _stretch(_shape(bore), phis)
        cx, cy = bore_axis(bore, z_exit)
        cand = np.flatnonzero(~seg.final & (seg.bore == b))
        if cand.size == 0:
            continue
        w, bx, by = beam_width(seg, cand, z_exit)
        wf = np.where(np.isfinite(w), w, 0.0)
        near = np.hypot(bx - cx, by - cy) <= a_w * stretch.max() + window * wf
        idx = cand[near]
        if idx.size == 0:
            continue
        depth = min(a_w, (window + 2.0) * float(wf[near].max()))
        n_r = max(4, int(math.ceil(depth / ring_step)))
        dr = depth / n_r
        radii = a_w - (np.arange(n_r) + 0.5) * dr
        ga, gb = _halves(seg, idx, z_exit, cx, cy, radii, phis, k, z_src, window, stretch)
        measure = radii[:, None] * stretch[None, :] ** 2 * dr * (2.0 * math.pi / n_phi)
        full += float(np.sum(np.abs(ga + gb) ** 2 * measure))
        shot += float(np.sum(np.abs(ga - gb) ** 2 * measure))
    return full, shot


# ------------------------------------------------------------------ per mode
def mode_job(job):
    t0 = time.time()
    archive, parts = job["archive"], job["parts"]
    origin, rows, fates, total = read_mode(archive, parts, job["rays"])
    from ..simulation import Simulation
    sim = Simulation.from_dict(job["raw_cfg"])
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    centers = np.array([[float(b["center"][0]), float(b["center"][1])] for b in cap.bores])
    radii = np.array([float(b["radius"]) for b in cap.bores])
    seg = build_segments(rows, origin, job["z_rec"], bundle, job["zr_t"], job["zr_s"], job["waist_dz"],
                         job["fresnel"], job["k"], centers, radii, float(cap.z0))
    bores = sim.cfg.raw["capillary"]["bores"]
    length = job["z_exit"] - float(cap.z0)
    zs = (np.arange(job["sections"]) + 0.5) * length / job["sections"] + float(cap.z0)
    g_full, g_shot = wall_trace(seg, bores, job["a_w"], zs, job["k"], origin[2], job["ring_step"], job["window"])
    t_full, t_shot = exit_tails(seg, bores, job["a_w"], job["z_exit"], job["k"], origin[2], job["ring_step"], job["window"])
    norm = job["norm_per_ray"] * total                    # ||E_in|| in deposit units for this fan
    rms_field = norm / math.sqrt(job["area"])
    wall_area = length * 2.0 * math.pi * job["a_w"] * len(cap.bores)
    dz = length / job["sections"]
    g_coh = np.maximum(g_full - g_shot, 0.0)
    return {"mode": job["mode"], "rays": total, "fates": fates, "segments": int(seg.qxx.size),
            "G": math.sqrt(float(np.sum(g_coh)) * dz / wall_area) / rms_field,
            "G_shot": math.sqrt(float(np.sum(g_shot)) * dz / wall_area) / rms_field,
            "T": math.sqrt(max(t_full - t_shot, 0.0)) / norm, "T_shot": math.sqrt(t_shot) / norm,
            "trace_profile": (np.sqrt(g_coh / (2.0 * math.pi * job["a_w"] * len(cap.bores))) / rms_field).tolist(),
            "seconds": time.time() - t0}


def launch_jobs(sim, archive, index, opt, w0, w0_t, waist_z):
    cfg = sim.cfg
    cap = cfg.capillary
    k = float(sim.lines[0].k)
    z_src = float(cap.source.position[2])
    z0, z1 = float(cap.z0), float(cap.z1)
    z_in = z0 - z_src
    waist_dz = 0.0 if waist_z is None else float(waist_z) - z_src
    w0_t = w0 if w0_t is None else w0_t
    zr_t, zr_s = 0.5 * w0_t * w0_t * k, 0.5 * w0 * w0 * k
    # DIR-ell wall: hard wall at a + Re(ell), ell = i / (k q0)
    q0 = cmath.sqrt(complex(-2.0 * sim.delta_f, 2.0 * sim.beta_f))
    if q0.imag < 0:
        q0 = -q0
    a_w = float(cap.bores[0]["radius"]) + (1j / (k * q0)).real
    q_t, q_s = launch_q(zr_t, zr_s, waist_dz, z_in)
    d0, w_e = entrance_defect(k, a_w, z_in, q_t, q_s)
    area = sum(math.pi * float(b["radius"]) ** 2 for b in cap.bores)
    a_ent = abs(drift_amp(np.array([complex(-waist_dz, zr_t)]), np.array([0j]), np.array([complex(-waist_dz, zr_s)]), np.array([z_in]))[0])
    norm_per_ray = abs(kernel_integral(k, z_in, q_t, q_s)) * a_ent / math.sqrt(area)
    fres = FloatLineAmplitudes(cfg.material, sim.lines, cfg.precision).two_delta_beta[0]
    modes = index.modes("capillary")
    if opt["modes"] is not None:
        modes = modes[:opt["modes"]]
    screen_z = float((cap.screen).z)
    rec = index_screen_z(archive)
    base = dict(archive=archive, raw_cfg=cfg.raw, rays=opt["rays"], sections=opt["sections"],
                ring_step=float(opt["ring_step"]), window=float(opt["window"]), k=k, zr_t=zr_t, zr_s=zr_s,
                waist_dz=waist_dz, fresnel=fres, a_w=a_w, z_exit=z1, z_rec=rec, norm_per_ray=norm_per_ray, area=area)
    jobs = [dict(base, mode=m, parts=tuple(secs)) for m, secs in enumerate(modes)]
    meta = {"w0_m": w0, "w0_t_m": w0_t, "waist_z_m": waist_z, "waist_dz_m": waist_dz, "D0": d0,
            "rays_per_mode_production": int(max(secs[-1].r1 for secs in modes)),
            "width_at_entrance_m": w_e, "a_w_m": a_w, "z_in_m": z_in, "recorded_z_m": rec, "screen_z_m": screen_z}
    return jobs, meta


def index_screen_z(archive):
    meta = rays_v3.read_fingerprint(archive)
    g = meta["geometry"]
    screen = dict(g.get("screen") or {})
    screen.update((g.get("capillary") or {}).get("screen") or {})
    return float(screen["z"])


def run_defect(sim, archive, out_dir, opt, launches, log=print):
    """Per launch: D0 once, G and T per mode (processes over modes); writes the rows and
    the summary with the argmin launch."""
    index = rays_v3.load_index(archive)
    out = os.path.join(out_dir, RESULT_DIR)
    os.makedirs(out, exist_ok=True)
    summary = {"archive": archive, "options": opt, "launches": []}
    rows_path = os.path.join(out, "modes.jsonl")
    with open(rows_path, "w", encoding="utf-8") as rows:
        for w0, w0_t, waist_z in launches:
            jobs, meta = launch_jobs(sim, archive, index, opt, w0, w0_t, waist_z)
            t0 = time.time()
            results = []
            if opt["jobs"] > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=opt["jobs"]) as pool:
                    for res in pool.map(mode_job, jobs, chunksize=1):
                        results.append(res)
            else:
                results = [mode_job(j) for j in jobs]
            for res in results:
                res["D0"] = meta["D0"]
                # ray noise rescaled from the scanned to the production ray count
                scale = math.sqrt(res["rays"] / meta["rays_per_mode_production"])
                res["indicator"] = meta["D0"] + res["G"] + res["T"]
                res["indicator_prod"] = (meta["D0"] + math.hypot(res["G"], scale * res["G_shot"])
                                         + math.hypot(res["T"], scale * res["T_shot"]))
                res["launch"] = {"w0_m": w0, "w0_t_m": w0_t, "waist_z_m": waist_z}
                rows.write(json.dumps(res) + "\n")
            rms = lambda key: float(np.sqrt(np.mean([r[key] ** 2 for r in results])))
            rec = dict(meta, indicator_rms=rms("indicator"), indicator_prod_rms=rms("indicator_prod"),
                       indicator_max=float(max(r["indicator"] for r in results)),
                       G_rms=rms("G"), G_shot_rms=rms("G_shot"), T_rms=rms("T"), T_shot_rms=rms("T_shot"),
                       modes=len(results), seconds=time.time() - t0)
            summary["launches"].append(rec)
            log(f"  launch w0 {w0*1e6:.2f} um (w0_t {meta['w0_t_m']*1e6:.2f}) waist_z {waist_z}: D0 {meta['D0']:.4f} "
                f"G rms {rec['G_rms']:.4f} (noise {rec['G_shot_rms']:.4f}) T rms {rec['T_rms']:.4f} (noise {rec['T_shot_rms']:.4f}) "
                f"-> indicator rms {rec['indicator_rms']:.4f}, at production rays {rec['indicator_prod_rms']:.4f}; "
                f"{len(results)} modes, {rec['seconds']:.0f} s")
    best = min(summary["launches"], key=lambda r: r["indicator_prod_rms"])
    summary["argmin"] = {"w0_m": best["w0_m"], "w0_t_m": best["w0_t_m"], "waist_z_m": best["waist_z_m"],
                         "indicator_rms": best["indicator_rms"], "indicator_prod_rms": best["indicator_prod_rms"]}
    with open(os.path.join(out, "defect.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def parse_scan(text, z_src):
    launches = []
    for part in text.split(","):
        w, z = part.split(":")
        z = z.strip()
        launches.append((float(w) * 1e-6, None, None if z in ("", "null", "source") else float(z)))
    return launches


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage 11 beamlet defect indicator and launch scan")
    ap.add_argument("config"); ap.add_argument("archive"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--scan", default=None, help="extra launches w0_um:waist_z[,...] (waist_z 'source' = null)")
    ap.add_argument("--rays", type=int, default=None); ap.add_argument("--sections", type=int, default=None)
    ap.add_argument("--modes", type=int, default=None); ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    from .. import load
    from ..simulation import Simulation
    cfg = load(args.config)
    sim = Simulation(cfg)
    cfg.validate_beamlet()
    opt = options(cfg)
    for key in ("rays", "sections", "modes", "jobs"):
        if getattr(args, key) is not None:
            opt[key] = getattr(args, key)
    cap = cfg.capillary
    if cap is None:
        raise ValueError("beamlet_defect needs a capillary scene")
    w0_t, waist_dz = _launch(sim, "capillary", float(cap.screen.z))
    launches = [(cfg.beamlet_w0, w0_t, cfg.beamlet_waist_z)]
    for w0, _, wz in (opt["scan"] and [(p[0], None, p[1]) for p in opt["scan"]]) or []:
        launches.append((float(w0), None, wz))
    if args.scan:
        launches += parse_scan(args.scan, float(cap.source.position[2]))
    t0 = time.time()
    summary = run_defect(sim, os.path.abspath(args.archive), args.out, opt, launches)
    best = summary["argmin"]
    print(f"argmin indicator: w0 {best['w0_m']*1e6:.2f} um, waist_z {best['waist_z_m']} "
          f"(rms at production rays {best['indicator_prod_rms']:.4f}); {time.time() - t0:.0f} s; rows in {os.path.join(args.out, RESULT_DIR)}")


if __name__ == "__main__":
    main()

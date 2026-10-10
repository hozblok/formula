"""Stage 17 defect indicator: how far the canonical (archive_canonical) Gaussian
reconstruction of a scene is from a solution of its paraxial Dirichlet problem, per
source mode and window width, without a wave reference.

The reconstruction sums frozen Gaussian windows of width w along the archived rays:
psi_r(x, z) = c_r(z) exp(i k u_r.(x - X_r(z)) - |x - X_r(z)|^2 / 2 w^2), with the
canonical prefactor sqrt(det(gamma Q + i P)) e^{-i pi maslov / 2} of the variational
transport (Q = dX/dq, P = du/dq).  Its defects:
  D0  entrance representation: a uniform fan of windows on the entrance disc equals the
      disc indicator convolved with the Gaussian; D0 = ||1 - 1 * g / int g|| / ||1||.
  R   volume residual of the frozen windows, i d/dz psi + Lap psi / 2k, summed over the
      rays of a section and integrated over z inside the channel and up to the target
      screen (the windows do not spread, the prefactor does); the exact field has R = 0.
  G   wall trace: every ray segment continued along its straight line is summed on the
      bore wall r = a + Re(ell) at sections in z; RMS amplitude relative to the entrance
      field (the exact field has zero trace).
  T   exit tails: the map keeps the final segment of every ray; the windows of the
      earlier segments still reach the exit plane inside the bores.
indicator = D0 + R + G + T per mode and width; the scan picks the width with the
smallest RMS indicator over modes.  Reported as an indicator, not a certified bound.

Usage: python -m formula.capsysred.stages._b5_defect SCENE.yaml ARCHIVE -o OUT
         [--widths 0.5,1,2 (um)] [--rays N] [--sections S] [--modes M] [--jobs J]
Rows: OUT/stage17-defect/modes.jsonl, defect.json.
"""

import argparse
import concurrent.futures
import json
import math
import os
import time

import numpy as np

from .. import rays_v3
from ._b5_transport import count_free_caustics, reflect_variations, wall_normal_derivative

RESULT_DIR = "stage17-defect"
DEFAULTS = {"rays": 20000, "sections": 40, "samples": 300, "ring_step": 0.5e-6, "cutoff": 5.0,
            "modes": None, "jobs": 1}


def options(cfg):
    raw = dict(cfg.raw.get("b5_defect") or {})
    unknown = raw.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"b5_defect has unknown keys {sorted(unknown)}")
    opt = {**DEFAULTS, **raw}
    for key in ("rays", "sections", "samples", "jobs"):
        if not isinstance(opt[key], int) or isinstance(opt[key], bool) or opt[key] < 1:
            raise ValueError(f"b5_defect.{key}: a positive integer")
    if opt["modes"] is not None and (not isinstance(opt["modes"], int) or opt["modes"] < 1):
        raise ValueError("b5_defect.modes: null or a positive integer")
    for key in ("ring_step", "cutoff"):
        if not isinstance(opt[key], (int, float)) or opt[key] <= 0:
            raise ValueError(f"b5_defect.{key}: a positive number")
    return opt


# ------------------------------------------------------------------ entrance
def entrance_defect(a_w, width):
    """D0 of a uniform window fan on a disc: the disc indicator against its convolution
    with the normalised Gaussian of width w (FFT)."""
    h = min(0.05e-6, width / 10.0)
    half = a_w + 5.0 * width
    n = min(int(math.ceil(2.0 * half / h)), 6000)
    h = 2.0 * half / n
    x = (np.arange(n) - n // 2) * h
    X, Y = np.meshgrid(x, x)
    disc = (np.hypot(X, Y) <= a_w).astype(complex)
    kernel = np.exp(-(X * X + Y * Y) / (2.0 * width * width))
    conv = np.fft.ifft2(np.fft.fft2(disc) * np.fft.fft2(np.fft.ifftshift(kernel))) / kernel.sum()
    inside = disc.real > 0.5
    return float(np.sqrt(np.sum(np.abs(1.0 - conv[inside]) ** 2) / inside.sum()))


# ------------------------------------------------------------------ archive
def read_mode(archive, sections, n_rays):
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
    """Flat arrays of every ray segment: variational state at the segment start."""

    def __init__(self, n):
        self.p0 = np.zeros((n, 3)); self.u = np.zeros((n, 3)); self.z1 = np.zeros(n)
        self.r = np.zeros((n, 3, 2)); self.du = np.zeros((n, 3, 2))
        self.maslov = np.zeros(n, int); self.fres = np.zeros(n, complex); self.opl = np.zeros(n)
        self.amp = np.zeros(n); self.final = np.zeros(n, bool); self.bore = np.zeros(n, int); self.valid = np.ones(n, bool)
        self.ray = np.zeros(n, int)


def build_segments(rows, origin, z_rec, bores, z0_cap, k, fresnel):
    """Per-segment variational state from the archived path (transport as the map)."""
    d2, b2 = fresnel
    centers = np.array([[float(b["center"][0]), float(b["center"][1])] for b in bores])
    radii = np.array([float(b["radius"]) for b in bores])
    count = sum(len(r[6]) + 1 for r in rows)
    seg = Segments(count)
    source = np.asarray(origin, float)
    i = 0
    for ray, (x, y, dx, dy, opl, sins, refl) in enumerate(rows):
        dz = math.sqrt(max(1.0 - dx * dx - dy * dy, 0.0))
        points = [source] + [np.asarray(p, float) for p in refl] + [np.array([x, y, z_rec])]
        first = points[1]
        u = first - source
        dist = float(np.linalg.norm(u))
        u = u / dist
        ent = source[:2] + u[:2] / u[2] * (z0_cap - source[2])
        bore = int(np.argmin(np.sum((centers - ent) ** 2, axis=1)))
        inside = np.linalg.norm(centers[bore] - ent) <= radii[bore] * (1 + 1e-8)
        # derivatives with respect to the entrance coordinates q (flux measure dq)
        r = np.eye(3)[:, :2].copy()
        distance = (z0_cap - source[2]) / u[2]
        du = (r - u[:, None] * u[None, :2]) / distance
        amp = math.sqrt((distance / dist) / dz) / dist            # sqrt(uz0 / uz_out) / entrance distance
        maslov, fr, path, ok = 0, 1.0 + 0j, 0.0, bool(inside)
        n_seg = len(points) - 1
        for j in range(n_seg):
            pa, pb = points[j], points[j + 1]
            seg.p0[i] = pa; seg.u[i] = u; seg.z1[i] = pb[2]
            seg.r[i] = r; seg.du[i] = du; seg.maslov[i] = maslov; seg.fres[i] = fr; seg.opl[i] = path
            seg.amp[i] = amp; seg.final[i] = j == n_seg - 1; seg.bore[i] = bore; seg.ray[i] = ray
            flight = (pb[2] - pa[2]) / u[2]
            v = _velocity_derivative(u, du)
            crosses, ambiguous = count_free_caustics(r[:2][None], v[None], np.array([pb[2] - pa[2]]))
            maslov += int(crosses[0]); ok &= not bool(ambiguous[0])
            reached = pa + flight * u
            ok &= bool(np.linalg.norm(reached - pb) < 1e-8)
            r = r + flight * du
            path += flight
            if j < n_seg - 1:
                normal, dn, wall_distance = wall_normal_derivative(pb[None], bores[bore], z0_cap)
                ok &= bool(abs(wall_distance[0] - radii[bore]) < 1e-9)
                with np.errstate(divide="ignore", invalid="ignore"):
                    u2, r2, du2 = reflect_variations(u[None], r[None], du[None], normal, dn)
                u, r, du = u2[0], r2[0], du2[0]
                s = sins[j]
                root = np.sqrt(complex(s * s - d2, b2))
                fr *= (s - root) / (s + root)
            seg.valid[i] = ok
            i += 1
    return seg


def _velocity_derivative(u, du):
    slope = u[:2] / u[2]
    return (du[:2, :] - slope[:, None] * du[2, None, :]) / u[2]


def states(seg, idx, z):
    """Centre, slope, phase, Q, P, V and the Maslov count of the selected segments at z."""
    u = seg.u[idx]
    flight = (z - seg.p0[idx, 2]) / u[:, 2]
    X = seg.p0[idx, :2] + flight[:, None] * u[:, :2]
    r = seg.r[idx] + flight[:, None, None] * seg.du[idx]
    du = seg.du[idx]
    # fixed-z projection of the variations (as the transport does on the target plane)
    rz = r - u[:, :, None] * (r[:, 2, :] / u[:, 2, None])[:, None, :]
    Q, P = rz[:, :2, :], du[:, :2, :]
    V = (du[:, :2, :] - (u[:, :2] / u[:, 2, None])[:, :, None] * du[:, 2, None, :]) / u[:, 2, None, None]
    crosses, _ = count_free_caustics(seg.r[idx][:, :2], _velocity_derivative_batch(seg.u[idx], seg.du[idx]),
                                     np.maximum(z - seg.p0[idx, 2], 0.0))
    maslov = seg.maslov[idx] + crosses
    phase = seg.opl[idx] + flight - (z - seg.p0[idx, 2]) * 0.0           # excess carried below
    return X, u[:, :2] / u[:, 2, None], flight, Q, P, V, maslov


def _velocity_derivative_batch(u, du):
    slope = u[:, :2] / u[:, 2, None]
    return (du[:, :2, :] - slope[:, :, None] * du[:, 2, None, :]) / u[:, 2, None, None]


def prefactor(Q, P, maslov, gamma):
    m = gamma * Q + 1j * P
    det = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]
    det_q = Q[:, 0, 0] * Q[:, 1, 1] - Q[:, 0, 1] * Q[:, 1, 0]
    valid = np.isfinite(det) & (np.abs(det) > 0) & (det_q != 0)
    out = np.zeros(len(Q), complex)
    out[valid] = np.sqrt(np.sign(det_q[valid]) * det[valid]) * np.exp(-0.5j * np.pi * maslov[valid])
    return out, valid


def log_prefactor_derivative(Q, P, V, gamma):
    """d/dz log sqrt(det(gamma Q + i P)) = tr((gamma Q + i P)^-1 gamma V) / 2."""
    m = gamma * Q + 1j * P
    det = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]
    inv = np.empty_like(m)
    inv[:, 0, 0], inv[:, 1, 1] = m[:, 1, 1] / det, m[:, 0, 0] / det
    inv[:, 0, 1], inv[:, 1, 0] = -m[:, 0, 1] / det, -m[:, 1, 0] / det
    return 0.5 * gamma * np.einsum("nij,nji->n", inv, V)


def window_states(seg, idx, z, k, width, z_src):
    """Per-window centre, slope, coefficient and d/dz log-prefactor of the selected
    segments at z (computed once per section, reused for every evaluation point)."""
    X, slope, flight, Q, P, V, maslov = states(seg, idx, z)
    gamma = 1.0 / (k * width * width)
    pref, ok = prefactor(Q, P, maslov, gamma)
    coeff = (k / (2.0 * math.pi)) * seg.amp[idx] * pref * seg.fres[idx] * np.exp(1j * k * (seg.opl[idx] + flight - (z - z_src)))
    coeff = np.where(ok & seg.valid[idx], coeff, 0.0)
    return X, slope, coeff, log_prefactor_derivative(Q, P, V, gamma)


def window_values(seg, idx, z, px, py, k, width, z_src, residual=False, pre=None):
    """Frozen-window values of the selected segments at points [n_sel, n_pts]; with
    residual=True the values of i d/dz psi + Lap psi / 2k instead; `pre` = their
    window_states already restricted to idx."""
    X, slope, coeff, cz = window_states(seg, idx, z, k, width, z_src) if pre is None else pre
    dx = px - X[:, 0, None]
    dy = py - X[:, 1, None]
    rho2 = dx * dx + dy * dy
    psi = coeff[:, None] * np.exp(1j * k * (slope[:, 0, None] * dx + slope[:, 1, None] * dy) - rho2 / (2.0 * width * width))
    if not residual:
        return psi
    return psi * (1j * cz[:, None] + rho2 / (2.0 * k * width ** 4) - 1.0 / (k * width * width))


# ------------------------------------------------------------------ sampling
def bore_axis(bore, z):
    c = np.array([float(bore["center"][0]), float(bore["center"][1])])
    bend = bore.get("bend")
    if bend:
        t = np.array([float(bend["toward"][0]), float(bend["toward"][1])])
        t /= np.linalg.norm(t)
        c = c + t * z * z / (2.0 * float(bend["radius"]))
    return c


def _polar_sum(seg, idx, z, cx, cy, radii, phis, k, width, z_src, cutoff, residual=False):
    """Sum of the selected windows on the polar points around (cx, cy), each window only
    within its angular support; returns [n_r, n_phi]."""
    n_r, n_phi = radii.size, phis.size
    out = np.zeros((n_r, n_phi), complex)
    if idx.size == 0:
        return out
    X = states(seg, idx, z)[0]
    phi_b = np.arctan2(X[:, 1] - cy, X[:, 0] - cx)
    dphi = phis[1] - phis[0]
    h = min(int(math.ceil((cutoff * width) / (max(radii.min(), 1e-12) * dphi))) + 1, n_phi // 2)
    for start in range(0, idx.size, 1024):
        sel = idx[start:start + 1024]
        m0 = np.rint((phi_b[start:start + 1024] - phis[0]) / dphi).astype(int)
        cols = (m0[:, None] + np.arange(-h, h + 1)[None, :]) % n_phi
        px = cx + radii[None, :, None] * np.cos(phis[cols])[:, None, :]
        py = cy + radii[None, :, None] * np.sin(phis[cols])[:, None, :]
        vals = window_values(seg, sel, z, px.reshape(len(sel), -1), py.reshape(len(sel), -1), k, width, z_src, residual)
        np.add.at(out, (slice(None), cols), np.transpose(vals.reshape(len(sel), n_r, -1), (1, 0, 2)))
    return out


def near(seg, idx, z, cx, cy, a_w, reach):
    X = states(seg, idx, z)[0]
    return np.hypot(X[:, 0] - cx, X[:, 1] - cy) <= a_w + reach


def _halves(seg, idx, z, cx, cy, radii, phis, k, width, z_src, cutoff):
    """Window sums of the even and odd ray halves: |a + b|^2 is the full sum, |a - b|^2 its
    ray-quadrature noise."""
    even = seg.ray[idx] % 2 == 0
    return (_polar_sum(seg, idx[even], z, cx, cy, radii, phis, k, width, z_src, cutoff),
            _polar_sum(seg, idx[~even], z, cx, cy, radii, phis, k, width, z_src, cutoff))


def wall_trace(seg, bores, a_w, zs, k, width, z_src, ring_step, cutoff):
    """Per section: the wall integral of |window sum|^2 and of its ray-noise part."""
    n_phi = int(math.ceil(2.0 * math.pi * a_w / ring_step))
    phis = np.arange(n_phi) * 2.0 * math.pi / n_phi
    full, shot = np.zeros(zs.size), np.zeros(zs.size)
    for jz, z in enumerate(zs):
        for b, bore in enumerate(bores):
            cx, cy = bore_axis(bore, z)
            cand = np.flatnonzero(seg.bore == b)
            idx = cand[near(seg, cand, z, cx, cy, a_w, cutoff * width)]
            ra, rb = _halves(seg, idx, z, cx, cy, np.array([a_w]), phis, k, width, z_src, cutoff)
            measure = a_w * (2.0 * math.pi / n_phi)
            full[jz] += float(np.sum(np.abs(ra[0] + rb[0]) ** 2) * measure)
            shot[jz] += float(np.sum(np.abs(ra[0] - rb[0]) ** 2) * measure)
    return full, shot


def volume_residual(seg, bores, a_w, zs, k, width, z_src, samples, cutoff, final_only_after=None, seed=0):
    """||R(z)||^2 over the bore discs per section and its ray-noise part, by Monte Carlo:
    `samples` uniform points per disc, each summing the windows within cutoff * width
    (bucket join); after the exit (final_only_after = z_exit) only the final segments."""
    rng = np.random.default_rng(seed)
    reach = cutoff * width
    full, shot = np.zeros(zs.size), np.zeros(zs.size)
    for jz, z in enumerate(zs):
        for b, bore in enumerate(bores):
            cx, cy = bore_axis(bore, z)
            cand = np.flatnonzero(seg.bore == b)
            if final_only_after is not None and z > final_only_after:
                cand = cand[seg.final[cand]]
            idx = cand[near(seg, cand, z, cx, cy, a_w, reach)]
            if idx.size == 0:
                continue
            rad = a_w * np.sqrt(rng.random(samples))
            phi = 2.0 * math.pi * rng.random(samples)
            px, py = cx + rad * np.cos(phi), cy + rad * np.sin(phi)
            pre = window_states(seg, idx, z, k, width, z_src)
            X = pre[0]
            pt, sel = _bucket_pairs(X[:, 0] - cx, X[:, 1] - cy, px - cx, py - cy, a_w, reach)
            field = np.zeros((2, samples), complex)
            for start in range(0, pt.size, 1 << 18):
                p_, s_ = pt[start:start + (1 << 18)], sel[start:start + (1 << 18)]
                sub = tuple(arr[s_] for arr in pre)
                vals = window_values(seg, idx[s_], z, px[p_][:, None], py[p_][:, None], k, width, z_src, residual=True, pre=sub)[:, 0]
                np.add.at(field, (seg.ray[idx[s_]] % 2, p_), vals)
            area = math.pi * a_w * a_w
            full[jz] += float(np.mean(np.abs(field[0] + field[1]) ** 2) * area)
            shot[jz] += float(np.mean(np.abs(field[0] - field[1]) ** 2) * area)
    return full, shot


def _bucket_pairs(wx, wy, px, py, a_w, reach):
    """(point, window) pairs closer than `reach`: windows binned on cells of size reach,
    each point joined with its 3 x 3 neighbourhood, then filtered by distance."""
    n_cell = int(math.ceil(2.0 * (a_w + reach) / reach)) + 3
    lo = -(a_w + reach) - reach
    key = np.floor((wx - lo) / reach).astype(int) * n_cell + np.floor((wy - lo) / reach).astype(int)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    pcx, pcy = np.floor((px - lo) / reach).astype(int), np.floor((py - lo) / reach).astype(int)
    shift = np.array([-1, 0, 1])
    keys = ((pcx[:, None, None] + shift[None, :, None]) * n_cell + pcy[:, None, None] + shift[None, None, :]).reshape(-1)
    first = np.searchsorted(sorted_key, keys, "left")
    last = np.searchsorted(sorted_key, keys, "right")
    count = last - first
    pt = np.repeat(np.arange(px.size), count.reshape(px.size, -1).sum(axis=1))
    offsets = np.cumsum(count) - count
    sel = order[np.arange(count.sum()) - np.repeat(offsets, count) + np.repeat(first, count)]
    close = np.hypot(wx[sel] - px[pt], wy[sel] - py[pt]) < reach
    return pt[close], sel[close]


def exit_tails(seg, bores, a_w, z_exit, k, width, z_src, ring_step, cutoff):
    """T^2 and its ray-noise part: non-final windows on the exit plane inside the bores."""
    n_phi = int(math.ceil(2.0 * math.pi * a_w / ring_step))
    phis = np.arange(n_phi) * 2.0 * math.pi / n_phi
    depth = min(a_w, (cutoff + 1.0) * width)
    n_r = max(4, int(math.ceil(depth / ring_step)))
    dr = depth / n_r
    radii = a_w - (np.arange(n_r) + 0.5) * dr
    full = shot = 0.0
    for b, bore in enumerate(bores):
        cx, cy = bore_axis(bore, z_exit)
        cand = np.flatnonzero((seg.bore == b) & ~seg.final)
        idx = cand[near(seg, cand, z_exit, cx, cy, a_w, cutoff * width)]
        ga, gb = _halves(seg, idx, z_exit, cx, cy, radii, phis, k, width, z_src, cutoff)
        measure = radii[:, None] * dr * (2.0 * math.pi / n_phi)
        full += float(np.sum(np.abs(ga + gb) ** 2 * measure))
        shot += float(np.sum(np.abs(ga - gb) ** 2 * measure))
    return full, shot


# ------------------------------------------------------------------ per mode
def mode_job(job):
    t0 = time.time()
    from ..simulation import Simulation
    sim = Simulation.from_dict(job["raw_cfg"])
    cap = sim.cfg.capillary
    bores = sim.cfg.raw["capillary"]["bores"]
    origin, rows, fates, total = read_mode(job["archive"], job["parts"], job["rays"])
    seg = build_segments(rows, origin, job["z_rec"], bores, float(cap.z0), job["k"], job["fresnel"])
    z0, z1, z_t = float(cap.z0), job["z_exit"], job["z_target"]
    inside = (np.arange(job["sections"]) + 0.5) * (z1 - z0) / job["sections"] + z0
    n_after = max(1, int(round(job["sections"] * (z_t - z1) / (z1 - z0)))) if z_t > z1 else 0
    after = z1 + (np.arange(n_after) + 0.5) * (z_t - z1) / n_after if n_after else np.zeros(0)
    out = {}
    coh = lambda pair: np.maximum(pair[0] - pair[1], 0.0)
    for width in job["widths"]:
        g_pair = wall_trace(seg, bores, job["a_w"], inside, job["k"], width, origin[2], job["ring_step"], job["cutoff"])
        r_in = volume_residual(seg, bores, job["a_w"], inside, job["k"], width, origin[2], job["samples"], job["cutoff"])
        r_out = (volume_residual(seg, bores, job["a_w"] * 1.0, after, job["k"], width, origin[2], job["samples"], job["cutoff"], final_only_after=z1)
                 if n_after else (np.zeros(0), np.zeros(0)))
        t_full, t_shot = exit_tails(seg, bores, job["a_w"], z1, job["k"], width, origin[2], job["ring_step"], job["cutoff"])
        # entrance field norm of this fan in the map's units: (N/A) coeff_0 2 pi w^2 over the area
        amp0 = (job["k"] / (2.0 * math.pi)) * float(np.mean(seg.amp[seg.final])) * (1.0 / (job["k"] * width * width))
        rms_field = total / job["area"] * amp0 * 2.0 * math.pi * width * width
        norm = rms_field * math.sqrt(job["area"])
        wall_area = (z1 - z0) * 2.0 * math.pi * job["a_w"] * len(bores)
        dz_in, dz_out = (z1 - z0) / job["sections"], ((z_t - z1) / n_after if n_after else 0.0)
        r_int = lambda a, b: float(np.sum(np.sqrt(a)) * dz_in + np.sum(np.sqrt(b)) * dz_out) / norm
        g_scale = dz_in / wall_area
        out[f"{width*1e6:g}"] = {"G": math.sqrt(float(np.sum(coh(g_pair))) * g_scale) / rms_field,
                                 "G_shot": math.sqrt(float(np.sum(g_pair[1])) * g_scale) / rms_field,
                                 "R": r_int(coh(r_in), coh(r_out)), "R_shot": r_int(r_in[1], r_out[1]),
                                 "R_inside": r_int(coh(r_in), np.zeros(0)),
                                 "T": math.sqrt(max(t_full - t_shot, 0.0)) / norm, "T_shot": math.sqrt(t_shot) / norm}
    return {"mode": job["mode"], "rays": total, "fates": fates, "segments": int(seg.p0.shape[0]),
            "valid_segments": int(seg.valid.sum()), "widths": out, "seconds": time.time() - t0}


def run_defect(sim, archive, out_dir, opt, widths, log=print):
    cfg = sim.cfg
    cap = cfg.capillary
    k = float(sim.lines[0].k)
    b5 = cfg.validate_b5_estimator()
    screen = [cap.screen, *cap.screens][b5["screen_index"]]
    index = rays_v3.load_index(archive)
    modes = index.modes("capillary")
    if opt["modes"] is not None:
        modes = modes[:opt["modes"]]
    q0 = np.sqrt(complex(-2.0 * sim.delta_f, 2.0 * sim.beta_f))
    if q0.imag < 0:
        q0 = -q0
    a_w = float(cap.bores[0]["radius"]) + (1j / (k * q0)).real
    area = sum(math.pi * float(b["radius"]) ** 2 for b in cap.bores)
    meta = rays_v3.read_fingerprint(archive)["geometry"]
    rec = dict(meta.get("screen") or {}); rec.update((meta.get("capillary") or {}).get("screen") or {})
    base = dict(archive=archive, raw_cfg=cfg.raw, rays=opt["rays"], sections=opt["sections"], samples=int(opt["samples"]),
                ring_step=float(opt["ring_step"]), cutoff=float(opt["cutoff"]), k=k, fresnel=(2.0 * sim.delta_f, 2.0 * sim.beta_f),
                a_w=a_w, z_exit=float(cap.z1), z_target=float(screen.z), z_rec=float(rec["z"]), area=area, widths=list(widths))
    jobs = [dict(base, mode=m, parts=tuple(secs)) for m, secs in enumerate(modes)]
    out = os.path.join(out_dir, RESULT_DIR)
    os.makedirs(out, exist_ok=True)
    d0 = {f"{w*1e6:g}": entrance_defect(a_w, w) for w in widths}
    t0 = time.time()
    if opt["jobs"] > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=opt["jobs"]) as pool:
            results = list(pool.map(mode_job, jobs, chunksize=1))
    else:
        results = [mode_job(j) for j in jobs]
    rays_prod = int(max(b5.get("map_ray_budgets") or [b5["rays_per_mode"]]))
    summary = {"archive": archive, "options": opt, "screen_z_m": float(screen.z), "a_w_m": a_w,
               "rays_per_mode_production": rays_prod, "widths": []}
    with open(os.path.join(out, "modes.jsonl"), "w", encoding="utf-8") as rows:
        for res in results:
            scale = math.sqrt(res["rays"] / rays_prod)        # ray noise rescaled to the map's budget
            for key, terms in res["widths"].items():
                terms["D0"] = d0[key]
                terms["indicator"] = d0[key] + terms["R"] + terms["G"] + terms["T"]
                terms["indicator_prod"] = d0[key] + sum(math.hypot(terms[t], scale * terms[t + "_shot"]) for t in ("R", "G", "T"))
            rows.write(json.dumps(res) + "\n")
    for w in widths:
        key = f"{w*1e6:g}"
        rms = lambda term: float(np.sqrt(np.mean([r["widths"][key][term] ** 2 for r in results])))
        rec_w = {"width_m": w, "D0": d0[key], "indicator_rms": rms("indicator"), "indicator_prod_rms": rms("indicator_prod"),
                 "indicator_max": float(max(r["widths"][key]["indicator"] for r in results))}
        for term in ("R", "R_shot", "R_inside", "G", "G_shot", "T", "T_shot"):
            rec_w[term + "_rms"] = rms(term)
        summary["widths"].append(rec_w)
        log(f"  width {w*1e6:g} um: D0 {d0[key]:.4f} R rms {rec_w['R_rms']:.4f} (noise {rec_w['R_shot_rms']:.4f}, inside {rec_w['R_inside_rms']:.4f}) "
            f"G rms {rec_w['G_rms']:.4f} (noise {rec_w['G_shot_rms']:.4f}) T rms {rec_w['T_rms']:.4f} (noise {rec_w['T_shot_rms']:.4f}) "
            f"-> indicator rms {rec_w['indicator_rms']:.4f}, at the map's rays {rec_w['indicator_prod_rms']:.4f}")
    best = min(summary["widths"], key=lambda r: r["indicator_prod_rms"])
    summary["argmin"] = {"width_m": best["width_m"], "indicator_rms": best["indicator_rms"],
                         "indicator_prod_rms": best["indicator_prod_rms"]}
    summary["seconds"] = time.time() - t0
    summary["modes"] = len(results)
    with open(os.path.join(out, "defect.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage 17 canonical-map defect indicator and width scan")
    ap.add_argument("config"); ap.add_argument("archive"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--widths", default=None, help="window widths in um, comma separated (default: b5_estimator.widths_m)")
    ap.add_argument("--rays", type=int, default=None); ap.add_argument("--sections", type=int, default=None)
    ap.add_argument("--modes", type=int, default=None); ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None, help="residual sample points per disc and section")
    ap.add_argument("--cutoff", type=float, default=None, help="window support in widths")
    args = ap.parse_args(argv)
    from .. import load
    from ..simulation import Simulation
    cfg = load(args.config)
    sim = Simulation(cfg)
    opt = options(cfg)
    for key in ("rays", "sections", "modes", "jobs", "samples", "cutoff"):
        if getattr(args, key) is not None:
            opt[key] = getattr(args, key)
    if cfg.capillary is None:
        raise ValueError("b5_defect needs a capillary scene")
    widths = ([float(v) * 1e-6 for v in args.widths.split(",")] if args.widths
              else [float(v) for v in cfg.validate_b5_estimator()["widths_m"]])
    t0 = time.time()
    summary = run_defect(sim, os.path.abspath(args.archive), args.out, opt, widths)
    best = summary["argmin"]
    print(f"argmin indicator: width {best['width_m']*1e6:g} um (rms at the map's rays {best['indicator_prod_rms']:.4f}); {time.time() - t0:.0f} s; "
          f"rows in {os.path.join(args.out, RESULT_DIR)}")


if __name__ == "__main__":
    main()

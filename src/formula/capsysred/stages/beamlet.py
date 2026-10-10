"""Stage 11: beamlet (Gaussian-beam summation) estimator, float64.

Each ray becomes a Gaussian beamlet: the central ray is
the engine trace; the complex 2x2 beam tensor Q = Gamma^-1 rides the
segments and bounces by tensor ABCD (gamma.py, Arnaud-Kogelnik general
astigmatism — a bounce is the wall's projected curvature tensor plus the
mirror flip; skew bounces couple the planes). On the screen the beamlet
deposits an elliptic phase spot over the pixels inside the window_sigmas
ellipse's bounding box instead of one bin. Implicit bores carry no
closed-form curvature or normal: their bounces are no-ops (the scalar-q
model of stage 11a, exact for an isotropic launch on flat walls).

The estimator is honest — no ray self-pair subtraction: the mode field is a
true coherent sum of beamlet fields, mu = |W| / sqrt(I*I_ref) with
I = sum_m w_m |g_m|^2 as is. The rng stream matches the other estimators,
so the rays are the stage-2/14 rays.

One path for any CAPSYSRED_STAGE11_JOBS: the unit of work is a mode, each
mode's float64 W/I contribution enters an exact fixed-point accumulator (so
the totals do not depend on the order of completion), the float32 delete-one
rows go to a disk store, and the jackknife runs natively over pixel tiles.
jobs = 1 runs the same functions in-process; mu-beamlet.jsonl and the SVGs
are byte-identical for every jobs value.
"""

import cmath
import concurrent.futures
import copy
import ctypes
import json
import math
import mmap
import os
import shutil
import sys
import threading
import time
from array import array
from types import SimpleNamespace
from typing import NamedTuple

from .altcoh import FloatLineAmplitudes
from ..gamma import EXACT_KINDS, bounce_lenses, inv2, propagate
from ..native import exact_accumulator, jackknife_tile, make_beamlet_grid
from .. import rays_v3
from ..shared.progress import Progress
from ..rays import (geometry_metadata, metadata_equal, require_full_rows,
                    scene_stream)
from ..screen import ScreenGrid
from ..shared.utils import zeros

STAGING_PREFIX = "stage11-staging-"
TMP_PREFIX = "stage11-tmp-"
BACKUP_SUFFIX = ".prev"
PID_PREFIX = "pid-"
# ProcessPoolExecutor on Windows is limited by WaitForMultipleObjects.
_MAX_NT_WORKERS = 61
# Row-store bytes held per jackknife tile (12 B per pixel per mode, one copy).
_TILE_BYTES = 256 << 20
# Resource model: Python per worker, master accumulator, buffers.
_R_PY = 400 << 20
_ACC_BYTES_PER_PIXEL = 816
_PREFLIGHT_MARGIN = 1.3


def _flat_walls(optic) -> bool:
    walls = getattr(optic, "walls", None)
    return optic is not None and (
        walls is None or any(w.kind not in EXACT_KINDS for w in walls))


class BeamletField:
    """Per-mode complex beamlet fields per spectral line; honest mu totals."""

    def __init__(self, lines, screen: ScreenGrid, ref_pixel: int,
                 w0: float, n_sigmas: float, optic=None, use_native=True,
                 w0_t=None, waist_dz=0.0):
        self.kms = [float(l.k) for l in lines]
        # a beamlet fan sums to lambda * (point-source field): (k/k0)^2
        # restores the configured line weights in W and I
        self.wfs = [l.weight * (k / self.kms[0]) ** 2
                    for l, k in zip(lines, self.kms)]
        self.nl = len(self.kms)
        self.zrs = [0.5 * w0 * w0 * k for k in self.kms]   # z_R = w0^2*k/2
        self.w0, self.ns = w0, n_sigmas
        # anisotropic launch: w0_t along x, w0 along y, the same for every ray
        self.w0_t = w0 if w0_t is None else float(w0_t)
        self.zrt = [0.5 * self.w0_t * self.w0_t * k for k in self.kms]
        # waist plane waist_dz past the source: a virtual drift -waist_dz
        # before the first segment; its on-axis factor is divided out
        self.waist_dz = float(waist_dz)
        self.vfix = [(cmath.sqrt(1 + 1j * self.waist_dz / zt)
                      * cmath.sqrt(1 + 1j * self.waist_dz / zs)).conjugate()
                     for zt, zs in zip(self.zrt, self.zrs)]
        self.optic = optic
        walls = getattr(optic, "walls", None)
        self.flat_walls = _flat_walls(optic)
        self.curved = any(w.kind in EXACT_KINDS and w.kind != "polygon"
                          for w in walls or ())
        self.ref = ref_pixel
        self.grid = screen
        # delete-one cut: float32 rows leave ~nl*2^-24 relative residue
        # where exact 0 is due; x16 headroom, ~1e-6 at nl = 1
        self.loo_eps = (16 + self.nl) * 2.0 ** -24
        self.nx, self.ny = screen.nx, screen.ny
        self.zf = float(screen.z)
        self.xs = screen.xs()
        self.ys = screen.ys()
        self.dx = screen.exf / screen.nx
        self.dy = screen.eyf / screen.ny
        self.x0f, self.y0f = screen.x0f, screen.y0f
        # mu comes from the float64 totals; the delete-one-mode rows are
        # dense float32 arrays (sigma is statistical, 7 digits are plenty):
        # 12 B/pixel/mode
        self.W = {}         # pixel -> complex float64 total
        self.I = {}         # pixel -> float float64 total
        self.Ws = []        # [mode] array('f') interleaved re, im
        self.Is = []        # [mode] array('f')
        self.i_refs = []    # [mode] float32-consistent intensity at ref
        # arrival-point bins (uint32 per pixel); the shared path exports and
        # zeroes it per mode, the fold/finalize oracle keeps it running
        self.density = array("I", bytes(4 * self.nx * self.ny))
        self.w_sum, self.w_n = 0.0, 0   # mean spot width at screen (line 0)
        self.gamma_bad = 0  # deposits skipped: Im(G) lost negative-definiteness
        self.native = (make_beamlet_grid(screen.nx, screen.ny,
                                         screen.x0f, screen.y0f,
                                         screen.exf, screen.eyf,
                                         self.kms, self.zrs, self.zrt,
                                         n_sigmas)
                       if use_native else None)
        self._g = None

    def new_mode(self):
        if self.native is not None:
            self.native.clear()
        else:
            self._g = [{} for _ in range(self.nl)]

    def prep(self, rec, zf=None):
        """The shared per-record work: floats, segments to the plane zf (the
        recorded arrival plane; default this one), wall lenses. Extra planes
        shift the result arithmetically (straight final flight) instead of
        re-doing it."""
        zf = self.zf if zf is None else zf
        opl = float(rec.opl)
        x, y = float(rec.point[0]), float(rec.point[1])
        dxf, dyf = float(rec.direction[0]), float(rec.direction[1])
        dzf = float(rec.direction[2])
        pts = [tuple(float(c) for c in p) for p in rec.refl]
        if pts:
            segs = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
            segs.append(math.dist(pts[-1], (x, y, zf)))
            segs.insert(0, max(opl - sum(segs), 0.0))
            # outgoing direction per bounce (curved walls only): toward the
            # next hit, the last one the recorded final direction
            outs = [None] * len(pts)
            if self.curved:
                outs = [tuple(q - p for p, q in zip(a, b))
                        for a, b in zip(pts, pts[1:])]
                outs.append((dxf, dyf, dzf))
            lenses = bounce_lenses(self.optic, pts, outs)
        else:
            segs, lenses = [opl], []
        # one launch frame for every ray: GBS needs a ray-independent Q0
        return x, y, dxf, dyf, dzf, opl, 0.0, segs, lenses

    def add_ray(self, rec, amps):
        """Deposit one beamlet; rec.pixel None = tail only (the center lies
        outside the window), still deposited. The spot is elliptic from
        G = Q^-1 at the screen: phase = tilt + (k/2)·δᵀRe(G)δ, envelope
        exp((k/2)·δᵀIm(G)δ) — without the tilt term k*(d_x*δx + d_y*δy) the
        beamlets of one point source disagree at a pixel and the spherical
        front never reconstructs."""
        x, y, dxf, dyf, _, opl, psi, segs, lenses = self.prep(rec)
        self.deposit(x, y, dxf, dyf, opl, psi, segs, lenses, amps, rec.pixel)

    def deposit(self, x, y, dxf, dyf, opl, psi, segs, lenses, amps, pixel):
        if pixel is not None:
            self.density[pixel] += 1
        if self.waist_dz:
            segs = [-self.waist_dz] + segs
            lenses = [(math.nan, 0.0, 0.0, 0.0)] + lenses
            amps = [a * f for a, f in zip(amps, self.vfix)]
        if self.native is not None:
            w_spot, bad = self.native.add_ray(
                x, y, dxf, dyf, opl, psi, segs,
                [v for lens in lenses for v in lens], list(amps))
            self.gamma_bad += bad
            if w_spot >= 0.0:
                self.w_sum += w_spot
                self.w_n += 1
            return
        for m in range(self.nl):
            km = self.kms[m]
            q, a_geo = propagate((self.zrt[m], self.zrs[m], psi), segs, lenses)
            gm = inv2(q)
            gi = (gm[0].imag, gm[1].imag, gm[2].imag)
            mean = 0.5 * (gi[0] + gi[2])
            dev = math.hypot(0.5 * (gi[0] - gi[2]), gi[1])
            if mean + dev >= 0.0:      # beam blew up: no Gaussian to deposit
                self.gamma_bad += 1
                continue
            w_hi = math.sqrt(-2.0 / (km * (mean + dev)))   # widest spot axis
            if m == 0:
                w_lo = math.sqrt(-2.0 / (km * (mean - dev)))
                self.w_sum += math.sqrt(w_hi * w_lo)
                self.w_n += 1
            pref = amps[m] * a_geo.conjugate() * cmath.exp(1j * km * opl)
            tx, ty = km * dxf, km * dyf
            hxx, hxy, hyy = (0.5 * km * v for v in gm)   # (k/2)·G, complex
            # per-axis bounding box of the ns-sigma ellipse (det > 0 here)
            det_gi = gi[0] * gi[2] - gi[1] * gi[1]
            rx = self.ns * math.sqrt(-2.0 * gi[2] / (km * det_gi))
            ry = self.ns * math.sqrt(-2.0 * gi[0] / (km * det_gi))
            ix_lo = max(0, int(math.floor((x - rx - self.x0f) / self.dx)))
            ix_hi = min(self.nx - 1, int(math.floor((x + rx - self.x0f) / self.dx)))
            iy_lo = max(0, int(math.floor((y - ry - self.y0f) / self.dy)))
            iy_hi = min(self.ny - 1, int(math.floor((y + ry - self.y0f) / self.dy)))
            if ix_lo > ix_hi or iy_lo > iy_hi:
                continue
            g = self._g[m]
            for iy in range(iy_lo, iy_hi + 1):
                dy_off = self.ys[iy] - y
                cy = hyy * (dy_off * dy_off)
                phase_y = ty * dy_off
                row = iy * self.nx
                for ix in range(ix_lo, ix_hi + 1):
                    dx_off = self.xs[ix] - x
                    quad = (hxx * (dx_off * dx_off)
                            + 2.0 * hxy * (dx_off * dy_off) + cy)
                    val = pref * cmath.exp(complex(
                        quad.imag, quad.real + tx * dx_off + phase_y))
                    pix = row + ix
                    prev = g.get(pix)
                    g[pix] = val if prev is None else prev + val

    def fold_mode(self):
        """Oracle path (tests): fold into the retained native rows/totals,
        or into the Python dicts without the native grid."""
        if self.native is not None:
            # scan, totals AND the delete-one rows all stay in C++
            self.native.fold(self.wfs, self.ref)
            return
        npix = self.nx * self.ny
        w_row = array("f", bytes(8 * npix))
        i_row = array("f", bytes(4 * npix))
        for m in range(self.nl):
            wf = self.wfs[m]
            g = self._g[m]
            g_ref = g.get(self.ref)
            ref_c = g_ref.conjugate() if g_ref is not None else None
            for pixel, value in g.items():
                a2 = value.real * value.real + value.imag * value.imag
                self.I[pixel] = self.I.get(pixel, 0.0) + wf * a2
                i_row[pixel] += wf * a2
                if ref_c is not None:
                    cross = value * ref_c
                    self.W[pixel] = self.W.get(pixel, 0j) + wf * cross
                    w_row[2 * pixel] += wf * cross.real
                    w_row[2 * pixel + 1] += wf * cross.imag
        self.Ws.append(w_row)
        self.Is.append(i_row)
        self.i_refs.append(float(i_row[self.ref]))
        self._g = None

    def fold_export(self):
        """Shared path: one mode's (w_row f32, i_row f32, dW f64, dI f64,
        i_ref) exactly as fold builds the rows; nothing is retained."""
        if self.native is None:
            raise RuntimeError("fold_export needs the native BeamletGrid")
        return self.native.fold_export(self.wfs, self.ref)

    def take_density(self) -> bytes:
        """The mode's arrival counts (uint32 per pixel), then zeroed."""
        out = self.density.tobytes()
        self.density = array("I", bytes(4 * self.nx * self.ny))
        return out

    def _density_rows(self, nx: int, ny: int):
        density = [[0.0] * nx for _ in range(ny)]
        for pixel, count in enumerate(self.density):
            if count:
                iy, ix = divmod(pixel, nx)
                density[iy][ix] = float(count)
        return density

    def finalize(self, nx: int, ny: int):
        """Row-major [iy][ix] maps: mu, mu_err (delete-one-mode jackknife),
        dubious, intensity, density. No self-pair subtraction, so I itself
        is the mu denominator and every lit pixel is estimable; the trust
        flags are sigma > 1, |mu| pinned at 1 or 0 with sigma below loo_eps,
        fewer than 2 usable leave-one-out modes, or an unlit reference."""
        if self.native is not None:
            return self._finalize_native(nx, ny)
        n_modes = len(self.Ws)
        W, I = self.W, self.I
        i_ref = I.get(self.ref, 0.0)
        mu, err, dubious = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
        intensity = zeros(nx, ny)
        for pixel, value in I.items():
            iy, ix = divmod(pixel, nx)
            intensity[iy][ix] = value
        density = self._density_rows(nx, ny)
        Ws, Is, i_refs, eps = self.Ws, self.Is, self.i_refs, self.loo_eps
        if not i_ref > 0.0:   # unlit reference: no pixel is estimable
            for pixel, i_pix in I.items():
                if i_pix > 0.0:
                    iy, ix = divmod(pixel, nx)
                    dubious[iy][ix] = 1.0
        else:
            for pixel, i_pix in I.items():
                if i_pix <= 0.0:
                    continue
                w = W.get(pixel, 0j)
                iy, ix = divmod(pixel, nx)
                mu[iy][ix] = min(abs(w) / math.sqrt(i_pix * i_ref), 1.0)
                wr, wi = w.real, w.imag
                # sole-mode pixels are skipped; the cut is loo_eps-relative
                eps_i, eps_r = self.loo_eps * i_pix, self.loo_eps * i_ref
                loo = []
                for s in range(n_modes):
                    i_s = i_pix - Is[s][pixel]
                    iref_s = i_ref - i_refs[s]
                    if i_s > eps_i and iref_s > eps_r:
                        row_s = Ws[s]
                        dr = wr - row_s[2 * pixel]
                        di = wi - row_s[2 * pixel + 1]
                        loo.append(min(math.hypot(dr, di)
                                       / math.sqrt(i_s * iref_s), 1.0))
                if len(loo) > 1:
                    mean = sum(loo) / len(loo)
                    err[iy][ix] = math.sqrt(
                        sum((v - mean) ** 2 for v in loo)
                        * (len(loo) - 1) / len(loo))
                if (err[iy][ix] > 1.0 or len(loo) < 2
                        or ((mu[iy][ix] >= 1.0 - eps or mu[iy][ix] == 0.0)
                            and err[iy][ix] <= eps)):
                    dubious[iy][ix] = 1.0
        w_mean = self.w_sum / self.w_n if self.w_n else 0.0
        return {"mu": mu, "mu_err": err, "dubious": dubious,
                "intensity": intensity, "density": density,
                "ref_pixel": self.ref, "i_ref": i_ref, "w_mean": w_mean,
                "gamma_bad": self.gamma_bad, "flat_walls": self.flat_walls}

    def _finalize_native(self, nx: int, ny: int):
        """Oracle path: totals and the delete-one scan live in C++; here
        the dense results are only unpacked into row lists."""
        i_tot = array("d")
        i_tot.frombytes(self.native.totals()[1])
        mu_b, err_b, dub_b = self.native.jackknife(self.ref, self.loo_eps)
        mu_t = array("d")
        mu_t.frombytes(mu_b)
        err_t = array("d")
        err_t.frombytes(err_b)
        rows = lambda a: [list(a[iy * nx:(iy + 1) * nx]) for iy in range(ny)]
        w_mean = self.w_sum / self.w_n if self.w_n else 0.0
        return {"mu": rows(mu_t), "mu_err": rows(err_t),
                "dubious": rows(array("d", (float(b) for b in dub_b))),
                "intensity": rows(i_tot), "density": self._density_rows(nx, ny),
                "ref_pixel": self.ref, "i_ref": i_tot[self.ref],
                "w_mean": w_mean, "gamma_bad": self.gamma_bad,
                "flat_walls": self.flat_walls}


def _reference_pixel(grid, reference):
    xy = reference if reference else (grid.cxf, grid.cyf)
    pix = grid.pixel(xy)
    if pix is None:
        raise ValueError(f"screen reference {tuple(xy)} lies outside the window")
    return pix


def _scene_core(geometry: dict, scene: str) -> dict:
    """A scene's recorded source and bores, without screens and budgets."""
    core = copy.deepcopy(geometry.get(scene) or {})
    core.pop("screen", None)
    core.pop("screens", None)
    source = core.get("source")
    if isinstance(source, dict):
        source.pop("n_modes", None)
        source.pop("n_rays", None)
    return core


def recorded_plane(sim, scene, scr_cfg) -> float:
    """z of the recorded arrival plane. Every recorded part's scene geometry
    must match the config (screens, budgets and precision may differ), and
    the parts of a union must share one plane."""
    want = _scene_core(geometry_metadata(sim.cfg), scene)
    planes = set()
    for part in getattr(sim.rays, "parts", [sim.rays]):
        geo = part.meta.get("geometry", {})
        if scene in geo and not metadata_equal(_scene_core(geo, scene), want):
            raise ValueError(f"{part.path}: {scene} trace geometry differs "
                             "from the config")
        rec = {**geo.get("screen", {}), **(geo.get(scene) or {}).get("screen", {})}
        planes.add(float(rec.get("z", scr_cfg.z)))
    if len(planes) != 1:
        raise ValueError(f"{sim.rays.path}: union parts recorded {scene} "
                         f"arrivals on different planes {sorted(planes)}")
    return planes.pop()


# ------------------------------------------------------------ shared path

class ModeTask(NamedTuple):
    """One mode of one archive part: the unit of work."""
    part: str
    local_mode: int
    mode: int
    sections: tuple


def _mode_tasks(rays, scene):
    """Every mode of the scene across the reader's parts; a union offsets
    the mode ids like MultiRaysReader."""
    tasks, offset = [], 0
    for part in getattr(rays, "parts", [rays]):
        index = getattr(part, "index", None)
        if index is None:
            raise ValueError(f"{part.path}: stage 11 reads per-mode v3 archives "
                             "only; convert the recording with convert_rays_v3")
        modes = index.modes(scene)
        for local, sections in enumerate(modes):
            tasks.append(ModeTask(part.path, local, offset + local, tuple(sections)))
        offset += len(modes)
    return tasks


def _screen_spec(scr) -> dict:
    reference = getattr(scr, "reference", None)
    return {"z": float(scr.z), "nx": int(scr.nx), "ny": int(scr.ny),
            "center": [float(scr.center[0]), float(scr.center[1])],
            "edge_x": float(scr.edge_x), "edge_y": float(scr.edge_y),
            "reference": None if not reference else
            [float(reference[0]), float(reference[1])]}


def _scene_source(sim, scene):
    cfg = sim.cfg
    return cfg.free_source if scene == "free" else cfg.capillary.source


def _scene_optic(sim, scene):
    if scene != "capillary":
        return None
    from ..surfaces import CapillaryBundle
    cap = sim.cfg.capillary
    return CapillaryBundle(cap.bores, cap.z0, cap.z1)


def _launch(sim, scene, plane_z: float):
    """(w0_t, waist_dz) of the scene's beamlet launch, from the typed config."""
    cfg = sim.cfg
    z_src = float(_scene_source(sim, scene).position[2])
    w0_t = cfg.beamlet_w0_t
    if w0_t == "auto":   # Fresnel scale of the scene's source->screen flight
        w0_t = math.sqrt(float(sim.lam) * (plane_z - z_src) / math.pi)
    waist_dz = 0.0 if cfg.beamlet_waist_z is None else cfg.beamlet_waist_z - z_src
    return (None if w0_t is None else float(w0_t)), float(waist_dz)


def _stage11_contract(sim, scene, planes, n_modes) -> dict:
    """Everything the deposit depends on, from the typed config: the worker
    rebuilds it from cfg.raw and must obtain the same dict, else the run
    refuses (a programmatic drift between raw and typed config)."""
    from .stage14 import _physics_contract
    cfg = sim.cfg
    w0_t, waist_dz = _launch(sim, scene, planes[0]["z"])
    return {"scene": scene, "n_modes": int(n_modes), "planes": planes,
            "physics": _physics_contract(sim),
            "beamlet": {"w0": float(cfg.beamlet_w0), "w0_t": w0_t,
                        "ns": float(cfg.beamlet_ns), "waist_dz": waist_dz,
                        "loo_eps": (16 + len(sim.lines)) * 2.0 ** -24},
            "amplitude_min": float(cfg.amplitude_min)}


def _blank():
    return {"emitted": 0, "screen": 0, "absorbed": 0, "lost": 0,
            "off_window": 0}


def _memory_info(pid: int):
    """(peak working set, current working set) of a process on Windows;
    (0, 0) elsewhere or when the process cannot be opened."""
    if os.name != "nt":
        return 0, 0

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.OpenProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    handle = k32.OpenProcess(0x1000, False, int(pid))   # QUERY_LIMITED_INFORMATION
    if not handle:
        return 0, 0
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return int(pmc.PeakWorkingSetSize), int(pmc.WorkingSetSize)
        return 0, 0
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


def _peak_rss() -> int:
    """Peak working set of this process in bytes (0 where unsupported)."""
    if os.name == "nt":
        return _memory_info(os.getpid())[0]
    try:
        import resource
    except ImportError:
        return 0
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _working_set(pid: int) -> int:
    """Current working set of another process (Windows), 0 elsewhere."""
    return _memory_info(pid)[1]


class PeakSampler(threading.Thread):
    """1 Hz sum of the working sets of the master and every worker that
    announced itself with a pid-<pid> marker in a watched directory."""

    def __init__(self):
        super().__init__(daemon=True)
        self.pids = {os.getpid()}
        self.dirs = set()
        self.max_total = 0
        self.max_workers = 0
        self._halt = threading.Event()
        self._lock = threading.Lock()

    def watch_dir(self, path):
        with self._lock:
            self.dirs.add(path)

    def _discover(self):
        with self._lock:
            dirs = list(self.dirs)
        for folder in dirs:
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            for name in names:
                if name.startswith(PID_PREFIX):
                    try:
                        self.pids.add(int(name[len(PID_PREFIX):]))
                    except ValueError:
                        pass

    def sample(self):
        self._discover()
        own = _working_set(os.getpid())
        workers = [_working_set(pid) for pid in list(self.pids) if pid != os.getpid()]
        self.max_total = max(self.max_total, own + sum(workers))
        if workers:
            self.max_workers = max(self.max_workers, max(workers))

    def run(self):
        while not self._halt.wait(1.0):
            self.sample()

    def stop(self):
        self._halt.set()
        self.join()
        self.sample()


class _Worker:
    """Per-process state: the simulation rebuilt from cfg.raw and checked
    against the parent's contract, one BeamletField per plane, the verified
    reader and the row store; run() deposits one mode."""

    def __init__(self, raw_cfg, contract, z_rec, rows_paths, tmp_dir):
        from ..simulation import Simulation
        sim = Simulation.from_dict(raw_cfg)
        scene, planes = contract["scene"], contract["planes"]
        mine = _stage11_contract(sim, scene, planes, contract["n_modes"])
        if not metadata_equal(mine, contract):
            raise ValueError("stage-11 worker configuration differs from the "
                             "parent's contract (typed config changed after "
                             "loading, or the spectrum table drifted)")
        cfg = sim.cfg
        self.optic = _scene_optic(sim, scene)
        self.amps_of = FloatLineAmplitudes(cfg.material, sim.lines, cfg.precision)
        self.amplitude_min = contract["amplitude_min"]
        self.z_rec = float(z_rec)
        launch = contract["beamlet"]
        self.fields = []
        for spec in planes:
            grid = ScreenGrid(SimpleNamespace(**spec))
            self.fields.append(BeamletField(
                sim.lines, grid, _reference_pixel(grid, spec["reference"]),
                launch["w0"], launch["ns"], self.optic, w0_t=launch["w0_t"],
                waist_dz=launch["waist_dz"]))
            if self.fields[-1].native is None:
                raise RuntimeError("stage 11 needs the native BeamletGrid; "
                                   "rebuild the _formula extension")
        self.dz = [float(f.zf) - self.z_rec for f in self.fields]
        self.files = [open(p, "r+b") for p in rows_paths]
        self.maps = [mmap.mmap(f.fileno(), 0) for f in self.files]
        self.tmp_dir = tmp_dir

    def close(self):
        for m in self.maps:
            m.close()
        for f in self.files:
            f.close()
        self.maps, self.files, self.fields = [], [], []

    def run(self, task: ModeTask) -> dict:
        fields, dz_all = self.fields, self.dz
        for field in fields:
            field.new_mode()
        stats = [_blank() for _ in fields]
        amps_of, amin, first = self.amps_of, self.amplitude_min, fields[0]
        n = 0
        for entry in task.sections:
            for line in rays_v3.iter_section_lines(task.part, entry):
                rec = rays_v3._record(json.loads(line))
                n += 1
                fate, amps = rec.fate, None
                if fate == "screen":
                    amps = amps_of([float(s) for s in rec.sins])
                    if amin > 0.0 and max(abs(a) for a in amps) < amin:
                        fate = "absorbed"
                if fate == "screen":
                    # tail beamlets (center outside a window) still deposit there
                    x, y, dxf, dyf, dzf, opl, psi, segs, lenses = first.prep(
                        rec, self.z_rec)
                    for field, dz, st in zip(fields, dz_all, stats):
                        st["emitted"] += 1
                        if dz == 0.0:
                            xi, yi, opl_i, segs_i = x, y, opl, segs
                            # rec.pixel indexes the recording screen: re-bin here
                            pix = field.grid.pixel((x, y))
                        else:
                            step = dz / dzf
                            xi, yi = x + dxf * step, y + dyf * step
                            opl_i = opl + step
                            segs_i = ([opl_i] if not lenses and len(segs) == 1
                                      else segs[:-1] + [segs[-1] + step])
                            pix = field.grid.pixel((xi, yi))
                        field.deposit(xi, yi, dxf, dyf, opl_i, psi, segs_i, lenses,
                                      amps, pix)
                        st["screen" if pix is not None else "off_window"] += 1
                else:
                    for st in stats:
                        st["emitted"] += 1
                        st[fate] += 1
        out = {"mode": task.mode, "rows": n, "stats": stats, "w": [],
               "gamma_bad": [], "i_ref": [], "pid": os.getpid()}
        for k, field in enumerate(fields):
            w_row, i_row, dw, di, i_ref = field.fold_export()
            npix = field.nx * field.ny
            base = task.mode * 12 * npix
            store = self.maps[k]
            store[base:base + 8 * npix] = w_row
            store[base + 8 * npix:base + 12 * npix] = i_row
            start = base - base % mmap.PAGESIZE      # msync wants page alignment
            store.flush(start, 12 * npix + base - start)
            part = os.path.join(self.tmp_dir, f"s{k}-m{task.mode:06d}.part")
            with open(part + ".tmp", "wb") as fh:
                fh.write(dw)
                fh.write(di)
                fh.write(field.take_density())
            os.replace(part + ".tmp", part)
            out["w"].append((field.w_sum, field.w_n))
            out["gamma_bad"].append(field.gamma_bad)
            out["i_ref"].append(i_ref)
            field.w_sum, field.w_n, field.gamma_bad = 0.0, 0, 0
        out["peak_rss"] = _peak_rss()
        return out


_WORKER = None


def _worker_init(raw_cfg, contract, z_rec, rows_paths, tmp_dir):
    global _WORKER
    # announce the pid first: the sampler must see the worker before its
    # first mode completes
    with open(os.path.join(tmp_dir, f"{PID_PREFIX}{os.getpid()}"), "w"):
        pass
    _WORKER = _Worker(raw_cfg, contract, z_rec, rows_paths, tmp_dir)


def _worker_run(task):
    return _WORKER.run(task)


def _effective_jobs(jobs: int, n_modes: int) -> int:
    workers = max(1, min(int(jobs), n_modes))
    if os.name == "nt":
        workers = min(workers, _MAX_NT_WORKERS)
    return workers


def _tile_jackknife(rows_path, n_modes, npix, w_b, i_b, i_ref, irefs, eps, band):
    """mu, sigma, dubious and LOO-boundary flags over the pixel tiles of one
    row store; one tile buffer (12 B per pixel per mode) is held at a time."""
    tile = max(1, min(npix, _TILE_BYTES // (12 * max(1, n_modes))))
    mu, err = array("d", bytes(8 * npix)), array("d", bytes(8 * npix))
    dub, boundary = bytearray(npix), bytearray(npix)
    w_view, i_view = memoryview(w_b), memoryview(i_b)
    # one buffer pair for the whole scan; a shorter last tile uses a prefix
    # view of the same buffers (mode-major with the tile's own stride)
    w_rows, i_rows = bytearray(8 * n_modes * tile), bytearray(4 * n_modes * tile)
    with open(rows_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for p0 in range(0, npix, tile):
            p1 = min(npix, p0 + tile)
            n = p1 - p0
            for s in range(n_modes):
                base = s * 12 * npix
                w_rows[s * 8 * n:(s + 1) * 8 * n] = mm[base + 8 * p0:base + 8 * p1]
                i_rows[s * 4 * n:(s + 1) * 4 * n] = mm[base + 8 * npix + 4 * p0:
                                                       base + 8 * npix + 4 * p1]
            mu_t, err_t, dub_t, bnd_t = jackknife_tile(
                w_view[16 * p0:16 * p1], i_view[8 * p0:8 * p1], i_ref,
                memoryview(w_rows)[:8 * n_modes * n],
                memoryview(i_rows)[:4 * n_modes * n], irefs, eps, band)
            mu[p0:p1] = array("d", mu_t)
            err[p0:p1] = array("d", err_t)
            dub[p0:p1] = dub_t
            boundary[p0:p1] = bnd_t
    return mu, err, dub, boundary


def run_beamlet_stage(sim, label, scene, src_cfg, scr_cfg, optic, aim_factory,
                      seed_offset: int, extra_screens=(), work_dir=None,
                      jobs: int = 1, sampler=None):
    """The beamlet deposit over the scene's recorded modes, one path for any
    jobs: modes are the work units, their float64 W/I contributions enter
    exact accumulators, the float32 delete-one rows a disk store under
    work_dir, and the jackknife runs natively over pixel tiles.

    ONE pass serves every screen: the per-record prep (parse, segments, wall
    lenses) is shared, and each plane only shifts the straight final flight
    from the recorded arrival plane — arrival x + dx*s, opl + s, last
    segment + s with s = (z_i - z_rec)/dz. The launch is the same for every
    ray: waist w0 (w0_t along x) at z = beamlet.waist_z, the source plane by
    default. Returns the main-plane result with the extra planes under
    "extras"; "seconds" is the wall-clock of this whole call."""
    t_entry = time.time()
    cfg = sim.cfg
    cfg.validate_beamlet()
    n_modes, n_rays = src_cfg.budget()
    # the reader's budget/presence checks; the records themselves stream
    # per mode inside the workers
    _, rays_from = scene_stream(sim, scene, src_cfg, scr_cfg, optic,
                                aim_factory, seed_offset)
    require_full_rows(sim.rays, rays_from, "beamlet stage (refl segments)")
    z_rec = recorded_plane(sim, scene, scr_cfg)
    tasks = _mode_tasks(sim.rays, scene)
    if len(tasks) != n_modes:
        raise ValueError(f"{sim.rays.path}: scene {scene!r} holds {len(tasks)} "
                         f"modes, the budget promises {n_modes}")
    planes = [_screen_spec(s) for s in (scr_cfg, *extra_screens)]
    grids = [ScreenGrid(SimpleNamespace(**spec)) for spec in planes]
    refs = [_reference_pixel(g, spec["reference"]) for g, spec in zip(grids, planes)]
    npix = [g.nx * g.ny for g in grids]
    n_lines = len(sim.lines)
    loo_eps = (16 + n_lines) * 2.0 ** -24
    band = (n_modes * n_lines + 1) * 2.0 ** -53      # LOO-threshold boundary band
    contract = _stage11_contract(sim, scene, planes, n_modes)
    # the workers rebuild the configuration from cfg.raw: refuse here, before
    # any process starts, when the typed config no longer matches it
    from ..simulation import Simulation
    rebuilt = _stage11_contract(Simulation.from_dict(cfg.raw), scene, planes, n_modes)
    if not metadata_equal(rebuilt, contract):
        raise ValueError("stage 11: the typed configuration differs from cfg.raw "
                         "(changed after loading?) — the worker contract would not "
                         "match; reload the configuration")
    jobs_eff = _effective_jobs(jobs, n_modes)
    work_dir = os.getcwd() if work_dir is None else work_dir
    tmp = os.path.join(work_dir, f"{TMP_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}"
                                  f"-{os.getpid()}-{scene}")
    own_sampler = sampler is None
    if own_sampler:
        sampler = PeakSampler()
        sampler.start()
    timers = {"allocation": 0.0, "deposit": 0.0, "merge": 0.0, "jackknife": 0.0}
    os.makedirs(tmp)
    try:
        sampler.watch_dir(tmp)
        rows_paths = [os.path.join(tmp, f"s{k}.rows") for k in range(len(grids))]
        for path, pix in zip(rows_paths, npix):
            with open(path, "wb") as fh:
                fh.truncate(12 * n_modes * pix)
        init_args = (cfg.raw, contract, z_rec, rows_paths, tmp)
        accs = [exact_accumulator(pix) for pix in npix]
        i_refs = [[0.0] * n_modes for _ in grids]
        w_sums = [[0.0] * n_modes for _ in grids]
        w_ns = [[0] * n_modes for _ in grids]
        gamma_bad = [0] * len(grids)
        stats = [_blank() for _ in grids]
        peaks = {}
        rows_done = 0
        progress = Progress(label, n_modes * n_rays)
        timers["allocation"] = time.time() - t_entry
        t_deposit = time.time()

        def merge(res):
            nonlocal rows_done
            t = time.time()
            mode = res["mode"]
            for k in range(len(grids)):
                part = os.path.join(tmp, f"s{k}-m{mode:06d}.part")
                with open(part, "rb") as fh:
                    data = memoryview(fh.read())
                pix = npix[k]
                accs[k].add(data[:16 * pix], data[16 * pix:24 * pix],
                            data[24 * pix:28 * pix])
                data.release()
                os.remove(part)
                i_refs[k][mode] = res["i_ref"][k]
                w_sums[k][mode], w_ns[k][mode] = res["w"][k]
                gamma_bad[k] += res["gamma_bad"][k]
                for key, value in res["stats"][k].items():
                    stats[k][key] += value
            rows_done += res["rows"]
            peaks[res["pid"]] = max(peaks.get(res["pid"], 0), res["peak_rss"])
            timers["merge"] += time.time() - t
            progress.step(n_rays)

        if jobs_eff == 1:
            worker = _Worker(*init_args)
            try:
                for task in tasks:
                    merge(worker.run(task))
            finally:
                worker.close()
            del worker
        else:
            window = 2 * jobs_eff
            pending = set()
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=jobs_eff, initializer=_worker_init,
                    initargs=init_args) as pool:
                try:
                    queue = iter(tasks)
                    for task in queue:
                        pending.add(pool.submit(_worker_run, task))
                        if len(pending) >= window:
                            break
                    while pending:
                        done, pending = concurrent.futures.wait(
                            pending, return_when=concurrent.futures.FIRST_COMPLETED)
                        for fut in done:
                            merge(fut.result())
                            nxt = next(queue, None)
                            if nxt is not None:
                                pending.add(pool.submit(_worker_run, nxt))
                except concurrent.futures.process.BrokenProcessPool as exc:
                    for fut in pending:
                        fut.cancel()
                    raise ValueError(
                        "stage 11: a worker process died while starting or "
                        "running (its traceback is on stderr: configuration "
                        "contract, native library or memory)") from exc
                except BaseException:
                    for fut in pending:
                        fut.cancel()
                    raise
        timers["deposit"] = time.time() - t_deposit - timers["merge"]
        if rows_done != n_modes * n_rays:
            raise ValueError(f"{sim.rays.path}: scene {scene!r} yielded {rows_done} "
                             f"rows of {n_modes * n_rays} recorded — file rewritten "
                             "or truncated")
        progress.finish(f"on screen {stats[0]['screen']:,}")
        t = time.time()
        results = []
        flat_walls = _flat_walls(optic)
        for k, grid in enumerate(grids):
            # totals first, then the accumulator is released before the tiles
            w_b, i_b = accs[k].totals()
            dens = array("Q")
            dens.frombytes(accs[k].density())
            accs[k] = None
            i_tot = array("d")
            i_tot.frombytes(i_b)
            i_ref = i_tot[refs[k]]
            mu, err, dub, bnd = _tile_jackknife(rows_paths[k], n_modes, npix[k],
                                                w_b, i_b, i_ref, i_refs[k], loo_eps,
                                                band)
            del w_b, i_b
            nx, ny = grid.nx, grid.ny
            rows = lambda a: [list(a[iy * nx:(iy + 1) * nx]) for iy in range(ny)]
            total_n = sum(w_ns[k])
            w_mean = math.fsum(w_sums[k]) / total_n if total_n else 0.0
            boundary = [p for p, flag in enumerate(bnd) if flag]
            maps = {"mu": rows(mu), "mu_err": rows(err),
                    "dubious": rows(array("d", (float(b) for b in dub))),
                    "intensity": rows(i_tot),
                    "density": rows(array("d", (float(c) for c in dens))),
                    "ref_pixel": refs[k], "i_ref": i_ref, "w_mean": w_mean,
                    "gamma_bad": gamma_bad[k], "flat_walls": flat_walls,
                    "loo_boundary": {"count": len(boundary), "band_rel": band,
                                     "pixels": boundary[:20]}}
            results.append({"maps": maps, "screen": grid, "stats": stats[k],
                            "rays_from": rays_from,
                            "w0_t": contract["beamlet"]["w0_t"] if
                            contract["beamlet"]["w0_t"] is not None else cfg.beamlet_w0,
                            "n_modes": n_modes, "n_rays": n_rays})
            del mu, err, dub, bnd, dens, i_tot
        timers["jackknife"] = time.time() - t
    finally:
        if own_sampler:
            sampler.stop()
        shutil.rmtree(tmp)
    seconds = time.time() - t_entry
    timers["scene"] = seconds
    diag = {"jobs": jobs_eff, "timers": {k: round(v, 3) for k, v in timers.items()},
            "peak_rss": {"master": _peak_rss(),
                         "worker_max": max(peaks.values(), default=0),
                         "sampled_total_max": sampler.max_total}}
    for res in results:
        res["seconds"] = seconds
        res["diagnostics"] = diag
    out = results[0]
    out["extras"] = results[1:]
    return out


# -------------------------------------------------------- publication

def refuse_stage11_leftovers(out_dir):
    """Leftovers of an earlier run (backups, staging, tmp) stop the stage:
    a complete set of final names does not prove a completed publication,
    so nothing is removed automatically."""
    found = []
    try:
        names = os.listdir(out_dir)
    except FileNotFoundError:
        return
    for name in sorted(names):
        if (name.endswith(BACKUP_SUFFIX) or name.startswith(STAGING_PREFIX)
                or name.startswith(TMP_PREFIX)):
            found.append(name)
    if found:
        raise ValueError(
            f"{out_dir}: leftovers of an earlier stage-11 run: {', '.join(found)}. "
            f"Restore or remove them manually (a complete *{BACKUP_SUFFIX} set is "
            "the previous result; a staging directory holds an unpublished one), "
            "then rerun")


def _file_state(path) -> str:
    try:
        return f"{os.path.getsize(path)} B"
    except OSError:
        return "absent"


def _publication_state(staging, out_dir, names) -> str:
    """One line per file: final / backup / staged sizes, for recovery."""
    lines = []
    for name in names:
        final = os.path.join(out_dir, name)
        lines.append(f"{name}: final {_file_state(final)}, backup "
                     f"{_file_state(final + BACKUP_SUFFIX)}, staged "
                     f"{_file_state(os.path.join(staging, name))}")
    return "; ".join(lines)


def publish_stage11(staging, out_dir, names, replace=os.replace):
    """Install the stage files from staging into out_dir with per-file
    backups: phase 1 moves existing files to *.prev, phase 2 installs the new
    ones. Installing the last file is the boundary: before it any error rolls
    back (new files out, backups restored); after it only the cleanup of the
    backups and staging runs and its failures are reported, never rolled back."""
    backed, installed = [], []
    try:
        for name in names:
            final = os.path.join(out_dir, name)
            if os.path.lexists(final):
                replace(final, final + BACKUP_SUFFIX)
                backed.append(name)
        for name in names:
            replace(os.path.join(staging, name), os.path.join(out_dir, name))
            installed.append(name)
    except BaseException as exc:
        failures = []
        for name in names:
            final = os.path.join(out_dir, name)
            try:
                if name in backed:
                    replace(final + BACKUP_SUFFIX, final)
                elif name in installed:
                    os.remove(final)
            except OSError as err:
                failures.append(f"{name}: {err}")
        if failures:
            raise ValueError(
                f"stage 11 publication failed ({exc}) and the rollback did not "
                f"complete; {staging} and the *{BACKUP_SUFFIX} files are kept for "
                "manual recovery. Rollback errors: " + "; ".join(failures)
                + ". File state: " + _publication_state(staging, out_dir, names)) from exc
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"stage 11 publication failed before completion; the "
                         f"previous result was restored: {exc}") from exc
    leftovers = []
    for name in backed:
        try:
            os.remove(os.path.join(out_dir, name + BACKUP_SUFFIX))
        except OSError as err:
            leftovers.append(f"{name}{BACKUP_SUFFIX}: {err}")
    try:
        shutil.rmtree(staging)
    except OSError as err:
        leftovers.append(f"{staging}: {err}")
    if leftovers:
        raise ValueError("stage 11 published completely, but the cleanup left "
                         "remnants to remove manually: " + "; ".join(leftovers))


# ---------------------------------------------------------- preflight

def _available_ram() -> int | None:
    if os.name != "nt":
        return None

    class _MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    status = _MemStatus()
    status.dwLength = ctypes.sizeof(_MemStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullAvailPhys)
    return None


def stage11_resource_estimate(n_modes: int, n_lines: int, pixels: list[int],
                              jobs: int) -> dict:
    """Peak model for one scene run: bytes of RAM (max over the
    deposit, jackknife and publish phases) and of disk (row store, in-flight
    .part files, staging outputs). The jackknife phase holds one tile buffer,
    the totals and the finished maps; mapped row pages are page cache."""
    total = sum(pixels)
    jobs_eff = _effective_jobs(jobs, n_modes)
    deposit = jobs_eff * (total * (16 * n_lines + 64) + _R_PY) + total * (
        _ACC_BYTES_PER_PIXEL + 8 + 28 + 24)
    jackknife = _TILE_BYTES + total * (24 + 8 + 160)
    publish = total * 160
    disk = 12 * n_modes * total + 2 * jobs_eff * 28 * total + total * 210 + (4 << 20)
    return {"jobs": jobs_eff, "ram": int(max(deposit, jackknife, publish) * _PREFLIGHT_MARGIN),
            "disk": int(disk), "deposit": deposit, "jackknife": jackknife,
            "publish": publish}


def preflight_stage11(sim, out_dir, jobs: int) -> list[dict]:
    """Refuse a run whose estimated peak RAM or disk exceeds what is free."""
    cfg = sim.cfg
    scenes = []
    if cfg.free_source is not None:
        scenes.append(("free", cfg.free_source, [cfg.free_screen]))
    if cfg.capillary is not None:
        cap = cfg.capillary
        scenes.append(("capillary", cap.source, [cap.screen, *cap.screens]))
    estimates = []
    for scene, src, screens in scenes:
        n_modes, _ = src.budget()
        pixels = [int(s.nx) * int(s.ny) for s in screens]
        est = stage11_resource_estimate(n_modes, len(sim.lines), pixels, jobs)
        est["scene"] = scene
        estimates.append(est)
    if not estimates:
        return estimates
    need_ram = max(e["ram"] for e in estimates)
    need_disk = max(e["disk"] for e in estimates)
    avail_ram = _available_ram()
    if avail_ram is not None and need_ram > avail_ram:
        raise ValueError(f"stage 11 needs about {need_ram / 2 ** 30:.1f} GiB of RAM "
                         f"(estimate with margin), {avail_ram / 2 ** 30:.1f} GiB free; "
                         "lower CAPSYSRED_STAGE11_JOBS or free memory")
    free_disk = shutil.disk_usage(out_dir).free
    if need_disk > free_disk:
        raise ValueError(f"stage 11 needs about {need_disk / 2 ** 30:.1f} GiB on "
                         f"{out_dir}, {free_disk / 2 ** 30:.1f} GiB free")
    return estimates

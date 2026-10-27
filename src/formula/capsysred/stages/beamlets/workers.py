"""Stage 11 per-mode workers, bounded jackknife tiles and resource checks."""

import ctypes
import json
import mmap
import os
import shutil
import sys
import threading
from array import array
from types import SimpleNamespace
from typing import NamedTuple

from ...fresnel import FloatLineAmplitudes
from ...native import jackknife_tile
from ... import rays_v3
from ...rays import metadata_equal
from ...screen import ScreenGrid
from .field import BeamletField, _reference_pixel, _scene_optic, _stage11_contract


PID_PREFIX = "pid-"
# ProcessPoolExecutor on Windows is limited by WaitForMultipleObjects.
_MAX_NT_WORKERS = 61
# Row-store bytes held per jackknife tile (12 B per pixel per mode, one copy).
_TILE_BYTES = 256 << 20
# Resource model: Python per worker, master accumulator, buffers.
_R_PY = 400 << 20
_ACC_BYTES_PER_PIXEL = 816
_PREFLIGHT_MARGIN = 1.3


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
            raise ValueError(f"{part.path}: stage 11 reads per-mode v3 archives only")
        modes = index.modes(scene)
        for local, sections in enumerate(modes):
            tasks.append(ModeTask(part.path, local, offset + local, tuple(sections)))
        offset += len(modes)
    return tasks


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
        from ...simulation import Simulation
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
                waist_dz=launch["waist_dz"], self_pairs=launch["self_pairs"]))
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
                try:
                    rec = rays_v3._record(json.loads(line))
                except ValueError as exc:   # garbage that inflated without error
                    raise ValueError(
                        f"{rays_v3.section_path(task.part, entry)}: corrupt row "
                        "in section; remove it manually (verify the archive)") from exc
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
            w_row, i_row, dw, di, i_ref = field.fold_export(n)
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
        scenes.append(("capillary", cap.source, cap.all_screens))
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

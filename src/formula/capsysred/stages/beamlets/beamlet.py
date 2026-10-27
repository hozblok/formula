"""Stage 11 orchestration: per-mode beamlet deposition and exact reduction."""

import concurrent.futures
import math
import os
import shutil
import time
from array import array
from types import SimpleNamespace

from .._contracts import _screen_spec
from ...native import exact_accumulator
from ...shared.progress import Progress
from ...rays import metadata_equal, recorded_plane, require_full_rows, scene_stream
from ...screen import ScreenGrid
from .field import _flat_walls, _reference_pixel, _stage11_contract
from .publication import TMP_PREFIX
from .workers import (
    PeakSampler,
    _Worker,
    _blank,
    _effective_jobs,
    _mode_tasks,
    _peak_rss,
    _tile_jackknife,
    _worker_init,
    _worker_run,
)


def run_beamlet_stage(sim, label, scene, src_cfg, scr_cfg, optic, aim_factory,
                      extra_screens=(), work_dir=None, jobs: int = 1, sampler=None):
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
                                aim_factory)
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
    from ...simulation import Simulation
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

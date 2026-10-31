"""Stage 14 native finalization, result serialization, figures and publication."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
import statistics
import time

from .... import __version__, _formula
from ... import render
from ...rays import metadata_equal
from ...screen import ScreenGrid
from .flags import (
    FlagThresholds,
    PixelCounters,
    PixelStatistics,
    serialize_pixel,
    validate_counters,
    validate_ref,
    w_signal_status,
)
from ...shared.units import m_to_um
from ...shared.utils import durable_open
from .cache import (
    CachePart,
    META_NAME,
    ROWS_NAME,
    STAGE_ID,
    ScreenTarget,
    _array_bytes,
    _combine_aggregates,
    _estimated_peak_rss,
    _typed_array,
    _write_json,
)


RESULT_DIR = "stage14"
RESULT_PAYLOAD_NAMES = (
    "mu-jack.jsonl",
    "14-capillary-jack-mu.svg",
    "14-capillary-jack-mu-map.svg",
    "14-capillary-jack-mu-err.svg",
    "14-capillary-jack-mu-flags.svg",
    "14a-capillary-jack-slice.svg",
    "14b-capillary-jack-intensity.svg",
    "14b-capillary-jack-intensity-log.svg",
    "14b-capillary-jack-intensity-slice.svg",
    "14b-capillary-jack-intensity-log-slice.svg",
    "14b-capillary-jack-density.svg",
    "14c-capillary-jack-overlay.svg",
    "14d-capillary-ray-scatter.svg",
    "14e-capillary-ref-passport.svg",
    "14f-capillary-jack-ic.svg",
    "14f-capillary-jack-ic-log.svg",
    "14f-capillary-jack-ic-err.svg",
    "14f-capillary-jack-ic-err-log.svg",
    "14f-capillary-jack-ic-slice.svg",
    "14f-capillary-jack-ic-log-slice.svg",
)


def _publish_directory(partial: str, final: str, names: list[str]) -> None:
    """Publish files or child trees into an exclusively reserved directory.

    The metadata marker is supplied last by callers.  A crash can therefore
    leave an unmistakably incomplete directory, but can never replace a
    previously published result or cache.
    """
    try:
        os.mkdir(final)
    except FileExistsError as exc:
        raise ValueError(
            f"{final}: publication conflict; remove it manually"
        ) from exc
    try:
        for name in names:
            os.rename(os.path.join(partial, name), os.path.join(final, name))
        os.rmdir(partial)
    except BaseException:
        # Keep both paths as loud, fail-closed evidence.  The final directory
        # has no valid metadata until the last move succeeds.
        raise


def _publish_result_tree(partial: str, final: str,
                         fallback_names: list[str]) -> None:
    """Publish a complete result tree atomically where the OS permits it.

    Windows ``rename`` is an atomic no-replace directory move, which is the
    production platform for the large archives.  On platforms where rename
    may replace an empty destination, retain the older exclusive-directory,
    metadata-last fallback rather than weakening no-clobber semantics.
    """
    if os.name == "nt":
        try:
            os.rename(partial, final)
        except FileExistsError as exc:
            raise ValueError(
                f"{final}: publication conflict; remove it manually"
            ) from exc
        return
    _publish_directory(partial, final, fallback_names)


def _finalize(caches: list[CachePart], aggregate: dict, npix: int, ref: int):
    fn = getattr(_formula, "stage14_finalize", None)
    if fn is None:
        raise RuntimeError("the native extension has no Stage-14 finalizer")
    expected_hashes = [cache.meta["files"][ROWS_NAME]["sha256"] for cache in caches]
    return fn(
        [cache.rows_path for cache in caches],
        [cache.input.n_modes for cache in caches], npix, ref,
        _array_bytes(aggregate["I"]), _array_bytes(aggregate["w_re"]),
        _array_bytes(aggregate["w_im"]), _array_bytes(aggregate["ic"]),
        expected_hashes,
    )


def _ref_warnings(rows, grid: ScreenGrid, ref: int, n_modes: int):
    warnings, diagnostics = [], {}
    ref_row = rows[ref]
    lam = ref_row["n_rays"] / n_modes
    diagnostics["lambda_ref"] = lam
    diagnostics["n_rays_ref"] = ref_row["n_rays"]
    diagnostics["m_realizations_ref"] = ref_row["m_realizations"]
    diagnostics["m_pair_realizations_ref"] = ref_row["m_pair_realizations"]
    diagnostics["m_ref_realizations_histogram"] = {
        str(value): count for value, count in sorted(Counter(
            row["m_ref_realizations"] for row in rows).items())
    }
    if lam < 8.0:
        warnings.append("low-rays-per-realization-at-ref")
    pair_rows = [row for row in rows if row["m_pair_realizations"] > 0]
    if pair_rows:
        below = sum(row["n_rays"] <= ref_row["n_rays"] for row in pair_rows)
        percentile = 100.0 * below / len(pair_rows)
        diagnostics["ref_density_percentile"] = percentile
        if percentile < 50.0:
            warnings.append("below-median-ref-density")
        covered = sum(row["m_ref_realizations"] > 0 for row in pair_rows)
        diagnostics["w_cover"] = covered / len(pair_rows)
    iy, ix = divmod(ref, grid.nx)
    neighbors = []
    for jy in range(max(0, iy - 1), min(grid.ny, iy + 2)):
        for jx in range(max(0, ix - 1), min(grid.nx, ix + 2)):
            p = jy * grid.nx + jx
            if p != ref:
                neighbors.append(rows[p]["flag"] == "trusted")
    diagnostics["trusted_ref_neighbors"] = sum(neighbors)
    diagnostics["ref_neighbors"] = len(neighbors)
    if neighbors and not all(neighbors):
        warnings.append("untrusted-ref-neighborhood")
    return warnings, diagnostics


def _grid(values, nx, ny):
    return [list(values[i * nx:(i + 1) * nx]) for i in range(ny)]


def _figure_saver(result_dir: str):
    """Single gate for figure writes; the caller checks names against the manifest."""
    written = set()

    def save(name, fig):
        written.add(name)
        render.save(os.path.join(result_dir, name), fig)

    return written, save


def _stage14_figures(save, rows, aggregate, final, grid: ScreenGrid,
                     ref: int, flag_counts: Counter, n_modes: int,
                     screen_label: str = "capillary"):
    nx, ny = grid.nx, grid.ny
    mu = [row["mu_raw"] for row in rows]
    err = [row["mu_raw_err"] for row in rows]
    flags = [row["flag"] for row in rows]
    lit_flag_counts = Counter(
        row["flag"] for row in rows if row["n_rays"] > 0
    )
    density = [float(v) for v in aggregate["n_rays"]]
    intensity = [float(v) for v in aggregate["I"]]
    mu_grid, err_grid = _grid(mu, nx, ny), _grid(err, nx, ny)
    flag_grid = _grid(flags, nx, ny)
    i_grid, d_grid = _grid(intensity, nx, ny), _grid(density, nx, ny)
    extent = (m_to_um(grid.x0f), m_to_um(grid.x0f + grid.exf),
              m_to_um(grid.y0f), m_to_um(grid.y0f + grid.eyf))
    ref_xy = grid.pixel_xy(ref)
    mark = (m_to_um(ref_xy[0]), m_to_um(ref_xy[1]))
    sub = (f"{screen_label}; {n_modes} exact delete-one-mode units; "
           "raw μ, display clipped at 1")
    cell_x, cell_y = m_to_um(grid.exf) / nx, m_to_um(grid.eyf) / ny
    cell = (f"{cell_x:.3g} µm" if abs(cell_x - cell_y) < 5e-4
            else f"{cell_x:.3g} × {cell_y:.3g} µm")
    mu_sub = (f"ref cell at ({round(mark[0], 3) or 0.0:g}, "
              f"{round(mark[1], 3) or 0.0:g}) µm, cell {cell}")
    mu_fig = render.heatmap(mu_grid, extent, "|μ(P,P_ref)|", "x, µm", "y, µm",
                            mu_sub, "|μ|", vmax=1.0, mark=mark, w=448, equal=True)
    err_fig = render.heatmap(err_grid, extent, "σ_jack(μ)", "x, µm", "y, µm",
                             "", "σ", mark=mark, w=448, equal=True)
    flag_fig = render.category_map(flag_grid, extent, "Cell classification", "x, µm",
                                   "y, µm", "", mark=mark,
                                   counts=flag_counts,
                                   lit_counts=lit_flag_counts,
                                   lit_total=sum(row["n_rays"] > 0 for row in rows),
                                   w=448, equal=True)
    save("14-capillary-jack-mu.svg",
         render.hstack([mu_fig, err_fig, flag_fig]))
    save("14-capillary-jack-mu-map.svg", mu_fig)
    save("14-capillary-jack-mu-err.svg", err_fig)
    save("14-capillary-jack-mu-flags.svg", flag_fig)
    iy = ref // nx
    xs = [m_to_um(x) for x in grid.xs()]
    row_ids = range(iy * nx, (iy + 1) * nx)
    xs_row = [xs[p % nx] for p in row_ids]

    def _row(fn):
        # per-cell values along the reference row; None = gap, lines break
        return [fn(rows[p]) for p in row_ids]

    def banded(r):
        return r["mu_raw"] is not None and r["mu_raw_err"] is not None

    mu_band = _row(lambda r: min(r["mu_raw"], 1.0) if banded(r) else None)
    series = []
    if any(v is not None for v in mu_band):
        series.append({
            "xs": xs_row, "ys": mu_band,
            "lo": _row(lambda r: max(r["mu_raw"] - r["mu_raw_err"], 0.0)
                       if banded(r) else None),
            "hi": _row(lambda r: min(r["mu_raw"] + r["mu_raw_err"], 1.0)
                       if banded(r) else None),
            "label": "min(μ,1) ± σ_jack",
        })
    without_err = [p for p in row_ids if rows[p]["mu_raw"] is not None
                   and rows[p]["mu_raw_err"] is None]
    if without_err:
        series.append({"xs": [xs[p % nx] for p in without_err],
                       "ys": [min(rows[p]["mu_raw"], 1.0) for p in without_err],
                       "label": "μ; σ unavailable", "dots": True,
                       "color": "#CC79A7"})
    if not series:
        series.append({"xs": [xs[0], xs[-1]], "ys": [0.0, 0.0],
                       "label": "no defined μ on this slice", "dash": "2,3"})
    for flag in ("null-Ic", "noisy-Ic", "noisy-mu", "over-mu"):
        ids = [p for p in row_ids if rows[p]["flag"] == flag]
        if ids:
            series.append({"xs": [xs[p % nx] for p in ids],
                           "ys": [min(rows[p]["mu_raw"] or 0.0, 1.0) for p in ids],
                           "label": flag, "color": render.FLAG_COLORS[flag],
                           "dots": True})
    slice_label = f"slice y = {round(m_to_um(grid.ys()[iy]), 3) or 0.0:g} µm"
    slice_fig = render.line_chart(
        series, f"|μ(P,P_ref)| — {slice_label}",
        "x, µm", "|μ|", mu_sub, vlines=[(m_to_um(ref_xy[0]), "ref")], w=760)

    def _strip(vals, ylabel):
        return render.line_chart(
            [{"xs": xs_row, "ys": vals,
              "lo": [0.0 if v is not None else None for v in vals], "hi": vals,
              "color": "#ff7f0e", "width": 1.0}], "", "x, µm", ylabel,
            w=760, h=150)

    err_vals = _row(lambda r: r["mu_raw_err"])
    if any(v is not None for v in err_vals):
        slice_fig = render.vstack([slice_fig, _strip(err_vals, "σ_jack")])
    save("14a-capillary-jack-slice.svg", slice_fig)
    ref_line = [(m_to_um(ref_xy[0]), "ref")]

    def _slice_chart(vals, title, ylabel, empty, y_zero=True):
        series = ([{"xs": xs_row, "ys": vals, "label": ylabel}]
                  if any(v is not None for v in vals) else
                  [{"xs": [xs[0], xs[-1]], "ys": [0.0, 0.0],
                    "label": empty, "dash": "2,3"}])
        return render.line_chart(series, f"{title} — {slice_label}", "x, µm",
                                 ylabel, mu_sub, vlines=ref_line,
                                 y_zero=y_zero, w=760)

    row_i = [intensity[p] for p in row_ids]
    save("14b-capillary-jack-intensity-slice.svg",
         _slice_chart(row_i, "intensity", "I", "no rays"))
    save("14b-capillary-jack-intensity-log-slice.svg",
         _slice_chart([math.log10(v) if v > 0 else None for v in row_i],
                             "intensity, log scale", "log10 I",
                             "no lit cells", y_zero=False))
    ic_grid = _grid([row["ic"] for row in rows], nx, ny)
    ic_err_grid = _grid([row["ic_err"] for row in rows], nx, ny)
    for name, grid_, title, cbar, log_ in (
            ("14f-capillary-jack-ic.svg", ic_grid,
             "coherent intensity Ic", "Ic", False),
            ("14f-capillary-jack-ic-log.svg", ic_grid,
             "coherent intensity Ic, log scale", "Ic", True),
            ("14f-capillary-jack-ic-err.svg", ic_err_grid,
             "σ_jack(Ic)", "σ", False),
            ("14f-capillary-jack-ic-err-log.svg", ic_err_grid,
             "σ_jack(Ic), log scale", "σ", True)):
        save(name,
         render.heatmap(grid_, extent, title, "x, µm", "y, µm", "",
                                   cbar, w=518, equal=True, log=log_))
    ic_row = _row(lambda r: r["ic"])
    ic_fig = _slice_chart(ic_row, "coherent intensity Ic", "Ic", "no paired cells")
    ic_errs = _row(lambda r: r["ic_err"])
    if any(v is not None for v in ic_errs):
        ic_fig = render.vstack([ic_fig, _strip(ic_errs, "σ_jack(Ic)")])
    save("14f-capillary-jack-ic-slice.svg", ic_fig)
    ic_log_fig = _slice_chart(
        [math.log10(v) if v is not None and v > 0 else None for v in ic_row],
        "coherent intensity Ic, log scale", "log10 Ic", "no positive Ic",
        y_zero=False)
    rel = _row(lambda r: r["ic_err"] / r["ic"]
               if r["ic_err"] is not None and (r["ic"] or 0) > 0 else None)
    if any(v is not None for v in rel):
        ic_log_fig = render.vstack([ic_log_fig, _strip(rel, "σ_jack(Ic)/Ic")])
    save("14f-capillary-jack-ic-log-slice.svg",
         ic_log_fig)
    if ny > 1:
        intensity_fig = render.heatmap(i_grid, extent, "intensity", "x, µm",
                                       "y, µm", "", "I", w=518, equal=True)
        log_fig = render.heatmap(i_grid, extent, "intensity, log scale", "x, µm",
                                 "y, µm", "", "I", w=518, equal=True, log=True)
        density_fig = render.heatmap(d_grid, extent, "ray density", "x, µm",
                                     "y, µm", "", "rays", w=518, equal=True)
    else:
        imax, dmax = max(intensity) or 1.0, max(density) or 1.0
        intensity_fig = render.line_chart([
            {"xs": xs, "ys": [v / imax for v in intensity], "label": "I/max"}],
            "intensity", "x, µm", "normalized", sub, w=760)
        lit = [(x, v) for x, v in zip(xs, intensity) if v > 0]
        log_fig = render.line_chart(
            [{"xs": [x for x, _ in lit], "ys": [math.log10(v) for _, v in lit],
              "label": "log10 I"}] if lit else
            [{"xs": [xs[0], xs[-1]], "ys": [0.0, 0.0],
              "label": "no lit cells", "dash": "2,3"}],
            "intensity, log scale", "x, µm", "log10 I", "", y_zero=False, w=760)
        density_fig = render.line_chart([
            {"xs": xs, "ys": [v / dmax for v in density], "label": "rays/max",
             "dash": "6,4"}], "ray density", "x, µm", "normalized", "", w=760)
    save("14b-capillary-jack-intensity.svg",
         intensity_fig)
    save("14b-capillary-jack-intensity-log.svg",
         log_fig)
    save("14b-capillary-jack-density.svg",
         density_fig)
    save("14c-capillary-jack-overlay.svg",
         render.overlay_map(mu_grid, flag_grid, extent,
                                   "Stage-14 non-trusted overlay", "x, µm", "y, µm",
                                   sub, mark=mark, equal=True))
    scatter_meta = final["scatter"]
    scatter_grid = _grid(aggregate["scatter"], scatter_meta["nx"], scatter_meta["ny"])
    scatter_extent = (m_to_um(scatter_meta["x0"]),
                      m_to_um(scatter_meta["x0"] + scatter_meta["edge_x"]),
                      m_to_um(scatter_meta["y0"]),
                      m_to_um(scatter_meta["y0"] + scatter_meta["edge_y"]))
    save("14d-capillary-ray-scatter.svg",
         render.ray_scatter(scatter_grid, scatter_extent,
                                   f"{screen_label}: ray locations on target screen",
                                   "x, µm", "y, µm", sub))
    return {
        "mu_raw": mu_grid, "mu_raw_err": err_grid, "flag": flag_grid,
        "intensity": i_grid, "density": d_grid,
    }


def _ref_passport(save, rows, ref: int, ref_status: str, ref_warnings,
                  diagnostics: dict, thresholds: FlagThresholds, n_modes: int,
                  screen_label: str) -> None:
    """14e: the reference pixel's own numbers against the registered and
    guide thresholds; only recorded mu-jack/meta fields."""
    row = rows[ref]
    ic, err = row["ic"], row["ic_err"]
    snr = ic / err if ic is not None and err else None
    z_ref = thresholds.ref_ic_n_sigma
    pairs, lit, n_ref = (row["m_pair_realizations"], row["m_realizations"],
                         row["n_rays"])
    pair_rows = [r for r in rows if r["m_pair_realizations"] > 0]
    covered = sum(r["m_ref_realizations"] > 0 for r in pair_rows)
    others = [r["n_rays"] for r in pair_rows if not r["is_reference"]]
    median = statistics.median(others) if others else None
    pct, w_cover = diagnostics.get("ref_density_percentile"), diagnostics.get("w_cover")
    lam = diagnostics["lambda_ref"]
    checks = [
        {"name": "Ic_ref significance",
         "big": f"{snr:.1f}σ" if snr is not None else "n/a",
         "detail": (f"Ic_ref = {ic:.0f} ± {err:.0f}  ·  {snr / z_ref:.1f}× the weak line"
                    if snr is not None else f"ref_status = {ref_status}"),
         "value": snr, "lo": 0.0, "hi": 20.0, "threshold": z_ref,
         "threshold_label": f"≥ ref_ic_n_sigma = {z_ref:g} (registered)"},
        {"name": "pairness of the reference", "big": f"{pairs / n_modes:.0%}",
         "detail": (f"{pairs} of {n_modes} realizations gave ≥ 2 rays  ·  "
                    f"lit in {lit} ({lit / n_modes:.0%})"),
         "value": pairs / n_modes, "lo": 0.0, "hi": 1.0, "threshold": 0.5,
         "threshold_label": "≥ 50 % (guide)"},
        {"name": "rays per realization at ref", "big": f"{lam:.2f}",
         "detail": (f"n_rays = {n_ref}  ·  mean over lit realizations "
                    f"{n_ref / lit if lit else 0.0:.2f}  ·  max {row['max_rays_per_realization']}"),
         "value": lam, "lo": 0.0, "hi": 20.0, "threshold": 8.0,
         "threshold_label": "≥ 8 (guide 8–16)"},
        {"name": "density rank among paired cells",
         "big": f"{pct:.0f}%" if pct is not None else "n/a",
         "detail": (f"n_rays(ref) = {n_ref}  vs  map median {median:.0f}  ·  "
                    f"{n_ref / median:.1f}× median" if median else
                    f"n_rays(ref) = {n_ref}  ·  no other paired cells"),
         "value": pct, "lo": 0.0, "hi": 100.0, "threshold": 50.0,
         "threshold_label": "≥ 50 % (guide)"},
        {"name": "cross-link with the map  (w_cover)",
         "big": f"{w_cover:.2f}" if w_cover is not None else "n/a",
         "detail": f"paired cells sharing a realization with ref: {covered} of {len(pair_rows)}",
         "value": w_cover, "lo": 0.0, "hi": 1.0, "threshold": 0.9,
         "threshold_label": "≥ 0.9 (guide)"},
    ]
    title = (f"Reference-cell passport — "
             f"({row['x_um']:.1f}, {row['y_um']:.1f}) µm, N_jk = {n_modes}, "
             f"ref_ic_n_sigma = {z_ref:g}")
    save("14e-capillary-ref-passport.svg",
         render.gauge_table(checks, title))


def _finalize_screen(sim, target: ScreenTarget, output_dir: str,
                     fanout_metrics: dict,
                     *, write_meta: bool = True) -> dict:
    """Finalize and serialize one screen from its resolved cache parts."""
    screen_started = time.time()
    grid, ref = target.grid, target.ref
    npix = grid.nx * grid.ny
    parts = target.parts
    cache_parts = list(target.caches)
    if not cache_parts or any(cache is None for cache in cache_parts):
        raise ValueError(f"Stage-14 {target.label} caches are unresolved")
    scatter_meta = cache_parts[0].meta["scatter"]
    for cache in cache_parts[1:]:
        if (not metadata_equal(cache.meta["analysis_signature"], target.signature)
                or not metadata_equal(cache.meta["scatter"], scatter_meta)):
            raise ValueError(
                f"Stage-14 {target.label} union cache parts are incompatible")
    aggregate = _combine_aggregates(
        cache_parts, npix, sim.cfg.max_bounces,
        scatter_meta["nx"] * scatter_meta["ny"])
    native = _finalize(cache_parts, aggregate, npix, ref)
    # Dense native outputs: every undefined value has a separate mask; Python
    # never infers missingness from NaN or a plausible numeric zero.
    ic_err = _typed_array("d", native["ic_err"])
    w_err = _typed_array("d", native["w_err"])
    mu_raw = _typed_array("d", native["mu_raw"])
    mu_err = _typed_array("d", native["mu_raw_err"])
    n_valid = _typed_array("I", native["n_mu_loo_valid"])
    mu_defined = bytes(native["mu_raw_defined"])
    mu_err_defined = bytes(native["mu_raw_err_defined"])
    ref_ic_loo = _typed_array("d", native["ic_ref_loo"])
    n_modes = sum(part.n_modes for part in parts)
    for name, values in (("ic_err", ic_err), ("w_err", w_err),
                         ("mu_raw", mu_raw), ("mu_raw_err", mu_err),
                         ("n_mu_loo_valid", n_valid)):
        if len(values) != npix:
            raise ValueError(f"native Stage-14 {name} has wrong length")
    if len(mu_defined) != npix or len(mu_err_defined) != npix:
        raise ValueError("native Stage-14 validity masks have wrong length")
    if len(ref_ic_loo) != n_modes:
        raise ValueError("native Stage-14 reference LOO row has wrong length")
    hashes = list(native["payload_sha256"])
    expected_hashes = [cache.meta["files"][ROWS_NAME]["sha256"]
                       for cache in cache_parts]
    if hashes != expected_hashes:
        raise ValueError("Stage-14 payload SHA-256 mismatch")
    thresholds = FlagThresholds(**sim.cfg.stage14_flag_thresholds)
    ref_status = validate_ref(
        m_pair_realizations=int(aggregate["m_pair_realizations"][ref]),
        ic_ref=float(aggregate["ic"][ref]),
        ic_ref_err=float(ic_err[ref]),
        ic_ref_loo=ref_ic_loo,
        n_jackknife_units=n_modes,
        thresholds=thresholds,
    )
    rows = []
    for pixel in range(npix):
        counters = PixelCounters(
            int(aggregate["n_rays"][pixel]),
            int(aggregate["m_realizations"][pixel]),
            int(aggregate["m_pair_realizations"][pixel]),
            int(aggregate["m_ref_realizations"][pixel]),
            int(aggregate["max_rays_per_realization"][pixel]),
        )
        validate_counters(counters)
        I = float(aggregate["I"][pixel])
        ic = float(aggregate["ic"][pixel])
        w_abs_value = math.hypot(aggregate["w_re"][pixel], aggregate["w_im"][pixel])
        is_ref = pixel == ref
        # Apply the taxonomy's strict missingness before calling its serializer.
        if counters.n_rays == 0:
            stats = PixelStatistics(I, counters)
        elif counters.m_pair_realizations == 0:
            has_w = counters.m_ref_realizations > 0
            stats = PixelStatistics(
                I, counters, w_abs=w_abs_value if has_w else None,
                w_err=float(w_err[pixel]) if has_w else None)
        else:
            lower = ic - thresholds.ic_n_sigma * float(ic_err[pixel])
            upper = ic + thresholds.ic_n_sigma * float(ic_err[pixel])
            early_ic = upper < 0.0 or lower <= 0.0
            has_w = counters.m_ref_realizations > 0
            allow_mu = (not is_ref and not early_ic and ref_status == "ok" and has_w)
            stats = PixelStatistics(
                I, counters, ic=ic, ic_err=float(ic_err[pixel]),
                w_abs=w_abs_value if has_w else None,
                w_err=float(w_err[pixel]) if has_w else None,
                mu_raw=float(mu_raw[pixel]) if allow_mu and mu_defined[pixel] else None,
                mu_raw_err=(float(mu_err[pixel]) if allow_mu and mu_err_defined[pixel]
                            else None),
                n_mu_loo_valid=(int(n_valid[pixel]) if allow_mu and mu_defined[pixel]
                                else None),
            )
            if allow_mu and not mu_defined[pixel]:
                raise ValueError(
                    f"Stage-14 total μ is unexpectedly undefined at pixel {pixel}"
                )
        x, y = grid.pixel_xy(pixel)
        rows.append(serialize_pixel(
            stats, pixel=pixel, x_um=m_to_um(x), y_um=m_to_um(y),
            is_reference=is_ref, ref_status=ref_status,
            n_jackknife_units=n_modes, thresholds=thresholds,
            screen=target.label))
    flag_counts = Counter(row["flag"] for row in rows)
    ref_warnings, ref_diagnostics = _ref_warnings(rows, grid, ref, n_modes)
    w_census = (Counter(
        w_signal_status(row["w_abs"], row["w_err"], thresholds)
        for row in rows if not row["is_reference"])
        if ref_status == "ok" else
        Counter({"unknown": sum(not row["is_reference"] for row in rows)}))
    over_mu_partial_loo = sum(
        row["flag"] == "over-mu"
        and row["n_mu_loo_valid"] is not None
        and row["n_mu_loo_valid"] < n_modes
        for row in rows
    )
    remediation = {
        "estimator-errors": flag_counts["negative-Ic"],
        "statistics-limited": sum(flag_counts[name] for name in (
            "solo-rays-only", "noisy-Ic", "no-ref-realizations",
            "noisy-mu", "over-mu")),
        "measured-null": flag_counts["null-Ic"],
        "usable": flag_counts["trusted"],
        "background-no-rays": flag_counts["no-rays"],
        "unclassified": flag_counts[None],
    }
    part_provenance = []
    mode_offset = 0
    for part in parts:
        part_provenance.append({
            "input_id": part.input_id,
            "cache_id": part.analysis_id,
            "path": part.path,
            "seed": part.seed,
            "n_modes": part.n_modes,
            "n_rays": part.n_rays,
            "local_mode_start": 0,
            "local_mode_end": part.n_modes,
            "global_mode_start": mode_offset,
            "global_mode_end": mode_offset + part.n_modes,
        })
        mode_offset += part.n_modes
    scatter_cells = scatter_meta["nx"] * scatter_meta["ny"]
    estimated_peak_rss = _estimated_peak_rss(npix, len(sim.lines), scatter_cells)
    cache_bytes_written = sum(
        sum(entry["bytes"] for entry in cache.meta["files"].values())
        for cache in cache_parts if not cache.cache_hit)
    target_miss_ray_bytes = sum(
        part.bytes
        for part, cache in zip(parts, cache_parts) if not cache.cache_hit)
    result_final = {
        "scatter": scatter_meta,
        "pass1_seconds": float(native.get("pass1_seconds", 0.0)),
        "pass2_seconds": float(native.get("pass2_seconds", 0.0)),
        "mode_rows_bytes_read": int(native.get("bytes_read", 0)),
        # Physical gzip I/O is shared by fan-out and belongs to the root run,
        # not independently to every screen report.
        "ray_archive_bytes_read": (
            fanout_metrics["physical_ray_archive_bytes_read"]
            if target.index == 0 else 0),
        "fanout_physical_ray_archive_bytes_read":
            fanout_metrics["physical_ray_archive_bytes_read"],
        "target_cache_miss_ray_archive_bytes": target_miss_ray_bytes,
        "cache_bytes_written": cache_bytes_written,
    }
    if target.output_subdir:
        os.mkdir(output_dir)
    jsonl_path = os.path.join(output_dir, "mu-jack.jsonl")
    with durable_open(jsonl_path, encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    written, save = _figure_saver(output_dir)
    maps = _stage14_figures(
        save, rows, aggregate, result_final, grid, ref,
        flag_counts, n_modes, target.label)
    _ref_passport(save, rows, ref, ref_status, ref_warnings, ref_diagnostics,
                  thresholds, n_modes, target.label)
    expected = {name for name in RESULT_PAYLOAD_NAMES if name.endswith(".svg")}
    if written != expected:
        raise ValueError(
            "stage-14 figures drifted from RESULT_PAYLOAD_NAMES: "
            f"missing {sorted(expected - written)}, "
            f"unexpected {sorted(written - expected)}")
    screen_seconds = time.time() - screen_started
    result_meta = {
        "stage_id": STAGE_ID, "screen": target.label,
        "schema": 1, "capsysred_version": __version__,
        "yaml_file": sim.cfg.yaml_file,
        "analysis_signature": target.signature,
        "input_cache_ids": [part.analysis_id for part in parts],
        "input_ids": [part.input_id for part in parts],
        "input_paths": [part.path for part in parts],
        "part_order": list(range(len(parts))),
        "parts": part_provenance,
        "screen_geometry": target.contract,
        "flag_thresholds": sim.cfg.stage14_flag_thresholds,
        "jackknife_computed": True,
        "n_jackknife_units": n_modes,
        "jackknife_unit": "mode",
        "ref_status": ref_status,
        "ref_warnings": ref_warnings,
        "ref_diagnostics": ref_diagnostics,
        "over_mu_partial_loo": over_mu_partial_loo,
        "flag_counts": {("null" if key is None else key): value
                        for key, value in flag_counts.items()},
        "w_signal_census": dict(w_census),
        "remediation_counts": remediation,
        "stats": aggregate["stats"],
        "bounce_hist": aggregate["bounce_hist"],
        "cache": {
            "hits": sum(cache.cache_hit for cache in cache_parts),
            "misses": sum(not cache.cache_hit for cache in cache_parts),
            "directories": [cache.input.cache_dir for cache in cache_parts],
            "payload_bytes": sum(cache.meta["files"][ROWS_NAME]["bytes"]
                                 for cache in cache_parts),
        },
        "performance": {
            **result_final,
            "total_seconds": screen_seconds,
            "screen_finalize_seconds": screen_seconds,
            "estimated_peak_rss_bytes": estimated_peak_rss,
        },
    }
    if write_meta:
        _write_json(os.path.join(output_dir, META_NAME), result_meta)
    prefix = os.path.join(RESULT_DIR, target.output_subdir)
    return {
        "maps": maps, "rows": rows, "screen": grid,
        "screen_name": target.label,
        "stats": aggregate["stats"], "bounce_hist": aggregate["bounce_hist"],
        "n_modes": n_modes, "n_rays": parts[0].n_rays,
        "ref_pixel": ref, "ref_status": ref_status,
        "ref_warnings": ref_warnings, "ref_diagnostics": ref_diagnostics,
        "flag_counts": flag_counts, "w_signal_census": w_census,
        "remediation_counts": remediation,
        "over_mu_partial_loo": over_mu_partial_loo,
        "cache_parts": cache_parts, "cache_hits": result_meta["cache"]["hits"],
        "seconds": screen_seconds, "result_meta": result_meta,
        "files": [os.path.join(prefix, name)
                  for name in (META_NAME, *RESULT_PAYLOAD_NAMES)],
    }

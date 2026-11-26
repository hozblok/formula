"""Output of Stage 16: the rows, meta and maps of a screen, the report lines and the far field."""

import json
import math
import os

import numpy as np

from .... import __version__
from ....physical_constants import HC_KEV_ANGSTROM
from ... import render
from ...shared.units import m_to_um
from .providers.images import far_field_summary
from .screen_sampler import FAR_TAIL_RATIO

RESULT_DIR = "stage16"
FAR_FIELD = "far-field.json"
UNITS = ("|E|^2 of unit-amplitude point-source Fresnel fields (1/(i lambda z) per step), "
         "source weights normalized to 1, spectral weights normalized to 1; arbitrary but "
         "consistent within one run; I_coherent (area^2 scale) and I_pixel (area scale) "
         "are different detector quantities")


def _emit_far_field(out_dir, far, lines, ff, wave):
    """far-field.json (per line and weighted) and the marginal profiles of the weighted sum."""
    rows, wx, wy, wt, wi = [], None, None, 0.0, 0.0
    for (k, wl, *_), f in zip(lines, far):
        t, px, py, total, inside = f
        row = far_field_summary(t, px, py, total, inside)
        row.update(k_per_m=float(k),
                   e_kev=float(float(HC_KEV_ANGSTROM) / (2.0e10 * np.pi / k)),
                   weight=float(wl))
        rows.append(row)
        wx = wl * px if wx is None else wx + wl * px
        wy = wl * py if wy is None else wy + wl * py
        wt, wi = wt + wl * total, wl * inside + wi
    summary = {"theta_max_rad": ff["theta_max"], "pad": ff["pad"], "lines": rows,
               "weighted": far_field_summary(far[0][0], wx, wy, wt, wi),
               "convention": "P(theta) = |FFT2 u_exit|^2 on theta = lambda f, exit field inside the bore mask; "
                             "lines and nodes summed by intensity"}
    with open(os.path.join(out_dir, FAR_FIELD), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
        fh.write("\n")
    th = [float(v) * 1e3 for v in far[0][0]]
    sx, sy = float(np.max(wx)) or 1.0, float(np.max(wy)) or 1.0
    name = "16-far-field.svg"
    render.save(os.path.join(out_dir, name), render.line_chart(
        [{"xs": th, "ys": [float(v) / sx for v in wx], "label": "P(θ_x) / max"},
         {"xs": th, "ys": [float(v) / sy for v in wy], "label": "P(θ_y) / max", "dash": "6,4"}],
        "far field of the exit field (marginals)", "θ, mrad", "P / max",
        f"window |θ| < {ff['theta_max'] * 1e3:.2f} mrad, zero-padded to {ff['pad']}", w=760))
    return [FAR_FIELD, name]


def _report_lines(label, res, wave, extra):
    m = res["meta"]
    return [f"## Stage 16 — wave estimator [{label}]",
            f"- provider {m['provider']} ({m['model_class']}); observable {wave['observable']}; "
            f"source {m['source_rule']}; {m['n_nodes']} nodes",
            f"- trusted pixels {res['n_trusted']} of {res['nx'] * res['ny']} "
            f"(I ≥ {wave['intensity_floor']:g} × max I); |μ| > 1 on trusted: {res['over_unity']}",
            f"- status: unverified (declared target_error {wave['target_error']:g}; "
            "convergence ladders are not run inside this stage)",
            *([f"- WARNING: reference in a far diffraction tail (I_ref / max I = "
               f"{m['I_ref_over_max_I']:.1e} < {FAR_TAIL_RATIO:g}): W and μ there are dominated by the "
               "lattice representation of the aperture edges; accuracy not controlled"]
              if m.get("reference_in_far_tail") else []),
            *extra,
            f"- time: {res['seconds']:.1f} s"]


def _emit(partial, sub, res, wave, log):
    """Rows, meta and maps of one screen under partial/[sub/]."""
    out = os.path.join(partial, sub) if sub else partial
    os.makedirs(out, exist_ok=True)
    s, nx, ny = res["sampler"], res["nx"], res["ny"]
    xs_um, ys_um = [m_to_um(x) for x in s.px], [m_to_um(y) for y in s.py]
    ref_index = res["ref_index"]
    I, W, mu, tr = res["I"], res["W"], res["mu"], res["trusted"]
    with open(os.path.join(out, "mu-wave.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for iy in range(ny):
            for ix in range(nx):
                m = mu[ix, iy]
                ok = bool(np.isfinite(m.real))
                row = {"stage": res["meta"]["scene"], "pixel": iy * nx + ix,
                       "x_um": xs_um[ix], "y_um": ys_um[iy],
                       "W_re": float(W[ix, iy].real), "W_im": float(W[ix, iy].imag),
                       "I": float(I[ix, iy]), "I_ref": res["I_ref"],
                       "I_pixel": (None if res["I_pixel"] is None
                                   else float(res["I_pixel"][ix, iy])),
                       "mu_re": float(m.real) if ok else None,
                       "mu_im": float(m.imag) if ok else None,
                       "mu_abs": float(abs(m)) if ok else None,
                       "mu_null_reason": (None if ok else
                                          ("I_ref=0" if res["I_ref"] <= 0.0 else "I=0")),
                       "observable": wave["observable"], "trusted": bool(tr[ix, iy]),
                       "is_reference": ref_index is not None and (ix, iy) == tuple(ref_index)}
                if res["I_bore"] is not None:
                    row["I_bores"] = [float(v[ix, iy]) for v in res["I_bore"]]
                    row["I_int"] = float(I[ix, iy] - sum(v[ix, iy] for v in res["I_bore"]))
                if res["G12"] is not None:
                    row["G12_re"] = float(res["G12"][ix, iy].real)
                    row["G12_im"] = float(res["G12"][ix, iy].imag)
                fh.write(json.dumps(row) + "\n")
    meta = dict(res["meta"])
    meta.update({
        "wave_estimator": dict(wave),
        "screen": {"nx": nx, "ny": ny, "z_m": float(s.grid.z),
                   "cell_um": [m_to_um(s.hx), m_to_um(s.hy)],
                   "reference_requested_um": [m_to_um(res["ref_xy"][0]), m_to_um(res["ref_xy"][1])],
                   "reference_inside_window": bool(s.ref_inside),
                   "reference_cell_um": s.reference_cell_um(),
                   "reference_pixel": (None if ref_index is None
                                       else ref_index[1] * nx + ref_index[0])},
        "conventions": {"W": "<E(P) conj E(P_ref)>", "phasor": "exp(-i omega t), exp(+ikz)",
                        "I_coherent": "|integral_cell E dA|^2 (observable coherent_cell)",
                        "I_pixel": "integral_cell |sum_bores E|^2 dA = cell area x mean sub-pixel "
                                   "intensity of the total wave",
                        "units": UNITS},
        "status": "unverified", "I_ref": res["I_ref"], "n_trusted": res["n_trusted"],
        "seconds": res["seconds"], "capsysred_version": __version__})
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    names = ["mu-wave.jsonl", "meta.json"] + _maps(out, res, wave)
    prefix = (sub + "/") if sub else ""
    for n in names:
        log(f"  → {RESULT_DIR}/{prefix}{n}")
    return [prefix + n for n in names]


def _json_default(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    raise TypeError(f"not serializable: {type(v)}")


def _grid(arr, nx, ny, mask=None):
    """[iy][ix] rows for render; masked cells become None (checkerboard)."""
    return [[(float(arr[ix, iy]) if mask is None or mask[ix, iy] else None)
             for ix in range(nx)] for iy in range(ny)]


PHASE_MU_MIN = 1e-3     # phase of mu is drawn only where |mu| exceeds this (and is not exactly 0)


def _phase_mask(mu, trusted):
    """Trusted, finite, |mu| >= PHASE_MU_MIN and not an exact zero (W = 0 has no phase)."""
    a = np.abs(mu)
    return trusted & np.isfinite(a) & (a >= PHASE_MU_MIN) & (a > 0.0)


def _log_rows(values, imax):
    """log10(I / max): None for exact zeros (gap), true value for weak positives."""
    return [None if v <= 0.0 else math.log10(v / imax) for v in values]


def _maps(out, res, wave):
    """SVG maps of the contract: |mu|, I linear/log (zeros left blank), phase of mu with
    its own mask, I_pixel, per-bore intensities, Gamma_12 (Re/Im, modulus/phase) and the
    signed interference term. Empty curves are dropped, never faked."""
    s, nx, ny = res["sampler"], res["nx"], res["ny"]
    names = []
    ext = (m_to_um(s.grid.x0f), m_to_um(s.grid.x0f + s.grid.exf),
           m_to_um(s.grid.y0f), m_to_um(s.grid.y0f + s.grid.eyf))
    mark = (m_to_um(res["ref_xy"][0]), m_to_um(res["ref_xy"][1])) if s.ref_inside else None
    tr = res["trusted"]
    mu_abs, mu_arg = np.abs(res["mu"]), np.angle(res["mu"])
    pmask = _phase_mask(res["mu"], tr)
    sub = f"{res['meta']['provider']}, {res['meta']['n_nodes']} nodes, {wave['observable']}"
    two = res["I_bore"] is not None and len(res["I_bore"]) == 2
    zero = res["I"] <= 0.0

    def save(name, fig):
        render.save(os.path.join(out, name), fig)
        names.append(name)

    if ny > 1:
        row1 = [render.heatmap(_grid(mu_abs, nx, ny, tr), ext, "|μ(P, P_ref)| (wave)", "x, µm",
                               "y, µm", sub + f"; trusted {int(tr.sum())}/{nx * ny}", "|μ|",
                               mark=mark, vmax=1.0, w=430, equal=True),
                render.heatmap(_grid(res["I"], nx, ny), ext, "intensity", "x, µm", "y, µm",
                               "linear; " + wave["observable"], "I", w=430, equal=True),
                render.heatmap(_grid(res["I"], nx, ny, ~zero), ext, "intensity (log)", "x, µm",
                               "y, µm", "3 decades below max; exact zeros blank", "I",
                               w=430, equal=True, log=True)]
        save("16-wave-mu-intensity.svg", render.hstack(row1))
        row2 = [render.heatmap(_grid(mu_arg, nx, ny, pmask), ext, "arg μ(P, P_ref)", "x, µm", "y, µm",
                               f"trusted and |μ| ≥ {PHASE_MU_MIN:g}; masked elsewhere", "rad",
                               mark=mark, vmax=math.pi, w=430, equal=True, diverging=True)]
        if res["I_pixel"] is not None:
            row2.append(render.heatmap(_grid(res["I_pixel"], nx, ny), ext, "I_pixel (ordinary detector)",
                                       "x, µm", "y, µm", "∫cell |E|² dA of the total wave", "I_pixel",
                                       w=430, equal=True))
        save("16-wave-phase-pixel.svg", render.hstack(row2))
        if res["I_bore"] is not None:
            row3 = [render.heatmap(_grid(res["I_bore"][b], nx, ny), ext, f"I_{b + 1} (bore {b + 1} alone)",
                                   "x, µm", "y, µm", "same source, weights and observable", "I",
                                   w=430, equal=True) for b in range(len(res["I_bore"]))]
            if two:
                g12 = res["G12"]
                ga = np.abs(g12)
                gmask = (ga > 0.0) & (ga >= wave["intensity_floor"] * float(ga.max() or 1.0))
                i_int = res["I"] - res["I_bore"].sum(0)
                row3 += [render.heatmap(_grid(ga, nx, ny), ext, "|Γ₁₂(P)|", "x, µm", "y, µm",
                                        "cross term of the two bores at one point", "|Γ₁₂|",
                                        w=430, equal=True),
                         render.heatmap(_grid(np.angle(g12), nx, ny, gmask), ext, "arg Γ₁₂(P)", "x, µm",
                                        "y, µm", "masked where |Γ₁₂| = 0 or < floor × max", "rad",
                                        vmax=math.pi, w=430, equal=True, diverging=True),
                         render.heatmap(_grid(i_int, nx, ny), ext, "interference I_both − I₁ − I₂",
                                        "x, µm", "y, µm", "signed, symmetric scale", "I_int",
                                        w=430, equal=True, diverging=True)]
            save("16-wave-bores.svg", render.vstack(
                [render.hstack(row3[:3])] + ([render.hstack(row3[3:])] if len(row3) > 3 else [])))
    else:
        xs_um = [m_to_um(x) for x in s.px]
        vl = [(mark[0], "ref")] if mark else []
        series = []
        if tr.any():
            series.append({"xs": xs_um, "ys": _grid(mu_abs, nx, ny, tr)[0], "label": "wave |μ|"})
        if pmask.any():
            series.append({"xs": xs_um, "ys": [v / math.pi if v is not None else None
                                               for v in _grid(mu_arg, nx, ny, pmask)[0]],
                           "label": f"arg μ / π (|μ| ≥ {PHASE_MU_MIN:g})", "dash": "6,4"})
        if series:
            save("16-wave-mu.svg", render.line_chart(
                series, "|μ(x, x_ref)| and phase (wave)", "x, µm", "|μ|, arg μ/π",
                sub + f"; trusted {int(tr.sum())}/{nx}", vlines=vl, w=760))
        i_row = [float(v) for v in res["I"][:, 0]]
        imax = max(i_row)
        # coherent-observable scale S_coh = max(I, I_b): I, I_b, I_int and Gamma_12 share the
        # observable's units; I_pixel (ordinary detector, area scale) gets its own S_pixel
        s_coh = max([imax] + ([float(res["I_bore"].max())] if res["I_bore"] is not None else []))
        scale = s_coh if s_coh > 0.0 else 1.0           # technical divisor only; S_coh stays the true max
        pmax = float(res["I_pixel"].max()) if res["I_pixel"] is not None else None
        if s_coh > 0.0:
            note = "" if imax > 0.0 else "total I is zero everywhere; bores on the shared coherent scale"
        elif pmax:
            note = "coherent intensities are zero (shown without normalization); I_pixel is not"
        else:
            note = "all intensities are exactly zero"
        series = [{"xs": xs_um, "ys": [v / scale for v in i_row], "label": "I / S_coh"}]
        if res["I_bore"] is not None:
            for b in range(len(res["I_bore"])):
                series.append({"xs": xs_um, "label": f"I_{b + 1} / S_coh", "dash": "6,4",
                               "ys": [float(res["I_bore"][b][ix, 0]) / scale for ix in range(nx)]})
        if res["I_pixel"] is not None:
            series.append({"xs": xs_um, "ys": [float(v) / (pmax or 1.0) for v in res["I_pixel"][:, 0]],
                           "label": "I_pixel / S_pixel (ordinary detector, own scale)", "dash": "2,3"})
            note = (note + "; " if note else "") + f"S_pixel = {pmax:.3e} (area scale, not comparable)"
        save("16-wave-intensity.svg", render.line_chart(
            series, "intensity (profiles; two observables, two scales)", "x, µm", "I / S",
            (note + "; " if note else "") + f"S_coh = max(I, I_b) = {s_coh:.3e}", w=760))
        if imax > 0.0:
            save("16-wave-intensity-log.svg", render.line_chart(
                [{"xs": xs_um, "ys": _log_rows(i_row, imax), "label": "log10(I / max)"}],
                "intensity (log)", "x, µm", "log10(I / max)",
                f"exact zeros blank: {int(sum(v <= 0.0 for v in i_row))} of {nx}", y_zero=False, w=760))
        if two:
            i_int = [float(v) for v in (res["I"] - res["I_bore"].sum(0))[:, 0]]
            g12 = res["G12"][:, 0]
            save("16-wave-interference.svg", render.line_chart(
                [{"xs": xs_um, "ys": [v / scale for v in i_int], "label": "I_int / S_coh"},
                 {"xs": xs_um, "ys": [2.0 * float(v.real) / scale for v in g12],
                  "label": "2 Re Γ₁₂ / S_coh", "dash": "6,4"},
                 {"xs": xs_um, "ys": [2.0 * float(v.imag) / scale for v in g12],
                  "label": "2 Im Γ₁₂ / S_coh", "dash": "2,3"}],
                "interference term and complex Γ₁₂", "x, µm", "signed",
                f"S_coh = max(I, I_b) = {s_coh:.3e}", y_zero=False, w=760))
            ga = np.abs(g12)
            gmax = float(ga.max())
            gmask = (ga > 0.0) & (ga >= wave["intensity_floor"] * (gmax or 1.0))
            gseries = [{"xs": xs_um, "ys": [float(v) / (gmax or 1.0) for v in ga], "label": "|Γ₁₂| / max"}]
            if gmask.any():
                gseries.append({"xs": xs_um, "ys": [float(np.angle(v)) / math.pi if m else None
                                                    for v, m in zip(g12, gmask)],
                                "label": "arg Γ₁₂ / π (masked)", "dash": "6,4"})
            save("16-wave-gamma12.svg", render.line_chart(
                gseries, "Γ₁₂(P): modulus and phase", "x, µm", "|Γ₁₂|/max, arg/π",
                f"max |Γ₁₂| = {gmax:.3e}", y_zero=False, w=760))
    return names

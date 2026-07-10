"""Stages 1-6 of the capillary/coherence project on the Number engine.

One shared Monte-Carlo driver runs stages 4 (Lloyd wall) and 6 (capillaries);
stage 2 (free space) pushes the same ray stream through the stage-10 jackknife
estimator (optic = None), stage 12 keeps its pairwise Number ancestor.
Stages 3 and 5 are the deterministic references. Every optical stage cross-checks its analytic
Number hit against the RaySurface root-finding engine and the Fresnel factor
against xray.reflect_amplitude.
"""

import json
import math
import os
import sys
import time

from ..formula import Number
from .. import xray
from . import analytic, render, schematic
from .altcoh import run_alt_stage
from .beamlet import run_beamlet_stage
from .coherence import CoherenceAccumulator
from .jackknife import run_jack_stage
from .sketch import run_sketch_stage
from .validate import METHOD_LABELS, run_validate_stage
from .config import Config, load
from .nums import lift, solver, vunit
from .progress import Progress
from .screen import ScreenGrid
from .source import aim_disk_direction, slope_direction
from .spectrum import SpectralLine, spectral_lines, wavelength_m
from .surfaces import CapillaryBundle, Mirror, engine_hit_t, entrance_disk
from .symbolic import LineAmplitudes, ampl_template
from .fresnel import FresnelAmplitude
from .rays import RaysFile, scene_stream
from .types import RayRecord

ALL_STAGES = (1, 2, 3, 4, 5, 6)
KNOWN_STAGES = ALL_STAGES + (7, 8, 9, 10, 11, 12)  # 7 (alt), 8 (sketch), 9 (hit methods), 10 (jackknife), 11 (beamlets), 12 (pairwise free, ex-stage 2) — opt-in
_UM = 1e6


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _mm(x) -> str:
    return f"{float(x) * 1e3:g} mm"


def _um(x) -> str:
    return f"{float(x) * _UM:g} µm"


class Simulation:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lines = spectral_lines(cfg.spectrum, cfg.energy_kev)
        self.fresnel = FresnelAmplitude(cfg.material, cfg.energy_kev)
        # per_line: energy-dependent r(E_m) per spectral line via the symbolic template
        self.per_line = cfg.per_line_fresnel and len(self.lines) > 1
        self.line_amps = LineAmplitudes(cfg.material, self.lines, cfg.precision)
        self.lam = wavelength_m(cfg.energy_kev)
        p = cfg.precision
        self.theta_c = float(cfg.material.critical_angle(cfg.energy_kev, p))
        self.delta_f = float(cfg.material.delta(cfg.energy_kev, p))
        self.beta_f = float(cfg.material.beta(cfg.energy_kev, p))
        self.report = []
        self.files = []
        self.results = {}   # stage name -> MC result dict (maps, stats, ...)
        self.rays = None    # the run's RaysFile (trace once, stages consume)

    @classmethod
    def from_yaml(cls, path) -> "Simulation":
        return cls(load(path))

    @classmethod
    def from_dict(cls, raw: dict) -> "Simulation":
        return cls(load(raw))

    # ------------------------------------------------------------- helpers

    def _save(self, out_dir, name, fig):
        path = os.path.join(out_dir, name)
        render.save(path, fig)
        self.files.append(name)
        _log(f"  → {name}")

    def _spectrum_note(self) -> str:
        if len(self.lines) == 1:
            return f"monochromatic, E = {float(self.cfg.energy_kev):g} keV"
        kinds = {"gaussian": "Gaussian band", "lines": "discrete lines",
                 "table": "tabulated spectrum"}
        kind = kinds.get(self.cfg.spectrum.get("mode"), "lines")
        fres = "Fresnel per line" if self.per_line else "Fresnel at E₀"
        return (f"{len(self.lines)} lines around {float(self.cfg.energy_kev):g} keV "
                f"({kind}, {fres})")

    def _fresnel_check(self) -> str:
        theta = "1.0e-4"
        p = self.cfg.precision
        s = solver("sin(x)", p).number({"x": theta})
        r_fast = self.fresnel(s)
        r_ref = xray.reflect_amplitude(theta, str(self.cfg.energy_kev),
                                       self.cfg.material, p)
        r_sym = ampl_template(1, self.cfg.material, p).number(
            {"s1": str(s), "E": str(self.cfg.energy_kev)})
        diff = float(abs(r_fast - r_ref))
        diff_sym = float(abs(r_sym - r_ref))
        return (f"Fresnel r(θ=0.1 mrad): |r_CAPSYSred − r_xray| = {diff:.1e}; "
                f"|r_symbolic_template − r_xray| = {diff_sym:.1e}")

    # ------------------------------------------------------------- MC driver

    def _mc_stage(self, stage: str, label: str, src_cfg, scr_cfg, optic,
                  aim_factory, seed_offset: int, quick: int):
        cfg = self.cfg
        p = cfg.precision
        screen = ScreenGrid(scr_cfg)
        n_modes = max(2, src_cfg.n_modes // quick)
        n_rays = max(20, src_cfg.n_rays // quick)
        acc = CoherenceAccumulator(self.lines, screen.ref_pixel(scr_cfg.reference),
                                   cfg.precision)
        records, rays_from = scene_stream(self, stage, src_cfg, scr_cfg, optic,
                                          aim_factory, seed_offset, quick)
        stats = {"emitted": 0, "screen": 0, "absorbed": 0, "lost": 0,
                 "off_window": 0, "reflected_rays": 0, "reflections": 0,
                 "bounce_hist": {}}
        progress = Progress(label, n_modes * n_rays)
        t0 = time.time()
        mode_cur = None
        for rec in records:
            if rec.mode != mode_cur:
                if mode_cur is not None:
                    acc.fold_mode()
                acc.new_mode()
                mode_cur = rec.mode
            if not isinstance(rec.opl, Number):
                # file records carry strings; full-precision round-trip is exact
                rec = rec._replace(opl=Number(rec.opl, p),
                                   sins=tuple(Number(s, p) for s in rec.sins))
            stats["emitted"] += 1
            nb = len(rec.sins)
            fate, amps = rec.fate, None
            if fate == "screen":
                # geometry is energy-free; Fresnel enters here, per line or at E0
                amps = (self.line_amps(rec.sins) if self.per_line
                        else self.fresnel.product(rec.sins))
                if cfg.amplitude_min > 0.0:
                    peak = (max(float(abs(a)) for a in amps) if self.per_line
                            else float(abs(amps)))
                    if peak < cfg.amplitude_min:
                        fate = "absorbed"    # below threshold on every line
            if nb:
                stats["reflected_rays"] += 1
                stats["reflections"] += nb
                stats["bounce_hist"][nb] = stats["bounce_hist"].get(nb, 0) + 1
            if fate == "screen":
                if rec.pixel is None:
                    stats["off_window"] += 1
                else:
                    acc.add_ray(rec, amps)
                    stats["screen"] += 1
            else:
                stats[fate] += 1
            progress.step()
        if mode_cur is not None:
            acc.fold_mode()
        maps = acc.finalize(screen.nx, screen.ny)
        progress.finish(f"on screen {stats['screen']:,}")
        result = {"maps": maps, "stats": stats, "screen": screen,
                  "rays_from": rays_from, "n_modes": n_modes, "n_rays": n_rays,
                  "seconds": time.time() - t0, "src_cfg": src_cfg}
        return result

    # ------------------------------------------------------------- aiming

    def _aim_free(self, source, screen, rng):
        p = self.cfg.precision
        zscr = float(screen.z)

        def aim(origin):
            oxf, oyf, ozf = (float(c) for c in origin)
            dist = zscr - ozf
            pad_x, pad_y = 0.02 * screen.exf, 0.02 * screen.eyf
            mx = ((screen.x0f - pad_x - oxf) / dist,
                  (screen.x0f + screen.exf + pad_x - oxf) / dist)
            my = ((screen.y0f - pad_y - oyf) / dist,
                  (screen.y0f + screen.eyf + pad_y - oyf) / dist)
            return slope_direction(rng, mx, my, p)
        return aim

    def _aim_lloyd(self, source, screen, rng):
        p = self.cfg.precision
        z0f = float(self.cfg.lloyd.z0)
        zscr = float(screen.z)

        def aim(origin):
            oxf, oyf, ozf = (float(c) for c in origin)
            dist = zscr - ozf
            m_lo = 1.05 * (0.0 - oxf) / (z0f - ozf)   # a bit past the edge graze
            m_hi = (screen.x0f + screen.exf - oxf) / dist
            my = ((screen.y0f - oyf) / dist, (screen.y0f + screen.eyf - oyf) / dist)
            return slope_direction(rng, (m_lo, m_hi), my, p)
        return aim

    def _aim_capillary(self, source, screen, rng):
        cap = self.cfg.capillary
        z0f = float(cap.z0)
        disks = [entrance_disk(b, z0f) for b in cap.bores]
        weights = [a * a for _, _, a in disks]
        total = sum(weights)

        def aim(origin):
            u = rng.random() * total
            for (cx, cy, a), w in zip(disks, weights):
                u -= w
                if u <= 0.0:
                    return aim_disk_direction(rng, origin, cx, cy, a, z0f)
            cx, cy, a = disks[-1]
            return aim_disk_direction(rng, origin, cx, cy, a, z0f)
        return aim

    # ------------------------------------------------------------- stage 1

    def _stage1(self, out_dir):
        cfg = self.cfg
        cap = cfg.capillary
        src, scr = cfg.source, cfg.screen
        two_a = 2.0 * entrance_disk(cap.bores[0], float(cap.z0))[2]
        kinds = sorted({b.get("kind", "cylinder") for b in cap.bores})
        kind_note = "" if kinds == ["cylinder"] else f" [{', '.join(kinds)}]"
        d0 = float(cap.z0) - float(cap.source.position[2])
        d2 = float(cap.screen.z) - float(cap.z1)
        shape_ru = {"point": "point", "gaussian": "Gaussian",
                    "disk": "disk"}[src.shape]
        info = {
            "title": "Simulation layout: source → capillary(ies) → screen",
            "n_bores": len(cap.bores),
            "source_label": ["source",
                             f"{shape_ru}, {_um(src.size)}",
                             f"z = {_mm(src.position[2])}"],
            "capillary_title": (f"capillaries: {len(cap.bores)}, bore ⌀{two_a * _UM:g} µm, "
                                f"L = {_mm(float(cap.z1) - float(cap.z0))}{kind_note}"),
            "bore_label": f"2a = {two_a * _UM:g} µm",
            "screen_label": ["screen", f"{cap.screen.nx}×{cap.screen.ny} px"],
            "window_label": f"window {_um(cap.screen.edge_x)}",
            "d0_label": f"d₀ = {_mm(d0)} (capillary stage)",
            "len_label": f"L = {_mm(float(cap.z1) - float(cap.z0))}",
            "d2_label": f"d₂ = {_mm(d2)}",
            "description": [
                f"Energy E = {float(cfg.energy_kev):g} keV,  λ = {float(self.lam) * 1e10:.4f} Å;  spectrum: {self._spectrum_note()}.",
                f"Wall material: {cfg.material.name};  δ = {self.delta_f:.3e},  β = {self.beta_f:.3e},  θ_c = {self.theta_c * 1e3:.2f} mrad.",
                f"Source — a set of mutually incoherent point modes (van Cittert–Zernike method from a Monte-Carlo ensemble).",
                f"Engine precision: {cfg.precision} digits (Number/Solver, no float64 in the physics path);  seed = {cfg.seed}.",
                "Stages: 2 — |μ| on screen without optics (MC);  3 — van Cittert–Zernike analytics;  4 — Lloyd's mirror scheme",
                "(wall = capillary surface in the same tracer): |μ|, intensity, scheme;  5 — Lloyd analytics;",
                "6 — |μ| and intensity behind the capillary.",
                f"Free-field stage: source {_um(cfg.free_source.size)} at z = {_mm(cfg.free_source.position[2])}, screen z = {_mm(cfg.free_screen.z)}.",
                f"Lloyd: r₀ = {_um(cfg.lloyd.height)}, mirror z ∈ [{_mm(cfg.lloyd.z0)}, {_mm(cfg.lloyd.z1)}], source {_um(cfg.lloyd.source.size)}.",
            ],
        }
        self._save(out_dir, "01-scheme.svg", render.scheme_setup(info))
        # to-scale twin: real geometry, 10 traced rays, dimensioned axes
        G = schematic.build_geometry(cfg, "capillary")
        self._save(out_dir, "01a-scheme-traced.svg", schematic.compose(G))

    # ------------------------------------------------------------- stage 2+3

    def _stage2(self, out_dir, quick):
        """Free space through the stage-10 jackknife estimator: the same
        algorithm and outputs, optic = None."""
        res = run_jack_stage(self, "2/6 without optics (MC)", "free",
                             self.cfg.free_source, self.cfg.free_screen,
                             None, self._aim_free, 2, quick)
        self.results["free"] = res
        self._jack_outputs(out_dir, "02", "free", res)
        return res

    def _stage3(self, out_dir, res_free):
        cfg = self.cfg
        screen, maps = res_free["screen"], res_free["maps"]
        row = screen.ny // 2
        xs = screen.xs()
        xs_um = [x * _UM for x in xs]
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        src = res_free["src_cfg"]
        dist = float(screen.z) - float(src.position[2])
        lam_f = float(self.lam)
        mu_th = [analytic.vcz_mu(x - ref_xy[0], src.shape, float(src.size),
                                 lam_f, dist) for x in xs]
        mu_row, err_row = maps["mu"][row], maps["mu_err"][row]
        rms = analytic.rms_diff(mu_row, mu_th)
        if src.shape == "gaussian":
            xi = lam_f * dist / (2.0 * math.pi * float(src.size))
            note = f"ξ = λD/(2πσ) = {xi * _UM:.3f} µm;  "
        else:
            note = ""
        sub = (f"{note}RMS(MC − analytics) = {rms:.3f};  source: {src.shape}, "
               f"{_um(src.size)}, D = {_mm(dist)}")
        series = [{"xs": xs_um, "ys": mu_row, "label": "MC (stage 2) |μ| ± σ_jack",
                   "lo": [max(m - e, 0.0) for m, e in zip(mu_row, err_row)],
                   "hi": [min(m + e, 1.0) for m, e in zip(mu_row, err_row)]},
                  {"xs": xs_um, "ys": mu_th, "label": "van Cittert–Zernike analytics",
                   "dash": "6,4"}]
        dub_i = [i for i, d in enumerate(maps["dubious"][row]) if d > 0]
        if dub_i:
            series.append({"xs": [xs_um[i] for i in dub_i],
                           "ys": [mu_row[i] for i in dub_i],
                           "label": "don't trust: σ>1 / pinned at clamp",
                           "color": "#d62728", "dots": True})
        fig = render.line_chart(
            series, "Degree of coherence: analytics vs MC (without optics)",
            "x on screen, µm", "|μ|", sub,
            vlines=[(ref_xy[0] * _UM, "ref")], w=760)
        self._save(out_dir, "03-free-analytic-vs-mc.svg", fig)
        self.report += [
            "## Stage 3 — van Cittert–Zernike analytics",
            f"- RMS(|μ|_MC − |μ|_vCZ) = {rms:.4f}" + (f", ξ = {xi * _UM:.3f} µm" if src.shape == "gaussian" else ""),
        ]

    # ------------------------------------------------------------- stage 4+5

    def _stage4(self, out_dir, quick):
        cfg = self.cfg
        lloyd = cfg.lloyd
        mirror = Mirror(lloyd.z0, lloyd.z1)
        check = self._lloyd_engine_check(mirror)
        res = self._mc_stage("lloyd", "4/6 Lloyd (MC)", lloyd.source, lloyd.screen,
                             mirror, self._aim_lloyd, 3, quick)
        self.results["lloyd"] = res
        screen, maps = res["screen"], res["maps"]
        xs_um = [x * _UM for x in screen.xs()]
        row = screen.ny // 2
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        d_total = float(screen.z) - float(lloyd.source.position[2])
        dx_fringe = float(self.lam) * d_total / (2.0 * float(lloyd.height))
        x_ov = (float(lloyd.height) * d_total
                / (float(lloyd.z0) - float(lloyd.source.position[2]))
                - float(lloyd.height))
        st = res["stats"]
        sub = (f"{res['n_modes']} modes × {res['n_rays']} rays; reflected rays "
               f"{st['reflected_rays']:,}; x_ref = {ref_xy[0] * _UM:.2f} µm")
        vl = [(ref_xy[0] * _UM, "ref"), (x_ov * _UM, "overlap edge")]
        floor = 1.0 / math.sqrt(res["n_modes"])
        mu_fig = render.line_chart(
            [{"xs": xs_um, "ys": maps["mu"][row], "label": "MC |μ(x, x_ref)|"},
             {"xs": xs_um, "ys": [floor] * len(xs_um), "color": "#999",
              "dash": "2,4", "width": 1.0,
              "label": f"noise floor 1/√N modes ≈ {floor:.2f}"}],
            "Lloyd's mirror scheme: degree of coherence (MC, wall = capillary surface)",
            "x on screen, µm", "|μ|", sub, vlines=vl, w=680)
        self._save(out_dir, "04-lloyd-mc-coherence.svg", mu_fig)

        imax = max(maps["intensity"][row]) or 1.0
        dmax = max(max(r) for r in maps["density"]) or 1.0
        int_fig = render.line_chart(
            [{"xs": xs_um, "ys": [v / imax for v in maps["intensity"][row]],
              "label": "intensity (MC)"},
             {"xs": xs_um, "ys": [v / dmax for v in maps["density"][row]],
              "label": "ray density", "dash": "4,3"}],
            "Lloyd's mirror scheme: intensity on screen (MC)",
            "x on screen, µm", "I, arb. units",
            f"fringes Δx = λD/(2r₀) = {dx_fringe * _UM:.3f} µm; {sub}",
            vlines=vl, w=680)
        self._save(out_dir, "04a-lloyd-mc-intensity.svg", int_fig)

        src = lloyd.source
        info = {
            "title": "Lloyd's mirror scheme: wall instead of the capillary",
            "mirror_label": (f"mirror (glass, x<0), z ∈ [{_mm(lloyd.z0)}, {_mm(lloyd.z1)}] — "
                             "the same wall surface as the capillary"),
            "image_label": "virtual source (−r₀)",
            "source_label": ["source", f"Gaussian σ = {_um(src.size)}",
                             f"z = {_mm(src.position[2])}"],
            "height_label": f"r₀ = {_um(lloyd.height)}",
            "screen_label": ["screen", f"{screen.nx} px"],
            "window_label": f"window {_um(lloyd.screen.edge_x)}",
            "d0_label": f"d₀ = {_mm(float(lloyd.z0) - float(src.position[2]))}",
            "mirror_len_label": f"L = {_mm(float(lloyd.z1) - float(lloyd.z0))}",
            "total_label": f"D = {_mm(d_total)}",
            "description": [
                f"The direct and reflected rays interfere: Δx = λD/(2r₀) = {dx_fringe * _UM:.3f} µm,",
                f"overlap zone x ∈ [0, {x_ov * _UM:.2f} µm];  grazing angles "
                f"{float(lloyd.height) / (float(lloyd.z1) - float(src.position[2])) * 1e3:.3f}…"
                f"{float(lloyd.height) / (float(lloyd.z0) - float(src.position[2])) * 1e3:.3f} mrad ≪ θ_c = {self.theta_c * 1e3:.2f} mrad.",
                "arg r ≈ π below θ_c: a dark fringe at the mirror edge. Reflection is computed by the same tracer",
                "as the capillary (wall = capillary surface), with the complex Fresnel r.",
                check,
            ],
        }
        self._save(out_dir, "04b-lloyd-scheme.svg", render.scheme_lloyd(info))
        bh = ", ".join(f"{k}×: {v:,}" for k, v in sorted(st["bounce_hist"].items()))
        self.report += [
            "## Stage 4 — Lloyd's mirror scheme (MC)",
            f"- {res['n_modes']} modes × {res['n_rays']} rays; on screen {st['screen']:,}; absorbed {st['absorbed']:,}",
            f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
            f"- reflections per ray: {bh or 'none'}",
            f"- Δx (formula) = {dx_fringe * _UM:.3f} µm; overlap zone up to {x_ov * _UM:.2f} µm",
            f"- {check}",
            f"- time: {res['seconds']:.1f} s",
        ]
        return res

    def _lloyd_engine_check(self, mirror) -> str:
        lloyd = self.cfg.lloyd
        p = self.cfg.precision
        origin = lloyd.source.position
        z_mid = (float(lloyd.z0) + float(lloyd.z1)) / 2.0
        slope = -float(lloyd.height) / (z_mid - float(origin[2]))
        d = vunit((lift(slope, p), lift(0.0, p), lift(1.0, p)))
        event = mirror.next_event(origin, d)
        t_fast = event[1]
        t_engine = engine_hit_t(mirror.expr_um, origin, d,
                                1.2 * (float(lloyd.z1) - float(origin[2])),
                                method=self.cfg.engine_method)
        if t_engine is None:
            return "RaySurface engine check (mirror): no root found"
        rel = abs(float(t_fast - t_engine)) / float(t_fast)
        return (f"RaySurface engine check (mirror): |Δt|/t = {rel:.1e} "
                f"(analytics vs root finding)")

    def _stage5(self, out_dir, res_lloyd):
        cfg = self.cfg
        lloyd = cfg.lloyd
        screen, maps = res_lloyd["screen"], res_lloyd["maps"]
        row = screen.ny // 2
        xs = screen.xs()
        xs_um = [x * _UM for x in xs]
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        src = res_lloyd["src_cfg"]
        ref = analytic.lloyd_reference(
            xs, ref_xy[0], src.shape, float(src.size), float(lloyd.height),
            float(src.position[2]), float(lloyd.z0), float(lloyd.z1),
            float(screen.z), float(self.lam), self.delta_f, self.beta_f)
        mu_mc = maps["mu"][row]
        rms_mu = analytic.rms_diff(mu_mc, ref["mu"])
        i_mc = maps["intensity"][row]
        sm = [i_mc[max(0, i - 1)] * 0.25 + i_mc[i] * 0.5
              + i_mc[min(len(i_mc) - 1, i + 1)] * 0.25 for i in range(len(i_mc))]
        dx_mc = analytic.fringe_spacing(xs, sm)
        imax_mc = max(i_mc) or 1.0
        imax_th = max(ref["intensity"]) or 1.0
        x_ov = ref["x_overlap"]

        vis_win = [i for i, x in enumerate(xs) if 0.15 * x_ov < x < 0.85 * x_ov]

        def vis(curve):
            vals = [curve[i] for i in vis_win]
            return ((max(vals) - min(vals)) / (max(vals) + min(vals))
                    if vals and max(vals) + min(vals) > 0 else 0.0)

        corr = analytic.pearson(i_mc, ref["intensity"])
        sub_i = (f"Δx: analytics {ref['fringe_dx'] * _UM:.3f} µm"
                 + (f", MC {dx_mc * _UM:.3f} µm" if dx_mc else "")
                 + f";  I correlation: {corr:.3f};  visibility: MC {vis(sm):.2f}, "
                 f"analytics {vis(ref['intensity']):.2f}")
        int_fig = render.line_chart(
            [{"xs": xs_um, "ys": [v / imax_mc for v in i_mc], "label": "MC (stage 4)"},
             {"xs": xs_um, "ys": [v / imax_th for v in ref["intensity"]],
              "label": "analytics (2 paths, virtual source)", "dash": "6,4"}],
            "Lloyd: intensity — analytics vs MC",
            "x on screen, µm", "I, arb. units", sub_i, w=760)
        mu_fig = render.line_chart(
            [{"xs": xs_um, "ys": mu_mc, "label": "MC (stage 4)"},
             {"xs": xs_um, "ys": ref["mu"], "label": "analytics", "dash": "6,4"}],
            "Lloyd: degree of coherence — analytics vs MC",
            "x on screen, µm", "|μ|",
            f"RMS(MC − analytics) = {rms_mu:.3f};  x_ref = {ref_xy[0] * _UM:.2f} µm",
            vlines=[(ref_xy[0] * _UM, "ref")], w=760)
        self._save(out_dir, "05-lloyd-analytic-vs-mc.svg",
                   render.vstack([int_fig, mu_fig]))
        self.report += [
            "## Stage 5 — Lloyd analytics vs MC",
            f"- Δx: formula λD/(2r₀) = {ref['fringe_dx'] * _UM:.4f} µm"
            + (f", from MC peaks = {dx_mc * _UM:.4f} µm" if dx_mc else " (no MC peaks found)"),
            f"- Pearson correlation I_MC ↔ I_analytics: {corr:.3f}",
            f"- fringe visibility (0.15…0.85 of the overlap zone): MC {vis(sm):.3f}, analytics {vis(ref['intensity']):.3f} "
            "(MC minima are filled by shot noise; grows with the ray count)",
            f"- RMS(|μ|_MC − |μ|_analytics) = {rms_mu:.4f}",
        ]

    # ------------------------------------------------------------- stage 6

    def _stage6(self, out_dir, quick):
        cap = self.cfg.capillary
        bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1, self.cfg.engine_method)
        check = self._capillary_engine_check(bundle)
        res = self._mc_stage("capillary", "6/6 capillary (MC)", cap.source,
                             cap.screen, bundle, self._aim_capillary, 4,
                             quick)
        self.results["capillary"] = res
        screen, maps = res["screen"], res["maps"]
        st = res["stats"]
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        extent = (screen.x0f * _UM, (screen.x0f + screen.exf) * _UM,
                  screen.y0f * _UM, (screen.y0f + screen.eyf) * _UM)
        floor = 1.0 / math.sqrt(res["n_modes"])
        sub = (f"{res['n_modes']} modes × {res['n_rays']} rays; transmitted {st['screen']:,}; "
               f"absorbed {st['absorbed']:,}; reflections {st['reflections']:,}")
        sub_mu = (f"{res['n_modes']} modes × {res['n_rays']} rays; noise floor |μ| ≈ {floor:.2f}; "
                  "isolated bright pixels — low statistics")
        if screen.ny > 1:
            mu_fig = render.heatmap(maps["mu"], extent,
                                    "Capillary: degree of coherence |μ(P, P_ref)|",
                                    "x, µm", "y, µm", sub_mu, "|μ|",
                                    mark=(ref_xy[0] * _UM, ref_xy[1] * _UM), vmax=1.0,
                                    w=640)
            int_fig = render.heatmap(maps["intensity"], extent,
                                     "Capillary: intensity on screen",
                                     "x, µm", "y, µm", sub, "I, arb. units", w=640)
        else:
            xs_um = [x * _UM for x in screen.xs()]
            row = 0
            mu_fig = render.line_chart(
                [{"xs": xs_um, "ys": maps["mu"][row], "label": "MC |μ|"}],
                "Capillary: degree of coherence", "x, µm", "|μ|", sub)
            imax = max(maps["intensity"][row]) or 1.0
            int_fig = render.line_chart(
                [{"xs": xs_um, "ys": [v / imax for v in maps["intensity"][row]],
                  "label": "intensity"}],
                "Capillary: intensity", "x, µm", "I, arb. units", sub)
        self._save(out_dir, "06-capillary-mc-coherence.svg", mu_fig)
        self._save(out_dir, "06a-capillary-mc-intensity.svg", int_fig)
        bh = ", ".join(f"{k}×: {v:,}" for k, v in sorted(st["bounce_hist"].items()))
        mean_b = (st["reflections"] / st["reflected_rays"]
                  if st["reflected_rays"] else 0.0)
        self.report += [
            "## Stage 6 — capillary (MC)",
            f"- {res['n_modes']} modes × {res['n_rays']} rays; transmitted to the screen {st['screen']:,} "
            f"({100.0 * st['screen'] / st['emitted']:.1f}%); absorbed {st['absorbed']:,}",
            f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
            f"- reflections: total {st['reflections']:,}; per ray: {bh or 'none'}; mean {mean_b:.2f} per reflected ray",
            f"- {check}",
            f"- time: {res['seconds']:.1f} s",
        ]
        return res

    # ------------------------------------------------------------- stage 7

    def _stage7(self, out_dir, quick):
        """Alternative estimators (full W — axis C, Wigner — axis D) on the
        same ray streams as stages 2/6 (same seed offsets)."""
        cap = self.cfg.capillary
        scenes = [("free", "7 alt free (MC)", self.cfg.free_source,
                   self.cfg.free_screen, None, self._aim_free, 2)]
        if cap is None:
            self._skip_cap("## Stage 7 — alternative estimators [capillary]")
        else:
            scenes.append(
                ("capillary", "7 alt capillary (MC)", cap.source, cap.screen,
                 CapillaryBundle(cap.bores, cap.z0, cap.z1, self.cfg.engine_method),
                 self._aim_capillary, 4))
        rows = []
        for stage, label, src_cfg, scr_cfg, optic, aim_factory, off in scenes:
            if scr_cfg.ny != 1:
                _log(f"  7 [{stage}]: skipped — 2D screen (ny = {scr_cfg.ny}), estimators are 1D-only")
                self.report += [
                    f"## Stage 7 — alternative estimators [{stage}]",
                    f"- skipped: screen ny = {scr_cfg.ny}, estimators support ny = 1 only",
                ]
                continue
            res = run_alt_stage(self, label, stage, src_cfg, scr_cfg,
                                optic, aim_factory, off, quick)
            self.results[f"alt:{stage}"] = res
            maps, screen, st = res["maps"], res["screen"], res["stats"]
            xs_um = [x * _UM for x in screen.xs()]
            ref_x = xs_um[maps["ref_pixel"]]
            rms_full = analytic.rms_diff(maps["mu_pair"], maps["mu_full_col"])
            rms_wig = analytic.rms_diff(maps["mu_pair"], maps["mu_wigner"])
            imax = max(maps["intensity"]) or 1.0
            iwmax = max(maps["i_wigner"]) or 1.0
            rms_int = analytic.rms_diff(
                [v / imax for v in maps["intensity"]],
                [v / iwmax for v in maps["i_wigner"]])
            sub = (f"{res['n_modes']} modes × {res['n_rays']} rays; "
                   f"RMS pair↔fullW {rms_full:.1e}, pair↔Wigner {rms_wig:.3f}")
            fig = render.line_chart(
                [{"xs": xs_um, "ys": maps["mu_pair"], "label": "pairwise (reference)"},
                 {"xs": xs_um, "ys": maps["mu_full_col"],
                  "label": "full W, ref column", "dash": "6,4"},
                 {"xs": xs_um, "ys": maps["mu_wigner"],
                  "label": "Wigner phase space", "dash": "2,3"}],
                f"Stage 7 [{stage}]: |μ(x, x_ref)| by three estimators",
                "x on screen, µm", "|μ|", sub,
                vlines=[(ref_x, "ref")], w=760)
            self._save(out_dir, f"07-{stage}-alt-mu.svg", fig)
            extent = (xs_um[0], xs_um[-1], xs_um[0], xs_um[-1])
            fig = render.heatmap(
                maps["mu_full"], extent,
                f"Stage 7 [{stage}]: full |μ(x₁, x₂)| (no reference pixel)",
                "x₁, µm", "x₂, µm",
                "diagonal band width = coherence length; ray self-pairs off the diagonal",
                "|μ|", mark=(ref_x, ref_x), vmax=1.0, w=640)
            self._save(out_dir, f"07-{stage}-alt-fullw.svg", fig)
            grid, u_lo, u_hi = res["alt"].wigner_grid()
            fig = render.heatmap(
                grid, (xs_um[0], xs_um[-1], u_lo * 1e6, u_hi * 1e6),
                f"Stage 7 [{stage}]: phase space B(x, u) (ray histogram)",
                "x on screen, µm", "u = dx/dz, µrad",
                f"u bin {res['alt'].du * 1e6:.2f} µrad; intensity weights, no phases",
                "B", w=640)
            self._save(out_dir, f"07-{stage}-alt-wigner.svg", fig)
            for est, mu in (("pairwise", maps["mu_pair"]),
                            ("fullw_col", maps["mu_full_col"]),
                            ("wigner", maps["mu_wigner"])):
                for i, v in enumerate(mu):
                    rows.append({"stage": stage, "estimator": est, "pixel": i,
                                 "x_um": xs_um[i], "mu": v,
                                 "I": (maps["i_wigner"][i] if est == "wigner"
                                       else maps["intensity"][i])})
            self.report += [
                f"## Stage 7 — alternative estimators [{stage}]",
                f"- {res['n_modes']} modes × {res['n_rays']} rays; on screen {st['screen']:,} of {st['emitted']:,}",
                f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
                f"- RMS(|μ|_pair − |μ|_fullW_col) = {rms_full:.2e} (must be ~0: same sums)",
                f"- RMS(|μ|_pair − |μ|_Wigner) = {rms_wig:.4f}; RMS(I_pair − I_Wigner, normalized) = {rms_int:.2e}",
                f"- Fresnel float64 mirror vs Number (center line): |Δr| = {res['fresnel_check']:.1e}",
                f"- Wigner u bin: {res['alt'].du * 1e6:.3f} µrad",
                f"- time: {res['seconds']:.1f} s",
            ]
        path = os.path.join(out_dir, "mu-alt.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        self.files.append("mu-alt.jsonl")
        _log("  → mu-alt.jsonl")

    # ------------------------------------------------------------- stage 8

    def _stage8(self, out_dir, quick):
        """Streaming sketch of W (methods §3.10): pairwise reference column +
        Nystrom column + coherent-mode spectrum, 2D screens supported."""
        cap = self.cfg.capillary
        scenes = [("free", "8 sketch free (MC)", self.cfg.free_source,
                   self.cfg.free_screen, None, self._aim_free, 2)]
        if cap is None:
            self._skip_cap("## Stage 8 — sketch estimator [capillary]")
        else:
            scenes.append(
                ("capillary", "8 sketch capillary (MC)", cap.source, cap.screen,
                 CapillaryBundle(cap.bores, cap.z0, cap.z1, self.cfg.engine_method),
                 self._aim_capillary, 4))
        rows = []
        for stage, label, src_cfg, scr_cfg, optic, aim_factory, off in scenes:
            res = run_sketch_stage(self, label, stage, src_cfg, scr_cfg,
                                   optic, aim_factory, off, quick)
            self.results[f"sketch:{stage}"] = res
            maps, screen, st = res["maps"], res["screen"], res["stats"]
            flat = lambda grid: [v for row in grid for v in row]
            rms_engine = None
            num = self.results.get(stage)   # stage 2/6 Number maps, same rays
            if num is not None:
                mask = flat(maps["solid"])   # knife pixels: junk either way
                pairs = [(a, b) for a, b, s in zip(flat(maps["mu_pair"]),
                                                   flat(num["maps"]["mu"]), mask) if s]
                rms_engine = analytic.rms_diff([a for a, _ in pairs],
                                               [b for _, b in pairs])
            ref_xy = screen.pixel_xy(maps["ref_pixel"])
            sub = (f"{res['n_modes']} modes × {res['n_rays']} rays; rank r = {maps['rank']}; "
                   f"RMS pair↔sketch {maps['rms_pair_sketch']:.3f} on {maps['solid_px']} px")
            if screen.ny > 1:
                extent = (screen.x0f * _UM, (screen.x0f + screen.exf) * _UM,
                          screen.y0f * _UM, (screen.y0f + screen.eyf) * _UM)
                mark = (ref_xy[0] * _UM, ref_xy[1] * _UM)
                figs = [render.heatmap(maps["mu_pair"], extent,
                                       f"Stage 8 [{stage}]: pairwise |μ(P, P_ref)|",
                                       "x, µm", "y, µm", sub, "|μ|",
                                       mark=mark, vmax=1.0, w=430, equal=True),
                        render.heatmap(maps["mu_sketch"], extent,
                                       f"sketch r={maps['rank']}",
                                       "x, µm", "y, µm", "", "|μ|",
                                       mark=mark, vmax=1.0, w=430, equal=True),
                        render.heatmap(maps["mu_diff"], extent, "|pair − sketch|",
                                       "x, µm", "y, µm", "", "Δ", w=430, equal=True)]
                fig = render.hstack(figs)
            else:
                xs_um = [x * _UM for x in screen.xs()]
                fig = render.line_chart(
                    [{"xs": xs_um, "ys": maps["mu_pair"][0], "label": "pairwise"},
                     {"xs": xs_um, "ys": maps["mu_sketch"][0],
                      "label": f"sketch r={maps['rank']}", "dash": "6,4"}],
                    f"Stage 8 [{stage}]: |μ(x, x_ref)|", "x, µm", "|μ|", sub,
                    vlines=[(ref_xy[0] * _UM, "ref")], w=760)
            self._save(out_dir, f"08-{stage}-sketch-mu.svg", fig)
            lam = maps["lam"]
            top = min(len(lam), 60)
            l1 = lam[0] or 1.0
            fig = render.line_chart(
                [{"xs": list(range(1, top + 1)),
                  "ys": [v / l1 for v in lam[:top]], "label": "λ_n / λ_1"}],
                f"Stage 8 [{stage}]: coherent-mode spectrum of the field",
                "mode n", "λ_n / λ_1",
                f"N_eff = {maps['neff']:.1f}; 99% of energy in {maps['n99']} modes; "
                f"dark px {maps['dark_px']}", w=560)
            self._save(out_dir, f"08-{stage}-sketch-spectrum.svg", fig)
            ny, nx = screen.ny, screen.nx
            for iy in range(ny):
                for ix in range(nx):
                    rows.append({"stage": stage, "pixel": iy * nx + ix,
                                 "mu_pair": maps["mu_pair"][iy][ix],
                                 "mu_sketch": maps["mu_sketch"][iy][ix],
                                 "I": maps["intensity"][iy][ix]})
            self.report += [
                f"## Stage 8 — sketch estimator [{stage}]",
                f"- {res['n_modes']} modes × {res['n_rays']} rays; on screen {st['screen']:,} of {st['emitted']:,}",
                f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
                f"- rank r = {maps['rank']}; RMS(|μ|_pair − |μ|_sketch) = {maps['rms_pair_sketch']:.4f} "
                f"on {maps['solid_px']} solid px (dark: {maps['dark_px']})",
                f"- mode spectrum: N_eff = {maps['neff']:.2f}, 99% of energy in {maps['n99']} modes",
            ] + ([f"- RMS(|μ|_pair − |μ|_stage{'2' if stage == 'free' else '6'}_Number) = {rms_engine:.2e} (same rays, solid px)"]
                 if rms_engine is not None else []) + [
                f"- time: {res['seconds']:.1f} s",
            ]
        path = os.path.join(out_dir, "mu-sketch.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        self.files.append("mu-sketch.jsonl")
        _log("  → mu-sketch.jsonl")

    # ------------------------------------------------------------- stage 9

    def _stage9(self, out_dir, quick):
        """Hit-method cross-validation on the capillary scene: the first wall
        hit of each ray by python (reference) / C++ / implicit subdivision."""
        p = self.cfg.precision
        n_rays = max(100, self.cfg.validate_rays // quick)
        res = run_validate_stage(self, n_rays)
        self.results["validate"] = res
        st, per = res["stats"], res["per"]
        path = os.path.join(out_dir, "hit-validation.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in res["rows"]:
                fh.write(json.dumps(row) + "\n")
        self.files.append("hit-validation.jsonl")
        _log("  → hit-validation.jsonl")
        # match = same hit/pass call AND agreement to p digits minus 2 guard
        # digits of root-refinement wobble in the last places
        tol_exp = 2 - p
        lo, hi = -(p + 8), tol_exp + 5
        exps = [lo + 0.5 * k for k in range(int(2 * (hi - lo)) + 1)]
        series, agree = [], {}
        for m, s in per.items():
            denom = s["n"] + s["missing"] + s["extra"]
            if not denom:
                continue
            agree[m] = 100.0 * sum(1 for r in s["rels"]
                                   if r <= 10.0 ** tol_exp) / denom
            series.append({
                "xs": exps,
                "ys": [100.0 * sum(1 for r in s["rels"] if r <= 10.0 ** e) / denom
                       for e in exps],
                "label": f"{METHOD_LABELS[m]}: {agree[m]:.2f}% @1e{tol_exp}"})
        if series:
            fig = render.line_chart(
                series, "Stage 9: share of hits matching python analytics",
                "log₁₀ of the |Δt|/t tolerance", "matched, %",
                f"{st['hits']:,} wall hits of {st['rays']:,} rays; yaml precision "
                f"{p} digits − 2 guard ⇒ tol = 1e{tol_exp}; "
                "hit/pass mismatches never match",
                vlines=[(float(tol_exp), f"p = {p}")], w=760)
            self._save(out_dir, "09-hit-validation.svg", fig)
        self.report += [
            "## Stage 9 — hit-method cross-validation",
            f"- rays: {st['rays']:,}; wall hits {st['hits']:,}, passes {st['passes']:,}, "
            f"skipped {st['skipped']:,} (entrance web / `surface:` bores)",
            f"- python analytics (reference): {st['py_seconds']:.1f} s",
            f"- match tolerance from yaml precision: |Δt|/t ≤ 1e{tol_exp} "
            f"({p} digits − 2 guard)",
        ]
        if not res["native"]:
            self.report.append("- C++ twin: wall kind unsupported — engine method only")
        for m, s in per.items():
            self.report.append(
                f"- {METHOD_LABELS[m]}: matched {agree.get(m, 0.0):.2f}%; "
                f"max |Δt|/t = {s['max_rel']:.1e}, rms = {s['rms']:.1e} "
                f"on {s['n']:,} hits; missing/extra hits {s['missing']}/{s['extra']}; "
                f"{s['seconds']:.1f} s")
        self.report.append(f"- time: {res['seconds']:.1f} s")
        return res

    # ------------------------------------------------------------- stage 10

    def _stage10(self, out_dir, quick):
        """Stage-6 alternative (doc/mu28_legacy_fixed3_renamed.py, running-sum
        form): same capillary rays, |mu| plus a delete-one-mode jackknife map."""
        cap = self.cfg.capillary
        bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1, self.cfg.engine_method)
        res = run_jack_stage(self, "10 jackknife capillary (MC)", "capillary",
                             cap.source, cap.screen, bundle,
                             self._aim_capillary, 4, quick)
        self.results["jack:capillary"] = res
        self._jack_outputs(out_dir, "10", "capillary", res,
                           vs=self.results.get("capillary"))
        return res

    def _jack_outputs(self, out_dir, tag, scene, res, vs=None):
        """Jackknife scene outputs shared by stages 2 and 10: figures, report
        section, mu-jack.jsonl rows; vs = same-rays stage-6 result for Δμ."""
        maps, screen, st = res["maps"], res["screen"], res["stats"]
        nx, ny = screen.nx, screen.ny
        flat = lambda grid: [v for row in grid for v in row]
        n_lit = sum(1 for d in flat(maps["density"]) if d > 0)
        solid = [i for i, v in enumerate(flat(maps["solid"])) if v > 0]
        n_dub = sum(1 for v in flat(maps["dubious"]) if v > 0)
        errs = [flat(maps["mu_err"])[i] for i in solid]
        med_err = sorted(errs)[len(errs) // 2] if errs else 0.0
        floor = 1.0 / math.sqrt(res["n_modes"])
        rms6 = None
        if vs is not None and solid:
            a, b = flat(maps["mu"]), flat(vs["maps"]["mu"])
            rms6 = analytic.rms_diff([a[i] for i in solid], [b[i] for i in solid])
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        sub = (f"{res['n_modes']} modes × {res['n_rays']} rays; "
               f"σ_jack median {med_err:.3f}; noise floor |μ| ≈ {floor:.2f}; "
               f"solid px {len(solid)} of {n_lit} lit")
        if ny > 1:
            extent = (screen.x0f * _UM, (screen.x0f + screen.exf) * _UM,
                      screen.y0f * _UM, (screen.y0f + screen.eyf) * _UM)
            mark = (ref_xy[0] * _UM, ref_xy[1] * _UM)
            trust = [[1.0 if s > 0 and d == 0 else (0.5 if d > 0 else 0.0)
                      for s, d in zip(s_row, d_row)]
                     for s_row, d_row in zip(maps["solid"], maps["dubious"])]
            fig = render.hstack([
                render.heatmap(maps["mu"], extent,
                               "|μ(P, P_ref)| (jackknife)",
                               "x, µm", "y, µm", sub, "|μ|",
                               mark=mark, vmax=1.0, w=430, equal=True),
                render.heatmap(maps["mu_err"], extent, "σ_jack(P)",
                               "x, µm", "y, µm", "", "σ", w=430, equal=True),
                render.heatmap(trust, extent, "trust: 1 ok · ½ don't · 0 none",
                               "x, µm", "y, µm",
                               "½: σ>1, pinned at |μ|=1, or no jackknife; 0: no pairs",
                               "trust", vmax=1.0, w=430, equal=True)])
            self._save(out_dir, f"{tag}-{scene}-jack-mu.svg", fig)
            # y ≈ 0 slice of the three maps: |μ| ± σ_jack, σ_jack, trust
            iy0 = min(range(ny), key=lambda j: abs(screen.ys()[j]))
            y0_um = screen.ys()[iy0] * _UM
            xs_um = [x * _UM for x in screen.xs()]
            row_mu, row_err = maps["mu"][iy0], maps["mu_err"][iy0]
            dub_i = [i for i, d in enumerate(maps["dubious"][iy0]) if d > 0]
            mu_series = [{"xs": xs_um, "ys": row_mu, "label": "jackknife |μ| ± σ",
                          "lo": [max(m - e, 0.0) for m, e in zip(row_mu, row_err)],
                          "hi": [min(m + e, 1.0) for m, e in zip(row_mu, row_err)]}]
            err_series = [{"xs": xs_um, "ys": row_err, "label": "σ_jack"},
                          {"xs": xs_um, "ys": [floor] * nx,
                           "label": "1/√N_modes", "dash": "2,3"}]
            if dub_i:
                xd = [xs_um[i] for i in dub_i]
                mu_series.append({"xs": xd, "ys": [row_mu[i] for i in dub_i],
                                  "label": "don't trust: σ>1 / pinned at clamp",
                                  "color": "#d62728", "dots": True})
                err_series.append({"xs": xd, "ys": [row_err[i] for i in dub_i],
                                   "label": "don't trust",
                                   "color": "#d62728", "dots": True})
            vl = [(ref_xy[0] * _UM, "ref")] if maps["ref_pixel"] // nx == iy0 else []
            fig = render.hstack([
                render.line_chart(mu_series, "|μ(P, P_ref)| ± σ_jack", "x, µm",
                                  "|μ|", f"slice y = {y0_um:.2f} µm",
                                  vlines=vl, w=430),
                render.line_chart(err_series, "σ_jack(x)", "x, µm", "σ",
                                  f"slice y = {y0_um:.2f} µm", w=430),
                render.line_chart([{"xs": xs_um, "ys": trust[iy0]}],
                                  "trust: 1 ok · ½ don't · 0 none", "x, µm",
                                  "trust", f"slice y = {y0_um:.2f} µm", w=430)])
            self._save(out_dir, f"{tag}a-{scene}-jack-slice.svg", fig)
            fig = render.hstack([
                render.heatmap(maps["intensity"], extent, "intensity",
                               "x, µm", "y, µm", sub, "I, arb. units",
                               w=430, equal=True),
                render.heatmap(maps["density"], extent, "rays per pixel",
                               "x, µm", "y, µm", "", "rays", w=430, equal=True)])
            self._save(out_dir, f"{tag}b-{scene}-jack-intensity.svg", fig)
            if rms6 is not None:
                diff = [[abs(a - b) for a, b in zip(ra, rb)]
                        for ra, rb in zip(maps["mu"], vs["maps"]["mu"])]
                fig = render.heatmap(diff, extent,
                                     "|μ_jack − μ_stage6| (same rays)",
                                     "x, µm", "y, µm",
                                     f"RMS on solid px {rms6:.2e}; bright isolated px = "
                                     "stage-6 pairless residuals masked by the jackknife",
                                     "Δ", w=640)
                self._save(out_dir, f"{tag}c-{scene}-jack-vs6.svg", fig)
        else:
            xs_um = [x * _UM for x in screen.xs()]
            row_mu, row_err = maps["mu"][0], maps["mu_err"][0]
            dub_i = [i for i, d in enumerate(maps["dubious"][0]) if d > 0]
            series = [{"xs": xs_um, "ys": row_mu, "label": "jackknife |μ| ± σ",
                       "lo": [max(m - e, 0.0) for m, e in zip(row_mu, row_err)],
                       "hi": [min(m + e, 1.0) for m, e in zip(row_mu, row_err)]}]
            if vs is not None:
                series.append({"xs": xs_um, "ys": vs["maps"]["mu"][0],
                               "label": "stage 6 (Number)", "dash": "6,4"})
            if dub_i:
                series.append({"xs": [xs_um[i] for i in dub_i],
                               "ys": [row_mu[i] for i in dub_i],
                               "label": "don't trust: σ>1 / pinned at clamp",
                               "color": "#d62728", "dots": True})
            fig = render.line_chart(series,
                                    "|μ(x, x_ref)| with jackknife errors",
                                    "x, µm", "|μ|", sub,
                                    vlines=[(ref_xy[0] * _UM, "ref")], w=760)
            self._save(out_dir, f"{tag}-{scene}-jack-mu.svg", fig)
            err_series = [{"xs": xs_um, "ys": row_err, "label": "σ_jack"},
                          {"xs": xs_um, "ys": [floor] * nx,
                           "label": "1/√N_modes", "dash": "2,3"}]
            if dub_i:
                err_series.append({"xs": [xs_um[i] for i in dub_i],
                                   "ys": [row_err[i] for i in dub_i],
                                   "label": "don't trust",
                                   "color": "#d62728", "dots": True})
            fig = render.line_chart(
                err_series, "jackknife error by pixel", "x, µm", "σ", sub)
            self._save(out_dir, f"{tag}a-{scene}-jack-err.svg", fig)
            imax = max(maps["intensity"][0]) or 1.0
            dmax = max(maps["density"][0]) or 1.0
            fig = render.line_chart(
                [{"xs": xs_um, "ys": [v / imax for v in maps["intensity"][0]],
                  "label": "intensity / max"},
                 {"xs": xs_um, "ys": [v / dmax for v in maps["density"][0]],
                  "label": "rays / max", "dash": "6,4"}],
                "intensity and ray density", "x, µm", "normalized", sub)
            self._save(out_dir, f"{tag}b-{scene}-jack-intensity.svg", fig)
            if rms6 is not None:
                fig = render.line_chart(
                    [{"xs": xs_um,
                      "ys": [a - b for a, b in zip(row_mu, vs["maps"]["mu"][0])],
                      "label": "μ_jack − μ_stage6",
                      "lo": [-e for e in row_err], "hi": list(row_err)}],
                    "jackknife vs stage 6: Δμ with the ±σ_jack band", "x, µm", "Δμ",
                    f"RMS on solid px {rms6:.2e}; spikes = stage-6 pairless "
                    "residuals masked by the jackknife",
                    vlines=[(ref_xy[0] * _UM, "ref")], w=760, y_zero=False)
                self._save(out_dir, f"{tag}c-{scene}-jack-vs6.svg", fig)
        xs_um_all = [x * _UM for x in screen.xs()]
        ys_um_all = [y * _UM for y in screen.ys()]
        for iy in range(ny):
            for ix in range(nx):
                self.jack_rows.append({
                    "stage": scene, "pixel": iy * nx + ix,
                    "x_um": xs_um_all[ix], "y_um": ys_um_all[iy],
                    "mu": maps["mu"][iy][ix], "mu_err": maps["mu_err"][iy][ix],
                    "I": maps["intensity"][iy][ix],
                    "n_rays": int(maps["density"][iy][ix]),
                    "solid": bool(maps["solid"][iy][ix]),
                    "dubious": bool(maps["dubious"][iy][ix])})
        path = os.path.join(out_dir, "mu-jack.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in self.jack_rows:
                fh.write(json.dumps(row) + "\n")
        if "mu-jack.jsonl" not in self.files:
            self.files.append("mu-jack.jsonl")
        _log("  → mu-jack.jsonl")
        below = (100.0 * sum(1 for e in errs if e < floor) / len(errs)
                 if errs else 0.0)
        self.report += [
            f"## Stage {int(tag)} — jackknife estimator [{scene}]",
            f"- {res['n_modes']} modes × {res['n_rays']} rays; on screen {st['screen']:,} of {st['emitted']:,}",
            f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
            f"- solid pixels (≥2 same-mode rays, |μ| estimable): {len(solid)} of {n_lit} lit; "
            "the rest are masked to μ = σ = 0",
            f"- don't-trust estimates on solid px (σ > 1, pinned at |μ| = 1 with σ = 0, "
            f"or no usable jackknife): {n_dub} of {len(solid)}",
            f"- σ_jack on solid pixels: median {med_err:.4f}, max {max(errs, default=0.0):.4f}; "
            f"{below:.0f}% below the 1/√N floor ({floor:.3f})",
        ] + ([f"- RMS(|μ|_jack − |μ|_stage6) = {rms6:.2e} (same rays, solid pixels)"]
             if rms6 is not None else []) + [
            f"- time: {res['seconds']:.1f} s",
        ]

    # ------------------------------------------------------------- stage 11

    def _stage11(self, out_dir, quick):
        """Beamlet estimator (doc/2026-07-10-stage11-beamlets.ru.md):
        elliptic Gaussian phase spots instead of point bins, the 2x2 Gamma
        tensor through the bounces (general astigmatism), honest mu with no
        self-pair subtraction. Free scene validates against vCZ; the
        capillary scene compares to stage 6 on the same rays."""
        cap = self.cfg.capillary
        scenes = [("free", "11 beamlet free (MC)", self.cfg.free_source,
                   self.cfg.free_screen, None, self._aim_free, 2)]
        if cap is None:
            self._skip_cap("## Stage 11 — beamlet estimator [capillary]")
        else:
            scenes.append(
                ("capillary", "11 beamlet capillary (MC)", cap.source, cap.screen,
                 CapillaryBundle(cap.bores, cap.z0, cap.z1, self.cfg.engine_method),
                 self._aim_capillary, 4))
        rows = []
        for stage, label, src_cfg, scr_cfg, optic, aim_factory, off in scenes:
            res = run_beamlet_stage(self, label, stage, src_cfg, scr_cfg,
                                    optic, aim_factory, off, quick)
            self.results[f"beamlet:{stage}"] = res
            maps, screen, st = res["maps"], res["screen"], res["stats"]
            nx, ny = screen.nx, screen.ny
            flat = lambda grid: [v for row in grid for v in row]
            ref_xy = screen.pixel_xy(maps["ref_pixel"])
            sub = (f"{res['n_modes']} modes × {res['n_rays']} rays; "
                   f"w₀ = {self.cfg.beamlet_w0 * _UM:.2f} µm, "
                   f"mean w on screen = {maps['w_mean'] * _UM:.2f} µm")
            report = [f"## Stage 11 — beamlet estimator [{stage}]",
                      f"- {res['n_modes']} modes × {res['n_rays']} rays; on screen "
                      f"{st['screen']:,} of {st['emitted']:,} (tails off window: {st['off_window']:,})",
                      f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
                      f"- w₀ = {self.cfg.beamlet_w0 * _UM:.2f} µm; mean spot width on screen "
                      f"= {maps['w_mean'] * _UM:.2f} µm; Γ-tensor deposit; honest |μ| "
                      "(no self-pair subtraction)"]
            if maps["flat_walls"]:
                report.append("- implicit bore(s): no closed-form curvature — "
                              "flat-wall (scalar q) bounces")
            if maps["gamma_bad"]:
                report.append(f"- deposits skipped (beam blew up, Im G ⊁ 0): "
                              f"{maps['gamma_bad']:,}")
            row = ny // 2
            xs_um = [x * _UM for x in screen.xs()]
            if stage == "free":
                src = src_cfg
                dist = float(screen.z) - float(src.position[2])
                mu_th = [analytic.vcz_mu(x - ref_xy[0], src.shape,
                                         float(src.size), float(self.lam), dist)
                         for x in screen.xs()]
                rms = analytic.rms_diff(maps["mu"][row], mu_th)
                fig = render.line_chart(
                    [{"xs": xs_um, "ys": maps["mu"][row], "label": "beamlets |μ|"},
                     {"xs": xs_um, "ys": mu_th,
                      "label": "van Cittert–Zernike analytics", "dash": "6,4"}],
                    "Stage 11 [free]: beamlet |μ| vs vCZ analytics",
                    "x, µm", "|μ|", f"RMS(beamlets − vCZ) = {rms:.3f};  {sub}",
                    vlines=[(ref_xy[0] * _UM, "ref")], w=760)
                self._save(out_dir, "11-free-beamlet-mu.svg", fig)
                report.append(f"- RMS(|μ|_beamlet − |μ|_vCZ) = {rms:.4f}")
            else:
                num = self.results.get("capillary")   # stage 6 maps, same rays
                extent = (screen.x0f * _UM, (screen.x0f + screen.exf) * _UM,
                          screen.y0f * _UM, (screen.y0f + screen.eyf) * _UM)
                if ny > 1:
                    mark = (ref_xy[0] * _UM, ref_xy[1] * _UM)
                    fig = render.hstack([
                        render.heatmap(maps["mu"], extent,
                                       "Stage 11: beamlet |μ(P, P_ref)|",
                                       "x, µm", "y, µm", sub, "|μ|",
                                       mark=mark, vmax=1.0, w=430, equal=True),
                        render.heatmap(maps["intensity"], extent,
                                       "beamlet intensity", "x, µm", "y, µm",
                                       "coherent beamlet sum |Σ g|²",
                                       "I, arb. units", w=430, equal=True)])
                    self._save(out_dir, "11-capillary-beamlet-mu.svg", fig)
                else:
                    fig = render.line_chart(
                        [{"xs": xs_um, "ys": maps["mu"][0], "label": "beamlets |μ|"}],
                        "Stage 11: beamlet |μ(x, x_ref)|", "x, µm", "|μ|", sub,
                        vlines=[(ref_xy[0] * _UM, "ref")], w=760)
                    self._save(out_dir, "11-capillary-beamlet-mu.svg", fig)
                    imax = max(maps["intensity"][0]) or 1.0
                    fig = render.line_chart(
                        [{"xs": xs_um,
                          "ys": [v / imax for v in maps["intensity"][0]],
                          "label": "beamlet intensity"}],
                        "Stage 11: beamlet intensity", "x, µm", "I, arb. units", sub)
                    self._save(out_dir, "11a-capillary-beamlet-intensity.svg", fig)
                lit = ([i for i, d in enumerate(flat(num["maps"]["density"]))
                        if d > 0] if num is not None else [])
                if lit:
                    a, b = flat(maps["mu"]), flat(num["maps"]["mu"])
                    rms6 = analytic.rms_diff([a[i] for i in lit],
                                             [b[i] for i in lit])
                    sub6 = (f"RMS on lit px {rms6:.3f}; same rays, different "
                            "estimators: stage 6 subtracts ray self-pairs, "
                            "beamlets smear the field")
                    if ny > 1:
                        diff = [[abs(x - y) for x, y in zip(ra, rb)]
                                for ra, rb in zip(maps["mu"], num["maps"]["mu"])]
                        fig = render.heatmap(diff, extent,
                                             "|μ_beamlet − μ_stage6| (same rays)",
                                             "x, µm", "y, µm", sub6, "Δ", w=640)
                    else:
                        fig = render.line_chart(
                            [{"xs": xs_um,
                              "ys": [x - y for x, y in
                                     zip(maps["mu"][0], num["maps"]["mu"][0])],
                              "label": "μ_beamlet − μ_stage6"}],
                            "Stage 11 vs 6: Δμ (same rays)", "x, µm", "Δμ",
                            sub6, w=760, y_zero=False)
                    self._save(out_dir, "11b-capillary-beamlet-vs6.svg", fig)
                    report.append(
                        f"- RMS(|μ|_beamlet − |μ|_stage6) = {rms6:.4f} on "
                        f"{len(lit)} lit px (same rays; estimators differ — "
                        "stage 6 subtracts self-pairs, beamlets do not)")
            ys_um = [y * _UM for y in screen.ys()]
            for iy in range(ny):
                for ix in range(nx):
                    rows.append({"stage": stage, "pixel": iy * nx + ix,
                                 "x_um": xs_um[ix], "y_um": ys_um[iy],
                                 "mu": maps["mu"][iy][ix],
                                 "I": maps["intensity"][iy][ix],
                                 "n_rays": int(maps["density"][iy][ix])})
            report.append(f"- time: {res['seconds']:.1f} s")
            self.report += report
        path = os.path.join(out_dir, "mu-beamlet.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        self.files.append("mu-beamlet.jsonl")
        _log("  → mu-beamlet.jsonl")

    # ------------------------------------------------------------- stage 12

    def _stage12(self, out_dir, quick):
        """The pre-jackknife stage 2: pairwise Number estimator on the free
        scene (same rays as stage 2, no σ), kept for cross-checks."""
        res = self._mc_stage("free", "12 pairwise free (MC)", self.cfg.free_source,
                             self.cfg.free_screen, None, self._aim_free, 2,
                             quick)
        self.results["pairwise:free"] = res
        screen, maps = res["screen"], res["maps"]
        xs_um = [x * _UM for x in screen.xs()]
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        sub = (f"{res['n_modes']} modes × {res['n_rays']} rays, {self._spectrum_note()}, "
               f"x_ref = {ref_xy[0] * _UM:.2f} µm")
        row = screen.ny // 2
        floor = 1.0 / math.sqrt(res["n_modes"])
        mu_fig = render.line_chart(
            [{"xs": xs_um, "ys": maps["mu"][row], "label": "MC |μ(x, x_ref)|"},
             {"xs": xs_um, "ys": [floor] * len(xs_um), "color": "#999",
              "dash": "2,4", "width": 1.0,
              "label": f"noise floor 1/√N modes ≈ {floor:.2f}"}],
            "Degree of coherence without optics (pairwise Number)",
            "x on screen, µm", "|μ|", sub,
            vlines=[(ref_xy[0] * _UM, "ref")], w=640)
        imax = max(max(r) for r in maps["intensity"]) or 1.0
        dmax = max(max(r) for r in maps["density"]) or 1.0
        int_fig = render.line_chart(
            [{"xs": xs_um, "ys": [v / imax for v in maps["intensity"][row]],
              "label": "intensity"},
             {"xs": xs_um, "ys": [v / dmax for v in maps["density"][row]],
              "label": "ray density", "dash": "4,3"}],
            "Intensity and density (sampling check)",
            "x on screen, µm", "arb. units", sub, w=640)
        self._save(out_dir, "12-free-mc-coherence.svg",
                   render.hstack([mu_fig, int_fig]))
        st = res["stats"]
        self.report += [
            "## Stage 12 — |μ| without optics (pairwise MC)",
            f"- modes: {res['n_modes']}, rays/mode: {res['n_rays']}, on screen: {st['screen']:,} of {st['emitted']:,}",
            f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
            f"- time: {res['seconds']:.1f} s",
        ]
        return res

    def _capillary_engine_check(self, bundle) -> str:
        cap = self.cfg.capillary
        p = self.cfg.precision
        wall = bundle.walls[0]
        if wall.expr_um is None:
            return ("RaySurface engine check (capillary): generic `surface` bore — "
                    "hits come from the engine itself")
        bore = cap.bores[0]
        origin = (bore["center"][0], bore["center"][1], cap.z0)
        slope = 0.8 * wall.aim[2] / max((float(cap.z1) - float(cap.z0)) / 8.0, 1e-9)
        px, py = wall.probe_xy
        d = vunit((lift(slope * px, p), lift(slope * py, p), lift(1.0, p)))
        event = bundle.next_event(origin, d)
        if event[0] != "reflect":
            return "RaySurface engine check (capillary): probe ray did not reach the wall"
        t_fast = event[1]
        t_engine = engine_hit_t(wall.expr_um, origin, d,
                                1.2 * (float(cap.z1) - float(cap.z0)),
                                method=self.cfg.engine_method)
        if t_engine is None:
            return "RaySurface engine check (capillary): no root found"
        rel = abs(float(t_fast - t_engine)) / float(t_fast)
        return (f"RaySurface engine check (capillary wall, {wall.kind}): "
                f"|Δt|/t = {rel:.1e}")

    def _skip_cap(self, heading):
        """Skip note for a capillary stage/scene when the config has none."""
        _log(f"  {heading.lstrip('# ')}: skipped — no capillary in the config")
        self.report += [heading, "- skipped: no capillary section in the config"]

    # ------------------------------------------------------------- run

    def run(self, out_dir, stages=None, quick: int = 1) -> dict:
        cfg = self.cfg
        wanted = set(stages or ALL_STAGES)
        if 3 in wanted:
            wanted.add(2)
        if 5 in wanted:
            wanted.add(4)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        _log(f"CAPSYSred: stages {sorted(wanted)}, output to {out_dir}"
             + (f", speedup ×{quick}" if quick > 1 else ""))
        fres_check = self._fresnel_check()
        _log("  " + fres_check)
        self.report = [
            "# CAPSYSred report",
            "",
            f"- energy: {float(cfg.energy_kev):g} keV (λ = {float(self.lam) * 1e10:.4f} Å); spectrum: {self._spectrum_note()}",
            f"- material: {cfg.material.name}; δ = {self.delta_f:.3e}, β = {self.beta_f:.3e}, θ_c = {self.theta_c * 1e3:.2f} mrad",
            f"- precision: {cfg.precision} digits; seed = {cfg.seed}",
            f"- Fresnel: {'per spectral line (per_line_fresnel)' if self.per_line else 'at the central energy E₀'}"
            + (f"; records: every {cfg.sample_every}-th ray" if cfg.sample_every > 1 else ""),
            f"- {fres_check}",
            "",
        ]
        self.files = []
        self.jack_rows = []
        rays_name = "rays.jsonl.gz" if cfg.rays_gzip else "rays.jsonl"
        self.rays = (RaysFile(os.path.join(out_dir, rays_name), cfg, quick)
                     if cfg.rays_jsonl and wanted & {2, 4, 6, 7, 8, 10, 11, 12}
                     else None)
        try:
            if 1 in wanted:
                _log("Stage 1/6: simulation layout")
                if cfg.capillary is None:
                    _log("  skipped — no capillary in the config")
                else:
                    self._stage1(out_dir)
            res_free = None
            if 2 in wanted:
                _log("Stage 2/6: |μ| without optics (jackknife estimator, same tracer)")
                res_free = self._stage2(out_dir, quick)
            if 3 in wanted:
                _log("Stage 3/6: van Cittert–Zernike analytics")
                self._stage3(out_dir, res_free)
            res_lloyd = None
            if 4 in wanted:
                _log("Stage 4/6: Lloyd's mirror scheme — wall instead of the capillary (MC)")
                res_lloyd = self._stage4(out_dir, quick)
            if 5 in wanted:
                _log("Stage 5/6: Lloyd analytics vs MC")
                self._stage5(out_dir, res_lloyd)
            if 6 in wanted:
                _log("Stage 6/6: capillary (MC)")
                if cfg.capillary is None:
                    self._skip_cap("## Stage 6 — capillary (MC)")
                else:
                    self._stage6(out_dir, quick)
            if 7 in wanted:
                _log("Stage 7: alternative estimators — full W (axis C) + Wigner (axis D)")
                self._stage7(out_dir, quick)
            if 8 in wanted:
                _log("Stage 8: streaming sketch of W — column + mode spectrum")
                self._stage8(out_dir, quick)
            if 9 in wanted:
                _log("Stage 9: hit-method cross-validation — python / C++ / subdivision")
                if cfg.capillary is None:
                    self._skip_cap("## Stage 9 — hit-method cross-validation")
                else:
                    self._stage9(out_dir, quick)
            if 10 in wanted:
                _log("Stage 10: stage-6 estimator + delete-one-mode jackknife errors")
                if cfg.capillary is None:
                    self._skip_cap("## Stage 10 — jackknife estimator [capillary]")
                else:
                    self._stage10(out_dir, quick)
            if 11 in wanted:
                _log("Stage 11: beamlet estimator — elliptic phase spots (Γ tensor, general astigmatism)")
                self._stage11(out_dir, quick)
            if 12 in wanted:
                _log("Stage 12: pairwise Number estimator without optics (the pre-jackknife stage 2)")
                self._stage12(out_dir, quick)
        finally:
            if self.rays is not None:
                self.rays.close()
        if self.rays is not None:
            self.files.append(rays_name)
            _log(f"  → {rays_name}")
        self.report += ["", "## Files", ""]
        self.report += [f"- {name}" for name in self.files + ["report.md"]]
        report_path = os.path.join(out_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.report) + "\n")
        self.files.append("report.md")
        _log(f"  → report.md")
        _log(f"Done in {time.time() - t0:.0f} s.")
        return {"out_dir": out_dir, "files": list(self.files)}

    # ------------------------------------------------------------- replay

    def replay(self, records_path, out_dir) -> dict:
        """Re-evaluate recorded rays on THIS config's spectrum/material — no tracing.

        Ray geometry (fate, pixel, opl, angles) comes from rays.jsonl and is
        energy-independent; the spectrum and the material may differ from the
        recording run, the rest of the config must match it.
        """
        cfg = self.cfg
        os.makedirs(out_dir, exist_ok=True)
        screens = {"free": cfg.free_screen, "lloyd": cfg.lloyd.screen}
        if cfg.capillary is not None:
            screens["capillary"] = cfg.capillary.screen
        by_stage = {}
        with open(records_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if "stage" in row:   # skip the v2 meta line and scene trailers
                    by_stage.setdefault(row["stage"], []).append(row)
        if not by_stage:
            raise ValueError(f"no ray records in {records_path!r}")
        p = cfg.precision
        # Frozen-Fresnel replay evaluates one amplitude at E0 for every line.
        amps_of = (self.line_amps if self.per_line else LineAmplitudes(
            cfg.material, [SpectralLine(cfg.energy_kev, None, 1.0)], p))
        self.files = []
        report = [
            "# CAPSYSred report — replay",
            "",
            f"- records: {records_path}",
            f"- energy: {float(cfg.energy_kev):g} keV; spectrum: {self._spectrum_note()}",
            f"- material: {cfg.material.name}; precision: {cfg.precision} digits",
            "- ray geometry comes from the records (no tracing): the config must match"
            " the recording config in everything except the spectrum/material",
            "",
        ]
        for stage, rows in sorted(by_stage.items()):
            if stage not in screens:
                raise ValueError(f"unknown stage in records: {stage!r}")
            scr_cfg = screens[stage]
            screen = ScreenGrid(scr_cfg)
            acc = CoherenceAccumulator(self.lines,
                                       screen.ref_pixel(scr_cfg.reference), p)
            stats = {"rays": 0, "screen": 0, "absorbed": 0, "lost": 0,
                     "off_window": 0, "below_min": 0}
            mode_cur = None
            t0 = time.time()
            for row in rows:
                if row["mode"] != mode_cur:
                    if mode_cur is not None:
                        acc.fold_mode()
                    acc.new_mode()
                    mode_cur = row["mode"]
                stats["rays"] += 1
                if row["fate"] != "screen":
                    stats[row["fate"]] = stats.get(row["fate"], 0) + 1
                    continue
                if row["pixel"] is None:
                    stats["off_window"] += 1
                    continue
                amps = amps_of(row["sins"])
                if (cfg.amplitude_min > 0.0
                        and max(float(abs(a)) for a in amps) < cfg.amplitude_min):
                    stats["below_min"] += 1      # below threshold on every line
                    continue
                # v1 records carry no arrival point/direction/refl
                rec = RayRecord(row["mode"], row["ray"], "screen",
                                int(row["pixel"]), None, None,
                                Number(row["opl"], p), tuple(row["sins"]), None)
                acc.add_ray(rec, amps if self.per_line else amps[0])
                stats["screen"] += 1
            if mode_cur is not None:
                acc.fold_mode()
            maps = acc.finalize(screen.nx, screen.ny)
            self.results[f"replay:{stage}"] = {"maps": maps, "screen": screen,
                                               "stats": stats}
            self._replay_figs(out_dir, stage, screen, maps)
            report += [
                f"## {stage}",
                f"- rays in records: {stats['rays']:,}; to the screen: {stats['screen']:,};"
                f" absorbed/lost in records: {stats['absorbed']:,}/{stats['lost']:,};"
                f" below threshold on the new spectrum: {stats['below_min']:,}",
                f"- time: {time.time() - t0:.1f} s",
                "",
            ]
        report += ["## Files", ""] + [f"- {n}"
                                      for n in self.files + ["report-replay.md"]]
        with open(os.path.join(out_dir, "report-replay.md"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(report) + "\n")
        self.files.append("report-replay.md")
        _log("  → report-replay.md")
        return {"out_dir": out_dir, "files": list(self.files)}

    def _replay_figs(self, out_dir, stage, screen, maps):
        extent = (screen.x0f * _UM, (screen.x0f + screen.exf) * _UM,
                  screen.y0f * _UM, (screen.y0f + screen.eyf) * _UM)
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        sub = self._spectrum_note()
        if screen.ny > 1:
            mu_fig = render.heatmap(maps["mu"], extent,
                                    f"Replay {stage}: |μ(P, P_ref)|",
                                    "x, µm", "y, µm", sub, "|μ|",
                                    mark=(ref_xy[0] * _UM, ref_xy[1] * _UM),
                                    vmax=1.0, w=640)
            int_fig = render.heatmap(maps["intensity"], extent,
                                     f"Replay {stage}: intensity",
                                     "x, µm", "y, µm", sub, "I, arb. units", w=640)
        else:
            xs_um = [x * _UM for x in screen.xs()]
            mu_fig = render.line_chart(
                [{"xs": xs_um, "ys": maps["mu"][0], "label": "|μ| (replay)"}],
                f"Replay {stage}: degree of coherence", "x, µm", "|μ|", sub,
                vlines=[(ref_xy[0] * _UM, "ref")])
            imax = max(maps["intensity"][0]) or 1.0
            int_fig = render.line_chart(
                [{"xs": xs_um, "ys": [v / imax for v in maps["intensity"][0]],
                  "label": "I (replay)"}],
                f"Replay {stage}: intensity", "x, µm", "I, arb. units", sub)
        self._save(out_dir, f"replay-{stage}-coherence.svg", mu_fig)
        self._save(out_dir, f"replay-{stage}-intensity.svg", int_fig)

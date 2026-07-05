"""Stages 1-6 of the capillary/coherence project on the Number engine.

One shared Monte-Carlo driver runs stages 2 (free space), 4 (Lloyd wall) and
6 (capillaries) — the optic argument is the only difference. Stages 3 and 5 are
the deterministic references. Every optical stage cross-checks its analytic
Number hit against the RaySurface root-finding engine and the Fresnel factor
against xray.reflect_amplitude.
"""

import json
import math
import os
import random
import sys
import time

from ..formula import Number
from .. import xray
from . import analytic, render
from .coherence import CoherenceAccumulator
from .config import Config, load
from .nums import lift, solver, vunit
from .progress import Progress
from .screen import ScreenGrid
from .source import Source, aim_disk_direction, slope_direction
from .spectrum import SpectralLine, spectral_lines, wavelength_m
from .surfaces import CapillaryBundle, Mirror, engine_hit_t, entrance_disk
from .symbolic import LineAmplitudes, ampl_template
from .fresnel import FresnelAmplitude
from .trace import trace_ray

ALL_STAGES = (1, 2, 3, 4, 5, 6)
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
        return (f"Fresnel r(θ=0.1 mrad): |r_capsim − r_xray| = {diff:.1e}; "
                f"|r_symbolic_template − r_xray| = {diff_sym:.1e}")

    # ------------------------------------------------------------- MC driver

    def _mc_stage(self, stage: str, label: str, src_cfg, scr_cfg, optic,
                  aim_factory, seed_offset: int, quick: int, refl_fh,
                  rays_fh=None):
        cfg = self.cfg
        rng = random.Random(cfg.seed * 1000003 + seed_offset)
        source = Source(src_cfg, rng)
        screen = ScreenGrid(scr_cfg)
        n_modes = max(2, src_cfg.n_modes // quick)
        n_rays = max(20, src_cfg.n_rays // quick)
        acc = CoherenceAccumulator(self.lines, screen.ref_pixel(scr_cfg.reference),
                                   cfg.precision)
        aim = aim_factory(source, screen, rng)
        stats = {"emitted": 0, "screen": 0, "absorbed": 0, "lost": 0,
                 "off_window": 0, "reflected_rays": 0, "reflections": 0,
                 "bounce_hist": {}}
        progress = Progress(label, n_modes * n_rays)
        t0 = time.time()
        for mode in range(n_modes):
            origin = source.mode_origin()
            fields = acc.new_mode()
            for ray in range(n_rays):
                direction = aim(origin)
                tr = trace_ray(origin, direction, optic, screen.z,
                               cfg.max_bounces)
                stats["emitted"] += 1
                nb = len(tr.reflections)
                sins = [sin_g for _, sin_g in tr.reflections]
                fate, amps = tr.fate, None
                if fate == "screen":
                    # geometry is energy-free; Fresnel enters here, per line or at E0
                    amps = (self.line_amps(sins) if self.per_line
                            else self.fresnel.product(sins))
                    if cfg.amplitude_min > 0.0:
                        peak = (max(float(abs(a)) for a in amps) if self.per_line
                                else float(abs(amps)))
                        if peak < cfg.amplitude_min:
                            fate = "absorbed"    # below threshold on every line
                sampled = ray % cfg.sample_every == 0
                if nb:
                    stats["reflected_rays"] += 1
                    stats["reflections"] += nb
                    stats["bounce_hist"][nb] = stats["bounce_hist"].get(nb, 0) + 1
                    if refl_fh is not None and sampled:
                        for bounce, (P, sin_g) in enumerate(tr.reflections):
                            rc = complex(self.fresnel(sin_g))
                            refl_fh.write(json.dumps({
                                "stage": stage, "mode": mode, "ray": ray,
                                "bounce": bounce,
                                "x": str(P[0]), "y": str(P[1]), "z": str(P[2]),
                                "grazing_rad": math.asin(min(1.0, float(sin_g))),
                                "r_abs": abs(rc), "r_arg": math.atan2(rc.imag, rc.real),
                            }, ensure_ascii=False) + "\n")
                pixel = screen.pixel(tr.point) if fate == "screen" else None
                if rays_fh is not None and sampled:
                    rays_fh.write(json.dumps({
                        "stage": stage, "mode": mode, "ray": ray, "fate": fate,
                        "pixel": pixel, "opl": str(tr.opl),
                        "sins": [str(s) for s in sins],
                    }, ensure_ascii=False) + "\n")
                if fate == "screen":
                    if pixel is None:
                        stats["off_window"] += 1
                    else:
                        acc.add_ray(fields, pixel, amps, tr.opl)
                        stats["screen"] += 1
                else:
                    stats[fate] += 1
                progress.step()
            acc.fold_mode(fields)
        maps = acc.finalize(screen.nx, screen.ny)
        progress.finish(f"on screen {stats['screen']:,}")
        result = {"maps": maps, "stats": stats, "screen": screen,
                  "n_modes": n_modes, "n_rays": n_rays,
                  "seconds": time.time() - t0, "src_cfg": src_cfg}
        self.results[stage] = result
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
                "6 — |μ| and intensity behind the capillary."
                + (" All reflection points are written to reflections.jsonl."
                   if cfg.reflections_jsonl else ""),
                f"Free-field stage: source {_um(cfg.free_source.size)} at z = {_mm(cfg.free_source.position[2])}, screen z = {_mm(cfg.free_screen.z)}.",
                f"Lloyd: r₀ = {_um(cfg.lloyd.height)}, mirror z ∈ [{_mm(cfg.lloyd.z0)}, {_mm(cfg.lloyd.z1)}], source {_um(cfg.lloyd.source.size)}.",
            ],
        }
        self._save(out_dir, "01-scheme.svg", render.scheme_setup(info))

    # ------------------------------------------------------------- stage 2+3

    def _stage2(self, out_dir, quick, rays_fh):
        res = self._mc_stage("free", "2/6 without optics (MC)", self.cfg.free_source,
                             self.cfg.free_screen, None, self._aim_free, 2,
                             quick, None, rays_fh)
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
            "Degree of coherence without optics (n modes, MC)",
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
        self._save(out_dir, "02-free-mc-coherence.svg",
                   render.hstack([mu_fig, int_fig]))
        st = res["stats"]
        self.report += [
            "## Stage 2 — |μ| without optics (MC)",
            f"- modes: {res['n_modes']}, rays/mode: {res['n_rays']}, on screen: {st['screen']:,} of {st['emitted']:,}",
            f"- time: {res['seconds']:.1f} s",
        ]
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
        rms = analytic.rms_diff(maps["mu"][row], mu_th)
        if src.shape == "gaussian":
            xi = lam_f * dist / (2.0 * math.pi * float(src.size))
            note = f"ξ = λD/(2πσ) = {xi * _UM:.3f} µm;  "
        else:
            note = ""
        sub = (f"{note}RMS(MC − analytics) = {rms:.3f};  source: {src.shape}, "
               f"{_um(src.size)}, D = {_mm(dist)}")
        fig = render.line_chart(
            [{"xs": xs_um, "ys": maps["mu"][row], "label": "MC (stage 2)"},
             {"xs": xs_um, "ys": mu_th, "label": "van Cittert–Zernike analytics",
              "dash": "6,4"}],
            "Degree of coherence: analytics vs MC (without optics)",
            "x on screen, µm", "|μ|", sub,
            vlines=[(ref_xy[0] * _UM, "ref")], w=760)
        self._save(out_dir, "03-free-analytic-vs-mc.svg", fig)
        self.report += [
            "## Stage 3 — van Cittert–Zernike analytics",
            f"- RMS(|μ|_MC − |μ|_vCZ) = {rms:.4f}" + (f", ξ = {xi * _UM:.3f} µm" if src.shape == "gaussian" else ""),
        ]

    # ------------------------------------------------------------- stage 4+5

    def _stage4(self, out_dir, quick, refl_fh, rays_fh):
        cfg = self.cfg
        lloyd = cfg.lloyd
        mirror = Mirror(lloyd.z0, lloyd.z1)
        check = self._lloyd_engine_check(mirror)
        res = self._mc_stage("lloyd", "4/6 Lloyd (MC)", lloyd.source, lloyd.screen,
                             mirror, self._aim_lloyd, 3, quick, refl_fh, rays_fh)
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
                                1.2 * (float(lloyd.z1) - float(origin[2])))
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

    def _stage6(self, out_dir, quick, refl_fh, rays_fh):
        cap = self.cfg.capillary
        bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
        check = self._capillary_engine_check(bundle)
        res = self._mc_stage("capillary", "6/6 capillary (MC)", cap.source,
                             cap.screen, bundle, self._aim_capillary, 4,
                             quick, refl_fh, rays_fh)
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
            f"- reflections: total {st['reflections']:,}; per ray: {bh or 'none'}; mean {mean_b:.2f} per reflected ray",
            f"- {check}",
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
                                1.2 * (float(cap.z1) - float(cap.z0)))
        if t_engine is None:
            return "RaySurface engine check (capillary): no root found"
        rel = abs(float(t_fast - t_engine)) / float(t_fast)
        return (f"RaySurface engine check (capillary wall, {wall.kind}): "
                f"|Δt|/t = {rel:.1e}")

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
        _log(f"capsim: stages {sorted(wanted)}, output to {out_dir}"
             + (f", speedup ×{quick}" if quick > 1 else ""))
        fres_check = self._fresnel_check()
        _log("  " + fres_check)
        self.report = [
            "# capsim report",
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
        refl_path = os.path.join(out_dir, "reflections.jsonl")
        refl_fh = (open(refl_path, "w", encoding="utf-8")
                   if cfg.reflections_jsonl and wanted & {4, 6} else None)
        rays_path = os.path.join(out_dir, "rays.jsonl")
        rays_fh = (open(rays_path, "w", encoding="utf-8")
                   if cfg.rays_jsonl and wanted & {2, 4, 6} else None)
        try:
            if 1 in wanted:
                _log("Stage 1/6: simulation layout")
                self._stage1(out_dir)
            res_free = None
            if 2 in wanted:
                _log("Stage 2/6: |μ| without optics (MC, same tracer)")
                res_free = self._stage2(out_dir, quick, rays_fh)
            if 3 in wanted:
                _log("Stage 3/6: van Cittert–Zernike analytics")
                self._stage3(out_dir, res_free)
            res_lloyd = None
            if 4 in wanted:
                _log("Stage 4/6: Lloyd's mirror scheme — wall instead of the capillary (MC)")
                res_lloyd = self._stage4(out_dir, quick, refl_fh, rays_fh)
            if 5 in wanted:
                _log("Stage 5/6: Lloyd analytics vs MC")
                self._stage5(out_dir, res_lloyd)
            if 6 in wanted:
                _log("Stage 6/6: capillary (MC)")
                self._stage6(out_dir, quick, refl_fh, rays_fh)
        finally:
            for fh in (refl_fh, rays_fh):
                if fh is not None:
                    fh.close()
        for fh, name in ((refl_fh, "reflections.jsonl"), (rays_fh, "rays.jsonl")):
            if fh is not None:
                self.files.append(name)
                _log(f"  → {name}")
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
        screens = {"free": cfg.free_screen, "lloyd": cfg.lloyd.screen,
                   "capillary": cfg.capillary.screen}
        by_stage = {}
        with open(records_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                by_stage.setdefault(row["stage"], []).append(row)
        if not by_stage:
            raise ValueError(f"no ray records in {records_path!r}")
        p = cfg.precision
        # Frozen-Fresnel replay evaluates one amplitude at E0 for every line.
        amps_of = (self.line_amps if self.per_line else LineAmplitudes(
            cfg.material, [SpectralLine(cfg.energy_kev, None, 1.0)], p))
        self.files = []
        report = [
            "# capsim report — replay",
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
            mode_cur, fields = None, None
            t0 = time.time()
            for row in rows:
                if row["mode"] != mode_cur:
                    if fields is not None:
                        acc.fold_mode(fields)
                    fields, mode_cur = acc.new_mode(), row["mode"]
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
                acc.add_ray(fields, int(row["pixel"]),
                            amps if self.per_line else amps[0],
                            Number(row["opl"], p))
                stats["screen"] += 1
            if fields is not None:
                acc.fold_mode(fields)
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

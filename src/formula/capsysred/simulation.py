"""CAPSYSred stages on the Number engine.

Stage 2 (free space) pushes the same ray stream through the stage-10 jackknife
estimator (optic = None), stage 12 keeps its pairwise Number ancestor, and
stage 6 traces capillaries. Stage 3 is the deterministic free-space reference.
Every optical stage cross-checks its analytic Number hit against the RaySurface
root-finding engine and the Fresnel factor against xray.reflect_amplitude.
"""

import json
import math
import os
import sys
import time
import zlib

from .. import __version__
from ..formula import Number
from .. import xray
from . import analytic, render, schematic
from .altcoh import run_alt_stage
from .beamlet import run_beamlet_stage
from .coherence import CoherenceAccumulator
from .jackknife import run_jack_stage
from .sketch import run_sketch_stage
from .stage14 import preflight_stage14_output, run_stage14
from .validate import METHOD_LABELS, run_validate_stage
from .config import Config, load
from .nums import lift, solver, vunit
from .progress import Progress
from .screen import ScreenGrid
from .source import aim_disk_direction, slope_direction
from .spectrum import spectral_lines, wavelength_m
from .surfaces import CapillaryBundle, engine_hit_t, entrance_disk
from .symbolic import LineAmplitudes, ampl_template
from .fresnel import FresnelAmplitude
from . import rays_v3
from .rays import (RNG_SCHEME, MultiRaysReader, RaysReader, SceneSeed,
                   metadata_equal, metadata_path, read_metadata,
                   require_full_rows, scene_stream, sidecar_metadata)
from .types import HitMethod, RayRecord
from .units import (
    m_to_angstrom, m_to_mm, m_to_um, rad_to_mrad, rad_to_urad)

ALL_STAGES = (1, 2, 3, 6)
KNOWN_STAGES = ALL_STAGES + (7, 8, 9, 10, 11, 12, 14)  # opt-in estimators/validation


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _mm(x) -> str:
    return f"{m_to_mm(x):g} mm"


def _um(x) -> str:
    return f"{m_to_um(x):g} µm"


def _report_name(out_dir: str, base: str) -> str:
    """Timestamped report name, never colliding with an existing file."""
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    name, n = f"{base}-{stamp}.md", 2
    while os.path.exists(os.path.join(out_dir, name)):
        name = f"{base}-{stamp}-{n}.md"
        n += 1
    return name


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
        self.theta_c = float(cfg.material.critical_angle(cfg.energy_kev, precision=p))
        self.delta_f = float(cfg.material.delta(cfg.energy_kev, precision=p))
        self.beta_f = float(cfg.material.beta(cfg.energy_kev, precision=p))
        self.report = []
        self.files = []
        self.results = {}   # stage name -> MC result dict (maps, stats, ...)
        self.rays = None    # the run's RaysReader (stages consume, never trace)

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
                                       self.cfg.material, precision=p)
        r_sym = ampl_template(1, self.cfg.material, p).number(
            {"s1": str(s), "E": str(self.cfg.energy_kev)})
        diff = float(abs(r_fast - r_ref))
        diff_sym = float(abs(r_sym - r_ref))
        return (f"Fresnel r(θ=0.1 mrad): |r_CAPSYSred − r_xray| = {diff:.1e}; "
                f"|r_symbolic_template − r_xray| = {diff_sym:.1e}")

    def _record_stage14_result(self, result: dict) -> None:
        """Register one independently cached Stage-14 screen."""
        label = result["screen_name"]
        self.results[f"stage14:{label}"] = result
        for name in result["files"]:
            self.files.append(name)
            _log(f"  → {name}")
        counts = result["flag_counts"]
        lit = sum(row["n_rays"] > 0 for row in result["rows"])
        flag_names = ("trusted", "noisy-mu", "over-mu", "noisy-Ic",
                      "null-Ic", "negative-Ic", "solo-rays-only",
                      "no-ref-realizations", "no-rays")
        flag_line = ", ".join(
            f"{name}={counts.get(name, 0):,}"
            + (f" ({100.0 * counts.get(name, 0) / lit:.2f}% lit)"
               if lit and name != "no-rays" else "")
            for name in flag_names)
        unclassified = counts.get(None, 0)
        unclassified_lit = sum(
            row["n_rays"] > 0 and row["flag"] is None
            for row in result["rows"])
        unclassified_note = (
            f"{unclassified:,} ({100.0 * unclassified_lit / lit:.2f}% lit)"
            if lit else f"{unclassified:,}")
        stats = result["stats"]
        perf = result["result_meta"]["performance"]
        remediation = ", ".join(
            f"{name}={value:,}"
            for name, value in result["remediation_counts"].items())
        w_census = ", ".join(
            f"{name}={value:,}"
            for name, value in result["w_signal_census"].items())
        self.report += [
            f"## Stage 14 — exact disk-backed jackknife [{label}]",
            f"- {result['n_modes']} modes × {result['n_rays']} rays; "
            f"cache hits {result['cache_hits']} of {len(result['cache_parts'])}",
            f"- reference status: {result['ref_status']}; warnings: "
            f"{', '.join(result['ref_warnings']) or 'none'}",
            f"- reference diagnostics: {result['ref_diagnostics']}",
            f"- flags: {flag_line}; unclassified={unclassified_note}",
            f"- over-mu with incomplete LOO: "
            f"{result['over_mu_partial_loo']:,}; negative-Ic self-test: "
            f"{counts.get('negative-Ic', 0):,}",
            f"- remediation groups: {remediation}",
            f"- W significance channel: {w_census}",
            f"- stream: emitted={stats['emitted']:,}, "
            f"screen={stats['screen']:,}, off-window={stats['off_window']:,}, "
            f"absorbed={stats['absorbed']:,}, lost={stats['lost']:,}, "
            f"reflected rays={stats['reflected_rays']:,}, "
            f"reflections={stats['reflections']:,}",
            f"- thresholds: {self.cfg.stage14_flag_thresholds}",
            f"- I/O: target miss inputs "
            f"{perf['target_cache_miss_ray_archive_bytes']:,} B; shared fan-out "
            f"read {perf['fanout_physical_ray_archive_bytes_read']:,} B; "
            f"target cache written {perf['cache_bytes_written']:,} B; "
            f"mode rows read {perf['mode_rows_bytes_read']:,} B",
            f"- time: {result['seconds']:.1f} s "
            f"(payload passes {perf['pass1_seconds']:.3f} + "
            f"{perf['pass2_seconds']:.3f} s); estimated peak RSS "
            f"{perf['estimated_peak_rss_bytes'] / (1024 ** 3):.2f} GiB",
        ]

    # ------------------------------------------------------------- MC driver

    def _mc_stage(self, stage: str, label: str, src_cfg, scr_cfg, optic,
                  aim_factory, seed_offset: int):
        cfg = self.cfg
        p = cfg.precision
        screen = ScreenGrid(scr_cfg)
        n_modes, n_rays = src_cfg.budget()
        acc = CoherenceAccumulator(self.lines, screen.ref_pixel(scr_cfg.reference),
                                   cfg.precision)
        records, rays_from = scene_stream(self, stage, src_cfg, scr_cfg, optic,
                                          aim_factory, seed_offset)
        require_full_rows(self.rays, rays_from,
                          "Number-path estimator (full-precision opl/sins)")
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
        if cap is None:
            # no capillary: to-scale traced schematic of the free scene only
            G = schematic.build_geometry(cfg, "free")
            self._save(out_dir, "01a-scheme-traced.svg", schematic.compose(G))
            return
        src, scr = cap.source, cap.screen
        two_a = 2.0 * entrance_disk(cap.bores[0], float(cap.z0))[2]
        kinds = sorted({b.get("kind", "cylinder") for b in cap.bores})
        kind_note = "" if kinds == ["cylinder"] else f" [{', '.join(kinds)}]"
        d0 = float(cap.z0) - float(cap.source.position[2])
        d2 = float(cap.screen.z) - float(cap.z1)
        shape_ru = {"point": "point", "gaussian": "Gaussian",
                    "disk": "disk", "grid": "grid"}[src.shape]
        info = {
            "title": "Simulation layout: source → capillary(ies) → screen",
            "n_bores": len(cap.bores),
            "source_label": ["source",
                             f"{shape_ru}, {_um(src.size)}",
                             f"z = {_mm(src.position[2])}"],
            "capillary_title": (f"capillaries: {len(cap.bores)}, bore ⌀{m_to_um(two_a):g} µm, "
                                f"L = {_mm(float(cap.z1) - float(cap.z0))}{kind_note}"),
            "bore_label": f"2a = {m_to_um(two_a):g} µm",
            "screen_label": ["screen", f"{cap.screen.nx}×{cap.screen.ny} px"],
            "window_label": f"window {_um(cap.screen.edge_x)}",
            "d0_label": f"d₀ = {_mm(d0)} (capillary scene)",
            "len_label": f"L = {_mm(float(cap.z1) - float(cap.z0))}",
            "d2_label": f"d₂ = {_mm(d2)}",
            "description": [
                f"Energy E = {float(cfg.energy_kev):g} keV,  λ = {m_to_angstrom(self.lam):.4f} Å;  spectrum: {self._spectrum_note()}.",
                f"Wall material: {cfg.material.name};  δ = {self.delta_f:.3e},  β = {self.beta_f:.3e},  θ_c = {rad_to_mrad(self.theta_c):.2f} mrad.",
                f"Source — a set of mutually incoherent point modes (van Cittert–Zernike method from a Monte-Carlo ensemble).",
                f"Engine precision: {cfg.precision} digits (Number/Solver, no float64 in the physics path);  seed = {cfg.seed}.",
                "Pipeline: |μ| and intensity behind the capillary.",
            ],
        }
        if cfg.free_source is not None:
            info["description"].extend((
                "Free-field pipeline: |μ| without optics (MC) + van "
                "Cittert–Zernike analytics.",
                f"Free-field scene: source {_um(cfg.free_source.size)} at "
                f"z = {_mm(cfg.free_source.position[2])}, screen "
                f"z = {_mm(cfg.free_screen.z)}.",
            ))
        if cap.screens:
            info["screen_label"].append(
                "+" + ", ".join(f"z = {_mm(s.z)}" for s in cap.screens))
        self._save(out_dir, "01-scheme.svg", render.scheme_setup(info))
        # to-scale twin: real geometry, 10 traced rays, dimensioned axes
        G = schematic.build_geometry(cfg, "capillary")
        self._save(out_dir, "01a-scheme-traced.svg", schematic.compose(G))

    # ------------------------------------------------------------- stage 2+3

    def _stage2(self, out_dir):
        """Free space through the stage-10 jackknife estimator: the same
        algorithm and outputs, optic = None."""
        res = run_jack_stage(self, "2 without optics (MC)", "free",
                             self.cfg.free_source, self.cfg.free_screen,
                             None, self._aim_free, SceneSeed.FREE)
        self.results["free"] = res
        self._jack_outputs(out_dir, "02", "free", res)
        return res

    def _stage3(self, out_dir, res_free):
        screen, maps = res_free["screen"], res_free["maps"]
        row = maps["ref_pixel"] // screen.nx
        xs = screen.xs()
        xs_um = [m_to_um(x) for x in xs]
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
            note = f"ξ = λD/(2πσ) = {m_to_um(xi):.3f} µm;  "
        else:
            note = ""
        sub = (f"{note}RMS(MC − analytics) = {rms:.3f};  source: {src.shape}, "
               f"{_um(src.size)}, D = {_mm(dist)}")
        series = [{"xs": xs_um, "ys": mu_row, "label": "MC |μ| ± σ_jack",
                   "lo": [max(m - e, 0.0) for m, e in zip(mu_row, err_row)],
                   "hi": [min(m + e, 1.0) for m, e in zip(mu_row, err_row)]},
                  {"xs": xs_um, "ys": mu_th, "label": "van Cittert–Zernike analytics",
                   "dash": "6,4"}]
        dub_i = [i for i, d in enumerate(maps["dubious"][row]) if d > 0]
        if dub_i:
            series.append({"xs": [xs_um[i] for i in dub_i],
                           "ys": [mu_row[i] for i in dub_i],
                           "label": "don't trust: σ_jack>1 / pinned at clamp",
                           "color": "#ff7f0e", "dots": True})
        fig = render.line_chart(
            series, "Degree of coherence: analytics vs MC (without optics)",
            "x on screen, µm", "|μ|", sub,
            vlines=[(m_to_um(ref_xy[0]), "ref")], w=760)
        self._save(out_dir, "03-free-analytic-vs-mc.svg", fig)
        self.report += [
            "## Stage 3 — van Cittert–Zernike analytics",
            f"- RMS(|μ|_MC − |μ|_vCZ) = {rms:.4f}" + (f", ξ = {m_to_um(xi):.3f} µm" if src.shape == "gaussian" else ""),
        ]

    # ------------------------------------------------------------- stage 6

    def _stage6(self, out_dir):
        cap = self.cfg.capillary
        bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
        check = self._capillary_engine_check(bundle)
        res = self._mc_stage("capillary", "6/6 capillary (MC)", cap.source,
                             cap.screen, bundle, self._aim_capillary, SceneSeed.CAPILLARY)
        self.results["capillary"] = res
        screen, maps = res["screen"], res["maps"]
        st = res["stats"]
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        extent = (m_to_um(screen.x0f), m_to_um(screen.x0f + screen.exf),
                  m_to_um(screen.y0f), m_to_um(screen.y0f + screen.eyf))
        limit = 1.0 / math.sqrt(res["n_modes"])
        sub = (f"{res['n_modes']} modes × {res['n_rays']} rays; transmitted {st['screen']:,}; "
               f"absorbed {st['absorbed']:,}; reflections {st['reflections']:,}")
        sub_mu = (f"{res['n_modes']} modes × {res['n_rays']} rays; statistical limit |μ| ≈ {limit:.2f}; "
                  "isolated bright pixels — low statistics")
        if screen.ny > 1:
            mu_fig = render.heatmap(maps["mu"], extent,
                                    "Capillary: degree of coherence |μ(P, P_ref)|",
                                    "x, µm", "y, µm", sub_mu, "|μ|",
                                    mark=(m_to_um(ref_xy[0]), m_to_um(ref_xy[1])), vmax=1.0,
                                    w=640)
            int_fig = render.heatmap(maps["intensity"], extent,
                                     "Capillary: intensity on screen",
                                     "x, µm", "y, µm", sub, "I, arb. units", w=640)
        else:
            xs_um = [m_to_um(x) for x in screen.xs()]
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

    def _stage7(self, out_dir):
        """Alternative estimators (full W — axis C, Wigner — axis D) on the
        same ray streams as stages 2/6 (same seed offsets)."""
        cap = self.cfg.capillary
        scenes = []
        if self.cfg.free_source is not None:
            scenes.append(
                ("free", "7 alt free (MC)", self.cfg.free_source,
                 self.cfg.free_screen, None, self._aim_free, SceneSeed.FREE))
        if cap is not None:
            scenes.append(
                ("capillary", "7 alt capillary (MC)", cap.source, cap.screen,
                 CapillaryBundle(cap.bores, cap.z0, cap.z1),
                 self._aim_capillary, SceneSeed.CAPILLARY))
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
                                optic, aim_factory, off)
            self.results[f"alt:{stage}"] = res
            maps, screen, st = res["maps"], res["screen"], res["stats"]
            xs_um = [m_to_um(x) for x in screen.xs()]
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
                f"|μ(x, x_ref)| by three estimators [{stage}]",
                "x on screen, µm", "|μ|", sub,
                vlines=[(ref_x, "ref")], w=760)
            self._save(out_dir, f"07-{stage}-alt-mu.svg", fig)
            extent = (xs_um[0], xs_um[-1], xs_um[0], xs_um[-1])
            fig = render.heatmap(
                maps["mu_full"], extent,
                f"full |μ(x₁, x₂)| (no reference pixel) [{stage}]",
                "x₁, µm", "x₂, µm",
                "diagonal band width = coherence length; ray self-pairs off the diagonal",
                "|μ|", mark=(ref_x, ref_x), vmax=1.0, w=640)
            self._save(out_dir, f"07-{stage}-alt-fullw.svg", fig)
            grid, u_lo, u_hi = res["alt"].wigner_grid()
            fig = render.heatmap(
                grid, (xs_um[0], xs_um[-1], rad_to_urad(u_lo), rad_to_urad(u_hi)),
                f"phase space B(x, u) (ray histogram) [{stage}]",
                "x on screen, µm", "u = dx/dz, µrad",
                f"u bin {rad_to_urad(res['alt'].du):.2f} µrad; intensity weights, no phases",
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
                f"- Wigner u bin: {rad_to_urad(res['alt'].du):.3f} µrad",
                f"- time: {res['seconds']:.1f} s",
            ]
        path = os.path.join(out_dir, "mu-alt.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        self.files.append("mu-alt.jsonl")
        _log("  → mu-alt.jsonl")

    # ------------------------------------------------------------- stage 8

    def _stage8(self, out_dir):
        """Streaming sketch of W (methods §3.10): pairwise reference column +
        Nystrom column + coherent-mode spectrum, 2D screens supported."""
        cap = self.cfg.capillary
        scenes = []
        if self.cfg.free_source is not None:
            scenes.append(
                ("free", "8 sketch free (MC)", self.cfg.free_source,
                 self.cfg.free_screen, None, self._aim_free, SceneSeed.FREE))
        if cap is not None:
            scenes.append(
                ("capillary", "8 sketch capillary (MC)", cap.source, cap.screen,
                 CapillaryBundle(cap.bores, cap.z0, cap.z1),
                 self._aim_capillary, SceneSeed.CAPILLARY))
        rows = []
        for stage, label, src_cfg, scr_cfg, optic, aim_factory, off in scenes:
            res = run_sketch_stage(self, label, stage, src_cfg, scr_cfg,
                                   optic, aim_factory, off)
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
                extent = (m_to_um(screen.x0f), m_to_um(screen.x0f + screen.exf),
                          m_to_um(screen.y0f), m_to_um(screen.y0f + screen.eyf))
                mark = (m_to_um(ref_xy[0]), m_to_um(ref_xy[1]))
                figs = [render.heatmap(maps["mu_pair"], extent,
                                       f"pairwise |μ(P, P_ref)| [{stage}]",
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
                xs_um = [m_to_um(x) for x in screen.xs()]
                fig = render.line_chart(
                    [{"xs": xs_um, "ys": maps["mu_pair"][0], "label": "pairwise"},
                     {"xs": xs_um, "ys": maps["mu_sketch"][0],
                      "label": f"sketch r={maps['rank']}", "dash": "6,4"}],
                    f"|μ(x, x_ref)| [{stage}]", "x, µm", "|μ|", sub,
                    vlines=[(m_to_um(ref_xy[0]), "ref")], w=760)
            self._save(out_dir, f"08-{stage}-sketch-mu.svg", fig)
            lam = maps["lam"]
            top = min(len(lam), 60)
            l1 = lam[0] or 1.0
            fig = render.line_chart(
                [{"xs": list(range(1, top + 1)),
                  "ys": [v / l1 for v in lam[:top]], "label": "λ_n / λ_1"}],
                f"coherent-mode spectrum of the field [{stage}]",
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

    def _stage9(self, out_dir):
        """Hit-method cross-validation on the capillary scene: the first wall
        hit of each validate.methods entry against validate.reference."""
        p = self.cfg.precision
        n_rays = self.cfg.validate_rays
        res = run_validate_stage(self, n_rays)
        self.results["validate"] = res
        st, per = res["stats"], res["per"]
        validation_dir = os.path.join(out_dir, "hit-validation")
        os.makedirs(validation_dir, exist_ok=True)
        rows_name = "hit-validation/hit-validation.jsonl"
        with open(os.path.join(validation_dir, "hit-validation.jsonl"),
                  "w", encoding="utf-8") as fh:
            for row in res["rows"]:
                fh.write(json.dumps(row) + "\n")
        meta_name = "hit-validation/meta.json"
        meta = {
            "capsysred_version": __version__,
            "yaml_file": self.cfg.yaml_file,
            "validation": {
                "n_rays": st["rays"],
                "reference": st["reference"],
                "methods": [str(method) for method in per],
                "precision": self.cfg.precision,
                "precision_target": self.cfg.precision_target,
            },
        }
        with open(os.path.join(validation_dir, "meta.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
            fh.write("\n")
        self.files.extend((rows_name, meta_name))
        _log(f"  → {rows_name}")
        _log(f"  → {meta_name}")
        # match = same hit/pass call AND agreement to precision_target digits
        # (default: p - 2 guard - wall conditioning, config._conditioning_loss)
        target, loss = self.cfg.precision_target, self.cfg.precision_target_loss
        origin = ((f"auto: {p} − 2 − {loss} wall" if loss else "auto: p − 2")
                  if self.cfg.precision_target_auto else "yaml")
        tol_exp = -target
        lo, hi = -(p + 8), max(tol_exp, 2 - p) + 5
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
                "label": f"{METHOD_LABELS.get(m, m)}: {agree[m]:.2f}% @1e{tol_exp}"})
        if series:
            vlines = [(float(tol_exp), f"target = {target}")]
            if tol_exp != 2 - p:
                vlines.append((float(2 - p), f"p − 2 = {p - 2}"))
            fig = render.line_chart(
                series,
                f"share of hits matching the reference ({st['reference']})",
                "log₁₀ of the |Δt|/t tolerance", "matched, %",
                f"{st['hits']:,} wall hits of {st['rays']:,} rays; "
                f"precision_target = {target} ({origin}) ⇒ tol = 1e{tol_exp}; "
                "hit/pass mismatches never match",
                vlines=vlines, w=760)
            self._save(out_dir, "09-hit-validation.svg", fig)
        self.report += [
            "## Stage 9 — hit-method cross-validation",
            f"- rays: {st['rays']:,}; wall hits {st['hits']:,}, passes {st['passes']:,}, "
            f"skipped {st['skipped']:,} (entrance web)",
            f"- {METHOD_LABELS.get(st['reference'], st['reference'])} "
            f"(reference): {st['ref_seconds']:.1f} s",
            f"- match tolerance: |Δt|/t ≤ 1e{tol_exp} "
            f"(precision_target = {target}, {origin})",
        ]
        if not res["native"]:
            self.report.append("- C++ twin: wall kind unsupported — engine method only")
        for m, s in per.items():
            matched = f"{agree[m]:.2f}%" if m in agree else "n/a"
            self.report.append(
                f"- {METHOD_LABELS.get(m, m)}: matched {matched}; "
                f"max |Δt|/t = {s['max_rel']:.1e}, rms = {s['rms']:.1e} "
                f"on {s['n']:,} hits; missing/extra hits {s['missing']}/{s['extra']}; "
                f"{s['seconds']:.1f} s")
        self.report.append(f"- time: {res['seconds']:.1f} s")
        return res

    # ------------------------------------------------------------- stage 10

    def _stage10(self, out_dir):
        """Stage-6 alternative (doc/mu28_legacy_fixed3_renamed.py, running-sum
        form): same capillary rays, |mu| plus a delete-one-mode jackknife map."""
        cap = self.cfg.capillary
        bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
        res = run_jack_stage(self, "10 jackknife capillary (MC)", "capillary",
                             cap.source, cap.screen, bundle,
                             self._aim_capillary, SceneSeed.CAPILLARY)
        self.results["jack:capillary"] = res
        self._jack_outputs(out_dir, "10", "capillary", res,
                           vs=self.results.get("capillary"))
        # extra screens: the same records re-binned onto each plane
        for i, scr in enumerate(cap.screens, 1):
            res_i = run_jack_stage(self, f"10 jackknife capillary s{i} (MC)",
                                   "capillary", cap.source, cap.screen, bundle,
                                   self._aim_capillary, SceneSeed.CAPILLARY,
                                   screen_cfg=scr)
            self.results[f"jack:capillary-s{i}"] = res_i
            self._jack_outputs(out_dir, "10", f"capillary-s{i}", res_i,
                               note=f"screen {i}: z = {_mm(scr.z)}, "
                                    f"window {_um(scr.edge_x)} × {_um(scr.edge_y)}, "
                                    f"{scr.nx}×{scr.ny} px")
        return res

    def _jack_outputs(self, out_dir, tag, scene, res, vs=None, note=None):
        """Jackknife scene outputs shared by stages 2 and 10: figures, report
        section, mu-jack.jsonl rows; vs = same-rays stage-6 result for Δμ;
        note = extra-screen geometry line for the report."""
        maps, screen, st = res["maps"], res["screen"], res["stats"]
        nx, ny = screen.nx, screen.ny
        flat = lambda grid: [v for row in grid for v in row]
        n_lit = sum(1 for d in flat(maps["density"]) if d > 0)
        solid = [i for i, v in enumerate(flat(maps["solid"])) if v > 0]
        n_dub = sum(1 for v in flat(maps["dubious"]) if v > 0)
        flat_err = flat(maps["mu_err"])
        errs = [flat_err[i] for i in solid]
        med_err = sorted(errs)[len(errs) // 2] if errs else 0.0
        limit = 1.0 / math.sqrt(res["n_modes"])
        rms6 = None
        if vs is not None and solid:
            a, b = flat(maps["mu"]), flat(vs["maps"]["mu"])
            rms6 = analytic.rms_diff([a[i] for i in solid], [b[i] for i in solid])
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        sub = (f"{res['n_modes']} modes × {res['n_rays']} rays; "
               f"σ_jack median {med_err:.3f}; statistical limit |μ| ≈ {limit:.2f}; "
               f"solid px {len(solid)} of {n_lit} lit")
        if ny > 1:
            extent = (m_to_um(screen.x0f), m_to_um(screen.x0f + screen.exf),
                      m_to_um(screen.y0f), m_to_um(screen.y0f + screen.eyf))
            mark = (m_to_um(ref_xy[0]), m_to_um(ref_xy[1]))
            trust = [[1.0 if s > 0 and d == 0 else (0.5 if d > 0 else 0.0)
                      for s, d in zip(s_row, d_row)]
                     for s_row, d_row in zip(maps["solid"], maps["dubious"])]
            fig = render.hstack([
                render.heatmap(maps["mu"], extent,
                               "|μ(P, P_ref)| (jackknife)",
                               "x, µm", "y, µm", sub, "|μ|",
                               mark=mark, vmax=1.0, w=430, equal=True),
                render.heatmap(maps["mu_err"], extent, "σ_jack(P)",
                               "x, µm", "y, µm", "", "σ_jack", w=430, equal=True),
                render.heatmap(trust, extent, "trust: 1 ok · ½ don't · 0 none",
                               "x, µm", "y, µm",
                               "½: σ_jack>1, pinned at |μ|=1, or no usable data; 0: no pairs",
                               "trust", vmax=1.0, w=430, equal=True)])
            self._save(out_dir, f"{tag}-{scene}-jack-mu.svg", fig)
            # y ≈ 0 slice of the three maps: |μ| ± σ_jack, σ_jack, trust
            iy0 = min(range(ny), key=lambda j: abs(screen.ys()[j]))
            y0_um = m_to_um(screen.ys()[iy0])
            xs_um = [m_to_um(x) for x in screen.xs()]
            row_mu, row_err = maps["mu"][iy0], maps["mu_err"][iy0]
            dub_i = [i for i, d in enumerate(maps["dubious"][iy0]) if d > 0]
            mu_series = [{"xs": xs_um, "ys": row_mu, "label": "jackknife |μ| ± σ_jack",
                          "lo": [max(m - e, 0.0) for m, e in zip(row_mu, row_err)],
                          "hi": [min(m + e, 1.0) for m, e in zip(row_mu, row_err)]}]
            err_series = [{"xs": xs_um, "ys": row_err, "label": "σ_jack"},
                          {"xs": xs_um, "ys": [limit] * nx,
                           "label": "1/√N_modes", "dash": "2,3"}]
            if dub_i:
                xd = [xs_um[i] for i in dub_i]
                mu_series.append({"xs": xd, "ys": [row_mu[i] for i in dub_i],
                                  "label": "don't trust: σ_jack>1 / pinned at clamp",
                                  "color": "#d62728", "dots": True})
                err_series.append({"xs": xd, "ys": [row_err[i] for i in dub_i],
                                   "label": "don't trust",
                                   "color": "#d62728", "dots": True})
            vl = [(m_to_um(ref_xy[0]), "ref")] if maps["ref_pixel"] // nx == iy0 else []
            fig = render.hstack([
                render.line_chart(mu_series, "|μ(P, P_ref)| ± σ_jack", "x, µm",
                                  "|μ|", f"slice y = {y0_um:.2f} µm",
                                  vlines=vl, w=430),
                render.line_chart(err_series, "σ_jack(x)", "x, µm", "σ_jack",
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
            if res.get("scatter"):
                self._save(out_dir, f"{tag}d-{scene}-ray-scatter.svg",
                           render.ray_scatter(res["scatter"].counts,
                                              res["scatter"].extent_um(),
                                              f"{scene}: ray locations on screen",
                                              "x, µm", "y, µm", sub))
            if rms6 is not None:
                diff = [[abs(a - b) for a, b in zip(ra, rb)]
                        for ra, rb in zip(maps["mu"], vs["maps"]["mu"])]
                fig = render.heatmap(diff, extent,
                                     "|μ_jack − μ_pairwise| (same rays)",
                                     "x, µm", "y, µm",
                                     f"RMS on solid px {rms6:.2e}; bright isolated px = "
                                     "pairless residuals of the pairwise estimator, masked by the jackknife",
                                     "Δ", w=640)
                self._save(out_dir, f"{tag}c-{scene}-jack-vs6.svg", fig)
        else:
            xs_um = [m_to_um(x) for x in screen.xs()]
            row_mu, row_err = maps["mu"][0], maps["mu_err"][0]
            dub_i = [i for i, d in enumerate(maps["dubious"][0]) if d > 0]
            series = [{"xs": xs_um, "ys": row_mu, "label": "jackknife |μ| ± σ_jack",
                       "lo": [max(m - e, 0.0) for m, e in zip(row_mu, row_err)],
                       "hi": [min(m + e, 1.0) for m, e in zip(row_mu, row_err)]}]
            if vs is not None:
                series.append({"xs": xs_um, "ys": vs["maps"]["mu"][0],
                               "label": "pairwise (Number)", "dash": "6,4"})
            if dub_i:
                series.append({"xs": [xs_um[i] for i in dub_i],
                               "ys": [row_mu[i] for i in dub_i],
                               "label": "don't trust: σ_jack>1 / pinned at clamp",
                               "color": "#d62728", "dots": True})
            fig = render.line_chart(series,
                                    "|μ(x, x_ref)| with jackknife errors",
                                    "x, µm", "|μ|", sub,
                                    vlines=[(m_to_um(ref_xy[0]), "ref")], w=760)
            self._save(out_dir, f"{tag}-{scene}-jack-mu.svg", fig)
            err_series = [{"xs": xs_um, "ys": row_err, "label": "σ_jack"},
                          {"xs": xs_um, "ys": [limit] * nx,
                           "label": "1/√N_modes", "dash": "2,3"}]
            if dub_i:
                err_series.append({"xs": [xs_um[i] for i in dub_i],
                                   "ys": [row_err[i] for i in dub_i],
                                   "label": "don't trust",
                                   "color": "#d62728", "dots": True})
            fig = render.line_chart(
                err_series, "jackknife error by pixel", "x, µm", "σ_jack", sub)
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
                      "label": "μ_jack − μ_pairwise",
                      "lo": [-e for e in row_err], "hi": list(row_err)}],
                    "jackknife vs pairwise: Δμ with the ±σ_jack band", "x, µm", "Δμ",
                    f"RMS on solid px {rms6:.2e}; spikes = the pairwise estimator's "
                    "pairless residuals, masked by the jackknife",
                    vlines=[(m_to_um(ref_xy[0]), "ref")], w=760, y_zero=False)
                self._save(out_dir, f"{tag}c-{scene}-jack-vs6.svg", fig)
        xs_um_all = [m_to_um(x) for x in screen.xs()]
        ys_um_all = [m_to_um(y) for y in screen.ys()]
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
        below = (100.0 * sum(1 for e in errs if e < limit) / len(errs)
                 if errs else 0.0)
        refl = []
        if scene != "free":
            bh = ", ".join(f"{k}×: {v:,}" for k, v in sorted(st["bounce_hist"].items()))
            mean_b = (st["reflections"] / st["reflected_rays"]
                      if st["reflected_rays"] else 0.0)
            refl = [f"- reflections: total {st['reflections']:,}; per ray: {bh or 'none'}; "
                    f"mean {mean_b:.2f} per reflected ray"]
        self.report += [
            f"## Stage {int(tag)} — jackknife estimator [{scene}]",
        ] + ([f"- {note}"] if note else []) + [
            f"- {res['n_modes']} modes × {res['n_rays']} rays; on screen {st['screen']:,} of {st['emitted']:,}",
            f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
        ] + refl + [
            f"- solid pixels (≥2 same-mode rays, |μ| estimable): {len(solid)} of {n_lit} lit; "
            "the rest are masked to μ = σ_jack = 0",
            f"- don't-trust estimates on solid px (σ_jack > 1, pinned at |μ| = 1 with σ_jack = 0, "
            f"or no usable jackknife/cross data): {n_dub} of {len(solid)}",
            f"- σ_jack on solid pixels: median {med_err:.4f}, max {max(errs, default=0.0):.4f}; "
            f"{below:.0f}% below the 1/√N limit ({limit:.3f})",
        ] + ([f"- RMS(|μ|_jack − |μ|_stage6) = {rms6:.2e} (same rays, solid pixels)"]
             if rms6 is not None else []) + [
            f"- time: {res['seconds']:.1f} s",
        ]

    # ------------------------------------------------------------- stage 11

    def _stage11(self, out_dir):
        """Beamlet estimator (doc/2026-07-10-stage11-beamlets.ru.md):
        elliptic Gaussian phase spots instead of point bins, the 2x2 Gamma
        tensor through the bounces (general astigmatism), honest mu with no
        self-pair subtraction. Free scene validates against vCZ; the
        capillary scene compares to stage 6 on the same rays; extra
        capillary screens re-bin the same records onto each plane."""
        cap = self.cfg.capillary
        rows = []
        if self.cfg.free_source is not None:
            res = run_beamlet_stage(self, "11 beamlet free (MC)", "free",
                                    self.cfg.free_source, self.cfg.free_screen,
                                    None, self._aim_free, SceneSeed.FREE)
            self.results["beamlet:free"] = res
            maps, screen = res["maps"], res["screen"]
            ref_xy = screen.pixel_xy(maps["ref_pixel"])
            row = screen.ny // 2
            xs_um = [m_to_um(x) for x in screen.xs()]
            src = self.cfg.free_source
            dist = float(screen.z) - float(src.position[2])
            mu_th = [analytic.vcz_mu(x - ref_xy[0], src.shape, float(src.size),
                                     float(self.lam), dist)
                     for x in screen.xs()]
            rms = analytic.rms_diff(maps["mu"][row], mu_th)
            fig = render.line_chart(
                [{"xs": xs_um, "ys": maps["mu"][row],
                  "label": "beamlets |μ|"},
                 {"xs": xs_um, "ys": mu_th,
                  "label": "van Cittert–Zernike analytics", "dash": "6,4"}],
                "beamlet |μ| vs vCZ analytics [free]",
                "x, µm", "|μ|",
                f"RMS(beamlets − vCZ) = {rms:.3f};  {self._beamlet_sub(res)}",
                vlines=[(m_to_um(ref_xy[0]), "ref")], w=760)
            self._save(out_dir, "11-free-beamlet-mu.svg", fig)
            self._beamlet_outputs(
                out_dir, "free", res, rows,
                extra=[f"- RMS(|μ|_beamlet − |μ|_vCZ) = {rms:.4f}"])
        if cap is not None:
            bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
            # one pass over the records deposits the main and every extra
            # screen together (the shared prep feeds all planes)
            res = run_beamlet_stage(self, "11 beamlet capillary (MC)",
                                    "capillary", cap.source, cap.screen,
                                    bundle, self._aim_capillary,
                                    SceneSeed.CAPILLARY,
                                    extra_screens=cap.screens)
            self.results["beamlet:capillary"] = res
            self._beamlet_outputs(out_dir, "capillary", res, rows,
                                  vs=self.results.get("capillary"))
            for i, (scr, res_i) in enumerate(zip(cap.screens, res["extras"]), 1):
                self.results[f"beamlet:capillary-s{i}"] = res_i
                self._beamlet_outputs(
                    out_dir, f"capillary-s{i}", res_i, rows,
                    note=f"- screen {i}: z = {_mm(scr.z)}, "
                         f"window {_um(scr.edge_x)} × {_um(scr.edge_y)}, "
                         f"{scr.nx}×{scr.ny} px")
        path = os.path.join(out_dir, "mu-beamlet.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        self.files.append("mu-beamlet.jsonl")
        _log("  → mu-beamlet.jsonl")

    def _beamlet_sub(self, res):
        w0t = res.get("w0_t", self.cfg.beamlet_w0)
        aniso = (f" (sag) / {m_to_um(w0t):.2f} µm (tang)"
                 if w0t != self.cfg.beamlet_w0 else "")
        return (f"{res['n_modes']} modes × {res['n_rays']} rays; "
                f"w₀ = {m_to_um(self.cfg.beamlet_w0):.2f} µm{aniso}, "
                f"mean w on screen = {m_to_um(res['maps']['w_mean']):.2f} µm")

    def _beamlet_outputs(self, out_dir, tag, res, rows, vs=None, note=None,
                         extra=()):
        """Beamlet scene outputs: report section, figures (capillary tags),
        mu-beamlet.jsonl rows; vs = same-rays pairwise result for Δμ."""
        maps, screen, st = res["maps"], res["screen"], res["stats"]
        nx, ny = screen.nx, screen.ny
        flat = lambda grid: [v for row in grid for v in row]
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        sub = self._beamlet_sub(res)
        xs_um = [m_to_um(x) for x in screen.xs()]
        report = [f"## Stage 11 — beamlet estimator [{tag}]",
                  f"- {res['n_modes']} modes × {res['n_rays']} rays; on screen "
                  f"{st['screen']:,} of {st['emitted']:,} (tails off window: {st['off_window']:,})",
                  f"- rays: {'reused from the rays file' if res['rays_from'] == 'file' else 'traced'}",
                  f"- w₀ = {m_to_um(self.cfg.beamlet_w0):.2f} µm"
                  + (f" (sagittal), {m_to_um(res['w0_t']):.2f} µm (tangential)"
                     if res.get("w0_t", self.cfg.beamlet_w0)
                     != self.cfg.beamlet_w0 else "")
                  + f"; mean spot width on screen "
                  f"= {m_to_um(maps['w_mean']):.2f} µm; Γ-tensor deposit; honest |μ| "
                  "(no self-pair subtraction)"]
        if note:
            report.append(note)
        report += list(extra)
        if maps["flat_walls"]:
            report.append("- implicit bore(s): no closed-form curvature — "
                          "flat-wall (scalar q) bounces")
        if maps["gamma_bad"]:
            report.append(f"- deposits skipped (beam blew up, Im G ⊁ 0): "
                          f"{maps['gamma_bad']:,}")
        lit_px = [i for i, v in enumerate(flat(maps["intensity"])) if v > 0.0]
        flat_err = flat(maps["mu_err"])
        errs = sorted(flat_err[i] for i in lit_px)
        n_dub = int(sum(flat(maps["dubious"])))
        report.append(
            f"- σ_jack (delete-one-mode) on {len(lit_px)} lit px: median "
            f"{errs[len(errs) // 2] if errs else 0.0:.4f}, max "
            f"{errs[-1] if errs else 0.0:.4f}; don't-trust {n_dub}")
        if tag != "free":
            extent = (m_to_um(screen.x0f), m_to_um(screen.x0f + screen.exf),
                      m_to_um(screen.y0f), m_to_um(screen.y0f + screen.eyf))
            if ny > 1:
                mark = (m_to_um(ref_xy[0]), m_to_um(ref_xy[1]))
                fig = render.hstack([
                    render.heatmap(maps["mu"], extent,
                                   "beamlet |μ(P, P_ref)|",
                                   "x, µm", "y, µm", sub, "|μ|",
                                   mark=mark, vmax=1.0, w=430, equal=True),
                    render.heatmap(maps["intensity"], extent,
                                   "beamlet intensity", "x, µm", "y, µm",
                                   "coherent beamlet sum |Σ g|²",
                                   "I, arb. units", w=430, equal=True)])
                self._save(out_dir, f"11-{tag}-beamlet-mu.svg", fig)
            else:
                fig = render.line_chart(
                    [{"xs": xs_um, "ys": maps["mu"][0], "label": "beamlets |μ|"}],
                    "beamlet |μ(x, x_ref)|", "x, µm", "|μ|", sub,
                    vlines=[(m_to_um(ref_xy[0]), "ref")], w=760)
                self._save(out_dir, f"11-{tag}-beamlet-mu.svg", fig)
                imax = max(maps["intensity"][0]) or 1.0
                fig = render.line_chart(
                    [{"xs": xs_um,
                      "ys": [v / imax for v in maps["intensity"][0]],
                      "label": "beamlet intensity"}],
                    "beamlet intensity", "x, µm", "I, arb. units", sub)
                self._save(out_dir, f"11a-{tag}-beamlet-intensity.svg", fig)
            lit = ([i for i, d in enumerate(flat(vs["maps"]["density"]))
                    if d > 0] if vs is not None else [])
            if lit:
                a, b = flat(maps["mu"]), flat(vs["maps"]["mu"])
                rms6 = analytic.rms_diff([a[i] for i in lit],
                                         [b[i] for i in lit])
                sub6 = (f"RMS on lit px {rms6:.3f}; same rays, different "
                        "estimators: pairwise subtracts ray self-pairs, "
                        "beamlets smear the field")
                if ny > 1:
                    diff = [[abs(x - y) for x, y in zip(ra, rb)]
                            for ra, rb in zip(maps["mu"], vs["maps"]["mu"])]
                    fig = render.heatmap(diff, extent,
                                         "|μ_beamlet − μ_pairwise| (same rays)",
                                         "x, µm", "y, µm", sub6, "Δ", w=640)
                else:
                    fig = render.line_chart(
                        [{"xs": xs_um,
                          "ys": [x - y for x, y in
                                 zip(maps["mu"][0], vs["maps"]["mu"][0])],
                          "label": "μ_beamlet − μ_pairwise"}],
                        "beamlets vs pairwise: Δμ (same rays)", "x, µm", "Δμ",
                        sub6, w=760, y_zero=False)
                self._save(out_dir, f"11b-{tag}-beamlet-vs6.svg", fig)
                report.append(
                    f"- RMS(|μ|_beamlet − |μ|_stage6) = {rms6:.4f} on "
                    f"{len(lit)} lit px (same rays; estimators differ — "
                    "stage 6 subtracts self-pairs, beamlets do not)")
        if ny > 1:
            extent = (m_to_um(screen.x0f), m_to_um(screen.x0f + screen.exf),
                      m_to_um(screen.y0f), m_to_um(screen.y0f + screen.eyf))
            fig = render.heatmap(maps["mu_err"], extent, "σ_jack(P)",
                                 "x, µm", "y, µm", sub, "σ", w=640)
        else:
            fig = render.line_chart(
                [{"xs": xs_um, "ys": maps["mu_err"][0], "label": "σ_jack"}],
                "σ_jack by pixel", "x, µm", "σ", sub)
        self._save(out_dir, f"11c-{tag}-beamlet-err.svg", fig)
        ys_um = [m_to_um(y) for y in screen.ys()]
        for iy in range(ny):
            for ix in range(nx):
                rows.append({"stage": tag, "pixel": iy * nx + ix,
                             "x_um": xs_um[ix], "y_um": ys_um[iy],
                             "mu": maps["mu"][iy][ix],
                             "mu_err": maps["mu_err"][iy][ix],
                             "dubious": bool(maps["dubious"][iy][ix]),
                             "I": maps["intensity"][iy][ix],
                             "n_rays": int(maps["density"][iy][ix])})
        report.append(f"- time: {res['seconds']:.1f} s")
        self.report += report

    # ------------------------------------------------------------- stage 12

    def _stage12(self, out_dir):
        """The pre-jackknife stage 2: pairwise Number estimator on the free
        scene (same rays as stage 2, no σ_jack), kept for cross-checks."""
        res = self._mc_stage("free", "12 pairwise free (MC)", self.cfg.free_source,
                             self.cfg.free_screen, None, self._aim_free, SceneSeed.FREE)
        self.results["pairwise:free"] = res
        screen, maps = res["screen"], res["maps"]
        xs_um = [m_to_um(x) for x in screen.xs()]
        ref_xy = screen.pixel_xy(maps["ref_pixel"])
        sub = (f"{res['n_modes']} modes × {res['n_rays']} rays, {self._spectrum_note()}, "
               f"x_ref = {m_to_um(ref_xy[0]):.2f} µm")
        row = screen.ny // 2
        limit = 1.0 / math.sqrt(res["n_modes"])
        mu_fig = render.line_chart(
            [{"xs": xs_um, "ys": maps["mu"][row], "label": "MC |μ(x, x_ref)|"},
             {"xs": xs_um, "ys": [limit] * len(xs_um), "color": "#999",
              "dash": "2,4", "width": 1.0,
              "label": f"statistical limit 1/√N modes ≈ {limit:.2f}"}],
            "Degree of coherence without optics (pairwise Number)",
            "x on screen, µm", "|μ|", sub,
            vlines=[(m_to_um(ref_xy[0]), "ref")], w=640)
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
                                method=HitMethod.SUBDIVISION)
        if t_engine is None:
            return "RaySurface engine check (capillary): no root found"
        rel = abs(float(t_fast - t_engine)) / float(t_fast)
        return (f"RaySurface engine check (capillary wall, {wall.kind}): "
                f"|Δt|/t = {rel:.1e}")

    def _default_stages(self) -> set[int]:
        """Core stages for the scenes explicitly configured by the user."""
        wanted = {1}
        if self.cfg.free_source is not None:
            wanted.update((2, 3))
        if self.cfg.capillary is not None:
            wanted.add(6)
        return wanted

    def _validate_stage_scenes(self, wanted: set[int]) -> None:
        """Fail before creating output when a requested scene is absent."""
        if 1 in wanted and (self.cfg.free_source is None
                            and self.cfg.capillary is None):
            raise ValueError("stage 1 requires a free or capillary scene")
        free_stages = sorted(wanted & {2, 3, 12})
        if free_stages and self.cfg.free_source is None:
            raise ValueError(
                f"stages {free_stages} require a configured free.source"
            )
        capillary_stages = sorted(wanted & {6, 9, 10, 14})
        if capillary_stages and self.cfg.capillary is None:
            raise ValueError(
                f"stages {capillary_stages} require a configured "
                "capillary.source"
            )
        mixed_stages = sorted(wanted & {7, 8, 11})
        if (mixed_stages and self.cfg.free_source is None
                and self.cfg.capillary is None):
            raise ValueError(
                f"stages {mixed_stages} require a free or capillary scene"
            )

    # ------------------------------------------------------------- trace

    def _local_recording(self, out_dir: str):
        """The recording a stage run reads when no --replay is given:
        ``out_dir/rays-modes`` (v3) or ``out_dir/rays.jsonl.gz`` (v2), whose
        metadata must equal this config's; None when neither exists.
        Stages never trace: ``trace_v3`` is a separate command.
        """
        v3_dir = os.path.join(out_dir, "rays-modes")
        v2_path = os.path.join(out_dir, "rays.jsonl.gz")
        expected = sidecar_metadata(self.cfg)
        if os.path.isdir(v3_dir):
            try:
                actual = rays_v3.metadata(v3_dir)
            except ValueError as exc:
                raise ValueError(f"{v3_dir}: invalid v3 rays archive: {exc}") from exc
            probe = {"format": 2, "geometry": actual["geometry"],
                     "budgets": actual["budgets"], "rng": RNG_SCHEME}
            if actual.get("lean"):
                probe["lean"] = True
            if (not metadata_equal(probe, expected)
                    or (actual.get("rng") or {}).get("scheme") != RNG_SCHEME):
                raise ValueError(
                    f"{v3_dir}: recording metadata does not match this config "
                    "(geometry, seed, budgets, lean or rng scheme); remove the "
                    "recording manually, use explicit --replay or another output "
                    "directory"
                )
            return v3_dir
        if os.path.lexists(v2_path):
            try:
                actual = read_metadata(v2_path)
            except (OSError, UnicodeError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"{metadata_path(v2_path)}: rays metadata is missing or invalid; "
                    "remove the recording manually or choose another output directory"
                ) from exc
            if not metadata_equal(actual, expected):
                raise ValueError(
                    f"{v2_path}: recording metadata does not match this config; "
                    "remove the recording manually, use explicit --replay or "
                    "another output directory"
                )
            return v2_path
        return None

    # ------------------------------------------------------------- run

    def run(self, out_dir, stages=None, rays_src=None,
            stage14_paths=None) -> dict:
        cfg = self.cfg
        if stage14_paths is not None:
            stage14_paths = [os.fspath(path) for path in stage14_paths]
        wanted = self._default_stages() if stages is None else set(stages)
        if not wanted:
            raise ValueError("stages must not be empty")
        unknown = sorted(wanted - set(KNOWN_STAGES))
        if unknown:
            raise ValueError(f"unknown stages: {unknown}; available {list(KNOWN_STAGES)}")
        if 3 in wanted:
            wanted.add(2)
        self._validate_stage_scenes(wanted)
        if (rays_src is not None or stage14_paths is not None) and 9 in wanted:
            raise ValueError("stage 9 validates the tracers themselves and "
                             "cannot run from a rays file")
        os.makedirs(out_dir, exist_ok=True)
        if 14 in wanted:
            # Fail before a fresh trace or a many-hour cache build.
            preflight_stage14_output(out_dir)
        t0 = time.time()
        _log(f"CAPSYSred: stages {sorted(wanted)}, output to {out_dir}"
             + (f", rays from {rays_src.path}" if rays_src is not None else
                f", rays from {' + '.join(stage14_paths)}"
                if stage14_paths is not None else ""))
        fres_check = self._fresnel_check()
        _log("  " + fres_check)
        self.report = [
            "# CAPSYSred report",
            "",
            f"- energy: {float(cfg.energy_kev):g} keV (λ = {m_to_angstrom(self.lam):.4f} Å); spectrum: {self._spectrum_note()}",
            f"- material: {cfg.material.name}; δ = {self.delta_f:.3e}, β = {self.beta_f:.3e}, θ_c = {rad_to_mrad(self.theta_c):.2f} mrad",
            f"- precision: {cfg.precision} digits; certified target "
            f"{cfg.precision_target}{' (auto)' if cfg.precision_target_auto else ''}; "
            f"seed = {cfg.seed}",
            f"- Fresnel: {'per spectral line (per_line_fresnel)' if self.per_line else 'at the central energy E₀'}",
            f"- {fres_check}",
            "",
        ]
        self.files = []
        self.jack_rows = []
        if rays_src is not None:
            self.report.insert(-1, f"- replay: rays from {rays_src.path} "
                                   "(no tracing)")
            self.rays = rays_src
        elif stage14_paths is not None:
            self.report.insert(-1, "- replay: rays from "
                               + " + ".join(stage14_paths) + " (no tracing)")
            self.rays = None
        else:
            self.rays = None
            if wanted & {2, 6, 7, 8, 10, 11, 12, 14}:
                local = self._local_recording(out_dir)
                if local is None:
                    raise ValueError(
                        f"{out_dir}: no rays recording (rays-modes/ or rays.jsonl.gz); "
                        "trace first (python -m formula.capsysred.trace_v3 …) "
                        "or pass --replay"
                    )
                self.report.insert(-1, f"- rays from {local} (no tracing)")
                if wanted & {2, 6, 7, 8, 10, 11, 12}:
                    try:
                        self.rays = RaysReader(local)
                    except (OSError, EOFError, zlib.error, UnicodeError, ValueError,
                            KeyError, TypeError) as exc:
                        raise ValueError(
                            f"{local}: rays recording is incomplete or corrupt; "
                            "remove it manually or choose another output directory"
                        ) from exc
                if 14 in wanted:
                    stage14_paths = [local]
        try:
            if 1 in wanted:
                _log("Stage 1: simulation layout")
                self._stage1(out_dir)
            res_free = None
            if 2 in wanted:
                _log("Stage 2: |μ| without optics (jackknife estimator, same tracer)")
                res_free = self._stage2(out_dir)
            if 3 in wanted:
                _log("Stage 3: van Cittert–Zernike analytics")
                self._stage3(out_dir, res_free)
            if 6 in wanted:
                _log("Stage 6: capillary (MC)")
                self._stage6(out_dir)
            if 7 in wanted:
                _log("Stage 7: alternative estimators — full W (axis C) + Wigner (axis D)")
                self._stage7(out_dir)
            if 8 in wanted:
                _log("Stage 8: streaming sketch of W — column + mode spectrum")
                self._stage8(out_dir)
            if 9 in wanted:
                _log("Stage 9: hit-method cross-validation — "
                     f"{', '.join(self.cfg.validate_methods)} vs "
                     f"{self.cfg.validate_reference} reference")
                self._stage9(out_dir)
            if 10 in wanted:
                _log("Stage 10: stage-6 estimator + delete-one-mode jackknife errors")
                self._stage10(out_dir)
            if 11 in wanted:
                _log("Stage 11: beamlet estimator — elliptic phase spots (Γ tensor, general astigmatism)")
                self._stage11(out_dir)
            if 12 in wanted:
                _log("Stage 12: pairwise Number estimator without optics (the pre-jackknife stage 2)")
                self._stage12(out_dir)
        finally:
            if self.rays is not None:
                self.rays.close()
        if 14 in wanted:
            _log("Stage 14: exact disk-backed delete-one-mode jackknife")
            res14 = run_stage14(self, out_dir, stage14_paths, log=_log)
            self.results["stage14:capillary"] = res14
            for name in res14["files"]:
                self.files.append(name)
                _log(f"  → {name}")
            counts = res14["flag_counts"]
            lit = sum(row["n_rays"] > 0 for row in res14["rows"])
            flag_names = ("trusted", "noisy-mu", "over-mu", "noisy-Ic",
                          "null-Ic", "negative-Ic", "solo-rays-only",
                          "no-ref-realizations", "no-rays")
            flag_line = ", ".join(
                f"{name}={counts.get(name, 0):,}"
                + (f" ({100.0 * counts.get(name, 0) / lit:.2f}% lit)"
                   if lit and name != "no-rays" else "")
                for name in flag_names)
            unclassified = counts.get(None, 0)
            unclassified_lit = sum(
                row["n_rays"] > 0 and row["flag"] is None
                for row in res14["rows"]
            )
            unclassified_note = (f"{unclassified:,} "
                                 f"({100.0 * unclassified_lit / lit:.2f}% lit)"
                                 if lit else f"{unclassified:,}")
            stats14 = res14["stats"]
            perf14 = res14["result_meta"]["performance"]
            remediation = ", ".join(
                f"{name}={value:,}"
                for name, value in res14["remediation_counts"].items())
            w_census = ", ".join(
                f"{name}={value:,}"
                for name, value in res14["w_signal_census"].items())
            self.report += [
                "## Stage 14 — exact disk-backed jackknife [capillary]",
                f"- {res14['n_modes']} modes × {res14['n_rays']} rays; "
                f"cache hits {res14['cache_hits']} of {len(res14['cache_parts'])}",
                f"- reference status: {res14['ref_status']}; warnings: "
                f"{', '.join(res14['ref_warnings']) or 'none'}",
                f"- reference diagnostics: {res14['ref_diagnostics']}",
                f"- flags: {flag_line}; unclassified={unclassified_note}",
                f"- over-mu with incomplete LOO: "
                f"{res14['over_mu_partial_loo']:,}; negative-Ic self-test: "
                f"{counts.get('negative-Ic', 0):,}",
                f"- remediation groups: {remediation}",
                f"- W significance channel: {w_census}",
                f"- stream: emitted={stats14['emitted']:,}, "
                f"screen={stats14['screen']:,}, off-window={stats14['off_window']:,}, "
                f"absorbed={stats14['absorbed']:,}, lost={stats14['lost']:,}, "
                f"reflected rays={stats14['reflected_rays']:,}, "
                f"reflections={stats14['reflections']:,}",
                f"- thresholds: {self.cfg.stage14_flag_thresholds}",
                f"- I/O: physical fan-out rays read "
                f"{perf14['ray_archive_bytes_read']:,} B; cache written "
                f"{res14['fanout']['physical_cache_bytes_written']:,} B; "
                f"mode rows read {perf14['mode_rows_bytes_read']:,} B",
                f"- time: {res14['seconds']:.1f} s "
                f"(payload passes {perf14['pass1_seconds']:.3f} + "
                f"{perf14['pass2_seconds']:.3f} s); estimated peak RSS "
                f"{perf14['estimated_peak_rss_bytes'] / (1024 ** 3):.2f} GiB",
            ]
            for extra_result in res14.get("extra_results", []):
                self._record_stage14_result(extra_result)
        report_name = _report_name(out_dir, "report")
        self.report += ["", "## Files", ""]
        self.report += [f"- {name}" for name in self.files + [report_name]]
        report_path = os.path.join(out_dir, report_name)
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.report) + "\n")
        self.files.append(report_name)
        _log(f"  → {report_name}")
        _log(f"Done in {time.time() - t0:.0f} s.")
        return {"out_dir": out_dir, "files": list(self.files)}

    # ------------------------------------------------------------- replay

    def replay(self, records_path, out_dir, stages=None) -> dict:
        """Run stages from a recorded rays file — no tracing at all.

        Any streaming stage replays (2, 6-8, 10-12 and 14, plus analytics 3; stage
        9 validates live tracers and is refused); default = the Number stages
        of the scenes present (free -> 2, capillary -> 6). The
        spectrum and the material may differ from the recording — rays are
        energy-free; the geometry, seed and budgets must match it.
        """
        paths = ([records_path] if isinstance(records_path, str)
                 else list(records_path))
        if stages is None:
            reader = (RaysReader(paths[0]) if len(paths) == 1
                      else MultiRaysReader(paths))
            per_scene = {"free": 2, "capillary": 6}
            configured = set()
            if self.cfg.free_source is not None:
                configured.add("free")
            if self.cfg.capillary is not None:
                configured.add("capillary")
            stages = sorted(per_scene[sc] for sc in reader.done
                            if sc in per_scene and sc in configured)
            if not stages:
                raise ValueError(
                    f"no replayable configured scenes in {records_path!r}"
                )
            return self.run(out_dir, stages=stages, rays_src=reader)
        wanted = set(stages)
        if 14 in wanted and wanted == {14}:
            # No RaysReader: its constructor scans the whole gzip.  The
            # Stage-14 builder validates/deposits in one strict pass, while a
            # cache hit does not open the ray archive at all.
            return self.run(out_dir, stages=stages,
                            stage14_paths=paths)
        reader = (RaysReader(paths[0]) if len(paths) == 1
                  else MultiRaysReader(paths))
        return self.run(out_dir, stages=stages, rays_src=reader,
                        stage14_paths=paths if 14 in wanted else None)


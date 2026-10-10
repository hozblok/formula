"""YAML/dict -> typed simulation config.

The single Number boundary: every physical scalar becomes Number(precision) here;
downstream code reads precision from the Numbers it receives, never as a parameter.
Counts stay int, spectral weights stay float (not phase-critical).
"""

import copy
import math
import os
import warnings

from .._roots import get_backend
from ..formula import Number
from ..xray import FUSED_SILICA, OE2012_GLASS
from .spectrum import spectral_lines
from .shared.types import HitMethod

MATERIALS = {"fused_silica": FUSED_SILICA, "glass_oe2012": OE2012_GLASS}

_SOURCE_REQUIRED = frozenset({"shape", "size", "position", "n_modes", "n_rays"})
_GRID_SOURCE_REQUIRED = frozenset({"grid_n", "grid_step"})

FREE_DEFAULTS = {"screen": {}}

# 6 um bore and a screen immediately after the optic. The source is omitted
# deliberately: every configured scene must state its own complete source.
CAPILLARY_DEFAULTS = {
    "bores": [{"center": [0.0, 0.0], "radius": 6.0e-6}],
    "z0": 0.0,
    "z1": 0.05,
    "screen": {"z": 0.051, "edge_x": 1.6e-5, "edge_y": 1.6e-5,
               "nx": 41, "ny": 41},
}

DEFAULTS = {
    "precision": 32,
    # certified digits: the stage-9 match tolerance is 1e-precision_target;
    # None -> precision - 2 guard - torus conditioning loss (_conditioning_loss)
    "precision_target": None,
    "seed": 12345,
    "energy_kev": 8.0,
    # wall glass n = 1 - delta + i*beta: fused_silica | glass_oe2012 (Opt. Express 20, 3975)
    "material": "fused_silica",
    # monochromatic | gaussian {rel_fwhm, n_lines, n_sigma} | lines [{energy_kev, weight}]
    # | table {file}; per_line_fresnel (multi-line modes only): r(E_m) per
    # line instead of frozen r(E0)
    "spectrum": {"mode": "monochromatic", "rel_fwhm": 2.0e-4, "n_lines": 7,
                 "n_sigma": 3.0},
    # ny=1 is a thin detector strip: edge_y must keep the intra-pixel phase
    # spread k*y^2/(2D) well below a radian, or ray shot noise floods |mu|.
    "screen": {
        "z": 0.06,
        "center": [0.0, 0.0],
        "edge_x": 1.2e-5,
        "edge_y": 2.0e-6,
        "nx": 121,
        "ny": 1,
        "reference": None,            # [x, y] of the reference point; None -> window center
    },
    # lean_rays: drop refl and write opl/sins as float64 in the rays file —
    # the estimators read floats anyway (bit-identical); the file cannot
    # feed the beamlet stage.
    "trace": {"max_bounces": 200, "amplitude_min": 1.0e-6, "lean_rays": False},
    # stage 1: rays traced onto the to-scale schematic (01a-scheme-traced.svg)
    "schematic": {"n_rays": 10},
    # stage 8: number of sketch probe vectors (r ~ n99 modes, see methods §8)
    "sketch": {"rank": 96},
    # stage 9: rays for the hit-method cross-validation
    # (python / C++ / implicit subdivision)
    "validate": {
        "n_rays": 10000,
        "reference": "python-closed-form",
        "methods": ["cpp-closed-form", "subdivision"],
    },
    # stage 11: beamlet launch waists [m] and deposit window radius in beam
    # widths. w0 is the waist along y, w0_t along x: null = isotropic (= w0),
    # "auto" = the scene's Fresnel scale sqrt(lam*L/pi) of the
    # source->screen flight, or an explicit number. waist_z [m]: the launch
    # beam's free-space waist lies waist_z - z_source along the ray past the
    # source (the plane z = waist_z when walls before it are flat), null = at
    # the source.
    "beamlet": {"w0": 5.0e-7, "w0_t": None, "window_sigmas": 3.0},
    # stage 14: exact disk-backed delete-one-mode jackknife taxonomy.
    "stage14": {"flag_thresholds": {
        "ic_n_sigma": 3.0, "ref_ic_n_sigma": 3.0, "w_n_sigma": 3.0,
        "min_coherent_fraction": 0.05,
    }},
}

# stage 16 (`wave_estimator`): merged lazily by Config.validate_wave_estimator(),
# so a YAML without the section keeps its raw config unchanged for every other stage.
WAVE_DEFAULTS = {
    "provider": "auto",             # auto | uisk (regular-polygon bores) | fb (circular bores) | free
    "observable": "coherent_cell",  # coherent_cell (|integral_cell E|^2, as stage 14) | point
    "source_mode": "quadrature",    # quadrature (positive rule over `source`) | recorded_origins
    "target_error": 0.002,          # declared numerical budget on complex mu; recorded, not enforced
    "source_nodes": 768,            # rule size: disk 4 n_r^2 >= this, gaussian n^2 >= this
    "grid_dx": None,                # lattice step [m]; null -> lambda / (2 theta_max angle_margin)
    "angle_margin": 2.0,
    "pad": 1.0,                     # box margin in units of theta_max L + 4 sqrt(lambda L)
    "max_bounces": 2,               # image families up to this reflection order
    "mask_supersample": 4,          # polygon masks: fractional coverage from an SxS sub-lattice
    "pixel_subsamples": 3,          # I_pixel: SxS points per cell
    "intensity_floor": 1.0e-6,      # trusted: I and I_ref >= floor * max I
    "workers": None,                # scipy.fft threads; null = all cores
    "cache_gb": 2.0,                # transfer-function cache budget
    # provider fb (Fourier-Bessel modes of circular / torus bores)
    "fb_jmax": None,                # basis cut j_mn < jmax; null -> ceil(k (a + Re ell) fb_theta_cut)
    "fb_theta_cut": 3.6e-4,         # modal angular cut [rad] behind the jmax rule
    "fb_chebyshev_tol": 1.0e-12,    # Bessel-coefficient cutoff of the Chebyshev series of exp(-iHL)
    "fb_dr": 2.0e-8,                # fine radial table step [m] for the exit-field synthesis
    "fb_wall": "dir-ell",           # dir-ell (complex offset: phase + absorption) | dir-ell-real (no absorption)
    "fb_angle_margin": 1.1,         # lattice h = lambda / (2 (theta_modal + tilt) margin), rounded to pixel / b
    "fb_lattice_half": None,        # lattice half-width [m]; null -> windows + spread + Fresnel tails
    "fb_grid_dtype": "complex128",  # complex64 | complex128 lattice arrays
    "fb_jobs": 1,                   # processes over source nodes (each builds its own basis)
    "fb_per_bore_maps": None,       # per-bore drifts for I_bores / Gamma_12; null -> only for <= 2 bores
}

B5_DEFAULTS = {
    "provider": "archive_phase",
    "patch_space": "entrance",
    "charts": "adaptive",
    "max_modes": 4,
    "rays_per_mode": 20000,
    "degree": 4,
    "neighbors": 96,
    "min_neighbors": 48,
    "patches_per_family": 4,
    "phase_tolerance": 0.05,
    "max_condition": 1.0e8,
    "screen_index": 0,             # primary screen, then capillary.screens
    "seed": 17,
    "widths_m": [1.0e-6],
    "map_stride": 4,
    "map_jobs": 1,
    "map_ray_budgets": [],
    "map_snapshots": [],
}

B9_DEFAULTS = {
    "provider": "archive_contour",
    "carrier_groups": 0,
    "field_representation": "contour_p1",
    "phase_degree": 2,
    "triangle_quadrature_order": 8,
    "max_quadrature_nodes_per_batch": 500000,
    "phase_backend": "type3",
    "receiver_channels_per_batch": 4,
    "quadrature_safety": 1.5,
    "quadrature_max_order": 512,
    "cylinder_retrace": None,
    "adaptive_retrace": None,
    "curved_retrace": None,
    "max_missing_area_fraction": None,
    "amplitude_mode": "point_jacobian",
    "mode_start": 0,
    "max_modes": 4,
    "rays_per_mode": 20000,
    "screen_index": 0,
    "map_stride": 4,
    "map_jobs": 1,
    "map_ray_budgets": [],
    "map_snapshots": [],
    "phase_subdivisions": [1, 2],
    "edge_order": 8,
    "pixel_order": 4,
    "nufft_eps": 1.0e-9,
    "nufft_threads": 1,
    "max_triangles_per_batch": 40000,
    "holdout_stride": 5,
    "determinant_floor": 1.0e-10,
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class SourceCfg:
    def __init__(self, raw: dict, p: int):
        self.shape = raw["shape"]
        if self.shape not in ("point", "gaussian", "disk", "grid"):
            raise ValueError(f"unknown source shape: {self.shape!r}")
        self.size = Number(str(raw["size"]), p)
        self.position = tuple(Number(str(c), p) for c in raw["position"])
        self.n_modes = int(raw["n_modes"])
        self.n_rays = int(raw["n_rays"])
        if self.shape == "grid":
            # deterministic anode lattice: grid_n x grid_n nodes, gaussian weights
            self.grid_n = int(raw["grid_n"])
            self.grid_step = Number(str(raw["grid_step"]), p)
            # lattice orientation about the axis, degrees (45 = diamond)
            self.grid_rot_deg = float(raw.get("grid_rot_deg", 0.0))
            # optional disc trim: keep nodes with r <= grid_r_max (m)
            self.grid_r_max = (float(raw["grid_r_max"])
                               if "grid_r_max" in raw else None)

    def budget(self) -> tuple[int, int]:
        """(n_modes, n_rays) exactly as configured."""
        return self.n_modes, self.n_rays


def _required_source(raw: object, scene: str, p: int) -> SourceCfg:
    """Parse one explicit scene source; no values inherit across scenes."""
    if not isinstance(raw, dict):
        raise ValueError(f"{scene}.source must be a mapping")
    missing = _SOURCE_REQUIRED - raw.keys()
    if raw.get("shape") == "grid":
        missing |= _GRID_SOURCE_REQUIRED - raw.keys()
    if missing:
        raise ValueError(
            f"{scene}.source is missing required fields: {sorted(missing)}"
        )
    return SourceCfg(raw, p)


class ScreenCfg:
    def __init__(self, raw: dict, p: int):
        self.z = Number(str(raw["z"]), p)
        self.center = tuple(Number(str(c), p) for c in raw["center"])
        self.edge_x = Number(str(raw["edge_x"]), p)
        self.edge_y = Number(str(raw["edge_y"]), p)
        self.nx = int(raw["nx"])
        self.ny = int(raw["ny"])
        ref = raw.get("reference")
        self.reference = tuple(float(c) for c in ref) if ref else None


def _bore(raw: dict, p: int, idx: int) -> dict:
    """One bore -> typed wall spec; scalars become Number, `kind` selects the wall.

    Exactly one geometry: radius (cylinder), radius+bend (torus arc),
    radius+sides (regular polygon, radius = apothem), r2_poly (surface of
    revolution x'^2+y'^2 = c0+c1*z+c2*z^2), surface (implicit F(x,y,z)=0 in µm,
    F<0 inside; needs aim_radius for source aiming and engine_method for hits).
    """
    mods = [k for k in ("surface", "r2_poly", "bend", "sides", "funnel")
            if raw.get(k) is not None]
    if len(mods) > 1:
        raise ValueError(f"bore {idx}: {' + '.join(mods)} cannot be combined")
    out = {"center": tuple(Number(str(c), p) for c in raw.get("center", [0.0, 0.0]))}
    if raw.get("surface") is not None:
        if raw.get("radius") is not None:
            raise ValueError(f"bore {idx}: surface bore takes aim_radius, not radius")
        if raw.get("aim_radius") is None:
            raise ValueError(f"bore {idx}: surface bore needs aim_radius")
        if raw.get("engine_method") is None:
            raise ValueError(f"bore {idx}: surface bore needs engine_method")
        try:
            engine_method = HitMethod(str(raw["engine_method"]))
            get_backend(engine_method)
        except ValueError as exc:
            raise ValueError(
                f"bore {idx}: invalid engine_method {raw['engine_method']!r}"
            ) from exc
        out.update(kind="implicit", surface=str(raw["surface"]),
                   aim_radius=Number(str(raw["aim_radius"]), p),
                   engine_method=engine_method)
        return out
    if "engine_method" in raw:
        raise ValueError(f"bore {idx}: engine_method is only valid with surface")
    if raw.get("r2_poly") is not None:
        if raw.get("radius") is not None:
            raise ValueError(f"bore {idx}: r2_poly replaces radius")
        cs = list(raw["r2_poly"])
        if not 1 <= len(cs) <= 3:
            raise ValueError(f"bore {idx}: r2_poly takes 1-3 coefficients")
        cs += [0.0] * (3 - len(cs))
        out.update(kind="revolution",
                   r2_poly=tuple(Number(str(c), p) for c in cs))
        return out
    if raw.get("radius") is None:
        raise ValueError(f"bore {idx}: needs radius, r2_poly or surface")
    out["radius"] = Number(str(raw["radius"]), p)
    if raw.get("bend") is not None:
        bend = raw["bend"]
        if bend.get("radius") is None or bend.get("toward") is None:
            raise ValueError(f"bore {idx}: bend needs radius and toward [ux, uy]")
        out.update(kind="torus", bend={
            "radius": Number(str(bend["radius"]), p),
            "toward": tuple(Number(str(c), p) for c in bend["toward"]),
        })
    elif raw.get("sides") is not None:
        n = int(raw["sides"])
        if n < 3:
            raise ValueError(f"bore {idx}: sides must be >= 3")
        out.update(kind="polygon", sides=n,
                   rotation=Number(f"({raw.get('rotation_deg', 0)})*pi/180", p))
    elif raw.get("funnel") is not None:
        fn = raw["funnel"]
        g = list(fn.get("g", ()))
        if len(g) != 2:
            raise ValueError(f"bore {idx}: funnel needs g: [a, b] (per m, m^2)")
        f = list(fn.get("f", g))          # conformal f = g unless overridden
        if len(f) != 2:
            raise ValueError(f"bore {idx}: funnel f takes [a, b]")
        out.update(kind="funnel",
                   g=tuple(Number(str(c), p) for c in g),
                   f=tuple(Number(str(c), p) for c in f))
    else:
        out["kind"] = "cylinder"
    return out


def _conditioning_loss(bores, theta_c: float, z_span: float = 0.0) -> int:
    """Digits burnt by the worst conditioned bore.

    Torus: ceil(2*log10(R/a) + log10(1/theta_c) + 2) — the expanded quartic
    (w^2+K)^2 - 4R^2*(w^2-s^2) subtracts terms ~4R^4 (2.2e40 um^4 at
    R = 8625 m, a = 6 um), so at 64 digits their ULP ~2e-24 blurs the wall
    to ~1e-44 um, and grazing hits stretch the root jitter by another
    1/theta.
    Funnel: 2*log10(S/r0) + 2 with S = max(|c|*g, r0*f) over the z-span —
    the quartic mixes the dilated-axis scale against the r0-scale wall
    (measured ~2 digits at S/r0 ~ 19: exp/out/37-quick stage 9).
    """
    def poly_max(a, b):
        vals = [1.0, abs(1.0 + a * z_span + b * z_span * z_span)]
        if b != 0.0:
            zv = -a / (2.0 * b)
            if 0.0 < zv < z_span:
                vals.append(abs(1.0 + a * zv + b * zv * zv))
        return max(vals)

    loss = 0.0
    for bore in bores:
        if bore.get("kind") == "torus":
            ra = float(bore["bend"]["radius"]) / float(bore["radius"])
            loss = max(loss, 2 * math.log10(ra) + math.log10(1 / theta_c) + 2)
        elif bore.get("kind") == "funnel":
            r0 = float(bore["radius"])
            cr = math.hypot(float(bore["center"][0]), float(bore["center"][1]))
            gmax = poly_max(float(bore["g"][0]), float(bore["g"][1]))
            fmax = poly_max(float(bore["f"][0]), float(bore["f"][1]))
            s = max(cr * gmax, r0 * fmax)
            loss = max(loss, 2 * math.log10(max(s / r0, 1.0)) + 2)
    return math.ceil(loss)


def _precision_target(raw, p: int, bores, theta_c: float, z_span: float = 0.0):
    """Certified digits for the hit cross-checks (stage-9 tolerance 1e-target).

    Explicit yaml value, or the default max(4, ceiling) with
    ceiling = p - 2 guard - _conditioning_loss(bores). A target above the
    ceiling is unreachable by any method at this precision, so it warns and
    the run proceeds, reporting the shortfall. Returns (target, auto, loss).
    """
    loss = _conditioning_loss(bores, theta_c, z_span)
    ceiling = p - 2 - loss
    target = int(raw) if raw is not None else max(4, ceiling)
    if target < 1:
        raise ValueError("precision_target must be >= 1")
    if target > ceiling:
        warnings.warn(
            f"precision_target {target} exceeds the certifiable ceiling "
            f"{ceiling} (precision {p} - 2 guard - {loss} wall conditioning): "
            "stage-9 matches will undershoot; raise precision or lower the "
            "target")
    return target, raw is None, loss


class CapillaryCfg:
    def __init__(self, raw: dict, base_screen: dict, p: int):
        self.z0 = Number(str(raw["z0"]), p)
        self.z1 = Number(str(raw["z1"]), p)
        self.bores = [_bore(b, p, i) for i, b in enumerate(raw["bores"])]
        self.source = _required_source(raw.get("source"), "capillary", p)
        base = _merge(base_screen, raw.get("screen", {}))
        self.screen = ScreenCfg(base, p)
        # straight-flight invariant: source.z < z0 < z1 <= screen.z
        if not float(self.source.position[2]) < float(self.z0) < float(self.z1):
            raise ValueError(
                f"capillary: source z = {float(self.source.position[2])}, "
                f"z0 = {float(self.z0)}, z1 = {float(self.z1)} violate "
                "source.z < z0 < z1")
        if float(self.screen.z) < float(self.z1):
            raise ValueError(f"capillary screen: z = {float(self.screen.z)} "
                             f"is inside the optic (z1 = {float(self.z1)})")
        # extra screens (stages 11, 14): re-binned from the same trace, each merged
        # onto the main capillary screen; must sit past the exit (straight flight)
        self.screens = [ScreenCfg(_merge(base, s), p) for s in raw.get("screens", ())]
        for i, s in enumerate(self.screens):
            if float(s.z) < float(self.z1):
                raise ValueError(f"capillary screens[{i}]: z = {float(s.z)} "
                                 f"is inside the optic (z1 = {float(self.z1)})")


class Config:
    def __init__(self, raw: dict, yaml_file: str | None = None):
        if raw is None:
            raw = {}
        elif not isinstance(raw, dict):
            raise ValueError("config must be a mapping")
        if "lloyd" in raw:
            raise ValueError("lloyd was removed together with stages 4 and 5")
        if "source" in raw:
            raise ValueError(
                "top-level source was removed; configure free.source and/or "
                "capillary.source explicitly"
            )
        if "flag_thresholds" in raw:
            raise ValueError(
                "top-level flag_thresholds is not supported; use "
                "stage14.flag_thresholds"
            )
        trace = raw.get("trace")
        if isinstance(trace, dict) and "engine_method" in trace:
            raise ValueError(
                "trace.engine_method was removed; set engine_method on each "
                "`surface:` bore"
            )
        base_raw = {k: v for k, v in raw.items()
                    if k not in ("free", "capillary")}
        cfg = _merge(DEFAULTS, base_raw)
        for scene, defaults in (("free", FREE_DEFAULTS),
                                ("capillary", CAPILLARY_DEFAULTS)):
            if scene not in raw:
                continue
            if not isinstance(raw[scene], dict):
                raise ValueError(f"{scene} must be a mapping")
            cfg[scene] = _merge(defaults, raw[scene])
        self.raw = cfg
        self.yaml_file = yaml_file
        p = int(cfg["precision"])
        self.precision = p
        self.seed = int(cfg["seed"])
        self.energy_kev = Number(str(cfg["energy_kev"]), p)
        self.spectrum = cfg["spectrum"]
        mat = str(cfg["material"])
        if mat not in MATERIALS:
            raise ValueError(f"unknown material: {mat!r}; "
                             f"available {sorted(MATERIALS)}")
        self.material = MATERIALS[mat]
        self.screen = ScreenCfg(cfg["screen"], p)
        free = cfg.get("free")
        self.free_source = (_required_source(free.get("source"), "free", p)
                            if free is not None else None)
        self.free_screen = (ScreenCfg(_merge(cfg["screen"], free.get("screen", {})), p)
                            if free is not None else None)
        capillary = cfg.get("capillary")
        self.capillary = (CapillaryCfg(capillary, cfg["screen"], p)
                          if capillary is not None else None)
        # theta_c of the hardest spectral line (theta_c ~ 1/E): the smallest
        # critical angle bounds the grazing term of the conditioning loss
        e_max = max((ln.e_kev for ln in spectral_lines(cfg["spectrum"], self.energy_kev)),
                    key=float)
        theta_c = float(self.material.critical_angle(e_max, precision=p))
        (self.precision_target, self.precision_target_auto,
         self.precision_target_loss) = _precision_target(
            cfg["precision_target"], p,
            self.capillary.bores if self.capillary else [], theta_c,
            float(self.capillary.z1 - self.capillary.z0)
            if self.capillary else 0.0)
        self.max_bounces = int(cfg["trace"]["max_bounces"])
        self.amplitude_min = float(cfg["trace"]["amplitude_min"])
        self.lean_rays = bool(cfg["trace"]["lean_rays"])
        if ("per_line_fresnel" in (raw or {}).get("spectrum", {})
                and cfg["spectrum"]["mode"] == "monochromatic"):
            raise ValueError("spectrum: per_line_fresnel has no effect in "
                             "monochromatic mode; remove it")
        # The option only applies to multi-line spectra. Keeping its default
        # implicit lets a fully merged config be loaded again without looking
        # like the user explicitly enabled it for monochromatic mode.
        self.per_line_fresnel = bool(
            cfg["spectrum"].get("per_line_fresnel", True))
        self.schematic_rays = int(cfg["schematic"]["n_rays"])
        self.sketch_rank = int(cfg["sketch"]["rank"])
        self.validate_rays = int(cfg["validate"]["n_rays"])
        if self.validate_rays < 1:
            raise ValueError("validate: n_rays must be at least 1")
        vm = cfg["validate"].get("methods")
        if (not isinstance(vm, list) or not vm
                or not all(isinstance(m, str) for m in vm)):
            raise ValueError(
                "validate: requires methods, a non-empty list of method "
                f"names from {[m.value for m in HitMethod]}")
        if len(set(vm)) != len(vm):
            raise ValueError("validate: methods must not contain duplicates")
        self.validate_methods = tuple(HitMethod(m) for m in vm)
        self.validate_reference = HitMethod(str(cfg["validate"]["reference"]))
        if self.validate_reference in self.validate_methods:
            raise ValueError("validate: reference must not be among methods")
        for m in (*self.validate_methods, self.validate_reference):
            if m not in (HitMethod.CPP_CLOSED_FORM, HitMethod.PYTHON_CLOSED_FORM):
                get_backend(m)  # backend registry stays authoritative
        self.beamlet_w0 = float(cfg["beamlet"]["w0"])
        w0t = cfg["beamlet"].get("w0_t")
        if not (w0t is None or w0t == "auto" or isinstance(w0t, (int, float))):
            raise ValueError(f"beamlet w0_t: null, \"auto\" or a number, got {w0t!r}")
        self.beamlet_w0_t = float(w0t) if isinstance(w0t, (int, float)) else w0t
        self.beamlet_ns = float(cfg["beamlet"]["window_sigmas"])
        self.beamlet_waist_z = cfg["beamlet"].get("waist_z")
        stage14 = cfg.get("stage14")
        if not isinstance(stage14, dict):
            raise ValueError("stage14 must be a mapping")
        stage14_unknown = stage14.keys() - {"flag_thresholds"}
        if stage14_unknown:
            raise ValueError(
                f"stage14 has unknown keys {sorted(stage14_unknown)}"
            )
        thresholds = stage14.get("flag_thresholds")
        if not isinstance(thresholds, dict):
            raise ValueError("stage14.flag_thresholds must be a mapping")
        sigma_keys = ("ic_n_sigma", "ref_ic_n_sigma", "w_n_sigma")
        keys = sigma_keys + ("min_coherent_fraction",)
        required = set(keys)
        missing = required - thresholds.keys()
        unknown = thresholds.keys() - required
        if missing or unknown:
            detail = []
            if missing:
                detail.append(f"missing {sorted(missing)}")
            if unknown:
                detail.append(f"unknown {sorted(unknown)}")
            raise ValueError("stage14.flag_thresholds: " + "; ".join(detail))
        for key in keys:
            value = thresholds[key]
            if (not isinstance(value, (int, float))
                    or isinstance(value, bool)):
                raise ValueError(
                    f"stage14.flag_thresholds.{key} must be a number"
                )
        self.stage14_flag_thresholds = {
            key: float(thresholds[key]) for key in keys
        }
        for key in sigma_keys:
            value = self.stage14_flag_thresholds[key]
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"stage14.flag_thresholds.{key} must be finite and > 0"
                )
        fraction = self.stage14_flag_thresholds["min_coherent_fraction"]
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(
                "stage14.flag_thresholds.min_coherent_fraction must be finite and in (0, 1]"
            )

    def validate_beamlet(self):
        """Stage-11-only checks; other stages retain the legacy YAML parser."""
        def finite(v):
            return (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v))

        beamlet = self.raw["beamlet"]
        w0, w0t, wz = beamlet["w0"], beamlet.get("w0_t"), beamlet.get("waist_z")
        if not (finite(w0) and w0 > 0):
            raise ValueError(f"beamlet w0: a positive number, got {w0!r}")
        if not (w0t is None or w0t == "auto" or finite(w0t) and w0t > 0):
            raise ValueError(f"beamlet w0_t: null, \"auto\" or a positive number, got {w0t!r}")
        if not (wz is None or finite(wz)):
            raise ValueError(f"beamlet waist_z: null or a number, got {wz!r}")
        self.beamlet_waist_z = None if wz is None else float(wz)

    def validate_wave_estimator(self) -> dict:
        """Stage-16-only checks; returns the `wave_estimator` section merged with WAVE_DEFAULTS."""
        raw = self.raw.get("wave_estimator")
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("wave_estimator must be a mapping")
        unknown = raw.keys() - WAVE_DEFAULTS.keys()
        if unknown:
            raise ValueError(f"wave_estimator has unknown keys {sorted(unknown)}")
        w = _merge(WAVE_DEFAULTS, raw)

        def num(key, lo=None, hi=None, integer=False, optional=False, strict=False):
            v = w[key]
            if v is None and optional:
                return None
            if integer:
                ok = isinstance(v, int) and not isinstance(v, bool)
            else:
                ok = (isinstance(v, (int, float)) and not isinstance(v, bool)
                      and math.isfinite(v))
            if ok and lo is not None:
                ok = v > lo if strict else v >= lo
            if ok and hi is not None:
                ok = v <= hi
            if not ok:
                raise ValueError(f"wave_estimator.{key}: invalid value {v!r}")
            return v if integer else float(v)

        for key, choices in (("provider", ("auto", "uisk", "fb", "free")),
                             ("observable", ("coherent_cell", "point")),
                             ("source_mode", ("quadrature", "recorded_origins")),
                             ("fb_wall", ("dir-ell", "dir-ell-real")),
                             ("fb_grid_dtype", ("complex64", "complex128"))):
            if w[key] not in choices:
                raise ValueError(
                    f"wave_estimator.{key}: expected one of {choices}, got {w[key]!r}")
        w["target_error"] = num("target_error", lo=0.0, strict=True)
        w["source_nodes"] = num("source_nodes", lo=1, integer=True)
        w["grid_dx"] = num("grid_dx", lo=0.0, strict=True, optional=True)
        w["angle_margin"] = num("angle_margin", lo=1.0)
        w["pad"] = num("pad", lo=0.0)
        w["max_bounces"] = num("max_bounces", lo=0, integer=True)
        w["mask_supersample"] = num("mask_supersample", lo=1, integer=True)
        w["pixel_subsamples"] = num("pixel_subsamples", lo=1, integer=True)
        w["intensity_floor"] = num("intensity_floor", lo=0.0, hi=1.0)
        w["workers"] = num("workers", lo=1, integer=True, optional=True)
        w["cache_gb"] = num("cache_gb", lo=0.0)
        w["fb_jmax"] = num("fb_jmax", lo=8, integer=True, optional=True)
        w["fb_theta_cut"] = num("fb_theta_cut", lo=0.0, strict=True)
        w["fb_chebyshev_tol"] = num("fb_chebyshev_tol", lo=0.0, strict=True)
        w["fb_dr"] = num("fb_dr", lo=0.0, strict=True)
        w["fb_angle_margin"] = num("fb_angle_margin", lo=1.0)
        w["fb_lattice_half"] = num("fb_lattice_half", lo=0.0, strict=True, optional=True)
        w["fb_jobs"] = num("fb_jobs", lo=1, integer=True)
        if w["fb_per_bore_maps"] is not None and not isinstance(w["fb_per_bore_maps"], bool):
            raise ValueError(f"wave_estimator.fb_per_bore_maps: expected true, false or null, got {w['fb_per_bore_maps']!r}")
        self.wave = w
        return w

    def validate_b5_estimator(self) -> dict:
        """Stage-17-only contract for experimental archive phase-operator validation."""
        raw = self.raw.get("b5_estimator")
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("b5_estimator must be a mapping")
        unknown = raw.keys() - B5_DEFAULTS.keys()
        if unknown:
            raise ValueError(f"b5_estimator has unknown keys {sorted(unknown)}")
        b5 = _merge(B5_DEFAULTS, raw)
        for key, choices in (("provider", ("archive_phase", "archive_canonical")),
                             ("patch_space", ("entrance", "screen")),
                             ("charts", ("screen", "adaptive"))):
            if b5[key] not in choices:
                raise ValueError(f"b5_estimator.{key}: expected one of {choices}")
        for key, minimum in (("max_modes", 1), ("rays_per_mode", 1), ("degree", 2),
                             ("neighbors", 24), ("min_neighbors", 24),
                               ("patches_per_family", 1), ("screen_index", 0), ("seed", 0),
                               ("map_stride", 1), ("map_jobs", 1)):
            value = b5[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"b5_estimator.{key}: expected an integer >= {minimum}")
        if b5["degree"] > 5:
            raise ValueError("b5_estimator.degree must be from 2 through 5")
        if b5["min_neighbors"] > b5["neighbors"]:
            raise ValueError("b5_estimator.min_neighbors must not exceed neighbors")
        widths = b5["widths_m"]
        if (not isinstance(widths, list) or not widths or any(
                isinstance(v, bool) or not isinstance(v, (float, int))
                or not math.isfinite(v) or v <= 0 for v in widths)):
            raise ValueError("b5_estimator.widths_m must contain finite positive widths")
        if len(set(widths)) != len(widths):
            raise ValueError("b5_estimator.widths_m must not contain duplicates")
        for key, limit in (("map_ray_budgets", b5["rays_per_mode"]),
                           ("map_snapshots", b5["max_modes"])):
            values = b5[key]
            if (not isinstance(values, list) or any(isinstance(v, bool) or not isinstance(v, int)
                    or v < 1 or v > limit for v in values) or len(set(values)) != len(values)):
                raise ValueError(f"b5_estimator.{key}: invalid unique positive integer list")
        for key in ("phase_tolerance", "max_condition"):
            value = b5[key]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"b5_estimator.{key}: expected a finite positive number")
            b5[key] = float(value)
        self.b5 = b5
        return b5

    def validate_b9_estimator(self) -> dict:
        """Stage-18-only options for experimental archive contour maps."""
        raw = self.raw.get("b9_estimator")
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("b9_estimator must be a mapping")
        unknown = raw.keys() - B9_DEFAULTS.keys()
        if unknown:
            raise ValueError(f"b9_estimator has unknown keys {sorted(unknown)}")
        b9 = _merge(B9_DEFAULTS, raw)
        if b9["provider"] != "archive_contour":
            raise ValueError("b9_estimator.provider: expected archive_contour")
        if b9["amplitude_mode"] not in ("point_jacobian", "tube_flux", "shared_flux"):
            raise ValueError("b9_estimator.amplitude_mode: expected point_jacobian, tube_flux or shared_flux")
        if b9["field_representation"] not in ("contour_p1", "phase_quadrature", "curved_tubes"):
            raise ValueError("b9_estimator.field_representation: expected contour_p1, phase_quadrature or curved_tubes")
        if b9["phase_backend"] not in ("type3", "regular", "regular_mixed"):
            raise ValueError("b9_estimator.phase_backend: expected type3, regular or regular_mixed")
        if b9["phase_backend"] != "type3" and b9["field_representation"] not in ("phase_quadrature", "curved_tubes"):
            raise ValueError("b9_estimator.phase_backend=regular/regular_mixed requires phase_quadrature")
        for key, minimum in (("mode_start", 0), ("max_modes", 1), ("rays_per_mode", 3),
                             ("screen_index", 0), ("map_stride", 1), ("map_jobs", 1),
                             ("edge_order", 2), ("pixel_order", 1), ("nufft_threads", 1),
                             ("max_triangles_per_batch", 1), ("carrier_groups", 0),
                             ("phase_degree", 1), ("triangle_quadrature_order", 2),
                             ("max_quadrature_nodes_per_batch", 1),
                             ("receiver_channels_per_batch", 1),
                             ("quadrature_max_order", 2),
                             ("holdout_stride", 0)):
            value = b9[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"b9_estimator.{key}: expected an integer >= {minimum}")
        if b9["holdout_stride"] == 1:
            raise ValueError("b9_estimator.holdout_stride: expected 0 or an integer >= 2")
        for key, limit in (("map_ray_budgets", b9["rays_per_mode"]),
                           ("map_snapshots", b9["max_modes"]), ("phase_subdivisions", None)):
            values = b9[key]
            if (not isinstance(values, list)
                    or (key == "phase_subdivisions" and not values)
                    or any(isinstance(v, bool) or not isinstance(v, int)
                           or v < (3 if key == "map_ray_budgets" else 1)
                           or (limit is not None and v > limit) for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"b9_estimator.{key}: invalid unique positive integer list")
        for key in ("nufft_eps", "determinant_floor"):
            value = b9[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"b9_estimator.{key}: expected a finite non-negative number")
            b9[key] = float(value)
        if not 0 < b9["nufft_eps"] < 1:
            raise ValueError("b9_estimator.nufft_eps: expected a number strictly between 0 and 1")
        if b9["phase_degree"] not in (1, 2):
            raise ValueError("b9_estimator.phase_degree: expected 1 or 2")
        safety = b9["quadrature_safety"]
        if (isinstance(safety, bool) or not isinstance(safety, (int, float))
                or not math.isfinite(safety) or safety <= 0):
            raise ValueError("b9_estimator.quadrature_safety: expected a finite positive number")
        b9["quadrature_safety"] = float(safety)
        if b9["phase_backend"] == "regular_mixed" and b9["quadrature_max_order"] < b9["triangle_quadrature_order"]:
            raise ValueError("b9_estimator.quadrature_max_order must be at least triangle_quadrature_order")
        if b9["field_representation"] in ("phase_quadrature", "curved_tubes"):
            if b9["carrier_groups"] or b9["phase_subdivisions"] != [1]:
                raise ValueError("b9_estimator.phase_quadrature requires carrier_groups=0 and phase_subdivisions=[1]")
            if b9["max_quadrature_nodes_per_batch"] < b9["triangle_quadrature_order"]**2:
                raise ValueError("b9_estimator.max_quadrature_nodes_per_batch must fit one triangle rule")
        retrace = b9["cylinder_retrace"]
        if retrace is not None:
            defaults = dict(bores=[], angles=256, inner_rings=16, outer_rings=8,
                            boundary_relative_gap=1e-6, entrance_relative_inset=2e-6, precision=64)
            if not isinstance(retrace, dict) or retrace.keys()-defaults.keys():
                raise ValueError("b9_estimator.cylinder_retrace: invalid mapping or unknown keys")
            retrace = {**defaults, **retrace}
            ids = retrace["bores"]
            if (not isinstance(ids, list) or not ids
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("b9_estimator.cylinder_retrace.bores: expected unique nonnegative indices")
            for key, minimum in (("angles", 8), ("inner_rings", 1), ("outer_rings", 1), ("precision", 32)):
                value = retrace[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"b9_estimator.cylinder_retrace.{key}: expected integer >= {minimum}")
            for key in ("boundary_relative_gap", "entrance_relative_inset"):
                value = retrace[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 < value < .01):
                    raise ValueError(f"b9_estimator.cylinder_retrace.{key}: expected 0 < value < 0.01")
                retrace[key] = float(value)
            b9["cylinder_retrace"] = retrace
        adaptive = b9["adaptive_retrace"]
        if adaptive is not None:
            if b9["field_representation"] != "phase_quadrature" or b9["phase_degree"] != 2:
                raise ValueError("b9_estimator.adaptive_retrace requires phase_quadrature with phase_degree=2; its probes test that phase model")
            defaults = dict(bores=[], angles=128, radial_rings=4, max_depth=16, max_nodes=12000,
                            phase_tolerance_rad=.05, amplitude_relative_tolerance=.05,
                            fresnel_relative_tolerance=.05, geometry_relative_tolerance=.002,
                            flux_relative_tolerance=.05,
                            precision=64, entrance_relative_inset=2e-6)
            if not isinstance(adaptive, dict) or adaptive.keys()-defaults.keys():
                raise ValueError("b9_estimator.adaptive_retrace: invalid mapping or unknown keys")
            adaptive = {**defaults, **adaptive}
            ids = adaptive["bores"]
            if (not isinstance(ids, list) or not ids
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("b9_estimator.adaptive_retrace.bores: expected unique nonnegative indices")
            if set(ids) & set((b9["cylinder_retrace"] or {}).get("bores", [])):
                raise ValueError("b9_estimator: adaptive_retrace and cylinder_retrace bore sets must be disjoint")
            for key, minimum in (("angles", 8), ("radial_rings", 1), ("max_depth", 1),
                                 ("max_nodes", 7), ("precision", 32)):
                value = adaptive[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"b9_estimator.adaptive_retrace.{key}: expected integer >= {minimum}")
            for key in ("phase_tolerance_rad", "amplitude_relative_tolerance", "fresnel_relative_tolerance",
                          "geometry_relative_tolerance", "flux_relative_tolerance", "entrance_relative_inset"):
                value = adaptive[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"b9_estimator.adaptive_retrace.{key}: expected finite positive number")
                adaptive[key] = float(value)
            if adaptive["entrance_relative_inset"] >= .01:
                raise ValueError("b9_estimator.adaptive_retrace.entrance_relative_inset: expected < 0.01")
            b9["adaptive_retrace"] = adaptive
        curved = b9["curved_retrace"]
        if (curved is not None) != (b9["field_representation"] == "curved_tubes"):
            raise ValueError("b9_estimator.curved_retrace requires field_representation=curved_tubes and vice versa")
        if curved is not None:
            if b9["phase_degree"] != 2 or b9["phase_backend"] != "regular_mixed" or b9["amplitude_mode"] != "point_jacobian":
                raise ValueError("b9_estimator.curved_tubes requires phase_degree=2, phase_backend=regular_mixed and amplitude_mode=point_jacobian")
            if adaptive is not None:
                raise ValueError("b9_estimator.curved_retrace and adaptive_retrace cannot be combined")
            defaults = dict(bores=[], angles=128, radial_rings=4, max_depth=16, max_nodes=20000,
                phase_tolerance_rad=.05, density_relative_tolerance=.02, geometry_phase_tolerance_rad=.05,
                fresnel_relative_tolerance=.05, precision=64, entrance_relative_inset=2e-6)
            if not isinstance(curved, dict) or curved.keys()-defaults.keys():
                raise ValueError("b9_estimator.curved_retrace: invalid mapping or unknown keys")
            curved = {**defaults, **curved}
            ids = curved["bores"]
            if (not isinstance(ids, list) or not ids
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError("b9_estimator.curved_retrace.bores: expected unique nonnegative indices")
            if set(ids) & set((b9["cylinder_retrace"] or {}).get("bores", [])):
                raise ValueError("b9_estimator: curved_retrace and cylinder_retrace bore sets must be disjoint")
            for key, minimum in (("angles", 8), ("radial_rings", 1), ("max_depth", 1), ("max_nodes", 10), ("precision", 32)):
                value = curved[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"b9_estimator.curved_retrace.{key}: expected integer >= {minimum}")
            for key in ("phase_tolerance_rad", "density_relative_tolerance", "geometry_phase_tolerance_rad",
                        "fresnel_relative_tolerance", "entrance_relative_inset"):
                value = curved[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"b9_estimator.curved_retrace.{key}: expected finite positive number")
                curved[key] = float(value)
            if curved["entrance_relative_inset"] >= .01:
                raise ValueError("b9_estimator.curved_retrace.entrance_relative_inset: expected < 0.01")
            b9["curved_retrace"] = curved
        missing = b9["max_missing_area_fraction"]
        if missing is not None:
            if (isinstance(missing, bool) or not isinstance(missing, (int, float))
                    or not math.isfinite(missing) or not 0 <= missing < 1):
                raise ValueError("b9_estimator.max_missing_area_fraction: expected null or a number in [0,1)")
            b9["max_missing_area_fraction"] = float(missing)
        self.b9 = b9
        return b9


def load(path_or_dict: str | os.PathLike | dict) -> Config:
    """Build Config from a YAML file path or an already-parsed dict."""
    if isinstance(path_or_dict, dict):
        return Config(path_or_dict)
    import yaml
    path = os.fspath(path_or_dict)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config({} if raw is None else raw,
                  yaml_file=os.path.basename(path))

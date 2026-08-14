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
from .types import HitMethod

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
    # wall glass n = 1 - delta - i*beta: fused_silica | glass_oe2012 (Opt. Express 20, 3975)
    "material": "fused_silica",
    # monochromatic | gaussian {rel_fwhm, n_lines, n_sigma} | lines [{energy_kev, weight}]
    # | table {file}; per_line_fresnel (multi-line modes only): r(E_m) per
    # line instead of frozen r(E0)
    "spectrum": {"mode": "monochromatic", "rel_fwhm": 2.0e-4, "n_lines": 7,
                 "n_sigma": 3.0, "per_line_fresnel": True},
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
    # rays_jsonl: full-precision per-ray records (rays.jsonl.gz, replay
    # input). Records/multi-line runs trace with the amplitude_min kill off
    # (E0-truncation would bias other energies) and apply the threshold after
    # the per-line amplitudes are known.
    # lean_rays: drop refl and write opl/sins as float64 in the rays file —
    # stage 10/rescreen read floats anyway (bit-identical); the file cannot
    # feed the Number-path replay (stages 2/6) or the beamlet stage.
    "trace": {"max_bounces": 200, "amplitude_min": 1.0e-6,
              "rays_jsonl": True, "lean_rays": False},
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
    # widths. w0 is the sagittal (channel) waist; w0_t the tangential one:
    # null = isotropic (= w0), "auto" = the scene's Fresnel scale
    # sqrt(lam*L/pi) of the source->screen flight, or an explicit number.
    "beamlet": {"w0": 5.0e-7, "w0_t": None, "window_sigmas": 3.0},
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

    def budget(self, quick: int) -> tuple[int, int]:
        """(n_modes, n_rays) at reduction factor `quick`, with sampling floors."""
        return max(2, self.n_modes // quick), max(20, self.n_rays // quick)


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
    1/theta: doc/2026-07-13-torus-quartic-cancellation.ru.md.
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
            "target (doc/2026-07-13-torus-quartic-cancellation.ru.md)")
    return target, raw is None, loss


class CapillaryCfg:
    def __init__(self, raw: dict, base_screen: dict, p: int):
        self.z0 = Number(str(raw["z0"]), p)
        self.z1 = Number(str(raw["z1"]), p)
        self.bores = [_bore(b, p, i) for i, b in enumerate(raw["bores"])]
        self.source = _required_source(raw.get("source"), "capillary", p)
        base = _merge(base_screen, raw.get("screen", {}))
        self.screen = ScreenCfg(base, p)
        # extra screens (stage 10): re-binned from the same trace, each merged
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
            raise ValueError("configure free.source and/or capillary.source explicitly")
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
        self.rays_jsonl = bool(cfg["trace"]["rays_jsonl"])
        self.lean_rays = bool(cfg["trace"]["lean_rays"])
        if ("per_line_fresnel" in (raw or {}).get("spectrum", {})
                and cfg["spectrum"]["mode"] == "monochromatic"):
            raise ValueError("spectrum: per_line_fresnel has no effect in "
                             "monochromatic mode; remove it")
        self.per_line_fresnel = bool(cfg["spectrum"]["per_line_fresnel"])
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


def load(path_or_dict) -> Config:
    """Build Config from a YAML file path or an already-parsed dict."""
    if isinstance(path_or_dict, dict):
        return Config(path_or_dict)
    import yaml
    path = os.fspath(path_or_dict)
    with open(path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh) or {}, yaml_file=os.path.basename(path))

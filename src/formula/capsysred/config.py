"""YAML/dict -> typed simulation config.

The single Number boundary: every physical scalar becomes Number(precision) here;
downstream code reads precision from the Numbers it receives, never as a parameter.
Counts stay int, spectral weights stay float (not phase-critical).
"""

import copy

from .._roots import get_backend
from ..formula import Number
from ..xray import FUSED_SILICA

DEFAULTS = {
    "precision": 32,
    "seed": 12345,
    "energy_kev": 8.0,
    # monochromatic | gaussian {rel_fwhm, n_lines, n_sigma} | lines [{energy_kev, weight}]
    # | table {file}; per_line_fresnel: r(E_m) per line instead of frozen r(E0)
    "spectrum": {"mode": "monochromatic", "rel_fwhm": 2.0e-4, "n_lines": 7,
                 "n_sigma": 3.0, "per_line_fresnel": True},
    # extended incoherent source: n_modes coherent point modes, n_rays per mode
    "source": {
        "shape": "gaussian",          # point | gaussian (size=sigma) | disk (size=radius)
        "size": 2.1e-6,
        "position": [0.0, 0.0, -0.08],
        "n_modes": 100,
        "n_rays": 800,
    },
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
    "free": {"source": {}, "screen": {}},
    "lloyd": {
        "height": 1.0e-5,             # source axis above the mirror plane x=0
        "z0": 0.0,                    # mirror extent along z
        "z1": 0.06,
        "source": {"size": 1.5e-7, "n_modes": 80, "n_rays": 900},
        "screen": {"center": [6.0e-6, 0.0], "edge_x": 1.2e-5, "nx": 161},
    },
    # 6 um bore, source sigma 0.3 um a centimetre before it.
    "capillary": {
        "bores": [{"center": [0.0, 0.0], "radius": 6.0e-6}],
        "z0": 0.0,
        "z1": 0.05,
        "source": {"size": 3.0e-7, "position": [0.0, 0.0, -0.01], "n_modes": 80, "n_rays": 1000},
        "screen": {"z": 0.051, "edge_x": 1.6e-5, "edge_y": 1.6e-5, "nx": 41, "ny": 41},
    },
    # rays_jsonl: full-precision per-ray records (replay input); sample_every:
    # write every k-th ray. Records/multi-line runs trace with the amplitude_min
    # kill off (E0-truncation would bias other energies) and apply the threshold
    # after the per-line amplitudes are known.
    # engine_method: RaySurface root finder for `surface:` bores and the hit
    # cross-checks — subdivision (default: grazing-safe, any F) | sturm (exact,
    # polynomial F only) | chebyshev | sampling | auto.
    "trace": {"max_bounces": 200, "amplitude_min": 1.0e-6,
              "rays_jsonl": True, "sample_every": 1,
              "engine_method": "subdivision"},
    # stage 8: number of sketch probe vectors (r ~ n99 modes, see methods §8)
    "sketch": {"rank": 96},
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
        if self.shape not in ("point", "gaussian", "disk"):
            raise ValueError(f"unknown source shape: {self.shape!r}")
        self.size = Number(str(raw["size"]), p)
        self.position = tuple(Number(str(c), p) for c in raw["position"])
        self.n_modes = int(raw["n_modes"])
        self.n_rays = int(raw["n_rays"])


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


class LloydCfg:
    def __init__(self, raw: dict, base_source: dict, base_screen: dict, p: int):
        self.height = Number(str(raw["height"]), p)
        self.z0 = Number(str(raw["z0"]), p)
        self.z1 = Number(str(raw["z1"]), p)
        src = _merge(base_source, raw.get("source", {}))
        if "position" not in (raw.get("source") or {}):
            src["position"] = [raw["height"], 0.0, src["position"][2]]
        self.source = SourceCfg(src, p)
        self.screen = ScreenCfg(_merge(base_screen, raw.get("screen", {})), p)


def _bore(raw: dict, p: int, idx: int) -> dict:
    """One bore -> typed wall spec; scalars become Number, `kind` selects the wall.

    Exactly one geometry: radius (cylinder), radius+bend (torus arc),
    radius+sides (regular polygon, radius = apothem), r2_poly (surface of
    revolution x'^2+y'^2 = c0+c1*z+c2*z^2), surface (implicit F(x,y,z)=0 in µm,
    F<0 inside; needs aim_radius for source aiming).
    """
    mods = [k for k in ("surface", "r2_poly", "bend", "sides")
            if raw.get(k) is not None]
    if len(mods) > 1:
        raise ValueError(f"bore {idx}: {' + '.join(mods)} cannot be combined")
    out = {"center": tuple(Number(str(c), p) for c in raw.get("center", [0.0, 0.0]))}
    if raw.get("surface") is not None:
        if raw.get("radius") is not None:
            raise ValueError(f"bore {idx}: surface bore takes aim_radius, not radius")
        if raw.get("aim_radius") is None:
            raise ValueError(f"bore {idx}: surface bore needs aim_radius")
        out.update(kind="implicit", surface=str(raw["surface"]),
                   aim_radius=Number(str(raw["aim_radius"]), p))
        return out
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
    else:
        out["kind"] = "cylinder"
    return out


class CapillaryCfg:
    def __init__(self, raw: dict, base_source: dict, base_screen: dict, p: int):
        self.z0 = Number(str(raw["z0"]), p)
        self.z1 = Number(str(raw["z1"]), p)
        self.bores = [_bore(b, p, i) for i, b in enumerate(raw["bores"])]
        self.source = SourceCfg(_merge(base_source, raw.get("source", {})), p)
        self.screen = ScreenCfg(_merge(base_screen, raw.get("screen", {})), p)


class Config:
    def __init__(self, raw: dict):
        cfg = _merge(DEFAULTS, raw or {})
        self.raw = cfg
        p = int(cfg["precision"])
        self.precision = p
        self.seed = int(cfg["seed"])
        self.energy_kev = Number(str(cfg["energy_kev"]), p)
        self.spectrum = cfg["spectrum"]
        self.material = FUSED_SILICA
        self.source = SourceCfg(cfg["source"], p)
        self.screen = ScreenCfg(cfg["screen"], p)
        free = cfg["free"]
        self.free_source = SourceCfg(_merge(cfg["source"], free.get("source", {})), p)
        self.free_screen = ScreenCfg(_merge(cfg["screen"], free.get("screen", {})), p)
        self.lloyd = LloydCfg(cfg["lloyd"], cfg["source"], cfg["screen"], p)
        self.capillary = CapillaryCfg(cfg["capillary"], cfg["source"], cfg["screen"], p)
        self.max_bounces = int(cfg["trace"]["max_bounces"])
        self.amplitude_min = float(cfg["trace"]["amplitude_min"])
        self.rays_jsonl = bool(cfg["trace"]["rays_jsonl"])
        self.sample_every = max(1, int(cfg["trace"]["sample_every"]))
        self.engine_method = str(cfg["trace"]["engine_method"])
        get_backend(self.engine_method)  # fail fast on an unknown method name
        self.per_line_fresnel = bool(cfg["spectrum"]["per_line_fresnel"])
        self.sketch_rank = int(cfg["sketch"]["rank"])


def load(path_or_dict) -> Config:
    """Build Config from a YAML file path or an already-parsed dict."""
    if isinstance(path_or_dict, dict):
        return Config(path_or_dict)
    import yaml
    with open(path_or_dict, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh) or {})

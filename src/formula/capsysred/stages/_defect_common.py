"""Modal domains and field-error masks shared by defect diagnostics."""

import math

import numpy as np

from .wave_fb import Disc, FBChannel, mode_damping


_DISCS = {}


def _shape(bore, a_w=None):
    """(sides, rotation, apothem / radius per face) of a polygon bore, None for a circle; with
    `a_w` the outline offset outward by a_w - radius (each face apothem + offset, over a_w)."""
    sides = bore.get("sides")
    if not sides:
        return None
    rot = bore.get("rotation")
    rot = float(rot) if rot is not None else math.radians(float(bore.get("rotation_deg", 0.0)))
    aps = bore.get("apothems")
    if aps is None and a_w is None:
        return int(sides), rot, np.ones(int(sides))
    a = float(bore["radius"])
    apothems = np.array([float(v) for v in aps]) if aps is not None else np.full(int(sides), a)
    ratio = apothems / a if a_w is None else (apothems + (float(a_w) - a)) / float(a_w)
    return int(sides), rot, ratio


def _stretch(shape, phis):
    """Radius factor of the bore outline at the angles phis: the nearest face plane along the
    ray from the centre, r = min_k a_k / cos(phi - phi_k) over faces ahead; a circle 1."""
    if shape is None:
        return np.ones(phis.size)
    n, rot, ratio = shape
    cosr = np.cos(np.asarray(phis)[:, None] - (rot + 2.0 * math.pi * np.arange(n) / n)[None, :])
    r = np.where(cosr > 1e-12, ratio[None, :] / np.maximum(cosr, 1e-12), np.inf)
    return r.min(axis=1)


def _disc(a_w, jmax):
    key = (round(float(a_w), 15), float(jmax))
    if key not in _DISCS:
        _DISCS[key] = Disc(a_w, jmax)
    return _DISCS[key]


def _channel(bore, a_w, disc, k, fresnel=None):
    """FBChannel of a bore; `fresnel` = (2 delta, 2 beta) adds the DIR-ell per-mode damping of
    the fb wave (wave_fb.mode_damping), none = lossless Dirichlet."""
    c0 = np.array([float(bore["center"][0]), float(bore["center"][1])])
    gamma = None if fresnel is None else mode_damping(disc, k, 0.5 * fresnel[0], 0.5 * fresnel[1])
    bend = bore.get("bend")
    if not bend:
        return FBChannel(c0, None, None, disc, k, gamma)
    return FBChannel(c0, np.array([float(bend["toward"][0]), float(bend["toward"][1])]), float(bend["radius"]), disc, k, gamma)


def bore_area(bores, a_w):
    """Cross-section area of the modal domain per bore: pi a_w^2 (circle), 4 a_w^2 (square)."""
    return sum(4.0 * a_w * a_w if _shape(b, a_w) is not None else math.pi * a_w * a_w for b in bores)


def wall_distance(shape, a_w, bx, by):
    """Distance from points (bx, by) in the bore frame to the nearest wall point of the outline
    (circle of radius a_w, or the polygon of apothem a_w through its faces; 0 outside) and the
    unit normal towards that point."""
    if shape is None:
        r = np.hypot(bx, by)
        rs = np.where(r > 0, r, 1.0)
        return np.maximum(a_w - r, 0.0), np.where(r > 0, bx / rs, 1.0), np.where(r > 0, by / rs, 0.0)
    n, rot, ratio = shape
    ang = rot + 2.0 * math.pi * np.arange(n) / n
    gap = a_w * ratio[None, :] - (bx[:, None] * np.cos(ang)[None, :] + by[:, None] * np.sin(ang)[None, :])
    j = np.argmin(gap, axis=1)
    return np.maximum(gap[np.arange(bx.size), j], 0.0), np.cos(ang)[j], np.sin(ang)[j]


def uniform_mask(intensity, lit, ref, eps, threshold):
    """|d mu| <= 2 eta (1/sqrt(I) + 1/sqrt(I_ref)), eta^2 = eps^2 P / A_lit: the relative
    field error eps spread uniformly over the lit cells; masked = not below threshold."""
    I = np.asarray(intensity, float)
    eta = eps * math.sqrt(float(np.sum(np.maximum(I, 0.0))) / max(int(np.sum(lit)), 1))
    i_ref = float(I.flat[ref])
    if not i_ref > 0.0:              # unlit or U-statistic-negative reference: nothing estimable
        dmu = np.full(I.shape, np.inf)
        return dmu, np.ones(I.shape, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        dmu = 2.0 * eta * (1.0 / np.sqrt(np.where(I > 0, I, np.nan)) + 1.0 / math.sqrt(i_ref))
    dmu = np.where(I > 0, dmu, np.inf)
    return dmu, ~(dmu <= threshold)


def bound_eta(bound, spec, z_exit, averaged=False):
    """Field-error amplitude eta of a screen from the per-mode bounds B_s (deposit units):
    beyond the exit (z_s > 0) the Fresnel kernel bounds the field uniformly,
    |e(P)| <= k / (2 pi z_s) (sqrt|Omega| B_s ||E_in||_s + sqrt|A_out| ||E_dep||_out,s);
    on the exit plane only the cell average, |<e>_cell| <= B_s ||E_in||_s / sqrt(A_cell), so a
    map of point samples there gets (None, reason).  eta^2 = sum_s eps_s^2 (equal mode
    weights, as the map's W and I)."""
    z_s = float(spec["z"]) - z_exit
    if z_s <= 0.0 and not averaged:
        return None, "exit plane, point-sampled cells: the L2 bound gives no pointwise bound"
    if z_s > 0.0:
        pref = bound["k"] / (2.0 * math.pi * z_s)
        eps = [pref * (math.sqrt(bound["area"]) * m["B"] * m["E_in_dep"] + math.sqrt(m["O_area"]) * math.sqrt(max(m["O_abs2"], 0.0)))
               for m in bound["modes"]]
        kind = "fresnel"
    else:
        a_cell = float(spec["edge_x"]) / spec["nx"] * float(spec["edge_y"]) / spec["ny"]
        eps = [m["B"] * m["E_in_dep"] / math.sqrt(a_cell) for m in bound["modes"]]
        kind = "cell"
    return math.sqrt(sum(e * e for e in eps)), kind


def bound_mask(intensity, ref, eta, threshold):
    """|d mu| <= 2 eta (1 / sqrt(I) + 1 / sqrt(I_ref)) from a certified field-error amplitude
    eta; cells with I <= 0 (and every cell when I_ref <= 0) are not estimable: masked."""
    I = np.asarray(intensity, float)
    i_ref = float(I.flat[ref])
    if not i_ref > 0.0:
        return np.full(I.shape, np.inf), np.ones(I.shape, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        dmu = 2.0 * eta * (1.0 / np.sqrt(np.where(I > 0, I, np.nan)) + 1.0 / math.sqrt(i_ref))
    dmu = np.where(I > 0, dmu, np.inf)
    return dmu, ~(dmu <= threshold)

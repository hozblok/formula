"""Closed-form / deterministic references the Monte-Carlo maps are checked against.

Display-precision float math (no random sampling). Optical-path differences are
formed through (L1^2-L2^2)/(L1+L2) products, never by subtracting metre-scale
lengths, so float64 keeps the fringe phases to ~1e-15 rad.
"""

import cmath
import math


def bessel_j1(x: float) -> float:
    ax = abs(x)
    if ax < 1e-12:
        return x / 2.0
    if ax <= 14.0:
        term = x / 2.0
        total = term
        for m in range(1, 40):
            term *= -(x * x) / (4.0 * m * (m + 1))
            total += term
            if abs(term) < 1e-17 * abs(total):
                break
        return total
    return math.sqrt(2.0 / (math.pi * ax)) * math.cos(ax - 0.75 * math.pi) * (1 if x > 0 else -1)


def vcz_mu(delta_x: float, shape: str, size: float, lam: float, dist: float) -> float:
    """van Cittert-Zernike |mu| for two screen points separated by delta_x."""
    if shape == "point" or size <= 0.0:
        return 1.0
    k = 2.0 * math.pi / lam
    if shape == "gaussian":
        return math.exp(-0.5 * (k * size * delta_x / dist) ** 2)
    u = k * size * abs(delta_x) / dist          # disk of radius `size`
    return 1.0 if u < 1e-12 else abs(2.0 * bessel_j1(u) / u)


def fresnel_r(sin_theta: float, delta: float, beta: float) -> complex:
    root = cmath.sqrt(sin_theta * sin_theta - 2.0 * delta - 2.0j * beta)
    return (sin_theta - root) / (sin_theta + root)


def _source_quadrature(shape: str, size: float, n: int):
    """1-D x-projection of the source: [(offset, weight)], weights sum to 1."""
    if shape == "point" or size <= 0.0:
        return [(0.0, 1.0)]
    span = 4.0 * size if shape == "gaussian" else size
    pts = []
    for j in range(n):
        s = -span + 2.0 * span * (j + 0.5) / n
        if shape == "gaussian":
            w = math.exp(-0.5 * (s / size) ** 2)
        else:                                   # disk -> chord density
            w = math.sqrt(max(size * size - s * s, 0.0))
        pts.append((s, w))
    total = sum(w for _, w in pts)
    return [(s, w / total) for s, w in pts]


def lloyd_reference(xs, x_ref: float, shape: str, size: float, height: float,
                    z_src: float, z0: float, z1: float, z_scr: float,
                    lam: float, delta: float, beta: float, n_src: int = 201):
    """Deterministic two-path (image-source) Lloyd model on the screen strip.

    Per source point: U = e^{ikLd} + r(theta) e^{ikLr} when the reflection point
    falls on the mirror [z0, z1]; incoherent quadrature over the source. Returns
    normalized I(x), |mu(x, x_ref)| and the closed-form fringe spacing.
    """
    k = 2.0 * math.pi / lam
    D = z_scr - z_src

    def paths(x: float, s_abs: float):
        """(direct present, r or 0, delta_L = Lr - Ld, Ld) for one source point."""
        ld = math.hypot(x - s_abs, D)
        lr = math.hypot(x + s_abs, D)
        dl = 4.0 * x * s_abs / (lr + ld)
        z_cross = z_src + s_abs / (x + s_abs) * D if x + s_abs > 0 else None
        r = 0.0j
        if z_cross is not None and z0 <= z_cross <= z1:
            sin_t = s_abs / math.hypot(s_abs, z_cross - z_src)
            r = fresnel_r(sin_t, delta, beta)
        direct = 1.0
        if x < 0.0:                              # straight path dips under the edge
            zc = z_src + s_abs / (s_abs - x) * D
            if zc <= z1:
                direct = 0.0
        return direct, r, dl, ld

    quad = _source_quadrature(shape, size, n_src)
    intensity = [0.0] * len(xs)
    cross = [0.0j] * len(xs)
    i_ref = 0.0
    for s_off, w in quad:
        s_abs = height + s_off
        d_ref, r_ref, dl_ref, ld_ref = paths(x_ref, s_abs)
        u_ref = d_ref + r_ref * cmath.exp(1j * k * dl_ref)
        i_ref += w * abs(u_ref) ** 2
        for i, x in enumerate(xs):
            d_x, r_x, dl_x, ld_x = paths(x, s_abs)
            u_x = d_x + r_x * cmath.exp(1j * k * dl_x)
            intensity[i] += w * abs(u_x) ** 2
            # e^{ik(Ld(x)-Ld(ref))} via the stable product form
            dld = (x - x_ref) * (x + x_ref - 2.0 * s_abs) / (ld_x + ld_ref)
            cross[i] += w * u_x * u_ref.conjugate() * cmath.exp(1j * k * dld)
    mu = [abs(c) / math.sqrt(ii * i_ref) if ii > 0 and i_ref > 0 else 0.0
          for c, ii in zip(cross, intensity)]
    return {
        "intensity": intensity,
        "mu": mu,
        "fringe_dx": lam * D / (2.0 * height),
        "x_overlap": height * D / (z0 - z_src) - height if z0 > z_src else None,
    }


def fringe_spacing(xs, intensity):
    """Mean spacing of interference maxima (None when fewer than two peaks)."""
    mean = sum(intensity) / len(intensity)
    peaks = [xs[i] for i in range(1, len(xs) - 1)
             if intensity[i] > intensity[i - 1] and intensity[i] >= intensity[i + 1]
             and intensity[i] > mean]
    if len(peaks) < 2:
        return None
    return (peaks[-1] - peaks[0]) / (len(peaks) - 1)


def rms_diff(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def pearson(a, b) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)

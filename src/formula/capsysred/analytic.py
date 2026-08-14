"""Closed-form / deterministic references the Monte-Carlo maps are checked against.

Display-precision float math (no random sampling). Optical-path differences are
formed through (L1^2-L2^2)/(L1+L2) products, never by subtracting metre-scale
lengths, so float64 keeps the fringe phases to ~1e-15 rad.
"""

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
    if shape in ("gaussian", "grid"):
        return math.exp(-0.5 * (k * size * delta_x / dist) ** 2)
    u = k * size * abs(delta_x) / dist          # disk of radius `size`
    return 1.0 if u < 1e-12 else abs(2.0 * bessel_j1(u) / u)


def rms_diff(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))

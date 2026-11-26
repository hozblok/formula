"""Physics of Stage 16: the reflection amplitude of the wall, the entrance spot and the Fresnel kernels."""

import math

import numpy as np
from scipy.special import fresnel


def fresnel_r(theta, delta, beta):
    """Plane-wave amplitude (theta - q)/(theta + q), q = sqrt(theta^2 - 2delta + 2i beta), Im q >= 0."""
    theta = np.asarray(theta, dtype=float)
    q = np.sqrt(theta.astype(complex) ** 2 - 2.0 * delta + 2j * beta)
    q = np.where(q.imag < 0.0, -q, q)
    return (theta - q) / (theta + q)


def spot_axis(spot, k, x, axis):
    """One axis of the coherent entrance spot: gaussian of intensity FWHM `fwhm` at its waist, unit power,
    tilt exp(i k t x), waist at z = focus: exp(i k (x - c)^2 / 2q), q = -focus - i z_R (carrier e^{+ikz}),
    z_R = 2 k s^2, s = FWHM / sqrt(8 ln 2); c is the beam centre on the entrance plane."""
    s = spot["fwhm"] / math.sqrt(8.0 * math.log(2.0))
    c, t = spot["center"][axis], spot["tilt"][axis]
    if not spot.get("focus"):
        return (2.0 * math.pi * s * s) ** -0.25 * np.exp(-(x - c) ** 2 / (4.0 * s * s)) * np.exp(1j * k * t * x)
    zr = 2.0 * k * s * s
    q = complex(-spot["focus"], -zr)
    return ((2.0 * math.pi * s * s) ** -0.25 * np.sqrt(-1j * zr / q)
            * np.exp(1j * k * (x - c) ** 2 / (2.0 * q)) * np.exp(1j * k * t * x))


def spot_profile(spot, k, X, Y):
    """Coherent entrance spot on (X, Y): unit power; the waist form of the former providers when focus is 0."""
    if not spot.get("focus"):
        s = spot["fwhm"] / math.sqrt(8.0 * math.log(2.0))
        (cx, cy), (tx, ty) = spot["center"], spot["tilt"]
        return (np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (4.0 * s * s)) * np.exp(1j * k * (tx * X + ty * Y))
                / math.sqrt(2.0 * math.pi * s * s))
    return spot_axis(spot, k, X, 0) * spot_axis(spot, k, Y, 1)


def cell_kernel(edges_lo, edges_hi, xs, k, d):
    """int_lo^hi exp(ik (P - x)^2 / 2d) dP for every (cell, x): Fresnel integrals."""
    s = math.sqrt(k / (math.pi * d))
    s_hi, c_hi = fresnel((edges_hi[:, None] - xs[None, :]) * s)
    s_lo, c_lo = fresnel((edges_lo[:, None] - xs[None, :]) * s)
    return math.sqrt(math.pi * d / k) * ((c_hi - c_lo) + 1j * (s_hi - s_lo))


def point_kernel(pts, xs, k, d):
    """exp(ik (P - x)^2 / 2d) for every (point, x)."""
    return np.exp(1j * k * (pts[:, None] - xs[None, :]) ** 2 / (2.0 * d))


def _f1(u, k, d):
    """int_0^u exp(ik t^2/2d) dt."""
    s = math.sqrt(k / (math.pi * d))
    si, ci = fresnel(u * s)
    return math.sqrt(math.pi * d / k) * (ci + 1j * si)


def _f2(u, k, d):
    """int_0^u F1(t) dt = u F1(u) - (d/ik)(exp(ik u^2/2d) - 1)."""
    return u * _f1(u, k, d) - (d / (1j * k)) * (np.exp(1j * k * u * u / (2.0 * d)) - 1.0)


def point_cell_kernel(pts, xs, h, k, d):
    """Mean over the exit cell [x - h/2, x + h/2] of exp(ik (P - x')^2 / 2d): exact in x',
    so a fast chirp (far reference point) no longer aliases on the exit lattice."""
    lo, hi = xs[None, :] - h / 2.0, xs[None, :] + h / 2.0
    return (_f1(pts[:, None] - lo, k, d) - _f1(pts[:, None] - hi, k, d)) / h


def cell_cell_kernel(edges_lo, edges_hi, xs, h, k, d):
    """(1/h) int_cell_P int_cell_x exp(ik (P - x')^2 / 2d) dx' dP via the second antiderivative."""
    a, b = edges_lo[:, None], edges_hi[:, None]
    c, e = xs[None, :] - h / 2.0, xs[None, :] + h / 2.0
    return (_f2(b - c, k, d) - _f2(b - e, k, d) - _f2(a - c, k, d) + _f2(a - e, k, d)) / h


def point_pick_kernel(pts, xs, h):
    """Exit plane (d = 0): the lattice cell holding each point, weight 1/h."""
    lo, hi = xs[None, :] - h / 2.0, xs[None, :] + h / 2.0
    return ((pts[:, None] >= lo) & (pts[:, None] < hi)).astype(float) / h


def cell_overlap_kernel(edges_lo, edges_hi, xs, h):
    """Exit plane (d = 0): overlap length of a pixel cell with each lattice cell, over h."""
    lo = np.maximum(edges_lo[:, None], xs[None, :] - h / 2.0)
    hi = np.minimum(edges_hi[:, None], xs[None, :] + h / 2.0)
    return np.maximum(hi - lo, 0.0) / h

"""Leading semiclassical Lagrangian reconstruction with a finite Gaussian kernel.

The supplied tangent matrices and Maslov indices define the ray manifold.
This is not an exact boundary propagator or a sharp-aperture diffraction solver.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _cell_rule(order):
    return np.polynomial.legendre.leggauss(order)


def canonical_prefactor(Q, P, maslov, gamma):
    """Return the stable two-dimensional prefactor and its nonsingular mask."""
    Q, P = np.asarray(Q, float), np.asarray(P, float)
    if Q.shape != P.shape or Q.ndim != 3 or Q.shape[1:] != (2, 2):
        raise ValueError("Q and P must have shape (N, 2, 2)")
    maslov = np.broadcast_to(np.asarray(maslov, float), Q.shape[:1])
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be positive and finite")
    det_q = np.linalg.det(Q)
    matrix = gamma * Q + 1j * P
    determinant = matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
    valid = (np.isfinite(Q).all(axis=(1, 2)) & np.isfinite(P).all(axis=(1, 2))
             & np.isfinite(maslov) & (maslov == np.rint(maslov))
             & (det_q != 0) & np.isfinite(determinant) & (np.abs(determinant) > 0))
    result = np.zeros(len(Q), complex)
    # For real symmetric P Q^-1, the two eigenvalue arguments sum inside (-pi, pi).
    result[valid] = (np.sqrt(np.sign(det_q[valid]) * determinant[valid])
                     * np.exp(-0.5j * np.pi * maslov[valid]))
    return result, valid


def gaussian_cell_average(offset, direction, k, width, cell_width, order=12):
    """Coherent mean of exp(i k u t - t²/(2 width²)) over a receiver cell."""
    if (not np.isfinite([k, width, cell_width]).all() or k <= 0 or width <= 0
            or cell_width < 0 or isinstance(order, bool) or int(order) != order or order < 2):
        raise ValueError("invalid Gaussian cell parameters")
    offset, direction = np.broadcast_arrays(np.asarray(offset, float), np.asarray(direction, float))
    if not np.isfinite(offset).all() or not np.isfinite(direction).all():
        raise ValueError("cell offsets and directions must be finite")
    if cell_width == 0:
        return np.exp(1j * k * direction * offset - 0.5 * (offset / width)**2)
    nodes, weights = _cell_rule(int(order))
    result = np.zeros(offset.shape, complex)
    for node, weight in zip(nodes, weights):
        t = offset + 0.5 * cell_width * node
        result += (0.5 * weight) * np.exp(1j * k * direction * t - 0.5 * (t / width)**2)
    return result


def _axis(values, name):
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain at least two finite grid centers")
    steps = np.diff(values)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-10, atol=abs(steps[0])*1e-10):
        raise ValueError(f"{name} grid must be uniformly increasing")
    return values, float(steps[0])


def reconstruct_field(points, directions, phase, fresnel, Q, P, maslov, *,
                      area_weights, source_amplitude, k, width, x, y,
                      cell_width, cutoff=5.0, batch_size=4096, reference=(0.0, 0.0)):
    """Sum a fixed-source ray manifold into coherent rectangular receiver cells.

    Q=dX/dq and P=du/dq use entrance coordinates q, with paraxial flux measure dq.
    Source amplitude, entrance area and the Fresnel product remain separate.
    """
    points, directions = np.asarray(points, float), np.asarray(directions, float)
    if points.ndim != 2 or points.shape[1] != 2 or directions.shape != points.shape:
        raise ValueError("points and directions must have shape (N, 2)")
    n = len(points)
    phase = np.broadcast_to(np.asarray(phase, float), (n,))
    fresnel = np.broadcast_to(np.asarray(fresnel, complex), (n,))
    area = np.broadcast_to(np.asarray(area_weights, float), (n,))
    amplitude = np.broadcast_to(np.asarray(source_amplitude, complex), (n,))
    Q, P = np.asarray(Q, float), np.asarray(P, float)
    if Q.shape != (n, 2, 2) or P.shape != Q.shape:
        raise ValueError("tangent matrices must match the rays")
    if not all(np.isfinite(v).all() for v in (points, directions, phase, fresnel, area, amplitude)):
        raise ValueError("ray coordinates, phases, amplitudes and weights must be finite")
    if np.any(area < 0) or np.any(np.sum(directions**2, axis=1) >= 1):
        raise ValueError("area weights must be nonnegative and directions forward")
    if (not np.isfinite([k, width, cell_width, cutoff]).all() or k <= 0 or width <= 0
            or cell_width < 0 or cutoff < 3 or isinstance(batch_size, bool)
            or int(batch_size) != batch_size or batch_size < 1):
        raise ValueError("invalid reconstruction parameters")
    x, dx = _axis(x, "x")
    y, dy = _axis(y, "y")
    gamma = 1.0 / (k * width**2)
    prefactor, valid = canonical_prefactor(Q, P, maslov, gamma)
    coefficient = (k / (2 * np.pi)) * area * amplitude * prefactor * fresnel * np.exp(1j * phase)
    valid &= coefficient != 0
    half_phase = float(k * cell_width * np.abs(directions).max(initial=0) / 2)
    order = max(12, 4*math.ceil((0.75*half_phase + cell_width/width + 12)/4))
    if order > 256:
        raise ValueError("receiver cell phase requires more than 256 quadrature nodes")
    rx = math.ceil((cutoff * width + cell_width / 2) / dx) + 1
    ry = math.ceil((cutoff * width + cell_width / 2) / dy) + 1
    sx, sy = np.arange(-rx, rx + 1), np.arange(-ry, ry + 1)
    field = np.zeros((len(y), len(x)), complex)
    self_intensity = np.zeros(field.shape, float)
    self_cross = np.zeros(field.shape, complex)
    reference = np.asarray(reference, float)
    if reference.shape != (2,) or not np.isfinite(reference).all():
        raise ValueError("reference must contain two finite coordinates")
    ref_index = (int(np.argmin(abs(y-reference[1]))), int(np.argmin(abs(x-reference[0]))))
    ref_xy = np.array([x[ref_index[1]], y[ref_index[0]]])
    used, skipped = 0, 0
    ids = np.flatnonzero(valid)
    for start in range(0, len(ids), int(batch_size)):
        batch = ids[start:start + int(batch_size)]
        batch_phase = k*cell_width*np.abs(directions[batch]).max(initial=0)/2
        batch_order = max(12, 4*math.ceil((0.75*batch_phase + cell_width/width + 12)/4))
        center = points[batch]
        ix = (np.broadcast_to(np.arange(len(x)), (len(batch), len(x))) if len(x) <= len(sx)
              else np.floor((center[:, 0] - x[0]) / dx).astype(np.int64)[:, None] + sx)
        iy = (np.broadcast_to(np.arange(len(y)), (len(batch), len(y))) if len(y) <= len(sy)
              else np.floor((center[:, 1] - y[0]) / dy).astype(np.int64)[:, None] + sy)
        tx = x[np.clip(ix, 0, len(x)-1)] - center[:, 0, None]
        ty = y[np.clip(iy, 0, len(y)-1)] - center[:, 1, None]
        mx = ((ix >= 0) & (ix < len(x)) & (np.abs(tx) <= cutoff * width + cell_width / 2))
        my = ((iy >= 0) & (iy < len(y)) & (np.abs(ty) <= cutoff * width + cell_width / 2))
        inside = mx.any(axis=1) & my.any(axis=1)
        used += int(inside.sum())
        skipped += int((~inside).sum())
        gx = gaussian_cell_average(tx, directions[batch, 0, None], k, width, cell_width, batch_order)
        gy = gaussian_cell_average(ty, directions[batch, 1, None], k, width, cell_width, batch_order)
        ref_offsets = ref_xy-center
        ref_keep = np.all(np.abs(ref_offsets) <= cutoff*width+cell_width/2, axis=1)
        ref_kernel = gaussian_cell_average(ref_offsets, directions[batch], k, width, cell_width, batch_order)
        ref_value = coefficient[batch]*ref_kernel.prod(axis=1)*ref_keep
        keep = my[:, :, None] & mx[:, None, :]
        indices = iy[:, :, None] * len(x) + ix[:, None, :]
        values = coefficient[batch, None, None] * gy[:, :, None] * gx[:, None, :]
        np.add.at(field.ravel(), indices[keep], values[keep])
        np.add.at(self_intensity.ravel(), indices[keep], np.abs(values[keep])**2)
        cross = values*ref_value.conj()[:, None, None]
        np.add.at(self_cross.ravel(), indices[keep], cross[keep])
    diagnostics = {
        "model": "leading semiclassical Lagrangian Gaussian integral; paraxial entrance-area flux",
        "rays": n, "valid_nonzero_rays": int(valid.sum()), "contributing_rays": used,
        "outside_screen_kernel_support": skipped,
        "invalid_or_zero_weight_rays": int((~valid).sum()),
        "included_entrance_area_m2": float(area[valid].sum()),
        "excluded_entrance_area_m2": float(area[~valid].sum()),
        "width_m": float(width), "gamma_per_m": float(gamma), "cutoff_widths": float(cutoff),
        "kernel_at_cutoff": float(np.exp(-cutoff**2 / 2)),
        "cell_width_m": float(cell_width), "cell_rule": "coherent mean; separable Gauss-Legendre",
        "cell_quadrature_order": order, "cell_max_half_phase_rad": half_phase,
        "reference_requested_m": reference.tolist(), "reference_actual_m": ref_xy.tolist(),
        "self_terms": "sum of individual ray diagonal terms; use full sampling prefix N for U-statistic normalization",
        "limitations": [
            "Requires accurate single-branch Q, P, Maslov history and source amplitude supplied by caller.",
            "Exact zero det(Q) has no supplied limiting orientation and is excluded; near-zero determinants are finite.",
            "Finite Gaussian width smooths aperture edges; width and ray-density convergence are separate checks.",
            "A local kernel cutoff is not a relative field-error certificate in interference minima.",
        ],
    }
    return dict(field=field, self_intensity=self_intensity, self_cross=self_cross,
                ref_index=ref_index, metadata=diagnostics)

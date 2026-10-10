"""Scalar action fits and exact pair phases for the experimental B5 estimator."""

from dataclasses import dataclass

import numpy as np


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def _points(value, name, matrix=False):
    out = _real(value, name)
    if out.ndim < 1 or out.shape[-1] != 2 or (matrix and out.ndim != 2):
        raise ValueError(f"{name} must have shape {'(N, 2)' if matrix else '(..., 2)'}")
    if out.size == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _powers(degree):
    return tuple((px, total - px) for total in range(1, degree + 1)
                 for px in range(total, -1, -1))


def _values(q, powers):
    return np.stack([q[..., 0] ** px * q[..., 1] ** py for px, py in powers], axis=-1)


def _derivatives(q, powers):
    columns = []
    for px, py in powers:
        gx = px * q[..., 0] ** (px - 1) * q[..., 1] ** py if px else np.zeros(q.shape[:-1])
        gy = py * q[..., 0] ** px * q[..., 1] ** (py - 1) if py else np.zeros(q.shape[:-1])
        columns.append(np.stack((gx, gy), axis=-1))
    return np.stack(columns, axis=-1)


@dataclass(frozen=True)
class PhaseFit:
    """Real polynomial action in input coordinates; gradients use the input units."""

    degree: int
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    anchor_point: np.ndarray
    anchor_phase: float
    diagnostics: dict

    def _q(self, points):
        return (_points(points, "points") - self.center) / self.scale

    def _raw(self, points):
        return _values(self._q(points), _powers(self.degree)) @ self.coefficients

    def phase(self, points):
        return self.anchor_phase + (self._raw(points) - self._raw(self.anchor_point))

    def gradient(self, points):
        return (_derivatives(self._q(points), _powers(self.degree)) @ self.coefficients) / self.scale

    def pair_multiplier(self, left, right=None):
        """exp(i(phi(left)-phi(right))); shape (len(left), len(right))."""
        left = _points(left, "left", matrix=True)
        right = left if right is None else _points(right, "right", matrix=True)
        return np.exp(1j * (self._raw(left)[:, None] - self._raw(right)[None, :]))

    def residual_multiplier(self, left, right=None, *, baseline_gradient):
        """Exact pair phase divided by the caller's exp(i chord dot baseline kick)."""
        left = _points(left, "left", matrix=True)
        right = left if right is None else _points(right, "right", matrix=True)
        chord = left[:, None, :] - right[None, :, :]
        midpoint = (left[:, None, :] + right[None, :, :]) / 2.0
        kick = baseline_gradient(midpoint) if callable(baseline_gradient) else baseline_gradient
        kick = _real(kick, "baseline_gradient")
        try:
            kick = np.broadcast_to(kick, chord.shape)
        except ValueError as exc:
            raise ValueError("baseline_gradient must broadcast to (N, M, 2)") from exc
        delta = self._raw(left)[:, None] - self._raw(right)[None, :]
        return np.exp(1j * (delta - np.sum(chord * kick, axis=-1)))


def fit_phase_gradient(points, gradients, *, degree=3, anchor_point=None, anchor_phase=0.0):
    """Fit one scalar polynomial to both gradient components in physical units.

    Coordinates and design columns are scaled before least squares. Rank-deficient
    fits are returned with diagnostics; the caller must decide whether to accept them.
    """
    if not isinstance(degree, (int, np.integer)) or isinstance(degree, bool) or not 2 <= degree <= 5:
        raise ValueError("degree must be an integer from 2 through 5")
    points = _points(points, "points", matrix=True)
    gradients = _points(gradients, "gradients", matrix=True)
    if points.shape != gradients.shape:
        raise ValueError("gradients must have the same shape as points")
    center = points.min(axis=0) / 2.0 + points.max(axis=0) / 2.0
    scale = np.max(np.abs(points - center), axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    anchor = center.copy() if anchor_point is None else _points(anchor_point, "anchor_point")
    if anchor.shape != (2,):
        raise ValueError("anchor_point must have shape (2,)")
    phase0 = _real(anchor_phase, "anchor_phase")
    if phase0.ndim != 0:
        raise ValueError("anchor_phase must be a scalar")

    powers = _powers(int(degree))
    design = _derivatives((points - center) / scale, powers) / scale[None, :, None]
    design = design.reshape(-1, len(powers))
    column_norm = np.linalg.norm(design, axis=0)
    column_norm = np.where(column_norm > 0.0, column_norm, 1.0)
    scaled = design / column_norm
    solution, _, rank, singular = np.linalg.lstsq(scaled, gradients.reshape(-1), rcond=None)
    coefficients = solution / column_norm
    residual = design @ coefficients - gradients.reshape(-1)
    norm = np.linalg.norm(gradients)
    error = np.linalg.norm(residual)
    diagnostics = {
        "n_points": len(points), "n_coefficients": len(powers), "rank": int(rank),
        "full_rank": bool(rank == len(powers)),
        "condition": float(singular[0] / singular[-1]) if rank == len(powers) else float("inf"),
        "gradient_rms": float(error / np.sqrt(gradients.size)),
        "gradient_max_abs": float(np.max(np.abs(residual))),
        "gradient_relative_l2": float(error / norm) if norm > 0.0 else 0.0,
    }
    return PhaseFit(int(degree), center.copy(), scale.copy(), coefficients,
                    anchor.copy(), float(phase0), diagnostics)

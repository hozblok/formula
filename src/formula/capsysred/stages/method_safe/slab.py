"""Reduced 2D B5 operator: odd unfolding of a bent Dirichlet slab."""

import math
import operator

import numpy as np


class UnfoldedSlab:
    """Propagate fields or PSD factors along axis 0; curvature is signed c''.

    The physical frame equation is i dE/dz = [-d_x^2/(2k) + k c'' x] E.
    This backend describes two hard walls, not a circular 3D capillary.
    """

    def __init__(self, k, half_width, length, curvature, dz, n_inner):
        self.k = float(k)
        self.half_width = float(half_width)
        self.length = float(length)
        self.curvature = float(curvature)
        self.dz = float(dz)
        self.n_inner = operator.index(n_inner)
        values = (self.k, self.half_width, self.length,
                  self.curvature, self.dz)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("slab parameters must be finite")
        if self.k <= 0 or self.half_width <= 0 or self.dz <= 0:
            raise ValueError("k, half_width and dz must be positive")
        if self.length < 0 or self.n_inner < 1 or isinstance(n_inner, bool):
            raise ValueError("length must be nonnegative and n_inner positive")
        self.width = 2 * self.half_width
        self.period = 2 * self.width
        self.dx = self.width / (self.n_inner + 1)
        self.size = 2 * (self.n_inner + 1)
        self.s = np.arange(self.size) * self.dx
        self.x = -self.half_width + np.arange(1, self.n_inner + 1) * self.dx
        self.steps = int(math.ceil(self.length / self.dz))
        self.step = self.length / self.steps if self.steps else 0.0
        p = 2 * np.pi * np.fft.fftfreq(self.size, self.dx)
        self.drift_phase = np.exp(-0.5j * p * p * self.step / self.k)
        self.half_kick = np.exp(-0.5j * self.step * self.potential(self.s))

    def folded_coordinate(self, s):
        s = np.remainder(np.asarray(s, dtype=float), self.period)
        return self.width - np.abs(s - self.width) - self.half_width

    def potential(self, s):
        return self.k * self.curvature * self.folded_coordinate(s)

    def potential_gradient(self, s):
        s = np.remainder(np.asarray(s, dtype=float), self.period)
        # At a fold this diagnostic uses the right-hand derivative.
        return self.k * self.curvature * np.where(s < self.width, 1.0, -1.0)

    def chord_residual(self, left, right, step=None):
        """Exact pair phase minus affine ray phase on the chosen cover."""
        step = self.step if step is None else float(step)
        left, right = np.broadcast_arrays(np.asarray(left, dtype=float),
                                         np.asarray(right, dtype=float))
        midpoint = (left + right) / 2
        affine = (left - right) * self.potential_gradient(midpoint)
        return -step * (self.potential(left) - self.potential(right) - affine)

    @staticmethod
    def _values(values, size):
        values = np.asarray(values, dtype=np.complex128)
        if values.ndim not in (1, 2) or values.shape[0] != size:
            raise ValueError(f"expected shape ({size},) or ({size}, rank)")
        if not np.all(np.isfinite(values)):
            raise ValueError("fields must be finite")
        return values

    def unfold(self, fields):
        fields = self._values(fields, self.n_inner)
        result = np.zeros((self.size,) + fields.shape[1:], dtype=np.complex128)
        result[1:self.n_inner + 1] = fields
        result[self.n_inner + 2:] = -fields[::-1]
        return result

    def propagate_unfolded(self, factors):
        """Apply the complete pair operator through its field factorization."""
        factors = self._values(factors, self.size).copy()
        shape = (self.size,) + (1,) * (factors.ndim - 1)
        kick = self.half_kick.reshape(shape)
        drift = self.drift_phase.reshape(shape)
        for _ in range(self.steps):
            factors *= kick
            factors = np.fft.ifft(drift * np.fft.fft(factors, axis=0), axis=0)
            factors *= kick
        return factors

    def propagate(self, fields):
        return self.propagate_unfolded(self.unfold(fields))[1:self.n_inner + 1].copy()

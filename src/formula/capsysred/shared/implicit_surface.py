"""Formula-backed surface values and world-coordinate gradients for numbers and jets."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...backend import MP_TYPES
from ...formula import Formula, NumberContext
from .coordinates import surface_coordinates
from .formulas import cached_formula
from .jets import _Jet
from .nums import text


class Surface:
    """Smooth implicit surface; coordinate_scale converts metres to expression units."""

    def __init__(self, expression: str | Formula, precision: int = 64,
                 coordinate_scale=1.0) -> None:
        """Reuse Formula objects directly; precision applies only to expression text."""
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive and finite")
        self.coordinate_scale = coordinate_scale
        self.formula = cached_formula(expression, precision) if isinstance(expression, str) else expression
        if isinstance(coordinate_scale, MP_TYPES) and coordinate_scale.precision != self.precision:
            raise ValueError("surface and coordinate_scale precision must agree")
        self.numbers = NumberContext(self.precision)
        if set(self.formula.variables()) - set("xyz"):
            raise ValueError("surface variables must be x, y, z")

    @property
    def expression(self) -> str:
        return self.formula.expression

    @property
    def precision(self) -> int:
        """Effective precision of the stored Formula."""
        return self.formula.precision

    def _coordinates(
        self, point: Sequence,
    ) -> tuple:
        return tuple(v * self.coordinate_scale
                     for v in surface_coordinates(point, self.precision))

    def value(self, point: Sequence):
        """Evaluate the expression at a point given in world metres."""
        coordinates = self._coordinates(point)
        if not isinstance(coordinates[0], _Jet):
            values = {axis: text(v, self.precision) for axis, v in zip("xyz", coordinates)}
            return self.numbers(self.formula.get(values))
        alg = coordinates[0].alg
        coefficients, support = self.formula._evaluate_jet(
            alg.order, dict(zip("xyz", (v.c for v in coordinates))),
            dict(zip("xyz", (v.support for v in coordinates))))
        return _Jet(alg, coefficients, support)

    def gradient(
        self, point: Sequence,
    ) -> tuple:
        """Gradient with respect to world metres, including the coordinate-scale factor."""
        coordinates = self._coordinates(point)
        if not isinstance(coordinates[0], _Jet):
            values = {axis: text(v, self.precision) for axis, v in zip("xyz", coordinates)}
            axes = self.formula.variables()
            return tuple(self.numbers(self.formula.get_derivative(axis, values))
                         * self.coordinate_scale if axis in axes else self.numbers(0)
                         for axis in "xyz")
        alg, n = coordinates[0].alg, len(coordinates[0].c)
        gradient = []
        for axis in range(3):
            values, supports = {}, {}
            for j, name in enumerate("xyz"):
                values[name] = coordinates[j].c + ["0"] * n
                supports[name] = coordinates[j].support
                if axis == j:
                    values[name][n] = text(self.coordinate_scale, self.precision)
                    supports[name] |= 1 << n
            coefficients, support = self.formula._evaluate_jet(alg.order + 1, values, supports)
            gradient.append(_Jet(alg, coefficients[n:], (support >> n) | 1))
        return tuple(gradient)

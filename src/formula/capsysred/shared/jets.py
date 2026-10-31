"""Composable hyper-dual scalars backed by the Formula evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ...formula import Number
from .formulas import cached_formula


class _Algebra:
    order: int
    precision: int

    def __init__(self, order: int, precision: int) -> None:
        self.order, self.precision = order, precision

    def constant(self, value: str | int | float | Number) -> _Jet:
        return _Jet(self, [str(value)] + ["0"] * ((1 << self.order) - 1), 1)

    def variable(self, value: str | int | float | Number, seeds: Sequence[int], axis: int) -> _Jet:
        result = self.constant(value)
        for i, seed in enumerate(seeds):
            if seed == axis:
                result.c[1 << i] = "1"
                result.support |= 1 << (1 << i)
        return result

    def evaluate(self, expression: str, **values: _Jet | str | int | float | Number) -> _Jet:
        jets = {key: value if isinstance(value, _Jet) else self.constant(value)
                for key, value in values.items()}
        if any(jet.alg is not self for jet in jets.values()):
            raise ValueError("jets must belong to the same algebra")
        coefficients, support = cached_formula(expression, self.precision)._evaluate_jet(
            self.order, {key: jet.c for key, jet in jets.items()},
            {key: jet.support for key, jet in jets.items()})
        return _Jet(self, coefficients, support)


class _Jet:
    alg: _Algebra
    c: list[str]
    support: int

    def __init__(self, algebra: _Algebra, coefficients: Iterable[str], support: int) -> None:
        self.alg, self.c, self.support = algebra, list(coefficients), support

    def __add__(self, other: _Jet | str | int | float | Number) -> _Jet:
        return self.alg.evaluate("a+b", a=self, b=other)

    __radd__ = __add__

    def __sub__(self, other: _Jet | str | int | float | Number) -> _Jet:
        return self.alg.evaluate("a-b", a=self, b=other)

    def __rsub__(self, other: _Jet | str | int | float | Number) -> _Jet:
        return self.alg.evaluate("a-b", a=other, b=self)

    def __mul__(self, other: _Jet | str | int | float | Number) -> _Jet:
        return self.alg.evaluate("a*b", a=self, b=other)

    __rmul__ = __mul__

    def __truediv__(self, other: _Jet | str | int | float | Number) -> _Jet:
        return self.alg.evaluate("a/b", a=self, b=other)

    def __rtruediv__(self, other: _Jet | str | int | float | Number) -> _Jet:
        return self.alg.evaluate("a/b", a=other, b=self)

    def sqrt(self) -> _Jet:
        return self.alg.evaluate("sqrt(a)", a=self)

    @property
    def value(self) -> float:
        return float(self.c[0])

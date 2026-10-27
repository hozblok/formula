"""Experimental derivatives of one fixed reflection history; no SAFE amplitude correction.

Coordinates are metres. Formula's hyper-dual evaluator carries the mixed derivatives;
several seed sets recover the complete two-parameter Taylor polynomial of degree 1..4.

Example (after rebuilding the native extension)::

    wall = Surface("x-1")
    jet = transport_ray([.2, .1], [0., 0., -2.], 0., 10.,
                        [Hit(wall, [1., .5, 8.])], order=4)
    jet.Q                         # ((-6.0, 0.0), (0.0, 6.0))
    jet.predict_position([.001, 0.])

Inputs select the local branch; they do not prove that nearby rays hit the same walls.
No amplitude, Fresnel factor, caustic phase or SAFE a1 is computed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

from ...formula import Number
from ..shared.formulas import cached_formula
from ..shared.nums import vadd, vdot, vnorm, vscale, vsub, vunit
from ..shared.types import _GRAZING_SIN_MIN, _RAY_POSITION_TOL_M
from ..shared.validation import finite_vector


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


class Surface:
    """Smooth implicit surface; coordinate_scale converts metres to expression units."""

    def __init__(self, expression, precision=64, coordinate_scale=1.0):
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive and finite")
        self.expression, self.precision = expression, precision
        self.coordinate_scale = coordinate_scale
        self.formula = cached_formula(expression, precision)
        if set(self.formula.variables()) - set("xyz"):
            raise ValueError("surface variables must be x, y, z")

    def value(self, point):
        alg = point[0].alg
        return alg.evaluate(self.expression, **dict(zip("xyz", vscale(point, self.coordinate_scale))))

    def gradient(self, point):
        alg, n = point[0].alg, len(point[0].c)
        scaled = vscale(point, self.coordinate_scale)
        gradient = []
        for axis in range(3):
            values, supports = {}, {}
            for j, name in enumerate("xyz"):
                values[name] = scaled[j].c + ["0"] * n
                supports[name] = scaled[j].support
                if axis == j:
                    values[name][n] = str(self.coordinate_scale)
                    supports[name] |= 1 << n
            c, support = self.formula._evaluate_jet(alg.order + 1, values, supports)
            gradient.append(_Jet(alg, c[n:], (support >> n) | 1))
        return gradient


@dataclass(frozen=True)
class Hit:
    """Selected smooth surface and nominal point from the ordinary ray trace."""

    surface: Surface
    point: object


@dataclass(frozen=True)
class RayJet:
    """Raw derivatives in tuples: coordinate rows, then powers; path starts at source."""

    powers: tuple[tuple[int, int], ...]
    position: tuple[tuple[float, ...], ...]
    direction: tuple[tuple[float, ...], ...]
    path_length: tuple[float, ...]

    @property
    def Q(self) -> tuple[tuple[float, ...], ...]:
        columns = (self.powers.index((1, 0)), self.powers.index((0, 1)))
        return tuple(tuple(row[i] for i in columns) for row in self.position[:2])

    @property
    def P(self) -> tuple[tuple[float, ...], ...]:
        columns = (self.powers.index((1, 0)), self.powers.index((0, 1)))
        return tuple(tuple(row[i] for i in columns) for row in self.direction[:2])

    def _predict(self, rows: Sequence[Sequence[float]], delta: Iterable[float]) -> tuple[float, ...]:
        dx, dy = finite_vector(delta, 2, "delta")
        weights = [dx ** i * dy ** j / (math.factorial(i) * math.factorial(j)) for i, j in self.powers]
        return tuple(math.fsum(value * weight for value, weight in zip(row, weights, strict=True))
                     for row in rows)

    def predict_position(self, delta: Iterable[float]) -> tuple[float, ...]:
        return self._predict(self.position, delta)

    def predict_direction(self, delta: Iterable[float]) -> tuple[float, ...]:
        return self._predict(self.direction, delta)

    def predict_path_length(self, delta: Iterable[float]) -> float:
        return self._predict((self.path_length,), delta)[0]


def _implicit_hit(position, direction, hit):
    alg, surface = position[0].alg, hit.surface
    anchor = finite_vector(hit.point, 3, "hit")
    r0, u0 = tuple(v.value for v in position), tuple(v.value for v in direction)
    t0 = vdot(vsub(anchor, r0), u0)
    if t0 <= _RAY_POSITION_TOL_M or math.dist(vadd(r0, vscale(u0, t0)), anchor) > _RAY_POSITION_TOL_M:
        raise ValueError("hit must lie forward on the nominal ray beyond the position tolerance")
    base_r = [alg.constant(v.c[0]) for v in position]
    base_u = [alg.constant(v.c[0]) for v in direction]
    time = alg.constant(t0)
    # Polish only the selected root, then lift it in the truncated algebra.
    for _ in range(8):
        point = vadd(base_r, vscale(base_u, time))
        gradient = surface.gradient(point)
        normal_length = vnorm(gradient)
        ft = vdot(gradient, base_u)
        if not math.isfinite(normal_length.value) or normal_length.value <= 0:
            raise ValueError("surface must have a finite nonzero gradient")
        if abs(ft.value / normal_length.value) <= _GRAZING_SIN_MIN:
            raise ValueError("grazing hit has no regular implicit derivative")
        correction = surface.value(point) / ft
        if abs(correction.value) > _RAY_POSITION_TOL_M:
            raise ValueError("hit is not on the selected surface")
        time = time - correction
        if abs(correction.value) < 10.0 ** -min(30, alg.precision // 2):
            break
    else:
        raise ValueError("nominal intersection did not converge")
    # Newton doubles the first unresolved degree: 1 -> 2 -> 4 -> 8.
    for _ in range(alg.order.bit_length()):
        point = vadd(position, vscale(direction, time))
        ft = vdot(surface.gradient(point), direction)
        time = time - surface.value(point) / ft
    point = vadd(position, vscale(direction, time))
    if math.dist(tuple(v.value for v in point), anchor) > _RAY_POSITION_TOL_M:
        raise ValueError("refined intersection left the archived hit")
    normal = vunit(surface.gradient(point))
    projection = 2 * vdot(direction, normal)
    outgoing = vsub(direction, vscale(normal, projection))
    return point, outgoing, time


def transport_ray(entrance, source, z0, target_z, hits=(), order=3, precision=64):
    """One smooth archived branch, differentiated by entrance x/y; no root selection.

    This small-budget diagnostic returns float derivatives after multiprecision replay.
    It does not determine whether neighbouring rays retain the same reflection history.
    """
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 4:
        raise ValueError("order must be an integer from 1 to 4")
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 32:
        raise ValueError("precision must be an integer of at least 32 digits")
    entrance = finite_vector(entrance, 2, "entrance")
    source = finite_vector(source, 3, "source")
    if not all(math.isfinite(z) for z in (z0, target_z)) or not source[2] < z0 <= target_z:
        raise ValueError("source, entrance and target must be ordered in z")
    hits = tuple(Hit(hit.surface, finite_vector(hit.point, 3, "hit")) for hit in hits)
    for hit in hits:
        if hit.surface.precision != precision:
            raise ValueError("surface and transport precision must agree")
        if not z0 <= hit.point[2] <= target_z:
            raise ValueError("hits must lie between entrance and target")
    powers = tuple((d - j, j) for d in range(order + 1) for j in range(d + 1))
    result = {}
    for nx in range(order + 1):
        seeds = (0,) * nx + (1,) * (order - nx)
        alg = _Algebra(order, precision)
        position = [alg.variable(entrance[i], seeds, i) for i in range(2)] + [alg.constant(z0)]
        displacement = vsub(position, source)
        length = vnorm(displacement)
        direction = [v / length for v in displacement]
        for hit in hits:
            if direction[2].value <= 0:
                raise ValueError("ray must remain forward in z")
            position, direction, flight = _implicit_hit(position, direction, hit)
            length = length + flight
        if direction[2].value <= 0:
            raise ValueError("ray must remain forward in z")
        flight = (target_z - position[2]) / direction[2]
        if flight.value < -_RAY_POSITION_TOL_M:
            raise ValueError("target must lie forward of the final hit")
        position = vadd(position, vscale(direction, flight))
        length = length + flight
        outputs = (*position, *direction, length)
        for mask in range(1 << order):
            power = tuple(sum(seeds[i] == axis for i in range(order) if mask & (1 << i)) for axis in (0, 1))
            result.setdefault(power, [float(v.c[mask]) for v in outputs])
    if not all(math.isfinite(v) for coefficients in result.values() for v in coefficients):
        raise ValueError("nonfinite ray derivatives")
    values = tuple(zip(*(result[power] for power in powers), strict=True))
    return RayJet(powers, values[:3], values[3:6], values[6])

"""Experimental derivatives of one fixed reflection history; no SAFE amplitude correction.

Coordinates are metres. Formula's hyper-dual evaluator carries the mixed derivatives;
several seed sets recover the complete two-parameter Taylor polynomial of degree 1..4.

Examples::

    from formula import Number
    from formula.capsysred.stages._ray_jets import Hit, Surface, transport_ray
    from formula.capsysred.surfaces import ImplicitWall, Mirror
    from formula.capsysred.walls.wall_cylinder import CylinderWall
    from formula.capsysred.walls.wall_funnel import FunnelWall
    from formula.capsysred.walls.wall_polygon import PolygonWall
    from formula.capsysred.walls.wall_revolution import RevolutionWall
    from formula.capsysred.walls.wall_torus import TorusWall

    # Implicit surface: plane x=1.
    plane = Surface("x-1")
    plane_jet = transport_ray([.2, .1], [0., 0., -2.], 0., 10.,
                              [Hit(plane, [1., .5, 8.])], order=4)
    plane_jet.Q                   # ((-6.0, 0.0), (0.0, 6.0))
    plane_jet.predict_position([.001, 0.])

    # Straight cylinder of radius 1.
    zero, one = Number("0", 64), Number("1", 64)
    cylinder = CylinderWall((zero, zero), one)
    cylinder_jet = transport_ray([.5, 0.], [0., 0., -2.], 0., 5.,
                                 [Hit(cylinder, [1., 0., 2.])], order=4)
    cylinder_jet.Q                # ((-3.5, 0.0), (0.0, 0.5))
    cylinder_jet.predict_position([.001, 0.])

    # Torus: tube radius 1, bend radius 4, bending toward +x.
    torus = TorusWall((zero, zero), one, Number("4", 64), (one, zero), zero)
    torus_jet = transport_ray([0., 0.], [0., 0., -2.], 0., 4.,
                              [Hit(torus, [0., 0., 3.])], order=4)
    torus_jet.predict_position([.001, 0.])

    # Surface of revolution: r(z)^2 = 1 + 0.3125*z.
    revolution = RevolutionWall("revolution", (zero, zero),
                                (one, Number("0.3125", 64), zero))
    revolution_jet = transport_ray([.5, 0.], [0., 0., -2.], 0., 5.,
                                   [Hit(revolution, [1.5, 0., 4.])], order=4)
    revolution_jet.predict_position([.001, 0.])

    # Funnel: fixed axis, expanding radius 1 + z/16.
    funnel = FunnelWall((zero, zero), one, (zero, zero),
                        (Number("0.0625", 64), zero), zero)
    funnel_jet = transport_ray([.5, 0.], [0., 0., -2.], 0., 4.,
                               [Hit(funnel, [7/6, 0., 8/3])], order=4)
    funnel_jet.predict_position([.001, 0.])

    # Square bore; each archived hit fixes one face away from its edges.
    polygon = PolygonWall((zero, zero), one, 4, zero)
    polygon_jet = transport_ray([.5, .1], [0., 0., -2.], 0., 5.,
                                [Hit(polygon, [1., .2, 2.])], order=4)
    polygon_jet.predict_position([.001, 0.])  # Hit(polygon.face(0), ...) also works.

    # Existing implicit wall: its expression uses micrometres, inputs use metres.
    implicit = ImplicitWall("x-1000000", (zero, zero), one)
    implicit_jet = transport_ray([.2, .1], [0., 0., -2.], 0., 10.,
                                 [Hit(implicit, [1., .5, 8.])], order=4)
    implicit_jet.predict_position([.001, 0.])

    # Finite mirror x=0, illuminated from x>0; the hit is inside its z span.
    mirror = Mirror(zero, Number("4", 64))
    mirror_jet = transport_ray([.2, 0.], [.4, 0., -2.], 0., 4.,
                               [Hit(mirror, [0., 0., 2.])], order=4)
    mirror_jet.predict_position([.001, 0.])

Inputs select the local branch; they do not prove that nearby rays hit the same walls.
No amplitude, Fresnel factor, caustic phase or SAFE a1 is computed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

from ...backend import round_up_precision
from ...formula import Number
from ..shared.implicit_surface import Surface
from ..shared.jets import _Algebra, _Jet
from ..shared.nums import vadd, vdot, vnorm, vscale, vsub, vunit
from ..shared.types import _GRAZING_SIN_MIN, _RAY_POSITION_TOL_M
from ..shared.validation import finite_vector
from ..surfaces import ImplicitWall, Mirror
from ..walls.wall_cylinder import CylinderWall
from ..walls.wall_funnel import FunnelWall
from ..walls.wall_polygon import PolygonFace, PolygonWall
from ..walls.wall_revolution import RevolutionWall
from ..walls.wall_torus import TorusWall


@dataclass(frozen=True)
class Hit:
    """Selected smooth surface and nominal point from the ordinary ray trace."""

    surface: (Surface | CylinderWall | TorusWall | RevolutionWall | FunnelWall
              | PolygonWall | PolygonFace | ImplicitWall | Mirror)
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


def _prepare_hit(hit: Hit) -> Hit:
    """Resolve an archived wall to one smooth surface before seeding derivatives."""
    point = finite_vector(hit.point, 3, "hit")
    surface = hit.surface
    if isinstance(surface, PolygonWall):
        surface = surface.surface_at(point)
    return Hit(surface, point)


def _validate_hit(
    hit: Hit, point: Sequence[float], origin: Sequence[float], direction: Sequence[float],
) -> None:
    """Apply optional surface restrictions to the nominal incoming ray and hit."""
    validate = getattr(hit.surface, "validate", None)
    if validate is not None:
        validate(point, origin=origin, direction=direction)


def _implicit_hit(position, direction, hit):
    alg, surface = position[0].alg, hit.surface
    anchor = finite_vector(hit.point, 3, "hit")
    r0, u0 = tuple(v.value for v in position), tuple(v.value for v in direction)
    _validate_hit(hit, anchor, r0, u0)

    def gradient(point: Sequence[_Jet]) -> tuple[_Jet, ...]:
        return tuple(v if isinstance(v, _Jet) else alg.constant(v)
                     for v in surface.gradient(point))

    t0 = vdot(vsub(anchor, r0), u0)
    if t0 <= _RAY_POSITION_TOL_M or math.dist(vadd(r0, vscale(u0, t0)), anchor) > _RAY_POSITION_TOL_M:
        raise ValueError("hit must lie forward on the nominal ray beyond the position tolerance")
    base_r = [alg.constant(v.c[0]) for v in position]
    base_u = [alg.constant(v.c[0]) for v in direction]
    time = alg.constant(t0)
    # Polish only the selected root, then lift it in the truncated algebra.
    for _ in range(8):
        point = vadd(base_r, vscale(base_u, time))
        grad = gradient(point)
        normal_length = vnorm(grad)
        ft = vdot(grad, base_u)
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
    polished = vadd(base_r, vscale(base_u, time))
    _validate_hit(hit, tuple(v.value for v in polished), r0, u0)
    # Newton doubles the first unresolved degree: 1 -> 2 -> 4 -> 8.
    for _ in range(alg.order.bit_length()):
        point = vadd(position, vscale(direction, time))
        ft = vdot(gradient(point), direction)
        time = time - surface.value(point) / ft
    point = vadd(position, vscale(direction, time))
    if math.dist(tuple(v.value for v in point), anchor) > _RAY_POSITION_TOL_M:
        raise ValueError("refined intersection left the archived hit")
    normal = vunit(gradient(point))
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
    hits = tuple(_prepare_hit(hit) for hit in hits)
    for hit in hits:
        if round_up_precision(hit.surface.precision) != round_up_precision(precision):
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

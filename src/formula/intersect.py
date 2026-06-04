"""Find all intersections of a ray with an implicit surface F(x,y,z)=0.

A ray r(t)=O+t*d turns F into a single-variable g(t)=F(O+t*d); every
intersection is a real root of g on [0, t_max]. Pluggable root-finding
backends (see _roots/) locate all of them.
"""

from typing import List, Sequence, Tuple

from ._roots import get_backend
from .formula import Number, Solver

_AXES = ("x", "y", "z")


class RaySurfaceFunction:
    """g(t)=F(O+t*d) and g'(t)=grad F . d, built from a surface Solver.

    Reuses Solver.evaluate / get_derivative — no symbolic substitution.
    """

    def __init__(
        self,
        surface: Solver,
        origin: Sequence,
        direction: Sequence,
        precision: int,
    ):
        self.surface = surface
        self.precision = precision
        axes = surface.variables()
        unknown = axes - set(_AXES)
        if unknown:
            raise ValueError(f"surface variables must be in {_AXES}; got {unknown}")
        # Keep only the axes the surface actually depends on, in x,y,z order.
        self._axes = [a for a in _AXES if a in axes]
        self.origin = [self._num(origin[_AXES.index(a)]) for a in self._axes]
        d = [self._num(direction[_AXES.index(a)]) for a in self._axes]
        norm = abs(sum((c * c for c in d), self._num(0))) ** self._num("0.5")
        self.direction = [c / norm for c in d]

    def _num(self, value) -> Number:
        return value if isinstance(value, Number) else Number(value, self.precision)

    def _point(self, t: Number) -> dict:
        """Ray coordinates at parameter t, as Solver value strings."""
        return {
            a: str(o + t * dc)
            for a, o, dc in zip(self._axes, self.origin, self.direction)
        }

    def g(self, t: Number) -> Number:
        """g(t) = F(O + t*d)."""
        return Number.wrap(self.surface.evaluate(self._point(t)), self.precision)

    def gprime(self, t: Number) -> Number:
        """g'(t) = grad F . d via the chain rule."""
        point = self._point(t)
        total = self._num(0)
        for a, dc in zip(self._axes, self.direction):
            partial = Number(self.surface.get_derivative(a, point, 0), self.precision)
            total = total + partial * dc
        return total

    def point_at(self, t: Number) -> Tuple[Number, ...]:
        """The (x, y, z) point on the ray at parameter t."""
        full = {a: o + t * dc for a, o, dc in zip(self._axes, self.origin, self.direction)}
        return tuple(full.get(a, self._num(0)) for a in _AXES)


class RaySurface:
    """All intersections of a ray with an implicit surface F=0.

    Example:
        rs = RaySurface("x*x + y*y - 1", precision=64)
        ts = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10)
    """

    def __init__(
        self,
        expression: str,
        precision: int = 24,
        imaginary_unit: str = "i",
        case_insensitive: bool = False,
    ):
        self.surface = Solver(expression, precision, imaginary_unit, case_insensitive)
        self.precision = self.surface.precision

    def function(self, origin: Sequence, direction: Sequence) -> RaySurfaceFunction:
        """Build g(t)=F(O+t*d) for this ray."""
        return RaySurfaceFunction(self.surface, origin, direction, self.precision)

    def intersect(
        self,
        origin: Sequence,
        direction: Sequence,
        t_max,
        method: str = "auto",
        **options,
    ) -> List[Number]:
        """Return all ray parameters t in [t_min, t_max] where the ray hits F=0.

        method: "auto" | "sampling" | "sturm" | "chebyshev" | "interval".
        t_min (keyword, default 0) sets the lower bound.
        """
        t_min = options.pop("t_min", 0)
        func = self.function(origin, direction)
        backend = get_backend(method)
        t0 = Number(t_min, self.precision)
        t1 = Number(t_max, self.precision)
        roots = backend(func, t0, t1, self.precision, **options)
        return sorted(roots)

    def points(
        self, origin: Sequence, direction: Sequence, t_max, **kwargs
    ) -> List[Tuple[Number, ...]]:
        """Intersection points (x, y, z) on the surface, sorted by t."""
        func = self.function(origin, direction)
        return [func.point_at(t) for t in self.intersect(origin, direction, t_max, **kwargs)]

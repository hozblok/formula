"""Find all intersections of a ray with an implicit surface F(x,y,z)=0.

A ray r(t)=O+t*d turns F into a single-variable g(t)=F(O+t*d); every
intersection is a real root of g on [t_min, t_max]. Pluggable root-finding
backends (see _roots/) locate all of them.
"""

from typing import List, Sequence, Tuple

from ._roots import get_backend
from .constants import DEFAULT_CASE_INSENSITIVE, DEFAULT_IMAGINARY_UNIT
from .formula import Number, Solver

_AXES = ("x", "y", "z")


def _with_endpoint_roots(func, roots, t0, t1, precision):
    """Backend-agnostic net: add an exact root sitting on t0/t1 if a backend
    missed it (Sturm's (a,b] convention, interior Chebyshev nodes, etc.)."""
    gtol = Number(f"1e-{max(precision // 3, 4)}", precision)
    tol = Number(f"1e-{max(precision // 2, 6)}", precision)
    out = list(roots)
    for t in (t0, t1):
        mag = abs(func.g(t)).parts()[0]  # modulus as a real string (|re,im|)
        if "inf" in mag or "nan" in mag:
            continue
        if Number(mag, precision) <= gtol and all(abs(t - r) > tol for r in out):
            out.append(t)
    return out


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
        # Surface axes drive g/g' evaluation; origin and direction are kept over
        # the full x,y,z so point_at and the arc-length normalization stay correct
        # even when the surface ignores an axis.
        self._axes = [a for a in _AXES if a in axes]
        self.origin = {a: self._num(origin[_AXES.index(a)]) for a in _AXES}
        full = {a: self._num(direction[_AXES.index(a)]) for a in _AXES}
        norm = abs(sum((c * c for c in full.values()), self._num(0))) ** self._num("0.5")
        if norm == self._num(0):
            raise ValueError("direction must be a non-zero vector")
        self.direction = {a: c / norm for a, c in full.items()}

    def _num(self, value) -> Number:
        return value if isinstance(value, Number) else Number(value, self.precision)

    def _point(self, t: Number) -> dict:
        """Ray coordinates (surface axes only) at parameter t, as value strings."""
        return {a: str(self.origin[a] + t * self.direction[a]) for a in self._axes}

    def g(self, t: Number) -> Number:
        """g(t) = F(O + t*d)."""
        return Number.wrap(self.surface.evaluate(self._point(t)), self.precision)

    def gprime(self, t: Number) -> Number:
        """g'(t) = grad F . d via the chain rule."""
        point = self._point(t)
        total = self._num(0)
        for a in self._axes:
            partial = Number(self.surface.get_derivative(a, point, 0), self.precision)
            total = total + partial * self.direction[a]
        return total

    def point_at(self, t: Number) -> Tuple[Number, ...]:
        """The (x, y, z) point on the ray at parameter t."""
        return tuple(self.origin[a] + t * self.direction[a] for a in _AXES)


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
    ):
        self.surface = Solver(expression, precision)
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

        method: "auto" | "sampling" | "sturm" | "chebyshev" | "subdivision".
        t_min (keyword, default 0) sets the lower bound.
        """
        t_min = options.pop("t_min", 0)
        backend = get_backend(method)
        t0 = Number(t_min, self.precision)
        t1 = Number(t_max, self.precision)
        if t0 > t1:
            raise ValueError(f"t_min ({t_min}) must not exceed t_max ({t_max})")
        func = self.function(origin, direction)
        roots = backend(func, t0, t1, self.precision, **options)
        return sorted(_with_endpoint_roots(func, roots, t0, t1, self.precision))

    def points(
        self, origin: Sequence, direction: Sequence, t_max, **kwargs
    ) -> List[Tuple[Number, ...]]:
        """Intersection points (x, y, z) on the surface, sorted by t."""
        func = self.function(origin, direction)
        return [func.point_at(t) for t in self.intersect(origin, direction, t_max, **kwargs)]

"""Find all intersections of a ray with an implicit surface F(x,y,z)=0.

A ray r(t)=O+t*d turns F into a single-variable g(t)=F(O+t*d); every
intersection is a real root of g on [t_min, t_max]. Pluggable root-finding
backends (see _roots/) locate all of them.
"""

from collections import namedtuple
from typing import Iterable, List, Sequence, Tuple

from ._roots import get_backend
from .formula import Number, Solver

_AXES = ("x", "y", "z")

# Geometry-only record of one ray's life: source, every (reflection point, grazing
# angle), the screen hit + final direction, and the total geometric path length.
RayPath = namedtuple(
    "RayPath",
    ["source", "reflections", "screen_point", "screen_direction", "opl", "exited"],
)


def _normalize(components: Iterable[Number], precision, zero_msg="direction must be a non-zero vector"):
    """Unit-normalize components as Number; raises on a zero vector."""
    zero = Number(0, precision)
    nums = tuple(Number(c, precision) for c in components)
    norm = abs(sum((c * c for c in nums), zero)) ** 0.5
    if norm == zero:
        raise ValueError(zero_msg)
    return tuple(c / norm for c in nums)


def _with_endpoint_roots(func, roots, t0, t1, precision):
    """Backend-agnostic net: add an exact root sitting on t0/t1 if a backend
    missed it (Sturm's (a,b] convention, interior Chebyshev nodes, etc.)."""
    gtol = Number(f"1e-{max(precision // 3, 4)}", precision)
    tol = Number(f"1e-{max(precision // 2, 6)}", precision)
    out = list(roots)
    for t in (t0, t1):
        mag = abs(func.g(t)).parts[0]  # modulus as a real string (|re,im|)
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
        self.origin = {
            a: Number(origin[_AXES.index(a)], self.precision) for a in _AXES
        }
        self.direction = {
            a: c
            for a, c in zip(
                _AXES,
                _normalize(
                    (direction[_AXES.index(a)] for a in _AXES), self.precision
                ),
            )
        }

    def _point(self, t: Number) -> dict:
        """Ray coordinates (surface axes only) at parameter t, as value strings."""
        return {a: str(self.origin[a] + t * self.direction[a]) for a in self._axes}

    def g(self, t: Number) -> Number:
        """g(t) = F(O + t*d)."""
        return Number(self.surface.evaluate(self._point(t)), self.precision)

    def gprime(self, t: Number) -> Number:
        """g'(t) = grad F . d via the chain rule."""
        point = self._point(t)
        zero = Number(0, self.precision)
        total = zero
        for a in self._axes:
            partial = Number(self.surface.get_derivative(a, point, 0), self.precision)
            total = total + partial * self.direction[a]
        return total

    def point_at(self, t: Number) -> Tuple[Number, ...]:
        """The (x, y, z) point on the ray at parameter t."""
        return tuple(self.origin[a] + t * self.direction[a] for a in _AXES)

    def normal_at(self, t: Number) -> Tuple[Number, Number, Number]:
        """Unit surface normal grad F / |grad F| at the ray point r(t).

        Axes the surface does not mention contribute a zero gradient component.
        """
        point = self._point(t)
        grad = (
            Number(self.surface.get_derivative(a, point, 0), self.precision)
            if a in self._axes
            else Number(0, self.precision)
            for a in _AXES
        )
        return _normalize(
            grad,
            self.precision,
            zero_msg="surface gradient vanishes; normal is undefined",
        )

    def reflect_at(self, t: Number):
        """Specular reflection of the ray at r(t).

        Returns (point, reflected_unit_direction, grazing_angle); the grazing
        angle (radians) is measured from the surface tangent plane and the
        reflected direction is d - 2(d.n)n.
        """
        normal = self.normal_at(t)
        incident = tuple(self.direction[a] for a in _AXES)
        zero = Number(0, self.precision)
        dot = sum((incident[k] * normal[k] for k in range(3)), zero)
        reflected = tuple(
            incident[k] - Number(2, self.precision) * dot * normal[k] for k in range(3)
        )
        grazing = Number(f"asin({abs(dot)})", self.precision)
        return self.point_at(t), reflected, grazing


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

    def _as_surface(self, surface):
        """Coerce an expression string / RaySurface / None to a RaySurface or None."""
        if surface is None or isinstance(surface, RaySurface):
            return surface
        return RaySurface(surface, self.precision)

    def trace_path(self, origin, direction, t_max, exit_surface=None,
                   screen_surface=None, max_bounces=1000, t_min=None,
                   method="auto", **options) -> RayPath:
        """Exact geometric multi-reflection path of a ray off this surface F=0.

        The ray reflects specularly off F=0; the reflection sequence ends when it
        would cross `exit_surface` before the next reflection (the optic's exit
        aperture), when no further forward hit exists (a finite reflector), or after
        `max_bounces`. After it leaves, the ray is propagated to `screen_surface` when
        given. The direction is kept unit, so every step parameter is arc length and
        `opl` is the geometric path source -> ... -> screen.

        Returns RayPath(source, reflections, screen_point, screen_direction, opl,
        exited): `source` = (x,y,z); `reflections` = [(point, grazing_angle), ...] with
        the grazing angle in radians from the surface; `screen_point`/`screen_direction`
        = the hit on `screen_surface` and the final unit direction; `opl` = total
        geometric path length; `exited` = whether the ray left through a forward
        boundary. GEOMETRY ONLY -- no Fresnel/amplitude/energy.

        `exit_surface`/`screen_surface`: an F(x,y,z)=0 expression string or a
        RaySurface. `t_min` (default `t_max * 1e-9`) is the minimum forward step that
        skips the launch point lying on the surface. As elsewhere in the engine,
        root-finding is best conditioned when coordinates are O(1) (scale a metre-sized
        geometry to e.g. micrometres before tracing); use `method="subdivision"`/
        `"sturm"` for grazing (near-double-root) hits.
        """
        p = self.precision
        eps = (Number(t_min, p) if t_min is not None
               else Number(t_max, p) * Number("1e-9", p))
        bound = self._as_surface(exit_surface)
        screen = self._as_surface(screen_surface)

        source = tuple(Number(c, p) for c in origin)
        O, d = source, _normalize(direction, p)
        reflections, opl, exited = [], Number(0, p), False

        for _ in range(max_bounces):
            wall = self.intersect(O, d, t_max, method=method, t_min=eps, **options)
            t_wall = wall[0] if wall else None
            t_bound = None
            if bound is not None:
                hits = bound.intersect(O, d, t_max, t_min=eps)
                t_bound = hits[0] if hits else None
            if t_wall is None or (t_bound is not None and t_bound <= t_wall):
                exited = t_bound is not None or bound is None
                break
            point, d, grazing = self.function(O, d).reflect_at(t_wall)
            reflections.append((point, grazing))
            opl = opl + t_wall
            O = point

        screen_point = None
        if exited and screen is not None:
            hits = screen.intersect(O, d, t_max, t_min=eps)
            if hits:
                t_scr = hits[0]
                opl = opl + t_scr
                screen_point = tuple(O[k] + t_scr * d[k] for k in range(3))
        return RayPath(source, reflections, screen_point, d, opl, exited)

🇬🇧 **English** · [🇷🇺 Русский](ray-surface-design.ru.md)

# RaySurface: design & decisions

**Status:** shipped &middot; **See also:**
[usage reference](ray-surface-intersections.md) ·
[README](../README.md#finding-all-raysurface-intersections)

This is the *why* behind `RaySurface`. For *how to use it* — the API, every
backend, worked examples and the full limits of applicability — read the
[reference](ray-surface-intersections.md); this document does not repeat them.


## Goal

A reliable way to find **all** intersections of a ray with an implicit surface
`F(x, y, z) = 0`, living in `formula` next to `Solver` and `Number`, at arbitrary
precision — including the thin-feature and tangent cases a local Newton solver
misses.


## The reduction (why this is tractable)

A ray `r(t) = O + t·d` substituted into `F = 0` collapses the surface to a
single-variable `g(t) = F(O + t·d)`, so "all intersections" ≡ "all real roots of
`g` on `[t_min, t_max]`". `formula` already evaluates `F` and its partials
symbolically at arbitrary precision, so `g(t)` and `g'(t) = ∇F·d` are built
straight from `Solver.evaluate` / `get_derivative` — no new symbolic machinery.
That reduction is the whole design: everything else is univariate root-finding
behind one contract, `find_all(func, t_min, t_max, precision, **opts)`.


## Design decisions

- **Sturm over companion/colleague-matrix eigenvalues** for the algebraic path —
  no arbitrary-precision eigensolver exists in this project; Sturm stays inside
  `formula`'s precision world and gives a *provable* real-root count. The price is
  recovering the polynomial by interpolation, well-conditioned only to moderate
  degree.
- **`subdivision` (pure Python) as the reliability backbone** rather than a
  rigorous C++ interval type — it targets the real weak spot (completeness on
  transcendental/oscillatory surfaces) at zero risk to the core evaluator and no
  rebuild, trading formal rigor for a practical `g''` bound.
- **`chebyshev` isolates via a monomial proxy** (reusing the Sturm machinery)
  instead of a colleague-matrix eigensolve. This is the known weak link: the
  Chebyshev→monomial conversion is ill-conditioned at high degree, so `chebyshev`
  is for low-to-moderate oscillation and `auto` leans on `subdivision` for dense
  oscillation. A colleague-matrix or recursive-subdivision isolator is the
  principled future fix.
- **`auto` = exact-where-possible + union-elsewhere** — route algebraic surfaces
  to `sturm`, union `chebyshev` with `subdivision` otherwise so neither method's
  blind spot drops a root, and fall back to the general backends if `sturm` raises.


## Architecture

```
src/formula/intersect.py     RaySurface (public) + RaySurfaceFunction (g, g', point_at)
                             + the backend-agnostic endpoint-root net
src/formula/_roots/
  __init__.py                backend registry, auto routing, union, Sturm→general fallback
  sturm.py                   polynomial backend (exact)
  chebyshev.py               self-validating spectral backend
  subdivision.py             derivative-bound exclusion backend
  sampling.py                reference/oracle backend
  _poly.py                   arbitrary-precision polynomial ops (eval, divmod, gcd, square-free, interpolate)
  _isolate.py                shared Sturm chain / variations / isolate / isolate_roots / bisect / rtsafe
```

Every backend implements the one `find_all` contract and reuses
`RaySurfaceFunction`, `_poly` and `_isolate`, so each is an independent drop-in.

### Supporting change to `Number`

Added the minimal public API the backends need (previously only `_`-prefixed
internals existed): `Number.wrap(value, precision)`, `__neg__`, `.precision`,
`.is_complex`, `.parts()`.


## Backends, in one line each

Why each exists; the full table, options and limits are in the
[reference](ray-surface-intersections.md#backends).

- **`sturm`** — exact, complete root count for algebraic surfaces (quadrics,
  cones, tori). Also handles complex surfaces via `gcd(Re g, Im g)`.
- **`chebyshev`** — smooth analytic surfaces with low-to-moderate oscillation.
- **`subdivision`** — general/oscillatory surfaces; the practical completeness net.
- **`sampling`** — fast oracle and cross-check baseline; not rigorous.
- **`auto`** — the default dispatcher described above.


## Review & hardening log

A multi-agent adversarial review of the backends surfaced and fixed a set of
correctness bugs (each regression-tested):

- **Critical** — a root exactly at `t_min` made Sturm return a phantom root
  (`bisect` on an invalid bracket; Sturm's `(a,b]` convention drops the left
  endpoint). Fixed with endpoint-aware isolation (`isolate_roots`) plus a
  backend-agnostic endpoint net.
- **Normalization / `point_at`** — the direction was normalized over only the
  surface's axes, so `t` was wrong and `point_at` reported `0` for ignored axes;
  a zero direction produced a silent `nan`. Now normalized over the full
  `(x, y, z)` with a clear error on a degenerate direction.
- **`max_degree` off-by-one** (degree-16 wrongly rejected) and an unreliable cap
  (now uses a guard node).
- **Routing** — `is_polynomial` mis-handled variable-free denominators and
  non-natural powers (`x^-2`, `x^0.5`); `auto` now falls back to the general
  backends when `sturm` raises.
- **Non-finite samples** in `sampling` (phantom root at a singularity) are skipped.


## Deferred

- **`chebyshev` high-degree isolation** — replace the monomial proxy with a
  colleague-matrix or recursive-Chebyshev isolator to remove the dense-oscillation
  blind spot.

The remaining *behavioral* limits (degree/conditioning, `M2` bound, multiplicity,
clustering, poles) are inherent and documented with worked examples in the
[reference](ray-surface-intersections.md#worked-limit-examples).


## Tests

`tests/test_intersect*.py` — the baseline feature suite plus robustness and
diverse-surface tests (quadrics/conics, transcendentals, geometry, invariants,
multiplicity, higher-degree). Each hardening item above has a regression test, and
the documented limits each have a test asserting either the workaround or the
boundary.

# Add `RaySurface`: find all ray–surface intersections at arbitrary precision

## Goal

Add a **reliable method for finding *all* intersections of a ray with an implicit surface** `F(x, y, z) = 0`, living in `formula` next to `Solver` and `Number`. This generalizes the single-root, local Newton solvers in the `capsys`/`capsys-legacy` projects (3×3 Jacobian = surface gradient + two ray constraints, solved by Gauss/LU) into a backend that returns **every** intersection along the ray, including the thin-feature and tangent cases those solvers miss.

## Background / motivation

The legacy capillary-optics code (`linalg.cpp:NewtonMetodForSurfaceAndStraightLine`, `mathforcap.cpp:NewtonMetodForEquationAndLine`) converges to **one** root near an initial guess and hand-deflates known roots with the `(x²+y²−1)/(x−1)` trick. There was no way to robustly enumerate all intersections, and `formula` had no root-finder at all.

**Key reduction:** a ray `r(t) = O + t·d` substituted into `F = 0` collapses the surface to a single-variable function `g(t) = F(O + t·d)`. So "all intersections" ≡ "all real roots of `g` on `[t_min, t_max]`" — and `formula` already evaluates `F` and its partials symbolically at arbitrary precision, which is exactly what an all-roots solver needs. `g(t)` and `g'(t) = ∇F·d` (chain rule) are built from the existing `Solver.evaluate` / `get_derivative`, with no new symbolic substitution.

## Specification

### Public API

```python
from formula import RaySurface

rs = RaySurface(expression, precision=24, imaginary_unit="i", case_insensitive=False)

# all ray parameters t in [t_min, t_max] where the ray meets F = 0
ts  = rs.intersect(origin, direction, t_max, method="auto", t_min=0, **opts)  # -> sorted [Number, ...]

# the same hits mapped back to (x, y, z) points on the surface
pts = rs.points(origin, direction, t_max, method="auto", **opts)             # -> [(Number, Number, Number), ...]
```

- `origin` / `direction` are `(x, y, z)`; `direction` is normalized internally, so `t` is measured in units of `|d|` (distance for a unit `d`).
- Roots are returned as arbitrary-precision `Number` values, sorted ascending.

### Backends (`method=`)

| method | Approach | Best for | Guarantee |
|---|---|---|---|
| `auto` | dispatch per surface | default | Sturm for polynomials, else Chebyshev ∪ subdivision |
| `sturm` | interpolate → square-free → Sturm chain → bisect | algebraic surfaces (quadrics, cones, tori) | exact, complete real-root count |
| `chebyshev` | Chebyshev-Gauss fit → monomial proxy → isolate → Newton polish; **self-validating** degree | smooth analytic (`sin`/`exp`/`log`/`cosh`) | escalates degree until spectral tail converges; captures tangencies |
| `subdivision` | adaptive Taylor-bound exclusion `\|g(m)\| ≤ \|g'(m)\|·h + (M2/2)·h²` | general / oscillatory surfaces | practically reliable (bounded by the `g''` estimate) |
| `sampling` | grid sign-change + safeguarded Newton | quick, well-separated roots | none (oracle / cross-check baseline) |
| `interval` | rigorous interval isolation | — | **stub** (`NotImplementedError`); see "Deferred" |

- **`auto`** routes algebraic surfaces to `sturm` (exact) and other real surfaces to `chebyshev` reconciled with `subdivision` as a safety net (deduped union), so neither method's blind spot drops a root.
- **Complex-valued surfaces** are supported by `sturm`: real intersections are the common roots of `Re g` and `Im g` (their polynomial gcd). Other backends are real-only and raise `NotImplementedError` on complex surfaces.
- **Per-method options:** `max_degree` (sturm), `cheb_degree` (chebyshev start degree), `m2_samples` / `region_tol` (subdivision), `samples` (sampling).

## Architecture

```
src/formula/intersect.py        RaySurface (public) + RaySurfaceFunction (g, g', point_at)
src/formula/_roots/
  __init__.py                   backend registry + auto routing + union reconciliation
  sturm.py                      polynomial backend (exact)
  chebyshev.py                  self-validating spectral backend
  subdivision.py                derivative-bound exclusion backend
  sampling.py                   reference/oracle backend
  interval.py                   stub
  _poly.py                      arbitrary-precision polynomial ops (eval, divmod, gcd, square-free, interpolate)
  _isolate.py                   shared Sturm chain / variations / isolate / bisect / safeguarded Newton
```

All backends implement one contract — `find_all(func, t_min, t_max, precision, **opts) -> list[Number]` — and reuse `RaySurfaceFunction`, `_poly`, and `_isolate`, so each is an independent drop-in.

### Supporting change to `Number`

Added minimal public API used by the backends (previously only `_`-prefixed internals existed): `Number.wrap(value, precision)`, `__neg__`, `.precision`, `.is_complex`, `.parts()`.

## Testing

`tests/test_intersect.py` (22) + `tests/test_intersect_robustness.py` (66) +
`tests/test_intersect_surfaces.py` (44 — quadrics/conics, transcendentals,
ray geometry, invariants, multiplicity, higher-degree algebraic) +
`tests/test_intersect_capillary.py` (6 — the real CAPSYS X-ray surfaces) cover,
across all backends:

- cylinder / sphere (two roots), oblique ray (arc-length `t`), range clipping, point-mapping residuals, **point recovery on axes the surface ignores**;
- **endpoint roots** (ray origin on the surface; root at `t_min`/`t_max`/both) — found by every backend, no phantom;
- **tangent double root** (`x²+y²+z²−1` grazing): Sturm/subdivision find it, sampling correctly does **not**; odd multiplicity-3 (`x³`); clustered roots `(x−1)(x−1.001)`;
- quartic `(x²−1)(x²−4)`; degree 15/16 at the cap, degree 17 rejected, fallback when Sturm raises;
- **torus** (curved capillary, degree-4 quartic): four hits per ray, routing to exact Sturm, vertical/missing/grazing-tangent rays, on-surface residuals;
- **corrugated capillary wall** `x²+y²−(1+0.3·sin 4z)²` — twelve roots at `t = k·π/4`; turning-point rejection (`cos t + 1`);
- complex surface `(x²−1)(1+i)` via Sturm; complex-rejection for real-only backends;
- input validation (zero direction, `t_min>t_max`, unknown variable, `max_degree<2`); routing (`is_polynomial`, scaled quadric, non-natural powers);
- **documented limits**: sub-sample misses, high-multiplicity (use Sturm), oscillation (use subdivision/auto).

**Full suite: 815 passed, 2 xfailed.**

## Design decisions

- **Sturm over companion-matrix eigenvalues** for the polynomial path — no arbitrary-precision eigensolver is available; Sturm stays inside `formula`'s precision world and gives a provable root count.
- **`subdivision` (pure Python) as the reliability backbone** instead of a rigorous C++ interval type — it targets the actual weak spot (completeness on transcendental/oscillatory surfaces) at zero risk to the core evaluator and no rebuild.
- **Chebyshev isolates via a monomial proxy** (reusing the Sturm machinery) rather than a colleague-matrix eigensolve. This is the known weak link: the monomial conversion is ill-conditioned at high degree, so `chebyshev` is for low-to-moderate oscillation and `auto` leans on `subdivision` for dense oscillation. See the limits doc.

## Review & hardening in this revision

A multi-agent adversarial review of the backends surfaced and fixed a set of correctness bugs:

- **Critical** — a root exactly at `t_min` made Sturm return a phantom root (`bisect` on an invalid bracket; Sturm's `(a,b]` convention drops the left endpoint). Fixed with endpoint-aware isolation plus a backend-agnostic endpoint net.
- **Normalization / `point_at`** — the direction was normalized over only the surface's axes, so `t` was wrong and `point_at` reported `0` for ignored axes; a zero direction produced a silent `nan`. Now normalized over full `(x,y,z)` with a clear error on a degenerate direction.
- **`max_degree` off-by-one** (degree-16 wrongly rejected) and an unreliable cap (now uses a guard node).
- **Routing** — `is_polynomial` mis-handled variable-free denominators and non-natural powers (`x^-2`, `x^0.5`); `auto` now also falls back to the general backends when Sturm raises.
- **Non-finite samples** in `sampling` (phantom root at a singularity) are skipped.

## Deferred / caveats

See **[doc/ray-surface-intersections.md](doc/ray-surface-intersections.md)** for the full limits of applicability. In brief:

- **`interval` backend (rigorous) is a stub.** Needs a C++ `mp_interval` type; `cpp_dec_float` has no directed-rounding modes, so a naive interval type would not be rigorous. Sturm already proves the algebraic case.
- **`subdivision`** is *practically* reliable, not a formal proof — only as good as the sampled `g''` bound (`m2_samples`).
- **`chebyshev`** self-validates the fit, not the isolation; it can silently miss roots on densely oscillatory surfaces — prefer `subdivision`/`auto`.
- **`sturm`** is exact only up to a moderate algebraic degree.
- Complex surfaces are handled only by `sturm`.

## Files

New: `intersect.py`, `_roots/` package (8 modules), `tests/test_intersect.py`, `tests/test_intersect_robustness.py`, `doc/ray-surface-intersections.md`; modified: `formula.py` (`Number` API), `__init__.py` (exports), `README.md`.

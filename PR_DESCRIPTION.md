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

`tests/test_intersect.py` — 22 tests, including:

- cylinder / sphere (two roots), oblique ray (arc-length `t`), range clipping, point-mapping residuals;
- **tangent double root** (`x²+y²+z²−1` grazing): Sturm/subdivision find it, sampling correctly does **not**;
- quartic `(x²−1)(x²−4)` — four roots;
- **corrugated capillary wall** `x²+y²−(1+0.3·sin 4z)²` — twelve roots at `t = k·π/4`; subdivision finds all, and Chebyshev *starting at an inadequate degree 8* self-validates and also returns all twelve;
- turning-point rejection (`cos t + 1`: roots at π, 3π; the `g'=0`, `g=2` point at 2π is rejected);
- complex surface `(x²−1)(1+i)` via Sturm; complex-rejection for real-only backends;
- cross-checks: subdivision ↔ Sturm on polynomials, Chebyshev ↔ subdivision on transcendentals.

**Full suite: 749 passed, 2 xfailed. Pylint 10/10, isort clean.**

## Design decisions

- **Sturm over companion-matrix eigenvalues** for the polynomial path — no arbitrary-precision eigensolver is available; Sturm stays inside `formula`'s precision world and gives a provable root count.
- **Chebyshev without a colleague-matrix eigensolve** — roots are isolated by converting the proxy to monomial form and reusing the Sturm machinery, then polished on the true `g`.
- **`subdivision` (pure Python) as the reliability backbone** instead of a rigorous C++ interval type — it targets the actual weak spot (completeness on transcendental/oscillatory surfaces) at zero risk to the core evaluator and no rebuild.

## Deferred / caveats

- **`interval` backend (rigorous) is a stub.** A truly rigorous version needs a new C++ `mp_interval` type and an interval evaluation path through `cseval`. `formula`'s `cpp_dec_float` backend has **no directed-rounding modes**, so a naive `boost::numeric::interval` would silently assume exact arithmetic and not actually be rigorous — true rigor requires manual ULP inflation. Deferred because Sturm already proves the algebraic case.
- **`subdivision` (and the future `interval`) are *practically* reliable, not a formal proof** — their exclusion test is only as good as the estimated bound on `g''` (`m2_samples` / inflation knobs).
- Complex surfaces are handled only by `sturm` in this PR.

## Files

13 files changed, +997. New: `intersect.py`, `_roots/` package (7 modules), `tests/test_intersect.py`; modified: `formula.py` (`Number` API), `__init__.py` (exports), `README.md` (new "Finding all ray–surface intersections" section).

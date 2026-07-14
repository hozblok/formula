🇬🇧 **English** · [🇷🇺 Русский](ray-surface-intersections.ru.md)

# Ray–surface intersections (`RaySurface`)

`RaySurface` finds **every** intersection of a ray with an implicit surface
`F(x, y, z) = 0`, at arbitrary precision.

This document explains the mechanism, walks through each root-finding backend, and
states the **limits of applicability**. For the *why* — design decisions and the
review log — see [ray-surface-design.md](ray-surface-design.md).

- [Quick start](#quick-start)
- [Parameters](#parameters)
- [method — what to pick](#method--what-to-pick)
- [Reduction (how it works internally)](#reduction-how-it-works-internally)
- [Public API](#public-api)
- [How `g(t)` and `g'(t)` are built](#how-gt-and-gt-are-built)
- [Backends](#backends)
- [Worked examples](#worked-examples)
- [Strategy assessment](#strategy-assessment)
- [Limits of applicability](#limits-of-applicability)
- [Worked limit examples](#worked-limit-examples)
- [Choosing a method](#choosing-a-method)


## Quick start
```python
rs = RaySurface("x*x + y*y - 1", precision=24)
ts = rs.intersect(origin=(-2,0,0), direction=(1,0,0), t_max=10)
# -> [Number('1...'), Number('3...')]  — t parameters along the ray
pts = rs.points((-2,0,0), (1,0,0), t_max=10)
# -> [(x,y,z), ...] — the same intersections in coordinates
```


## Public API

```python
rs = RaySurface(expression, precision=24)

rs.intersect(origin, direction, t_max, method="auto", t_min=0, **options)
    # -> sorted list[Number] of ray parameters t where the ray meets F = 0

rs.points(origin, direction, t_max, **kwargs)
    # -> list[(x, y, z)] intersection points on the surface, sorted by t
```

- `origin` / `direction` are `(x, y, z)` triples; `direction` must be non-zero.
- Roots come back as arbitrary-precision `Number`s, ascending.
- `points()` reconstructs the **full** 3-D point even for axes the surface
  ignores (a cylinder is independent of `z`, but a ray's `z` is still reported):

```python
>>> rs = RaySurface("x*x - 1", precision=24)   # depends on x only
>>> rs.points((-2, 5, 7), (1, 0, 0), t_max=10, method="sturm")
[(Number('-0.999...'), Number('5'), Number('7')),
 (Number('0.999...'),  Number('5'), Number('7'))]
```

**`method`** — the root-finding backend:

- `"auto"` (default)
- `"sturm"`, `"chebyshev"`, `"subdivision"`, `"sampling"`

Per-method options:

- **`sturm`:** `max_degree`
- **`chebyshev`:** `cheb_degree`
- **`subdivision`:** `m2_samples`, `region_tol`
- **`sampling`:** `samples`

## Parameters
| Argument | Meaning |
|---|---|
| origin | ray start (x,y,z) |
| direction | direction; non-zero; length doesn't matter |
| t_max | right end of the search interval |
| t_min | left end (default 0) |
| method | see below; default "auto" |


## method — what to pick

| method | When | Options (rarely needed) |
|---|---|---|
| `auto` | unsure; default | — |
| `sturm` | polynomial, quadric, complex-valued | `max_degree` (16) |
| `chebyshev` | smooth analytic (`sin`, `exp`, …), few roots | `cheb_degree` (32) |
| `subdivision` | oscillatory, corrugated, unknown | `m2_samples` (200), `region_tol` (1e-6·Δt) |
| `sampling` | quick check; **not** rigorous | `samples` (256) |

See [Backends](#backends) and [Choosing a method](#choosing-a-method) for details.

## Reduction (how it works internally)

A ray is `r(t) = O + t·d`. Substituting it into the surface equation collapses
the three-variable surface into a single-variable function

```
g(t) = F(O + t·d)
```

and an intersection is exactly a **real root of `g` on `[t_min, t_max]`**. So
"find all intersections" becomes "find all real roots of `g`", and the whole
problem is delegated to pluggable univariate root-finders in
[`src/formula/_roots/`](../src/formula/_roots/).

Before tracing, `direction` `d` is scaled to unit length. So `t` is the actual
distance along the ray (**arc length**) in the same units as the coordinates, and
the length of `d` itself has no effect on the result — only its direction matters.
Hence `(1, 0, 0)` and `(7, 0, 0)` give the same `t`.



```python
>>> from formula import RaySurface
>>> rs = RaySurface("x*x + y*y - 1", precision=24)       # unit cylinder
>>> rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10)        # enters at x=-1 (t=1), exits at x=1 (t=3)
[Number('1.00000000000000000000002', precision=24), Number('2.99999999999999999999999', precision=24)]
>>> rs.intersect((-2, 0, 0), (7, 0, 0), t_max=10)        # longer d, same direction -> same t
[Number('1.00000000000000000000002', precision=24), Number('2.99999999999999999999999', precision=24)]
```


## How `g(t)` and `g'(t)` are built

No symbolic substitution is performed. `RaySurfaceFunction` reuses the existing
evaluator:

- `g(t) = F(O + t·d)` via `Solver.evaluate` on the point `O + t·d`.
- `g'(t) = ∇F · d` via `Solver.get_derivative` for each axis (chain rule), summed
  against the normalized direction. Only the axes the surface actually depends on
  contribute (the rest have `∂F/∂a = 0`).

Both are evaluated at full precision, which is what an all-roots solver needs to
separate close roots and certify a root count.


## Backends

Every backend implements one contract,
`find_all(func, t_min, t_max, precision, **opts) -> list[Number]`, and may be
selected directly or reached through `auto`.

### `sturm` — exact, for algebraic surfaces

The polynomial path. It is **complete** (provable real-root count) for algebraic
`g`:

1. Sample `g` at `max_degree + 2` nodes and recover its monomial coefficients by
   Newton divided differences in the scaled coordinate `u ∈ [-1, 1]`,
   `t = mid + span·u` (the extra node lets an over-degree surface be detected
   and rejected, not silently under-fit; the scaling keeps the coefficients
   comparable so the degree cutoff cannot clip a true leading term on badly
   scaled polynomials).
2. Make it **square-free** (`p / gcd(p, p')`) so every root is simple.
3. Build the **Sturm chain** and count roots in any sub-interval by the
   difference of sign variations at its ends.
4. **Isolate** each root in its own bracket and **bisect** to precision.

Complex-valued surfaces are supported: the real intersections are the common
roots of `Re g` and `Im g`, i.e. the roots of their polynomial gcd.

### `chebyshev` — for smooth analytic surfaces (`sin`, `exp`, `log`, …)

`g` is approximated on `[t_min, t_max]` by a Chebyshev interpolant at
Chebyshev–Gauss nodes. The degree is grown until the spectral tail converges
(self-validating *fit*). The interpolant is converted to a monomial proxy, whose
roots are isolated by the same Sturm machinery and then polished on the **true**
`g` by Newton. It captures even-multiplicity (tangent) roots that sampling steps
over. See [its limit](#chebyshev-limits) — the monomial conversion bounds where
this backend is trustworthy.

### `subdivision` — general / oscillatory surfaces

A pure-Python middle ground. On each sub-interval a Taylor bound excludes regions
that provably cannot hold a root:

```
|g(t) − g(m)| ≤ |g'(m)|·h + (M2/2)·h²       (m = midpoint, h = half-width)
```

so a root requires `|g(m)| ≤ |g'(m)|·h + (M2/2)·h²`. `M2` is an estimate of
`max|g''|`, sampled from `g'` and inflated. Surviving intervals are resolved to
simple roots (bracketed Newton) or tangencies (`g'=0` with `g≈0`); turning points
where `g≠0` are rejected. Practically reliable, **not** a formal proof — only as
good as the `M2` estimate.

### `sampling` — reference oracle

Grid sign-changes refined by safeguarded Newton. Not rigorous: even-multiplicity
roots and sub-sample features are missed by construction. It exists as the
baseline the rigorous backends are cross-checked against. Non-finite samples
(near a singularity) are skipped rather than reported as phantom roots.

### `auto` — default dispatch

- **Algebraic** surfaces → `sturm` (exact). If Sturm raises (e.g. degree above
  the cap), `auto` **falls back** to the general backends instead of failing.
- **Other real** surfaces → the deduped **union of `chebyshev` and
  `subdivision`**, so neither method's blind spot drops a root. Either may fail
  numerically and drop out; a complex surface still raises informatively.

## Worked examples

```python
from formula import RaySurface, Number

# 1) Sphere, two crossings
rs = RaySurface("x*x + y*y + z*z - 1", precision=48)
rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10)            # -> [1, 3]

# 2) Tangent (grazing) ray: an even-multiplicity root
rs.intersect((-2, 1, 0), (1, 0, 0), t_max=10, method="sturm")   # -> [2]
rs.intersect((-2, 1, 0), (1, 0, 0), t_max=10, method="sampling")# -> []  (missed)

# 3) Quartic (x^2-1)(x^2-4): four roots along the ray
RaySurface("(x*x - 1) * (x*x - 4)", 48).intersect(
    (-3, 0, 0), (1, 0, 0), t_max=6, method="sturm")      # -> [1, 2, 4, 5]

# 4) Corrugated capillary wall, radius 1 + 0.3 sin(4z): 12 roots at z = k·pi/4
RaySurface("x*x + y*y - (1 + 0.3*sin(4*z))^2", 32).intersect(
    (1, 0, 0), (0, 0, 1), t_max=10, t_min="0.1", method="auto") # -> 12 roots

# 5) Torus (a curved capillary): a degree-4 quartic, up to four hits per ray.
#    (x^2+y^2+z^2 + R^2 - r^2)^2 - 4 R^2 (x^2+y^2) = 0, here R=2, r=1.
torus = RaySurface("(x*x+y*y+z*z+3)^2 - 16*(x*x+y*y)", 48)
torus.intersect((-4, 0, 0), (1, 0, 0), t_max=10)         # -> [1, 3, 5, 7]
torus.intersect((3, -3, 0), (0, 1, 0), t_max=10, method="sturm")  # grazing -> [3]
# A bent capillary is a torus; "part of the torus" is an angular [t_min, t_max] slice.

# 6) Complex-valued surface; real hits are common roots of Re g and Im g
RaySurface("(x*x - 1) * (1 + i)", 48).intersect(
    (-2, 0, 0), (1, 0, 0), t_max=10, method="sturm")     # -> [1, 3]

# 7) Ray origin lies on the surface: t = 0 is a genuine root
RaySurface("x*x + y*y - 1", 48).intersect(
    (-1, 0, 0), (1, 0, 0), t_max=10, t_min=0)            # -> [0, 2]
```


## Strategy assessment

**Is the chosen strategy right?** Largely yes, with one backend (`chebyshev`)
that is the weak link.

- **Reduction to `g(t)` reusing `Solver.evaluate` / `get_derivative`** — correct
  and economical. It inherits arbitrary precision for free and adds no symbolic
  machinery. The cost is one surface evaluation per `g` sample.

- **Sturm over companion/colleague-matrix eigenvalues** for the algebraic path —
  the right call here. No arbitrary-precision eigensolver exists in this project;
  Sturm stays inside `formula`'s precision world and yields a *provable* root
  count rather than approximate eigenvalues. The price is the interpolation step
  (recovering the polynomial), which is well-conditioned only up to moderate
  degree.

- **`subdivision` as the reliability backbone** instead of a rigorous C++
  interval type — a pragmatic choice. It targets the real weak spot
  (completeness on transcendental/oscillatory surfaces) at zero risk to the core
  evaluator and with no rebuild. It trades formal rigor for a practical `M2`
  bound.

- **`chebyshev` converting to a monomial proxy** — the questionable decision.
  The Chebyshev *fit* is excellent, but converting it to the monomial basis and
  running Sturm there throws away exactly the conditioning advantage Chebyshev
  exists to provide. At high degree the monomial coefficients overflow the
  working precision and roots are silently dropped (see below). A colleague-matrix
  eigensolve or recursive Chebyshev subdivision would be the principled fix; until
  then `subdivision` (and therefore `auto`) is the dependable route for dense
  oscillation, and `chebyshev` is best reserved for low-to-moderate oscillation.

- **`auto` dispatch with union + fallback** — sound. Routing algebraic surfaces
  to the exact backend and unioning the two general backends elsewhere is the
  correct default, and the Sturm→general fallback removes a hard-failure mode.


## Limits of applicability

The geometric contract first, then per-backend conditions and failure modes.

**Geometric / API**

| Requirement | Outside it |
|---|---|
| Surface variables ⊆ `{x, y, z}` | `ValueError` |
| `direction` is non-zero | `ValueError` (was a silent `nan`) |
| `t_min ≤ t_max` | `ValueError` |
| Roots reported only in `[t_min, t_max]` | roots outside are not returned |
| `t` is arc length in `|d|` units | — |

### `sturm` limits

- **Degree.** Reliable for algebraic `g` up to the default cap (degree ≤ 16 with
  `max_degree=16`); degree **17+** raises a clear error, and `auto` then falls
  back. Raising `max_degree` extends the range, but equally-spaced
  divided-difference interpolation is itself ill-conditioned: well above ~degree
  20–30 the recovered coefficients (and even the degree *detection*) lose meaning
  **silently**, so a far-over-cap surface can return a wrong count rather than an
  error. *Mitigation:* keep algebraic ray-degrees modest, or split the interval.
- **Tolerance-based square-free / gcd.** `deg()` and the gcd use a numeric
  tolerance, so surfaces with a tiny leading coefficient relative to the others,
  or two roots far closer than that tolerance, can be mis-degreed or merged.
- **Complex surfaces.** Real hits are the gcd of `Re g` and `Im g`; the common
  factor is detected up to tolerance, so near-common (but not exactly common)
  factors are fragile.

### `chebyshev` limits

- **Monomial-conversion conditioning (the hard limit).** The self-validation
  guarantees the *fit*, not the *root isolation*. Converting a high-degree
  Chebyshev interpolant to the monomial basis blows the coefficient dynamic range
  past the working precision, and Sturm on those numerically-meaningless
  coefficients **silently drops roots**. Concretely, a dense oscillation such as
  `sin(60·x)` on `[0.1, 5]` returns far fewer roots than the true 94 via
  `method="chebyshev"`. *Mitigation:* use `subdivision` or `auto` for dense
  oscillation, or raise `precision` to push the breakdown to higher degree.
- **Analyticity.** Assumes `g` is smooth on the whole interval. A pole/singularity
  in range (e.g. a rational surface `1/(x−a)`) defeats the fit: the degree
  escalates to the cap, the run is slow, and the result is unreliable.
- **Endpoints.** Chebyshev–Gauss nodes are strictly interior, so a root exactly
  at `t_min`/`t_max` is recovered by the backend-agnostic endpoint net, not by the
  fit itself.
- Real surfaces only (`NotImplementedError` on complex).

### `subdivision` limits

- **Non-rigorous `M2`.** `max|g''|` is estimated from `g'` on a grid and inflated
  ×2. For oscillation finer than the grid the estimate **aliases low**, the
  exclusion test wrongly discards a root-bearing interval, and roots are **missed**
  (raise `m2_samples` for high-frequency surfaces). It is "practically reliable",
  never a proof.
- **Sharp/narrow features.** A bump narrower than the candidate resolution (e.g.
  a tall thin Gaussian) can be excluded entirely.
- **Clustered roots.** Roots closer than `region_tol` (default `span·1e-6`) merge.
- **Multiplicity.** Only simple (sign-change) and even tangency (`g'=0`, `g≈0`)
  roots are resolved; multiplicity ≥ 3 patterns may be missed or mis-handled.
- Real surfaces only.

### `sampling` limits

- A non-rigorous oracle. **Even-multiplicity roots are invisible** (no sign
  change) and any feature between samples is missed; widen `samples` to reduce,
  never to eliminate, the gap.
- Real surfaces only; non-finite samples are skipped.

### `auto` limits

- **Routing is syntactic.** `is_polynomial` rejects transcendentals, non-natural
  powers (`x^-2`, `x^0.5`) and variable denominators, but exotic encodings (a
  fractional power written as a nested fraction, a parenthesised variable
  exponent) can misroute. Pass `method=` explicitly when in doubt.
- The Sturm→general **fallback covers a Sturm *exception*, not a Sturm *silent
  misfit*** (degree far above the cap, per above).
- **Union tolerance.** Two genuinely distinct roots closer than `1e-(precision/2)`
  are merged into one.

### Cross-cutting: Newton-based polishing

`sampling`, `subdivision` and `chebyshev` polish roots with Newton, which needs
`g'`. A surface whose **symbolic derivative is singular at a root** breaks this —
e.g. `(x−1)^5`, whose derivative is formed as `u^5·5·u'/u` and is `nan` at `u=0`.
Use `sturm`, which removes the multiplicity via square-free and never evaluates
the surface derivative.


## Worked limit examples

Each example is a minimal **failure** (default call that returns a wrong/incomplete
result) followed by the **workaround**. Outputs are the observed values.

### 1. Sturm degree cap (`degree > max_degree` raises)

Reliable when the true degree of `g` is `≤ max_degree` (default 16).

```python
rs = RaySurface("x^17 - x", precision=40)            # g(t) = t^17 - t, degree 17
rs.intersect((0, 0, 0), (1, 0, 0), t_max=2, method="sturm")
# -> ValueError: polynomial degree exceeds max_degree; not a low-degree polynomial
rs.intersect((0, 0, 0), (1, 0, 0), t_max=2, method="sturm", max_degree=17)
# -> [0, 1]                                           (workaround: lift the cap)
```

Caveat — a degree *two or more* above the cap can **silently misfit** if its
high-order content aliases through the `max_degree+2` sample nodes; only `cap+1`
is reliably *detected* (the single guard node). Under `auto` the `ValueError`
case falls back to the general backends automatically.

### 2. Sturm high-degree conditioning (silent miscount)

Reliable when the recovered polynomial is low-degree *relative to the precision*.
Equally-spaced interpolation is ill-conditioned (Runge), so high degree at modest
precision loses roots with no error.

```python
expr = "*".join(f"(x-{k})" for k in range(1, 19))    # 18 distinct roots t=1..18
rs = RaySurface(expr, precision=24)
len(rs.intersect((0,0,0),(1,0,0), t_max=19, method="sturm", max_degree=20))
# -> 15            (wrong: three roots lost, others drifted by up to ~0.4)
```

Workaround: raise precision (`precision=40` recovers all 18) or split the interval
so each piece is low-degree.

### 3. Chebyshev monomial-conversion conditioning (silent undercount)

Reliable only for few oscillations on the interval. The Chebyshev→monomial step
is ill-conditioned, so dense oscillation drops roots silently.

```python
rs = RaySurface("sin(7*x)")                           # 23 roots on [0,10]
len(rs.intersect((0,0,0),(1,0,0), 10.0, method="chebyshev"))     # -> 10  (13 lost)
len(rs.intersect((0,0,0),(1,0,0), 10.0, method="subdivision"))   # -> 23  (workaround)
len(rs.intersect((0,0,0),(1,0,0), 10.0, method="auto"))          # -> 23
```

(`sin(k·x)` on `[0,10]` is exact for `k ≤ 6`; `k = 7` is the first to undercount.)

### 4. Chebyshev near/inside a pole

Reliable only when `g` is analytic on the interval *and* the nearest singularity is
well separated from it.

```python
rs = RaySurface("1/(x-5) - 1/2", precision=30)        # pole at x=5, root at x=7
rs.intersect((0,0,0),(1,0,0), t_max=10, t_min=Number("5.5",30), method="chebyshev")
# -> []            (silent miss: the pole at 5 is just left of t_min=5.5)
rs.intersect((0,0,0),(1,0,0), t_max=10, t_min=Number("5.5",30), method="subdivision")
# -> [7]           (workaround: bracketing backends tolerate a nearby pole)
```

A pole *strictly inside* the interval is worse — the fit never converges, escalates
to the degree cap, and the run is slow/unreliable.

### 5. Subdivision `M2` grid resonance (silent miss)

Reliable when `m2_samples` resolves the curvature of `g`. The danger is not raw
frequency but a **resonance** between the oscillation and the sampling grid.

```python
rs = RaySurface("sin(126*x)", precision=24)           # 402 roots on [0,10]
len(rs.intersect((0,0,0),(1,0,0), t_max=10, method="subdivision"))
# -> 137           (k·step = 126·(10/200) ≈ 2π: g' samples alias, M2 under-estimated)
len(rs.intersect((0,0,0),(1,0,0), t_max=10, method="subdivision", m2_samples=2000))
# -> 402           (workaround: a finer M2 grid)
```

### 6. Subdivision narrow feature (silent miss)

Reliable when the narrowest feature is resolved by the `m2_samples` grid.

```python
# tall Gaussian bump: two roots 1.4934.. and 1.5066.. (≈0.013 apart)
rs = RaySurface("exp(-16000*(x-1.5)*(x-1.5)) - 0.5", precision=30)
rs.intersect((0,0,0),(1,0,0), 3.0, method="subdivision")              # -> []   (both missed)
len(rs.intersect((0,0,0),(1,0,0), 3.0, method="subdivision", m2_samples=2000))  # -> 2
```

(Tightening `region_tol` does **not** help — the bump's region is excluded at a
coarse level before bisection ever drills in.)

### 7. `auto` union tolerance vs precision

`_union` merges roots closer than `1e-(precision/2)`. In practice this is finer
than what Sturm can resolve at the same precision, so to **keep** a genuinely close
pair you raise precision — which lifts both the union tolerance and Sturm's
resolution together.

```python
expr = "(x - 1.0)*(x - 1.0000000000001)"              # roots 1e-13 apart
RaySurface(expr, 24).intersect((0,0,0),(1,0,0), t_max=2, method="auto")   # -> [1.0]      (merged)
len(RaySurface(expr, 40).intersect((0,0,0),(1,0,0), t_max=2, method="auto"))  # -> 2  (precision=40)
```

### 8. Newton-path derivative singularity (use Sturm)

```python
rs = RaySurface("(x-1)^5", precision=24)
rs.intersect((0,0,0),(1,0,0), t_max=3, method="sampling")
# -> ValueError ('nan'): d/dx of u^5 is formed as u^5·5·u'/u -> 0/0 at the root
rs.intersect((0,0,0),(1,0,0), t_max=3, method="sturm")    # -> [1]  (square-free, exact)
```

### 9. Sampling misses tangent / sub-step roots

```python
rsA = RaySurface("x*x - 2*x + 1", precision=40)       # g(t)=(t-1)^2, tangent
rsA.intersect((0,0,0),(1,0,0), t_max=3, method="sampling")    # -> []   (no sign change)
rsA.intersect((0,0,0),(1,0,0), t_max=3, method="sturm")       # -> [1]
# two roots 0.005 apart inside one sample step are likewise missed by sampling.
```

### 10. `auto` routing of exotic forms

`is_polynomial` is syntactic. It correctly rejects `x^-2`, `x^0.5` and variable
denominators (routing them to the general backends), but an unusual encoding can
slip through — pass `method=` explicitly when in doubt.

```python
from formula._roots import is_polynomial
is_polynomial(RaySurface("x^-2 - 0.25", 24).surface)   # -> False (general path)
is_polynomial(RaySurface("1/(x-1) - 0.5", 24).surface) # -> False
is_polynomial(RaySurface("x^2 - 2", 24).surface)       # -> True  (natural power)
```


## Choosing a method

| Surface | Recommended |
|---|---|
| Polynomial / quadric / cone / torus (modest degree) | `sturm` (or `auto`) |
| Smooth analytic, few roots, low oscillation | `chebyshev` or `auto` |
| Oscillatory / corrugated / unknown | `subdivision` or `auto` |
| Complex-valued | `sturm` |
| Quick check, well-separated roots | `sampling` |
| Don't know | `auto` |

When in doubt, `auto` is the safe default: it is exact on algebraic surfaces and
reconciles both general backends elsewhere.

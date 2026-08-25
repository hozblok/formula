# Differentiation in `formula`: the mechanism and its domain of applicability

## 1. What is computed

`get_derivative` returns a single number — the partial derivative `∂f/∂x` of the
parsed expression `f` with respect to one named variable `x`, evaluated at one
supplied point, in multiprecision.

```python
from formula import Formula, Solver

Formula("x^2", precision=24).get_derivative("x", {"x": "3"})   # "6"
Solver("3*x^2 + 2*x + 7")({"x": "5"}, derivative="x")          # "32"
```

This is neither a symbolic derivative (no formula is returned) nor a finite
difference (no step is introduced). It is forward-mode automatic differentiation:
exact local rules, applied to the expression tree and evaluated numerically at the
point.

## 2. The mechanism

The expression is parsed into a tree: `3*x^2 + 2*x + 7` is a `+` over `*`, `^`, the
variable `x`, and constants. The engine traverses the tree and associates with each
node two quantities — its value at the point, and its derivative (the rate of
change with respect to `x`).

The traversal rests on two elementary facts:

1. the derivative of the variable `x` equals `1`;
2. the derivative of any constant equals `0`.

The remaining nodes combine the derivatives of their operands by the standard rules
of differentiation:

| node | value | derivative |
|------|-------|------------|
| `u + v` | `u + v` | `u' + v'` |
| `u - v` | `u - v` | `u' − v'` |
| `u * v` | `u·v` | `u'·v + u·v'` |
| `u / v` | `u/v` | `(u'·v − u·v') / v²` |
| `sin u` | `sin u` | `cos(u)·u'` |
| `u ^ v` | `u^v` | `v·u^(v−1)·u' + u^v·ln(u)·v'` |

The unary functions `exp, log, sqrt, cos, tan, asin, acos, atan` follow the same
pattern: a known local derivative multiplied by the derivative `u'` of the
argument. This factor `u'` is the chain rule, applied at every level of the tree.

We emphasize: wherever the function is differentiable and no intermediate rule is
evaluated at a singular point, the result is exact to the working precision — the
method introduces no truncation error of its own.

Example. For `f = 3x² + 2x + 7` at `x = 5`:
`f' = (3·2x)·1 + 2·1 + 0 = 30 + 2 = 32`.

## 3. An essential simplification: the structural zero

If an operand does not contain the variable `x`, it is constant, and its derivative
is identically zero. The engine recognizes this *structurally* — by testing whether
`x` occurs in the subtree — and omits the corresponding term without evaluating it.

This is not merely an optimization. It is what keeps a singular but irrelevant
factor out of the result. In the power rule the second term carries `ln(u)`; for
`x²` the exponent is constant, the term is structurally absent, and `ln(u)` is never
formed. Were it formed at `x = 0`, one would obtain `ln(0)·0 = (−∞)·0`, an
indeterminate `NaN`. With the dead term omitted, `d/dx x²` at `0` is exactly `0`, as
it must be.

## 4. Domain of applicability

It is necessary to distinguish four cases.

**(a) Where the method is exact.** At any point at which `f` is differentiable and
no intermediate rule meets a singularity. The result is one partial derivative, at
one point, to the requested precision. Since a number is returned rather than an
expression, there is no second derivative by direct iteration.

**(b) Where the method refuses.** At a singular point the engine raises an
exception; it does not return a substitute value.

| expression | at | reason |
|------------|-----|--------|
| `log(x)`, `sqrt(x)` | `0` | infinite derivative |
| `asin(x)`, `acos(x)` | `±1` | infinite derivative |
| `0^x`, `x^x` | `0` | of the form `0^0`: `ln(0)` arises |
| `x^(1/2)`, `(x^2)^(1/2)`, `x^-2` | `0` | base `0`, exponent `< 1`: infinite derivative |

**(c) Conventions.** Certain non-differentiable points are assigned a value by
convention, stated here explicitly: `abs'(0) = 0` and `sign' = 0` (the symmetric
sub-derivative); the relational and logical operators `> < = | &`, being piecewise
constant, are assigned derivative `0`.

**(d) A limitation of principle.** There exist points at which the derivative exists
and is finite, yet the engine nevertheless refuses. The clearest instance is
`(x^2)^(3/4) = |x|^(3/2)`, differentiable at `0` with derivative `0`. At `x = 0` it
cannot be distinguished — by a procedure that carries a single number — from

- `(x^2)^(1/2) = |x|` — a corner, derivative undefined;
- `(x^2)^(1/4) = |x|^(1/2)` — a vertical tangent, derivative infinite.

All three present identical local data: base `0`, exponent `< 1`, derivative of the
argument `0`. The correct answer is fixed by the *order of vanishing* of the
argument, and that order is not carried by a single pointwise value. We stress that
this is a limitation of principle, not a defect of the implementation: to resolve
such cases one must carry more than a number — the whole expression (a
computer-algebra system, which can pass to a limit) or the leading asymptotic
order. The present method carries one number deliberately; therein lie its speed and
its exactness everywhere else.

**Complex values.** The same algorithm operates over the complex field, with two
reservations: complex exponentiation drifts slightly, being computed as
`exp(b·ln a)`; and the base-`0` guard of case (b) is real-only, since the condition
"exponent `< 1`" presupposes an order, which the complex field does not possess.

## 5. Summary

The engine carries, through the expression tree, the value and its derivative,
computed by exact local rules. Within its domain of applicability the result is
exact; at a singular point it refuses, rather than return an unjustified value.

---

**Source.** The rules are defined in `src/cpp/cseval/cseval.hpp` and its complex
counterpart; every statement above is fixed by a test in
`tests/test_derivatives_audit.py`.

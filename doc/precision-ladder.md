# The precision ladder

## Where it came from

This started inside a big project for simulating physical processes. From around
2011 I kept wanting to reuse the C++ work over in Python. Eventually I cut away
everything that did not belong and concentrated on small, local modules. One of
them became `formula`: hand it a string with an expression and it evaluates it to
more than 16 digits. Arbitrary-precision libraries for Python may have already existed by
then. But this code had grown on its own, and carrying Boost's muscle and speed
into Python through pybind11 looked like a fine experiment. So I tried it.

We use Boost in headers only. That is the simple road: nothing extra to compile
and link, and no licensing tangle, since the Boost license is permissive. The
backends that would hand us true runtime precision (GMP, MPFR, MPC) are separate
libraries you must build and link, and they carry the LGPL. Skip them and two
things follow: we stay header-only and permissive, and we give up arbitrary
precision. What is left is the ladder. For every physical-modeling problem I have
thrown at it, the ladder has been more than enough.

## Compared to the alternatives

How does it stand in 2026?

| library | precision | multiply backend | license | string parser | derivatives |
| ------- | --------- | ---------------- | ------- | ------------- | ----------- |
| `mpmath` | runtime, arbitrary | pure Python (gmpy2 optional) | BSD | no | no |
| `gmpy2` | runtime, arbitrary | GMP / MPFR / MPC, FFT | LGPL | no | no |
| `python-flint` | runtime, arbitrary | FLINT / Arb, FFT, ball arithmetic | LGPL | no | no |
| `SymPy` | runtime, arbitrary | mpmath | BSD | yes (symbolic) | yes (symbolic) |
| `decimal` | runtime, arbitrary | pure Python | PSF | no | no |
| **`formula`** | **ladder, up to 8192** | **Boost cpp_dec_float, Karatsuba** | **BSL** | **yes (numeric)** | **yes (numeric)** |

Every alternative offers true runtime precision. The fast ones (gmpy2, python-flint)
use FFT multiplication and beat our Karatsuba at large N, but they pull in GMP,
MPFR or FLINT as compiled LGPL dependencies. `formula` trades that away on
purpose: a free-form string parser plus Boost's `cpp_dec_float`, shipped
header-only and permissive, with a fixed ladder instead of unbounded precision.
That is the niche. Not the fastest at a million digits, but self-contained, and
plenty for the job it was built for.

Where `formula` specifically wins:

1. **Numeric derivatives from a string.** `Solver("x^2 + sin(x)")({"x": "1"}, derivative="x")`
   returns the derivative as a string, computed analytically from the expression
   tree in C++. No other numeric library in the table does this. SymPy can, but
   it is a symbolic engine: it carries a CAS, a pattern-matcher, and a full
   expression system. `formula` just parses the string once and differentiates the
   tree.

2. **Automatic real/complex detection.** Write `Formula("x + i*y")` and the parser
   detects the imaginary unit and switches to complex arithmetic by itself. One
   expression type, two number kinds, no manual switching. The imaginary unit is
   also configurable: `j`, `k`, any single character.

3. **Zero runtime dependencies, permissive license.** Ships as a wheel. No
   libgmp, libmpfr, or libmpc to install or link. The Boost headers are bundled;
   the license is BSL-1.0. Dropping it into a project does not introduce LGPL
   obligations or a compiled dependency chain.

Speed is the other axis. Measured against the Python field, `formula` is
mid-pack on raw arithmetic and far slower on high-precision transcendentals; it
wins on parsing and at modest precision. Full numbers and methodology:
[benchmarks.md](benchmarks.md).

## The problem

We want to evaluate a formula to whatever accuracy the user asks for. Thirty
digits, three hundred digits, thirty thousand digits. The catch is that the type
holding the number, `cpp_dec_float<N>`, fixes N at compile time. N is a template
parameter. There is no single "big number" type you can dial at runtime.

That is the whole reason for the ladder. Since we cannot mint a type per request,
we build a fixed set of types ahead of time and, at runtime, round the request up
to the nearest one we have. Every rung is a real C++ type baked into the binary,
collected into a `boost::variant`, and picked by a switch.

Always round up, never down. Ask for 100 digits and you get a type with at least
100. Giving you fewer would be lying about the accuracy.

## Why these rungs

Choosing the rungs fixes three things at once, and they pull against each other:

* the **range**, lowest to highest precision;
* the **count**, how many rungs, which is how many full evaluator types the
  compiler must instantiate, which is build time and memory;
* the **overshoot**, the extra work you do when a request lands between two rungs.

For a geometric ladder all three are tied to one number, the ratio `r` between
neighbours. Wide range plus few rungs forces a big `r`, and a big `r` means more
wasted digits. You cannot win all three. You pick.

We picked doubling. `r = 2`. Each rung is twice the one below:

```
     16 |  default zone
     24 |* (library default)
     32 |
     64 |
    128 |
    256 |
    512 |
   1024 |
   2048 |
   4096 |
   8192 |  ceiling
```

The reasons, plainly:

1. **Doubling caps the waste.** Land between two rungs and you overshoot by at
   most 2x. Predictable, and 2x is cheap insurance against under-delivering.
2. **The numbers are the ones people type.** Nobody asks for 600 digits on
   purpose; they ask for 512 or 1024.
3. **The 24 is the odd one out, on purpose.** 24 is the library default, so it
   gets its own rung and the common case never rounds. Below 64 the steps go
   16, 24, 32, which is exactly where most real work lives (a double is about 16
   digits), so a bit of extra density there is free and handy.

The range spans 16 to 8192, a factor of 512. That is 11 rungs: 11 evaluator
instantiations for the real types and 11 for the complex ones.

## Why 8192 is the ceiling

Three facts decide it.

First, `pi` is a baked-in constant with 8198 digits. Every rung must be fully
covered by it, or `pi` silently zero-pads past the constant and the rung lies
about its own precision. A higher ceiling means baking in a longer constant.

Second, rungs are not free at build time. Each one instantiates a full real and
complex evaluator; the compiler's memory peak grows with both the count of
rungs and the size of the top types.

Third, the arithmetic itself. Multiplication in this Boost has no FFT path:
schoolbook below ~1024 digits, Karatsuba above, which is `O(N^1.585)`. We
measured up to a million digits (full table in
[benchmark-dec-float-scaling.md](benchmark-dec-float-scaling.md)): at 8192 a
multiply is ~0.4 ms and a formula evaluates instantly; at 262144 a single
multiply is already a tenth of a second. Past the ceiling the right move is not
a bigger rung but a different number type, one with FFT multiply (MPFR or GMP).

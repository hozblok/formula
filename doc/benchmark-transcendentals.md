# Benchmark: transcendentals

Raw arithmetic is not where real formulas spend their time — transcendental
functions are, and Boost's are slow. Evaluating one expression at a point, to
1000 digits:

| 1000 digits | `formula` | `gmpy2`-backed `mpmath` |
| ----------- | --------: | ----------------------: |
| `x^2`       |   0.04 ms |                0.012 ms |
| `sin(x)`    |   12.5 ms |                 0.12 ms |

The arithmetic gap is 3x; the `sin` gap is about 100x. For high-precision work
the transcendental functions, not multiplication, are the real ceiling on speed.
`SymPy` evaluates through `gmpy2`-backed `mpmath`, so it inherits that speed.

## Reproduce it

`pip install sympy gmpy2` (so `mpmath`, pulled by `sympy`, uses `gmpy2`), then
run with `formula` importable (`PYTHONPATH=src`):

```python
import time
from formula import Solver
from sympy import sympify, Symbol

def ms(fn, budget=0.4):                       # best per-call time, milliseconds
    fn()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n): fn()
        dt = time.perf_counter() - t0
        if dt > 0.05 or n > 1 << 22: break
        n *= 4
    best = float("inf")
    for _ in range(max(3, int(budget / dt))):
        t0 = time.perf_counter()
        for _ in range(n): fn()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1000

P = 1000; x = Symbol("x")
for fe, se in (("x^2", "x**2"), ("sin(x)", "sin(x)")):
    s = Solver(fe, precision=P); k = [0]
    def f(): k[0] += 1; return s({"x": str(k[0])})
    e = sympify(se); m = [0]
    def g(): m[0] += 1; return e.evalf(P, subs={x: m[0]})
    print(fe, ms(f), ms(g))
```

Do not set `MPMATH_NOGMPY` here — `SymPy` should evaluate through `gmpy2` as it
normally would.

## Environment

* CPU: Intel Xeon Gold 6230 @ 2.10 GHz, single threaded
* RAM: 14 GB
* OS: Ubuntu 24.04.4 LTS
* `formula` built with g++ 13.3.0, `-O2`, `-std=c++17`, Boost 1.79.0
* Python 3.12.3; sympy 1.14.0, mpmath 1.3.0, gmpy2 2.3.0
* Measured: 2026-05-22

## Contact

Ivan Ergunov <hoz.blok@gmail.com>

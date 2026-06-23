🇬🇧 **English** · [🇷🇺 Русский](benchmark-end-to-end-sympy.ru.md)

# Benchmark: end to end vs SymPy

The job `formula` is built for: hand it the string `x^2 + sin(x)`, get the value,
and the derivative, at a point to N digits. Only `SymPy` does the same from a
string, so it is the only fair opponent. Parse done once, then evaluated at
varying points.

| digits | `formula` value | `SymPy` value | `formula` d/dx | `SymPy` d/dx |
| -----: | --------------: | ------------: | -------------: | -----------: |
|     24 |           12 µs |         49 µs |          37 µs |        64 µs |
|    100 |           79 µs |         53 µs |         487 µs |        72 µs |
|   1000 |         12.3 ms |       0.15 ms |          96 ms |      0.16 ms |

Parsing the string once: `formula` 10 µs, `SymPy` 478 µs — about 48x.

The crossover is around 50 digits. Below it `formula` wins: lighter per-call
overhead and a far cheaper parser. Above it `SymPy` wins, and once a
transcendental is in the expression it wins by hundreds, because under the hood
it evaluates through `gmpy2`. `formula`'s edge is light weight, fast parse, and
zero dependencies at modest precision — not high-precision throughput.

## Reproduce it

`pip install sympy gmpy2` (so `mpmath`, pulled by `sympy`, uses `gmpy2`), then
run with `formula` importable (`PYTHONPATH=src`):

```python
import time
from formula import Solver
from sympy import sympify, diff, Symbol

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

x = Symbol("x"); e = sympify("x**2 + sin(x)"); de = diff(e, x)
for P in (24, 100, 1000):
    s = Solver("x^2 + sin(x)", precision=P); k = [0]
    def fv(): k[0] += 1; return s({"x": str(k[0])})
    def sv(): k[0] += 1; return e.evalf(P, subs={x: k[0]})
    def fd(): k[0] += 1; return s({"x": str(k[0])}, derivative="x")
    def sd(): k[0] += 1; return de.evalf(P, subs={x: k[0]})
    print(P, ms(fv), ms(sv), ms(fd), ms(sd))

# parse once
print("parse", ms(lambda: Solver("x^2 + sin(x)", precision=100)),
      ms(lambda: (lambda ex: (ex, diff(ex, x)))(sympify("x**2 + sin(x)"))))
```

## Environment

* CPU: Intel Xeon Gold 6230 @ 2.10 GHz, single threaded
* RAM: 14 GB
* OS: Ubuntu 24.04.4 LTS
* `formula` built with g++ 13.3.0, `-O2`, `-std=c++17`, Boost 1.79.0
* Python 3.12.3; sympy 1.14.0, mpmath 1.3.0, gmpy2 2.3.0
* Measured: 2026-05-22

## Contact

Ivan Ergunov <hoz.blok@gmail.com>

# Benchmark: raw arithmetic vs the Python field

One multiply of two full-precision numbers, matched on decimal digits across
libraries. The feature comparison is in
[the precision ladder](precision-ladder.md#compared-to-the-alternatives); this is
the speed side. Single thread, best of many runs. `mpmath` runs with its `gmpy2`
backend off (`MPMATH_NOGMPY=1`), to show what an uncompiled backend costs; leave
it on and `mpmath` rides `gmpy2`.

| digits | `formula` | `gmpy2` | `decimal` | `mpmath` |
| -----: | --------: | ------: | --------: | -------: |
|    256 |    2.3 µs | 0.28 µs |    2.0 µs |   2.6 µs |
|   1024 |     15 µs |  1.2 µs |     28 µs |    15 µs |
|   4096 |    134 µs |   10 µs |    454 µs |   135 µs |
|  16384 |   1.45 ms |   80 µs |   1.54 ms |  1.18 ms |
|  65536 |   11.7 ms | 0.62 ms |    7.3 ms |  11.4 ms |

(multiply; for `formula` a divide costs ~4x a multiply, same ordering.)

The 16384 and 65536 rows were measured on a build with extra rungs; the shipped
ladder now tops out at 8192 (see
[Why 8192 is the ceiling](precision-ladder.md#why-8192-is-the-ceiling)).

Up to ~1000 digits every library is sub-0.1 ms and the choice does not matter.
Past that `gmpy2`'s FFT pulls away — about 19x at 65536 digits — while `formula`
sits level with pure-Python `mpmath` and stdlib `decimal`, all three
Karatsuba/NTT-class. On raw arithmetic `formula` is mid-pack, not fast.

## Reproduce it

`pip install mpmath gmpy2`, then run with `formula` importable
(`PYTHONPATH=src`):

```python
import math, time
import gmpy2, mpmath
from gmpy2 import mpfr
from mpmath import mp, mpf
from decimal import Decimal, getcontext
from formula import Number

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

for P in (256, 1024, 4096, 8192):
    a = Number("1/3", P)._value; b = Number("1/7", P)._value
    gmpy2.get_context().precision = math.ceil(P * math.log2(10))
    g1, g2 = mpfr(1) / 3, mpfr(1) / 7
    mp.dps = P; m1, m2 = mpf(1) / 3, mpf(1) / 7
    getcontext().prec = P; d1, d2 = Decimal(1) / Decimal(3), Decimal(1) / Decimal(7)
    print(P, ms(lambda: a * b), ms(lambda: g1 * g2),
             ms(lambda: d1 * d2), ms(lambda: m1 * m2))   # swap * for / to time divide
```

Set `MPMATH_NOGMPY=1` in the environment for the pure-Python `mpmath` column.

## Environment

* CPU: Intel Xeon Gold 6230 @ 2.10 GHz, single threaded
* RAM: 14 GB
* OS: Ubuntu 24.04.4 LTS
* `formula` built with g++ 13.3.0, `-O2`, `-std=c++17`, Boost 1.79.0
* Python 3.12.3; gmpy2 2.3.0, mpmath 1.3.0
* Measured: 2026-05-22

## Contact

Ivan Ergunov <hoz.blok@gmail.com>

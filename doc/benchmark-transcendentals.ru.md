[🇬🇧 English](benchmark-transcendentals.md) · 🇷🇺 **Русский**

# Бенчмарк: трансцендентные функции

Основное время реальные формулы тратят не на простую арифметику, а на
трансцендентные функции, и в Boost они медленные. Вычисление одного выражения
в точке с точностью 1000 знаков:

| 1000 знаков | `formula` | `mpmath` на базе `gmpy2` |
| ----------- | --------: | -----------------------: |
| `x^2`       |   0.04 ms |                 0.012 ms |
| `sin(x)`    |   12.5 ms |                  0.12 ms |

Разрыв по арифметике — 3x; разрыв по `sin` — около 100x. Для работы с высокой
точностью настоящий потолок скорости — это трансцендентные функции, а не
умножение. `SymPy` вычисляет через `mpmath` на базе `gmpy2`, поэтому наследует
эту скорость.

## Как воспроизвести

`pip install sympy gmpy2` (чтобы `mpmath`, подтягиваемый `sympy`, использовал
`gmpy2`), затем запустите с импортируемым `formula` (`PYTHONPATH=src`):

```python
import time
from formula import Solver
from sympy import sympify, Symbol

def ms(fn, budget=0.4):                       # лучшее время на вызов, миллисекунды
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

Не задавайте здесь `MPMATH_NOGMPY` — `SymPy` должен вычислять через `gmpy2`, как
он это делает обычно.

## Окружение

* CPU: Intel Xeon Gold 6230 @ 2.10 GHz, однопоточно
* RAM: 14 GB
* OS: Ubuntu 24.04.4 LTS
* `formula` собрана с g++ 13.3.0, `-O2`, `-std=c++17`, Boost 1.79.0
* Python 3.12.3; sympy 1.14.0, mpmath 1.3.0, gmpy2 2.3.0
* Измерено: 2026-05-22

## Контакты

Ivan Ergunov <hoz.blok@gmail.com>

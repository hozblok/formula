[🇬🇧 English](benchmark-end-to-end-sympy.md) · 🇷🇺 **Русский**

# Бенчмарк: сквозной сценарий против SymPy

Задача, под которую создан `formula`: передать строку `x^2 + sin(x)` и получить значение,
а также производную в точке с N знаками. Только `SymPy` делает то же самое из
строки, поэтому это единственный честный соперник. Разбор выполняется один раз, затем
происходит вычисление в разных точках.

| знаков | `formula` значение | `SymPy` значение | `formula` d/dx | `SymPy` d/dx |
| -----: | -----------------: | ---------------: | -------------: | -----------: |
|     24 |               12 µs |            49 µs |          37 µs |        64 µs |
|    100 |               79 µs |            53 µs |         487 µs |        72 µs |
|   1000 |             12.3 ms |          0.15 ms |          96 ms |      0.16 ms |

Разбор строки один раз: `formula` 10 µs, `SymPy` 478 µs — примерно в 48 раз.

Точка перелома — около 50 знаков. Ниже неё выигрывает `formula`: меньше накладных
расходов на вызов и куда более дешёвый парсер. Выше неё выигрывает `SymPy`, и как только
в выражении появляется трансцендентная функция, он выигрывает в сотни раз, потому что под капотом
вычисляет через `gmpy2`. Преимущество `formula` — лёгкость, быстрый разбор и
нулевые зависимости при умеренной точности, а не пропускная способность при высокой точности.

## Как воспроизвести

`pip install sympy gmpy2` (чтобы `mpmath`, подтягиваемый `sympy`, использовал `gmpy2`), затем
запустить с импортируемым `formula` (`PYTHONPATH=src`):

```python
import time
from formula import Solver
from sympy import sympify, diff, Symbol

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

x = Symbol("x"); e = sympify("x**2 + sin(x)"); de = diff(e, x)
for P in (24, 100, 1000):
    s = Solver("x^2 + sin(x)", precision=P); k = [0]
    def fv(): k[0] += 1; return s({"x": str(k[0])})
    def sv(): k[0] += 1; return e.evalf(P, subs={x: k[0]})
    def fd(): k[0] += 1; return s({"x": str(k[0])}, derivative="x")
    def sd(): k[0] += 1; return de.evalf(P, subs={x: k[0]})
    print(P, ms(fv), ms(sv), ms(fd), ms(sd))

# разбор один раз
print("parse", ms(lambda: Solver("x^2 + sin(x)", precision=100)),
      ms(lambda: (lambda ex: (ex, diff(ex, x)))(sympify("x**2 + sin(x)"))))
```

## Окружение

* CPU: Intel Xeon Gold 6230 @ 2.10 GHz, однопоточно
* RAM: 14 GB
* OS: Ubuntu 24.04.4 LTS
* `formula` собран с g++ 13.3.0, `-O2`, `-std=c++17`, Boost 1.79.0
* Python 3.12.3; sympy 1.14.0, mpmath 1.3.0, gmpy2 2.3.0
* Измерено: 2026-05-22

## Контакты

Ivan Ergunov <hoz.blok@gmail.com>

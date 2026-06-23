[🇬🇧 English](benchmark-raw-arithmetic.md) · 🇷🇺 **Русский**

# Бенчмарк: чистая арифметика против Python-поля

Одно умножение двух чисел полной точности, согласованное по десятичным разрядам
между библиотеками. Сравнение возможностей приведено в
[лестнице точности](precision-ladder.ru.md#compared-to-the-alternatives); здесь же
речь о скорости. Один поток, лучший результат из множества прогонов. `mpmath`
работает с выключенным бэкендом `gmpy2` (`MPMATH_NOGMPY=1`), чтобы показать, во
сколько обходится нескомпилированный бэкенд; оставьте его включённым — и `mpmath`
поедет на `gmpy2`.

| разрядов | `formula` | `gmpy2` | `decimal` | `mpmath` |
| -------: | --------: | ------: | --------: | -------: |
|      256 |    2.3 µs | 0.28 µs |    2.0 µs |   2.6 µs |
|     1024 |     15 µs |  1.2 µs |     28 µs |    15 µs |
|     4096 |    134 µs |   10 µs |    454 µs |   135 µs |
|    16384 |   1.45 ms |   80 µs |   1.54 ms |  1.18 ms |
|    65536 |   11.7 ms | 0.62 ms |    7.3 ms |  11.4 ms |

(умножение; для `formula` деление стоит ~4x умножения, тот же порядок.)

Строки 16384 и 65536 измерены на сборке с дополнительными ступенями; поставляемая
лестница теперь упирается в потолок на 8192 (см.
[Почему 8192 — это потолок](precision-ladder.ru.md#why-8192-is-the-ceiling)).

Вплоть до ~1000 разрядов любая библиотека укладывается в доли миллисекунды (меньше
0.1 мс), и выбор не имеет значения. Дальше FFT в `gmpy2` отрывается — примерно в
19x на 65536 разрядах — тогда как `formula` идёт вровень с чисто-питоновским
`mpmath` и `decimal` из стандартной библиотеки, все трое класса Karatsuba/NTT. На
чистой арифметике `formula` середнячок, не из быстрых.

## Как воспроизвести

`pip install mpmath gmpy2`, затем запустите так, чтобы `formula` импортировался
(`PYTHONPATH=src`):

```python
import math, time
import gmpy2, mpmath
from gmpy2 import mpfr
from mpmath import mp, mpf
from decimal import Decimal, getcontext
from formula import Number

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

for P in (256, 1024, 4096, 8192):
    a = Number("1/3", P)._value; b = Number("1/7", P)._value
    gmpy2.get_context().precision = math.ceil(P * math.log2(10))
    g1, g2 = mpfr(1) / 3, mpfr(1) / 7
    mp.dps = P; m1, m2 = mpf(1) / 3, mpf(1) / 7
    getcontext().prec = P; d1, d2 = Decimal(1) / Decimal(3), Decimal(1) / Decimal(7)
    print(P, ms(lambda: a * b), ms(lambda: g1 * g2),
             ms(lambda: d1 * d2), ms(lambda: m1 * m2))   # замените * на / для замера деления
```

Установите `MPMATH_NOGMPY=1` в окружении для колонки чисто-питоновского `mpmath`.

## Окружение

* CPU: Intel Xeon Gold 6230 @ 2.10 GHz, один поток
* RAM: 14 GB
* OS: Ubuntu 24.04.4 LTS
* `formula` собрана с g++ 13.3.0, `-O2`, `-std=c++17`, Boost 1.79.0
* Python 3.12.3; gmpy2 2.3.0, mpmath 1.3.0
* Измерено: 2026-05-22

## Контакты

Ivan Ergunov <hoz.blok@gmail.com>

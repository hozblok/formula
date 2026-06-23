[🇬🇧 English](benchmark-dec-float-scaling.md) · 🇷🇺 **Русский**

# Бенчмарк: масштабирование cpp_dec_float

Как `cpp_dec_float` из Boost масштабируется с ростом точности — данные, стоящие за
потолком лестницы на 8192 (см.
[Почему потолок именно 8192](precision-ladder.ru.md#почему-потолок-именно-8192)). Без
FFT-пути: школьное умножение ниже ~1024 цифр, Карацуба выше, `O(N^1.585)`.

| точность | байт/число | одно умножение | одно деление | вердикт |
| --------- | -----------: | -----------: | ---------: | ------- |
| 1024      | 0.5 KB       | 0.015 ms     | 0.069 ms   | быстро |
| 8192      | 4 KB         | 0.42 ms      | 1.8 ms     | быстро, **потолок** |
| 65536     | 32 KB        | 11.7 ms      | 44.9 ms    | пригодно |
| 262144    | 128 KB       | 95.7 ms      | 402.6 ms   | пригодно |
| 1048576   | 512 KB       | 907 ms       | 3.6 s      | медленно |

Однопоточно, `-O2`. Удвоение числа цифр замедляет умножение примерно в 3 раза; одно
деление стоит примерно как 4 умножения.

## Как воспроизвести

Сохраните как `bench_dec.cpp`:

```cpp
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <chrono>
#include <cstdio>
#include <memory>
using namespace boost::multiprecision;
using clk = std::chrono::steady_clock;

template <unsigned N>
void bench(const char* tag) {
  using T = number<cpp_dec_float<N>, et_off>;
  auto a = std::make_unique<T>(1);
  auto b = std::make_unique<T>(1);
  *a = 1; *a /= 3;          // 0.3333... заполняет все лимбы
  *b = 1; *b /= 7;
  auto c = std::make_unique<T>();
  int reps = N <= 8192 ? 200 : (N <= 65536 ? 20 : 3);
  auto t0 = clk::now();
  for (int i = 0; i < reps; ++i) { *c = (*a) * (*b); *a += 1; }
  auto t1 = clk::now();
  double ms_mul = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
  reps = N <= 8192 ? 100 : (N <= 65536 ? 10 : 2);
  t0 = clk::now();
  for (int i = 0; i < reps; ++i) { *c = (*a) / (*b); *a += 1; }
  t1 = clk::now();
  double ms_div = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
  printf("%-9s  sizeof=%8zu B (%6.0f KB)   mul=%9.3f ms   div=%9.3f ms\n",
         tag, sizeof(T), sizeof(T) / 1024.0, ms_mul, ms_div);
}

int main() {
  bench<1024>("1024");
  bench<8192>("8192");
  bench<65536>("65536");
  bench<262144>("262144");
  bench<1048576>("1048576");
  return 0;
}
```

Соберите и запустите (направьте `-I` на ваши заголовки Boost):

```sh
g++ -O2 -std=c++17 -I boost_headers bench_dec.cpp -o bench_dec
./bench_dec
```

## Окружение

* CPU: Intel Xeon Gold 6230 @ 2.10 GHz, однопоточно
* RAM: 14 GB
* ОС: Ubuntu 24.04.4 LTS
* Компилятор: g++ 13.3.0, `-O2`, `-std=c++17`
* Boost: 1.79.0
* Измерено: 2026-05-22

## Контакты

Ivan Ergunov <hoz.blok@gmail.com>

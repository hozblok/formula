🇬🇧 **English** · [🇷🇺 Русский](benchmark-dec-float-scaling.ru.md)

# Benchmark: cpp_dec_float scaling

How Boost's `cpp_dec_float` scales with precision — the data behind the ladder's
ceiling at 8192 (see
[Why 8192 is the ceiling](precision-ladder.md#why-8192-is-the-ceiling)). No
FFT path: schoolbook below ~1024 digits, Karatsuba above, `O(N^1.585)`.

| precision | bytes/number | one multiply | one divide | verdict |
| --------- | -----------: | -----------: | ---------: | ------- |
| 1024      | 0.5 KB       | 0.015 ms     | 0.069 ms   | fast |
| 8192      | 4 KB         | 0.42 ms      | 1.8 ms     | fast, **ceiling** |
| 65536     | 32 KB        | 11.7 ms      | 44.9 ms    | usable |
| 262144    | 128 KB       | 95.7 ms      | 402.6 ms   | usable |
| 1048576   | 512 KB       | 907 ms       | 3.6 s      | slow |

Single threaded, `-O2`. Double the digits and a multiply gets ~3x slower; a
divide costs ~4x a multiply.

## Reproduce it

Save as `bench_dec.cpp`:

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
  *a = 1; *a /= 3;          // 0.3333... fills all limbs
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

Build and run (point `-I` at your Boost headers):

```sh
g++ -O2 -std=c++17 -I boost_headers bench_dec.cpp -o bench_dec
./bench_dec
```

## Environment

* CPU: Intel Xeon Gold 6230 @ 2.10 GHz, single threaded
* RAM: 14 GB
* OS: Ubuntu 24.04.4 LTS
* Compiler: g++ 13.3.0, `-O2`, `-std=c++17`
* Boost: 1.79.0
* Measured: 2026-05-22

## Contact

Ivan Ergunov <hoz.blok@gmail.com>

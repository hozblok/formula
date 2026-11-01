// Complex power, one rule for the bindings and the engine: a real exponent of
// integral value within 2**62 by repeated squaring, except a zero base with a
// negative one; anything else exp(b log a).
#pragma once

template <class Complex>
Complex complex_pow(const Complex &a, const Complex &b) {
  const auto &e = b.real();
  if (b.imag() == 0 && e == floor(e) && abs(e) <= (1LL << 62) &&
      !(a == 0 && e < 0)) {
    const long long n = e.template convert_to<long long>();
    unsigned long long m = n < 0 ? -n : n;
    Complex r = 1, s = a;
    for (; m; m >>= 1) {
      if (m & 1) r = r * s;
      if (m > 1) s = s * s;
    }
    return n < 0 ? Complex(1) / r : r;
  }
  return pow(a, b);
}

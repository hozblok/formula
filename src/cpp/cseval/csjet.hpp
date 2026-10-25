#ifndef CSJET_H
#define CSJET_H

#include <boost/multiprecision/number.hpp>
#include <cstdint>
#include <vector>

constexpr unsigned kMaxDerivativeOrder = 5;

template <typename T>
constexpr bool is_complex_number =
    boost::multiprecision::number_category<T>::value ==
    boost::multiprecision::number_kind_complex;

// Truncated Taylor number in nilpotent e1..eN (ek^2 = 0): c[m] is the
// coefficient at the product of the e with bits in m (c[0] value).
template <typename T>
struct csjet {
  std::vector<T> c;
  // Bit m: c[m] depends on the seeded variables by the formula structure.
  uint32_t support;

  csjet(unsigned order, const T &value)
      : c(std::size_t(1) << order, T(0)), support(1u) {
    c[0] = value;
  }

  unsigned order() const {
    unsigned n = 0;
    while ((std::size_t(1) << n) < c.size()) ++n;
    return n;
  }

  bool has(unsigned m) const { return (support >> m) & 1u; }

  // Constant w.r.t. every seeded variable.
  bool is_constant() const { return support == 1u; }
};

template <typename T>
csjet<T> operator+(csjet<T> a, const csjet<T> &b) {
  for (std::size_t m = 0; m < a.c.size(); ++m) a.c[m] = a.c[m] + b.c[m];
  a.support |= b.support;
  return a;
}

template <typename T>
csjet<T> operator-(csjet<T> a, const csjet<T> &b) {
  for (std::size_t m = 0; m < a.c.size(); ++m) a.c[m] = a.c[m] - b.c[m];
  a.support |= b.support;
  return a;
}

// r[m] = sum over submasks s of m: a[s] * b[m ^ s].
template <typename T>
csjet<T> operator*(const csjet<T> &a, const csjet<T> &b) {
  csjet<T> r(a.order(), T(0));
  r.support = 0;
  for (unsigned m = 0; m < r.c.size(); ++m) {
    for (unsigned s = m;; s = (s - 1) & m) {
      if (a.has(s) && b.has(m ^ s)) {
        r.c[m] = r.c[m] + a.c[s] * b.c[m ^ s];
        r.support |= 1u << m;
      }
      if (s == 0) break;
    }
  }
  return r;
}

// r = a / b from r * b = a: r[m] = (a[m] - sum over proper submasks s of m:
// r[s] * b[m ^ s]) / b[0].
template <typename T>
csjet<T> operator/(const csjet<T> &a, const csjet<T> &b) {
  csjet<T> r(a.order(), a.c[0] / b.c[0]);
  for (unsigned m = 1; m < r.c.size(); ++m) {
    T s = a.c[m];
    bool present = a.has(m);
    for (unsigned t = (m - 1) & m;; t = (t - 1) & m) {
      if (r.has(t) && b.has(m ^ t)) {
        s = s - r.c[t] * b.c[m ^ t];
        present = true;
      }
      if (t == 0) break;
    }
    if (present) {
      r.c[m] = s / b.c[0];
      r.support |= 1u << m;
    }
  }
  return r;
}

// f(x) by Faa di Bruno: r[m] = sum_k f^(k)(x0) * (sum over k-block partitions
// of m of prod x[block]); deriv(k) is called for k = 1, 2, ... while present.
template <typename T, typename Deriv>
csjet<T> compose(const csjet<T> &x, const T &f0, const Deriv &deriv) {
  const unsigned size = x.c.size();
  csjet<T> r(x.order(), f0);
  std::vector<T> p(x.c), q(size, T(0));
  uint32_t present = x.support & ~1u;
  for (unsigned k = 1; present; ++k) {
    T fk = deriv(k);
    for (unsigned m = 1; m < size; ++m) {
      if (!((present >> m) & 1u)) continue;
      r.c[m] = r.has(m) ? r.c[m] + fk * p[m] : fk * p[m];
      r.support |= 1u << m;
    }
    // P_{k+1}[m]: the block s holding the lowest bit of m times P_k[m ^ s].
    uint32_t next = 0;
    for (unsigned m = 1; m < size; ++m) {
      unsigned low = m & (~m + 1);
      for (unsigned s = (m - 1) & m; s; s = (s - 1) & m) {
        if ((s & low) && x.has(s) && ((present >> (m ^ s)) & 1u)) {
          T term = x.c[s] * p[m ^ s];
          q[m] = ((next >> m) & 1u) ? q[m] + term : term;
          next |= 1u << m;
        }
      }
    }
    p.swap(q);
    present = next;
  }
  return r;
}

//+ f^(k)(a), k = 0..n, of the one-argument functions:
// Taylor coefficients of a / b up to degree n.
template <typename T>
std::vector<T> taylor_div(const std::vector<T> &a, const std::vector<T> &b,
                          unsigned n) {
  std::vector<T> c(n + 1, T(0));
  for (unsigned k = 0; k <= n; ++k) {
    T s = k < a.size() ? a[k] : T(0);
    for (unsigned j = 1; j <= k && j < b.size(); ++j) s = s - b[j] * c[k - j];
    c[k] = s / b[0];
  }
  return c;
}

// Taylor coefficients of s^r up to degree n, w0 = s[0]^r on the caller's
// branch.
template <typename T>
std::vector<T> taylor_pow(const std::vector<T> &s, const T &r, const T &w0,
                          unsigned n) {
  std::vector<T> w(n + 1, T(0));
  w[0] = w0;
  for (unsigned k = 1; k <= n; ++k) {
    T acc(0);
    for (unsigned j = 1; j <= k && j < s.size(); ++j) {
      acc = acc + (r * int(j) - int(k - j)) * s[j] * w[k - j];
    }
    w[k] = acc / (s[0] * int(k));
  }
  return w;
}

// f^(k) = (k-1)! g[k-1] from the Taylor coefficients g of f'.
template <typename T>
std::vector<T> from_derivative_series(const T &f0, const std::vector<T> &g) {
  std::vector<T> d(g.size() + 1);
  d[0] = f0;
  T factorial(1);
  for (unsigned k = 1; k < d.size(); ++k) {
    d[k] = g[k - 1] * factorial;
    factorial = factorial * int(k);
  }
  return d;
}

template <typename T>
std::vector<T> d_cycle(const std::vector<T> &cycle, unsigned n) {
  std::vector<T> d(n + 1);
  for (unsigned k = 0; k <= n; ++k) d[k] = cycle[k % cycle.size()];
  return d;
}

template <typename T>
std::vector<T> d_zero(const T &, unsigned n) {
  return std::vector<T>(n + 1, T(0));
}

template <typename T>
std::vector<T> d_sin(const T &a, unsigned n) {
  T s = sin(a), c = cos(a);
  return d_cycle<T>({s, c, -s, -c}, n);
}

template <typename T>
std::vector<T> d_cos(const T &a, unsigned n) {
  T s = sin(a), c = cos(a);
  return d_cycle<T>({c, -s, -c, s}, n);
}

template <typename T>
std::vector<T> d_exp(const T &a, unsigned n) {
  return d_cycle<T>({exp(a)}, n);
}

template <typename T>
std::vector<T> d_sinh(const T &a, unsigned n) {
  return d_cycle<T>({sinh(a), cosh(a)}, n);
}

template <typename T>
std::vector<T> d_cosh(const T &a, unsigned n) {
  return d_cycle<T>({cosh(a), sinh(a)}, n);
}

template <typename T>
std::vector<T> d_log(const T &a, unsigned n) {
  std::vector<T> d(n + 1);
  d[0] = log(a);
  T p = T(1) / a;
  for (unsigned k = 1; k <= n; ++k) {
    d[k] = p;
    p = p * (-int(k)) / a;
  }
  return d;
}

template <typename T>
std::vector<T> d_log2(const T &a, unsigned n) {
  std::vector<T> d = d_log(a, n);
  T ln2 = log(T(2));
  for (T &v : d) v = v / ln2;
  return d;
}

template <typename T>
std::vector<T> d_log10(const T &a, unsigned n) {
  std::vector<T> d = d_log(a, n);
  T ln10 = log(T(10));
  for (T &v : d) v = v / ln10;
  return d;
}

// (1/2)(1/2 - 1)...(1/2 - k + 1) a^(1/2 - k).
template <typename T>
std::vector<T> d_sqrt(const T &a, unsigned n) {
  std::vector<T> d(n + 1);
  d[0] = sqrt(a);
  for (unsigned k = 1; k <= n; ++k) {
    d[k] = d[k - 1] * (T(1) / 2 - int(k - 1)) / a;
  }
  return d;
}

// Taylor v of tan (sign = 1) or tanh (sign = -1) from v0 = f(a), v1 = f'(a):
// k v_k = sign * sum_j v_j v_(k-1-j) for k >= 2, by f' = 1 + sign f^2.
template <typename T>
std::vector<T> d_tan_like(const T &v0, const T &v1, int sign, unsigned n) {
  std::vector<T> v(n + 2, T(0)), d(n + 1);
  v[0] = v0;
  v[1] = v1;
  d[0] = v0;
  T factorial(1);
  for (unsigned k = 1; k <= n; ++k) {
    if (k >= 2) {
      T s(0);
      for (unsigned j = 0; j < k; ++j) s = s + v[j] * v[k - 1 - j];
      v[k] = sign * s / int(k);
    }
    factorial = factorial * int(k);
    d[k] = v[k] * factorial;
  }
  return d;
}

// v1 = 1/cos^2, 1/cosh^2 as in the first derivative.
template <typename T>
std::vector<T> d_tan(const T &a, unsigned n) {
  T c = cos(a);
  return d_tan_like(tan(a), T(1) / (c * c), 1, n);
}

template <typename T>
std::vector<T> d_tanh(const T &a, unsigned n) {
  T c = cosh(a);
  return d_tan_like(tanh(a), T(1) / (c * c), -1, n);
}

// Real: atan' = 1/(1 + u^2) at u = a + t. Complex, by the simple poles +-i:
// atan^(k)(a) = (-1)^(k-1) (k-1)! / (2i) * ((a - i)^-k - (a + i)^-k).
template <typename T>
std::vector<T> d_atan(const T &a, unsigned n) {
  if constexpr (is_complex_number<T>) {
    T i("0.0", "1.0"), half_over_i("0.0", "-0.5");
    T p = T(1) / (a - i), q = T(1) / (a + i), pk = p, qk = q, c(1);
    std::vector<T> d(n + 1);
    d[0] = atan(a);
    for (unsigned k = 1; k <= n; ++k) {
      d[k] = c * ((pk - qk) * half_over_i);
      pk = pk * p;
      qk = qk * q;
      c = c * (-int(k));
    }
    return d;
  } else {
    std::vector<T> s = {1 + a * a, 2 * a, T(1)};
    return from_derivative_series(atan(a), taylor_div({T(1)}, s, n - 1));
  }
}

// By the simple poles +-1: atanh^(k)(a) = (k-1)!/2 * ((1 - a)^-k +
// (-1)^(k-1) (1 + a)^-k).
template <typename T>
std::vector<T> d_atanh(const T &a, unsigned n) {
  T p = T(1) / (1 - a), q = T(1) / (1 + a), pk = p, qk = q, factorial(1);
  std::vector<T> d(n + 1);
  d[0] = atanh(a);
  for (unsigned k = 1; k <= n; ++k) {
    d[k] = factorial * (k % 2 ? pk + qk : pk - qk) / 2;
    pk = pk * p;
    qk = qk * q;
    factorial = factorial * int(k);
  }
  return d;
}

// asin' = (1 - u^2)^(-1/2), asinh' = (1 + u^2)^(-1/2), acosh' = (u^2 -
// 1)^(-1/2) at u = a + t, continued from w0 = f'(a) on the branch of f.
template <typename T>
std::vector<T> d_inverse_sqrt_like(const T &f0, const T &w0,
                                   const std::vector<T> &s, unsigned n) {
  return from_derivative_series(f0, taylor_pow(s, T(-1) / 2, w0, n - 1));
}

// 1 - a^2 as (1 - a)(1 + a).
template <typename T>
std::vector<T> d_asin(const T &a, unsigned n) {
  return d_inverse_sqrt_like(asin(a), T(1) / (sqrt(1 - a) * sqrt(1 + a)),
                             {(1 - a) * (1 + a), -2 * a, T(-1)}, n);
}

template <typename T>
std::vector<T> d_acos(const T &a, unsigned n) {
  std::vector<T> d = d_asin(a, n);
  for (T &v : d) v = -v;
  d[0] = acos(a);
  return d;
}

// Complex: 1 + a^2 as (1 + ia)(1 - ia).
template <typename T>
std::vector<T> d_asinh(const T &a, unsigned n) {
  if constexpr (is_complex_number<T>) {
    T ia = T("0.0", "1.0") * a;
    return d_inverse_sqrt_like(asinh(a), T(1) / (sqrt(1 + ia) * sqrt(1 - ia)),
                               {(1 + ia) * (1 - ia), 2 * a, T(1)}, n);
  } else {
    std::vector<T> s = {1 + a * a, 2 * a, T(1)};
    return d_inverse_sqrt_like(asinh(a), T(1) / sqrt(s[0]), s, n);
  }
}

// acosh'(a) = 1/(sqrt(a - 1) sqrt(a + 1)), the branch with cut (-inf, 1);
// a^2 - 1 as (a - 1)(a + 1).
template <typename T>
std::vector<T> d_acosh(const T &a, unsigned n) {
  return d_inverse_sqrt_like(acosh(a), T(1) / (sqrt(a - 1) * sqrt(a + 1)),
                             {(a - 1) * (a + 1), 2 * a, T(1)}, n);
}
//- f^(k)(a) of the one-argument functions

#endif  // CSJET_H

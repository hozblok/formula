// Normalizes signed-zero output so '-0', '-0.0', '-0.000…' all render as
// the unsigned form. MPFR keeps a sign bit on zero per IEEE-754, but +0 == -0
// at the value level — exposing the sign breaks string-based equality.
#pragma once

#include <string>

inline std::string strip_neg_zero(std::string s) {
  if (!s.empty() && s.front() == '-' &&
      s.find_first_of("123456789") == std::string::npos) {
    s.erase(0, 1);
  }
  return s;
}

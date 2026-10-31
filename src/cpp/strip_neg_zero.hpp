// A signed zero ('-0', '-0.0', '-0e+00') renders unsigned, as +0 == -0;
// anything else, '-inf' included, keeps its sign.
#pragma once

#include <string>

inline std::string strip_neg_zero(std::string s) {
  if (!s.empty() && s.front() == '-' && s.find('0') != std::string::npos &&
      s.find_first_of("123456789") == std::string::npos) {
    s.erase(0, 1);
  }
  return s;
}

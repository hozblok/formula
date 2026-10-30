// Python operands of the mp bindings: int and float as decimal text in
// arithmetic, ints within 2**53 in comparisons, other mp precisions refused,
// anything else left to Python's NotImplemented fallback.
#pragma once

#include <pybind11/pybind11.h>

#include <cctype>
#include <functional>
#include <optional>
#include <string>
#include <utility>

// int exactly; float as repr (shortest round-trip decimal); bool refused.
inline std::optional<std::string> scalar_text(pybind11::handle h) {
  if (PyBool_Check(h.ptr()))
    throw pybind11::type_error("bool is not a valid numeric operand");
  if (PyFloat_Check(h.ptr())) {
    char *s = PyOS_double_to_string(PyFloat_AS_DOUBLE(h.ptr()), 'r', 0,
                                    Py_DTSF_ADD_DOT_0, nullptr);
    if (!s) throw pybind11::error_already_set();
    std::string text(s);
    PyMem_Free(s);
    return text;
  }
  if (PyIndex_Check(h.ptr())) {
    auto n = pybind11::reinterpret_steal<pybind11::object>(
        PyNumber_Index(h.ptr()));
    if (!n) throw pybind11::error_already_set();
    auto text = pybind11::reinterpret_steal<pybind11::str>(
        PyNumber_ToBase(n.ptr(), 10));
    if (!text) throw pybind11::error_already_set();
    return text.cast<std::string>();
  }
  return std::nullopt;
}

inline std::string type_name(pybind11::handle h) {
  return pybind11::str(pybind11::type::of(h).attr("__name__"));
}

// Fallback for an operand that is neither the same precision nor a scalar.
inline pybind11::object other_operand(pybind11::handle other,
                                      const std::string &self_name) {
  const std::string name = type_name(other);
  if (name.rfind("mp_real_", 0) == 0 || name.rfind("mp_complex_", 0) == 0)
    throw pybind11::type_error("precision mismatch: " + self_name + " and " +
                               name);
  return pybind11::reinterpret_borrow<pybind11::object>(Py_NotImplemented);
}

// Python int that every mp type holds exactly: |n| <= 2**53. Other Python
// numbers do not compare: TypeError, never a silent False.
inline std::optional<long long> small_int(pybind11::handle h,
                                          const std::string &self_name) {
  if (PyBool_Check(h.ptr()))
    throw pybind11::type_error("bool is not a valid numeric operand");
  if (PyLong_Check(h.ptr())) {
    int overflow = 0;
    const long long n = PyLong_AsLongLongAndOverflow(h.ptr(), &overflow);
    if (PyErr_Occurred()) throw pybind11::error_already_set();
    const long long limit = 1LL << 53;
    if (!overflow && -limit <= n && n <= limit) return n;
    throw pybind11::type_error("an int beyond 2**53 compares with " +
                               self_name + " only as " + self_name +
                               "(str(n))");
  }
  if (PyFloat_Check(h.ptr()) || PyComplex_Check(h.ptr()))
    throw pybind11::type_error("float and complex do not compare with " +
                               self_name + "; compare with " + self_name +
                               "(repr(x))");
  return std::nullopt;
}

inline bool numeric_literal(const std::string &text) {
  bool digit = false, point = false, exponent = false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (std::isdigit(static_cast<unsigned char>(c))) {
      digit = true;
    } else if (c == '+' || c == '-') {
      if (i != 0 && text[i - 1] != 'e' && text[i - 1] != 'E') return false;
    } else if (c == '.') {
      if (point || exponent) return false;
      point = true;
    } else if (c == 'e' || c == 'E') {
      if (exponent || !digit) return false;
      exponent = true;
      digit = false;
    } else {
      return false;
    }
  }
  return digit;
}

inline void require_literal(const std::string &text) {
  if (!numeric_literal(text))
    throw pybind11::value_error("not a numeric literal: '" + text + "'");
}

// "(re,im)" split at its comma, or a real literal with a zero imaginary part.
inline std::pair<std::string, std::string> complex_literal(
    const std::string &text) {
  const std::size_t comma = text.find(',');
  if (text.size() > 2 && text.front() == '(' && text.back() == ')' &&
      comma != std::string::npos) {
    std::string re = text.substr(1, comma - 1),
                im = text.substr(comma + 1, text.size() - comma - 2);
    require_literal(re);
    require_literal(im);
    return {re, im};
  }
  require_literal(text);
  return {text, "0"};
}

// Python's hash of the value: an integer within 2**53 as Python hashes that
// int (the int itself, -1 mapped to -2), anything else by its canonical text.
template <class Real>
inline Py_hash_t real_hash(const Real &x) {
  const long long limit = 1LL << 53;
  if (x == floor(x) && x >= -limit && x <= limit) {
    const long long n = x.template convert_to<long long>();
    return n == -1 ? -2 : static_cast<Py_hash_t>(n);
  }
  return static_cast<Py_hash_t>(std::hash<std::string>{}(x.str()));
}

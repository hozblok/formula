// Python operands of the mp bindings: int and float as decimal text, other
// mp precisions refused, anything else left to Python's NotImplemented fallback.
#pragma once

#include <pybind11/pybind11.h>

#include <optional>
#include <string>

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

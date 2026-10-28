#pragma once

#include <pybind11/pybind11.h>

#include <limits>
#include <string>

template <class T>
static T multiply_python_int(const T &value, const pybind11::int_ &integer) {
  if (PyBool_Check(integer.ptr()))
    throw pybind11::type_error("bool is not a valid numeric operand");

  int overflow = 0;
  const auto scalar = PyLong_AsLongLongAndOverflow(integer.ptr(), &overflow);
  if (PyErr_Occurred())
    throw pybind11::error_already_set();
  if (!overflow && scalar >= std::numeric_limits<int>::min() &&
      scalar <= std::numeric_limits<int>::max())
    return value * static_cast<int>(scalar);

  auto text = pybind11::reinterpret_steal<pybind11::str>(
      PyNumber_ToBase(integer.ptr(), 10));
  if (!text)
    throw pybind11::error_already_set();
  const T operand(text.cast<std::string>());
  return value * operand;
}

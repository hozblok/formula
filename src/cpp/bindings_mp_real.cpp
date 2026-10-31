// mp_real_<P> bindings, one py::class_ per AllowedPrecisions. Split into its
// own translation unit to keep per-compile memory bounded.
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <functional>
#include <string>
#include <utility>

#include <cseval/cseval.hpp>
#include <cseval/cseval_complex.hpp>

#include "mp_scalar.hpp"
#include "strip_neg_zero.hpp"

namespace py = pybind11;

template <unsigned P>
static void register_one_mp_real(py::module_ &m) {
  using R = mp_real<P>;
  using C = mp_complex<P>;
  const std::string class_name = "mp_real_" + std::to_string(P);
  // Operands: R; C of the same P (a promoted through its decimal text);
  // int and float as text.
  auto binary = [class_name](auto op, bool reflected) {
    return [op, reflected, class_name](const R &a, py::object other) {
      if (py::isinstance<R>(other)) {
        const R &b = other.cast<const R &>();
        return py::cast(reflected ? op(b, a) : op(a, b));
      }
      if (py::isinstance<C>(other)) {
        const C &b = other.cast<const C &>();
        const C ca(a.str());
        return py::cast(reflected ? op(b, ca) : op(ca, b));
      }
      if (auto text = scalar_text(other)) {
        R b(*text);
        return py::cast(reflected ? op(b, a) : op(a, b));
      }
      return other_operand(other, class_name);
    };
  };
  // Comparisons: R natively and ints within 2**53 exactly. C raises: the
  // decimal type cannot hold every binary value (exponent range), so no
  // conversion compares them faithfully. Other Python numbers raise.
  auto ordering = [class_name](auto native) {
    return [native, class_name](const R &a, py::object other) {
      if (py::isinstance<R>(other))
        return py::cast(native(a, other.cast<const R &>()));
      if (py::isinstance<C>(other))
        throw py::type_error("real and complex values do not compare");
      if (auto n = small_int(other, class_name)) return py::cast(native(a, R(*n)));
      return other_operand(other, class_name);
    };
  };
  auto equality = [class_name](bool negate) {
    return [negate, class_name](const R &a, py::object other) {
      bool same;
      if (py::isinstance<R>(other)) {
        same = a == other.cast<const R &>();
      } else if (py::isinstance<C>(other)) {
        throw py::type_error("real and complex values do not compare");
      } else if (auto n = small_int(other, class_name)) {
        same = a == R(*n);
      } else {
        return other_operand(other, class_name);
      }
      return py::cast(same != negate);
    };
  };
  auto plus = [](const auto &x, const auto &y) { return x + y; };
  auto minus = [](const auto &x, const auto &y) { return x - y; };
  auto times = [](const auto &x, const auto &y) { return x * y; };
  auto over = [](const auto &x, const auto &y) { return x / y; };
  auto power = [](const auto &x, const auto &y) { return pow(x, y); };
  py::class_<R>(m, class_name.c_str(), py::is_final())
      .def(py::init([](const std::string &text) {
             require_literal(text);
             return R(text);
           }),
           "A real literal: [+-] digits [. digits] [e [+-] digits].")
      // Storage precision and kind, on the class and on every value.
      .def_property_readonly_static("precision",
                                    [](const py::object &) { return P; })
      .def_property_readonly_static("is_complex",
                                    [](const py::object &) { return false; })
      .def("__add__", binary(plus, false))
      .def("__radd__", binary(plus, false))
      .def("__sub__", binary(minus, false))
      .def("__rsub__", binary(minus, true))
      .def("__mul__", binary(times, false))
      .def("__rmul__", binary(times, false))
      .def("__truediv__", binary(over, false))
      .def("__rtruediv__", binary(over, true))
      .def("__pow__", binary(power, false))
      .def("__rpow__", binary(power, true))
      .def(-py::self)
      .def("__eq__", equality(false))
      .def("__ne__", equality(true))
      .def("__lt__", ordering([](const R &x, const R &y) { return x < y; }))
      .def("__le__", ordering([](const R &x, const R &y) { return x <= y; }))
      .def("__gt__", ordering([](const R &x, const R &y) { return x > y; }))
      .def("__ge__", ordering([](const R &x, const R &y) { return x >= y; }))
      .def("__bool__", [](const R &x) { return x != 0; })
      .def("__abs__", [](const R &x) { return abs(x); })
      .def("sqrt", [](const R &x) { return sqrt(x); },
           "Native mp square root (much faster than pow(x, 0.5)).")
      .def_property_readonly("real", [](const R &x) { return x; })
      .def_property_readonly("imag", [](const R &) { return R(0); })
      .def("__float__",
           [](const R &x) { return x.template convert_to<double>(); })
      .def("__complex__",
           [](const R &x) {
             return std::complex<double>(x.template convert_to<double>(), 0.0);
           })
      // Python's hash of the value: an int within 2**53 hashes as that int.
      .def("__hash__", [](const R &x) { return real_hash(x); })
      .def("__str__", [](const R &x) { return strip_neg_zero(x.str()); })
      .def("__repr__",
           [class_name](const R &x) {
             return class_name + "('" + x.str() + "')";
           })
      .def(
          "str",
          [](const R &x, std::streamsize digits,
             std::ios_base::fmtflags format) {
            return strip_neg_zero(x.str(digits, format));
          },
          "Returns the number formatted as a string, with at least precision "
          "digits.",
          py::arg("digits") = 0, py::arg("format") = std::ios_base::fmtflags(0));
}

template <unsigned... Ps>
static void register_all_mp_real(py::module_ &m,
                                 std::integer_sequence<unsigned, Ps...>) {
  (register_one_mp_real<Ps>(m), ...);
}

// mp_real_16 … mp_real_8192.
void register_mp_real(py::module_ &m) {
  register_all_mp_real(m, AllowedPrecisionsSeq{});
}

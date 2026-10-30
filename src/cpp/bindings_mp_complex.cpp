// mp_complex_<P> bindings, one py::class_ per AllowedPrecisions. Split into its
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
static void register_one_mp_complex(py::module_ &m) {
  using C = mp_complex<P>;
  using R = mp_real<P>;
  const std::string class_name = "mp_complex_" + std::to_string(P);
  // Operands: C; R of the same P (promoted through its decimal text); int and
  // float as text.
  auto operand = [](py::object other) -> std::optional<C> {
    if (py::isinstance<C>(other)) return other.cast<C>();
    if (py::isinstance<R>(other)) return C(other.cast<R>().str());
    if (auto text = scalar_text(other)) return C(*text);
    return std::nullopt;
  };
  auto binary = [class_name, operand](auto op, bool reflected) {
    return [op, reflected, class_name, operand](const C &a, py::object other) {
      if (auto b = operand(other))
        return py::cast(reflected ? op(*b, a) : op(a, *b));
      return other_operand(other, class_name);
    };
  };
  // Equality: C natively and ints within 2**53 exactly. R raises: no
  // conversion between the decimal and the binary type keeps every value.
  // Other Python numbers raise.
  auto equality = [class_name](bool negate) {
    return [negate, class_name](const C &a, py::object other) {
      bool same;
      if (py::isinstance<C>(other)) {
        same = a == other.cast<C>();
      } else if (py::isinstance<R>(other)) {
        throw py::type_error("real and complex values do not compare");
      } else if (auto n = small_int(other, class_name)) {
        same = a == C(*n);
      } else {
        return other_operand(other, class_name);
      }
      return py::cast(same != negate);
    };
  };
  auto plus = [](const C &x, const C &y) { return x + y; };
  auto minus = [](const C &x, const C &y) { return x - y; };
  auto times = [](const C &x, const C &y) { return x * y; };
  auto over = [](const C &x, const C &y) { return x / y; };
  auto power = [](const C &x, const C &y) { return pow(x, y); };
  py::class_<C>(m, class_name.c_str())
      .def(py::init([](const std::string &text) {
             auto parts = complex_literal(text);
             return C(parts.first, parts.second);
           }),
           "A real literal or the pair \"(re,im)\" of two.")
      .def(py::init([](const std::string &real, const std::string &imag) {
             require_literal(real);
             require_literal(imag);
             return C(real, imag);
           }),
           py::arg("real"), py::arg("imag"))
      // Storage precision and kind, on the class and on every value.
      .def_property_readonly_static("precision",
                                    [](const py::object &) { return P; })
      .def_property_readonly_static("is_complex",
                                    [](const py::object &) { return true; })
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
      .def("__bool__", [](const C &x) { return !(x == 0); })
      // |z| as mp_real, through its decimal text.
      .def("__abs__", [](const C &x) { return R(abs(x).str()); })
      // x ** 0.5, the former Number path; sqrt(x) differs in the last bits.
      .def("sqrt", [](const C &x) { return pow(x, C("0.5")); })
      .def_property_readonly("real", [](const C &x) { return R(x.real().str()); })
      .def_property_readonly("imag", [](const C &x) { return R(x.imag().str()); })
      .def("__float__",
           [](const C &x) {
             if (x.imag() != 0)
               throw py::type_error("cannot convert a complex value to float");
             return x.real().template convert_to<double>();
           })
      .def("__complex__",
           [](const C &x) {
             return std::complex<double>(x.real().template convert_to<double>(),
                                         x.imag().template convert_to<double>());
           })
      // An integer within 2**53 hashes as that int, as in equality.
      .def("__hash__",
           [](const C &x) {
             if (x.imag() == 0) return real_hash(x.real());
             return static_cast<Py_hash_t>(std::hash<std::string>{}(x.str()));
           })
      .def("__str__", [](const C &x) { return strip_neg_zero(x.str()); })
      .def("__repr__",
           [class_name](const C &x) {
             return class_name + "('" + x.str() + "')";
           })
      .def(
          "str",
          [](const C &x, std::streamsize digits,
             std::ios_base::fmtflags format) {
            return strip_neg_zero(x.str(digits, format));
          },
          "Returns the number formatted as a string. Non-zero imaginary part "
          "renders as (real,imag); a zero one renders as just real.",
          py::arg("digits") = 0, py::arg("format") = std::ios_base::fmtflags(0));
}

template <unsigned... Ps>
static void register_all_mp_complex(py::module_ &m,
                                    std::integer_sequence<unsigned, Ps...>) {
  (register_one_mp_complex<Ps>(m), ...);
}

// mp_complex_16 … mp_complex_8192.
void register_mp_complex(py::module_ &m) {
  register_all_mp_complex(m, AllowedPrecisionsSeq{});
}

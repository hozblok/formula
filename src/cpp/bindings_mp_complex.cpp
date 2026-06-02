// mp_complex_<P> bindings, one py::class_ per AllowedPrecisions. Split into its
// own translation unit to keep per-compile memory bounded.
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <string>
#include <utility>

#include <cseval/cseval_complex.hpp>

#include "strip_neg_zero.hpp"

namespace py = pybind11;

template <unsigned P>
static void register_one_mp_complex(py::module_ &m) {
  using C = mp_complex<P>;
  const std::string class_name = "mp_complex_" + std::to_string(P);
  py::class_<C>(m, class_name.c_str())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("real"), py::arg("imag"))
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(-py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(
          "__pow__", [](const C &a, const C &b) { return pow(a, b); },
          py::is_operator())
      // abs() of a complex is its real magnitude, wrapped back into mp_complex.
      .def("__abs__", [](const C &x) { return C(abs(x)); })
      .def("__hash__",
           [](const C &x) { return std::hash<std::string>{}(x.str()); })
      .def("__repr__",
           [class_name](const C &x) {
             return class_name + "('" + x.str() + "')";
           })
      .def(
          "real",
          [](const C &x, std::streamsize digits,
             std::ios_base::fmtflags format) {
            return strip_neg_zero(x.real().str(digits, format));
          },
          "Real part formatted as a string.", py::arg("digits") = 0,
          py::arg("format") = std::ios_base::fmtflags(0))
      .def(
          "imag",
          [](const C &x, std::streamsize digits,
             std::ios_base::fmtflags format) {
            return strip_neg_zero(x.imag().str(digits, format));
          },
          "Imaginary part formatted as a string.", py::arg("digits") = 0,
          py::arg("format") = std::ios_base::fmtflags(0))
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

// mp_complex_16 … mp_complex_262144.
void register_mp_complex(py::module_ &m) {
  register_all_mp_complex(m, AllowedPrecisionsSeq{});
}

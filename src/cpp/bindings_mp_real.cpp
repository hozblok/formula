// mp_real_<P> bindings, one py::class_ per AllowedPrecisions. Split into its
// own translation unit to keep per-compile memory bounded.
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <string>
#include <utility>

#include <cseval/cseval.hpp>

#include "strip_neg_zero.hpp"

namespace py = pybind11;

template <unsigned P>
static void register_one_mp_real(py::module_ &m) {
  using R = mp_real<P>;
  const std::string class_name = "mp_real_" + std::to_string(P);
  py::class_<R>(m, class_name.c_str())
      .def(py::init<const std::string &>())
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(-py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self < py::self)
      .def(py::self <= py::self)
      .def(py::self > py::self)
      .def(py::self >= py::self)
      .def(
          "__pow__", [](const R &a, const R &b) { return pow(a, b); },
          py::is_operator())
      .def("__abs__", [](const R &x) { return abs(x); })
      .def("__hash__",
           [](const R &x) { return std::hash<std::string>{}(x.str()); })
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

// mp_real_16 … mp_real_262144.
void register_mp_real(py::module_ &m) {
  register_all_mp_real(m, AllowedPrecisionsSeq{});
}

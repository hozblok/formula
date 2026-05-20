#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <string>
#include <utility>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include <csformula/csformula.cpp>

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
      .def("str",
           (std::string(R::*)(std::streamsize, std::ios_base::fmtflags) const) &
               R::str,
           "Returns the number formatted as a string, with at least precision "
           "digits.",
           py::arg("digits") = 0, py::arg("format") = std::ios_base::fmtflags(0));
}

template <unsigned... Ps>
static void register_all_mp_real(py::module_ &m,
                                 std::integer_sequence<unsigned, Ps...>) {
  (register_one_mp_real<Ps>(m), ...);
}

PYBIND11_MODULE(_formula, m) {
  m.doc() = R"pbdoc(
        Arbitrary-precision formula parser and solver.
        -----------------------

        .. currentmodule:: formula

        .. autosummary::
           :toctree: _generate
    )pbdoc";

  py::class_<std::ios_base::fmtflags>(m, "FmtFlags")
      .def(py::self | py::self)
      .def(py::self & py::self)
      .def(py::self ^ py::self)
      .def(py::self |= py::self)
      .def(py::self &= py::self)
      .def(py::self ^= py::self)
      .def(~py::self)
      .def_property_readonly_static("default",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::fmtflags(0);
                                    })
      .def_property_readonly_static("fixed",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::fixed;
                                    })
      .def_property_readonly_static("internal",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::internal;
                                    })
      .def_property_readonly_static("left",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::left;
                                    })
      .def_property_readonly_static("right",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::right;
                                    })
      .def_property_readonly_static("scientific",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::scientific;
                                    })
      .def_property_readonly_static("showpoint",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::showpoint;
                                    })
      .def_property_readonly_static("showpos",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::showpos;
                                    })
      .def_property_readonly_static("skipws",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::skipws;
                                    })
      .def_property_readonly_static("unitbuf",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::unitbuf;
                                    })
      .def_property_readonly_static("uppercase",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::uppercase;
                                    })
      .def_property_readonly_static("adjustfield",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::adjustfield;
                                    })
      .def_property_readonly_static("floatfield",
                                    [](py::object) -> std::ios_base::fmtflags {
                                      return std::ios_base::floatfield;
                                    });

  py::class_<Formula>(m, "Formula")
      .def(py::init<const std::string &, const unsigned, const char,
                    const bool>(),
           py::arg("expression"), py::arg("precision") = 24,
           py::arg("imaginary_unit") = 'i', py::arg("case_insensitive") = false)
      .def("get_precision", &Formula::get_precision)
      .def("set_precision", &Formula::set_precision)
      .def_property("precision", &Formula::get_precision,
                    &Formula::set_precision,
                    "Precision with which the calculations will be performed.")
      .def("get_expression", &Formula::get_expression)
      .def("set_expression", &Formula::set_expression)
      .def_property("expression", &Formula::get_expression,
                    &Formula::set_expression,
                    "Formula expression that is ready for calculations.")
      .def("copy", &Formula::copy, py::return_value_policy::take_ownership,
           "Create copy of the Formula object.")
      .def("variables",
           (std::unordered_set<std::string>(Formula::*)() const) &
               Formula::variables,
           py::return_value_policy::automatic,
           "Parsed variables from the expression.")
      .def(
          "get",
          (std::string(Formula::*)(const std::map<std::string, std::string> &,
                                   std::streamsize, std::ios_base::fmtflags)
               const) &
              Formula::get,
          "Calculate the value of the parsed formula string \
using the passed string values of the variables.",
          py::arg("variables_to_values") = std::map<std::string, std::string>(),
          py::arg("digits") = 0, py::arg("format") = std::ios_base::fmtflags(0))
      .def(
          "get_pair", &Formula::get_pair,
          "Calculate the value and return it as a (real_part, imag_part) pair \
of strings, both formatted with the given digits/format. For real-only \
expressions the imaginary part is zero formatted with the same shape so the \
pair can be compared byte-for-byte against another get_pair() result.",
          py::arg("variables_to_values") = std::map<std::string, std::string>(),
          py::arg("digits") = 0, py::arg("format") = std::ios_base::fmtflags(0))
      .def(
          "get_derivative",
          (std::string(Formula::*)(
              const std::string, const std::map<std::string, std::string> &,
              std::streamsize, std::ios_base::fmtflags) const) &
              Formula::get_derivative,
          "Calculate the value of the partial derivative of the formula.",
          py::arg("variable"),
          py::arg("variables_to_values") = std::map<std::string, std::string>(),
          py::arg("digits") = 0, py::arg("format") = std::ios_base::fmtflags(0))
      .def("get_from_float",
           (std::string(Formula::*)(const std::map<std::string, double> &,
                                    std::streamsize, std::ios_base::fmtflags)
                const) &
               Formula::get,
           "Calculate the value of the parsed formula string \
using the passed real values of the variables.",
           py::arg("variables_to_values") = std::map<std::string, double>(),
           py::arg("digits") = 0,
           py::arg("format") = std::ios_base::fmtflags(0));

  // One py::class_<mp_real<P>> per precision, exposed as mp_real_16 … mp_real_8192.
  register_all_mp_real(m, AllowedPrecisionsSeq{});

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

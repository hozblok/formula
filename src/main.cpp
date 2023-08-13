#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include <csformula/csformula.cpp>

namespace py = pybind11;

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

  // TODO support all mp_real
  py::class_<mp_real<24>>(m, "mp_real_24")
      .def(py::init<const std::string &>())
      .def("str",
           (std::string(mp_real<24>::*)(std::streamsize digits,
                                        std::ios_base::fmtflags) const) &
               mp_real<24>::str,
           "Returns the number formatted as a string, with at least precision "
           "digits.",
           py::arg("digits") = 0, py::arg("f") = std::ios_base::fmtflags(0));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

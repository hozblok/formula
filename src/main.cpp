#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cseval.hpp>
#include <csformula.cpp>

namespace py = pybind11;

PYBIND11_MODULE(formula, m) {
  m.doc() = R"pbdoc(
        Arbitrary-precision formula parser and solver.
        -----------------------

        .. currentmodule:: formula

        .. autosummary::
           :toctree: _generate
    )pbdoc";

  enum fmtflags {
    boolalpha = std::ios_base::boolalpha,
    dec = std::ios_base::dec,
    fixed = std::ios_base::fixed,
    hex = std::ios_base::hex,
    internal = std::ios_base::internal,
    left = std::ios_base::left,
    oct = std::ios_base::oct,
    right = std::ios_base::right,
    scientific = std::ios_base::scientific,
    showbase = std::ios_base::showbase,
    showpoint = std::ios_base::showpoint,
    showpos = std::ios_base::showpos,
    skipws = std::ios_base::skipws,
    unitbuf = std::ios_base::unitbuf,
    uppercase = std::ios_base::uppercase,
    adjustfield = std::ios_base::adjustfield,
    basefield = std::ios_base::basefield,
    floatfield = std::ios_base::floatfield,
  };

  py::enum_<fmtflags>(m, "fmtflags", "Specifies available formatting flags.")
      .value("skipws", fmtflags::skipws)
      .value("unitbuf", fmtflags::unitbuf)
      .value("uppercase", fmtflags::uppercase)
      .value("showbase", fmtflags::showbase)
      .value("showpoint", fmtflags::showpoint)
      .value("showpos", fmtflags::showpos)
      .value("left", fmtflags::left)
      .value("right", fmtflags::right)
      .value("internal", fmtflags::internal)
      .value("dec", fmtflags::dec)
      .value("oct", fmtflags::oct)
      .value("hex", fmtflags::hex)
      .value("scientific", fmtflags::scientific)
      .value("fixed", fmtflags::fixed)
      .value("boolalpha", fmtflags::boolalpha)
      .value("adjustfield", fmtflags::adjustfield)
      .value("basefield", fmtflags::basefield)
      .value("floatfield", fmtflags::floatfield)
      .export_values();

  py::class_<Formula>(m, "formula")
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
      .def("variables", (std::unordered_set<std::string>(Formula::*)()
                const) &Formula::variables, py::return_value_policy::automatic,
           "Parsed variables from the expression.")
      .def("get",
           (std::string(Formula::*)(const std::map<std::string, std::string> &,
                                    std::streamsize, std::ios_base::fmtflags)
                const) &
               Formula::get,
           "Calculate the value of the parsed formula string \
using the passed string values of the variables.",
           py::arg("variables_to_values") = std::map<std::string, std::string>(),
           py::arg("digits") = 0,
           py::arg("format") = std::ios_base::fmtflags(0))
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
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

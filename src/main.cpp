#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cseval.hpp>
#include <csformula.cpp>

namespace py = pybind11;

PYBIND11_MODULE(formula, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: formula

        .. autosummary::
           :toctree: _generate

           add
           subtract
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

  py::enum_<fmtflags>(m, "fmtflags")
      .value("boolalpha", fmtflags::boolalpha)
      .value("dec", fmtflags::dec)
      .value("fixed", fmtflags::fixed)
      .value("hex", fmtflags::hex)
      .value("internal", fmtflags::internal)
      .value("left", fmtflags::left)
      .value("oct", fmtflags::oct)
      .value("right", fmtflags::right)
      .value("scientific", fmtflags::scientific)
      .value("showbase", fmtflags::showbase)
      .value("showpoint", fmtflags::showpoint)
      .value("showpos", fmtflags::showpos)
      .value("skipws", fmtflags::skipws)
      .value("unitbuf", fmtflags::unitbuf)
      .value("uppercase", fmtflags::uppercase)
      .value("adjustfield", fmtflags::adjustfield)
      .value("basefield", fmtflags::basefield)
      .value("floatfield", fmtflags::floatfield)
      .export_values();

  py::class_<csformula>(m, "formula")
      .def(py::init<const std::string &, const unsigned, const char,
                    const bool>(),
           py::arg("_expression"), py::arg("_precision") = 24,
           py::arg("_imaginary_unit") = 'i',
           py::arg("_case_insensitive") = false)
      .def("getPrecision", &csformula::getPrecision, "TODO")
      .def("setPrecision", &csformula::setPrecision, "TODO")
      .def_property("precision", &csformula::getPrecision,
                    &csformula::setPrecision, "TODO")
      .def("getExpression", &csformula::getExpression, "TODO")
      .def("setExpression", &csformula::setExpression, "TODO")
      .def_property("expression", &csformula::getExpression,
                    &csformula::setExpression)
      .def("get",
           (std::string(csformula::*)(
               const std::map<std::string, std::string> &, std::streamsize,
               std::ios_base::fmtflags) const) &
               csformula::get,
           "Calculate the value of the parsed formula string \
using the passed string values of the variables.",
           py::arg("mapVariableValues") = std::map<std::string, std::string>(),
           py::arg("digits") = 0,
           py::arg("format") = std::ios_base::fmtflags(0))
      .def("get_from_float",
           (std::string(csformula::*)(
               const std::map<std::string, double> &, std::streamsize,
               std::ios_base::fmtflags) const) &
               csformula::get,
           "Calculate the value of the parsed formula string \
using the passed real values of the variables.",
           py::arg("mapVariableValues") = std::map<std::string, double>(),
           py::arg("digits") = 0,
           py::arg("format") = std::ios_base::fmtflags(0));

  // TODO
  py::class_<mp_real<1>>(m, "mp_real_1")
      .def(py::init<const std::string &>())
      .def("str",
           (std::string(mp_real<1>::*)(std::streamsize digits,
                                       std::ios_base::fmtflags) const) &
               mp_real<1>::str,
           "Returns the number formatted as a string, with at least precision "
           "digits, and in scientific format if scientific is true. ",
           py::arg("digits") = 0,
           py::arg("f") = fmtflags::dec /*std::_Ios_Fmtflags::_S_scientific*/
           /*1L << 8*/ /*std::ios_base::fmtflags(0)*/);

  //            std::string str(std::streamsize digits = 0,
  //            std::ios_base::fmtflags f = std::ios_base::fmtflags(0))const
  //    {
  //       return m_backend.str(digits, f);
  //    }

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

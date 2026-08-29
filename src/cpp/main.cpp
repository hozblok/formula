#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include <csformula/csformula.cpp>

namespace py = pybind11;

// One py::class_ per precision; defined in their own translation units to keep
// per-compile memory bounded.
void register_mp_real(py::module_ &m);
void register_mp_complex(py::module_ &m);
void register_trace(py::module_ &m);

// Evaluate a formula node and return the value as the registered mp_real_<P> /
// mp_complex_<P> Python object, so its type carries the real/complex kind.
struct GetValueVisitor : public boost::static_visitor<py::object> {
  const std::map<std::string, std::string> *variables_to_values;
  template <typename CSEval>
  py::object operator()(const CSEval &eval) const {
    return py::cast(eval->calculate(*variables_to_values));
  }
};

PYBIND11_MODULE(_formula, m) {
  m.doc() = R"pbdoc(
        Arbitrary-precision formula parser and solver.
        -----------------------

        .. currentmodule:: formula

        .. autosummary::
           :toctree: _generate
    )pbdoc";

  // Engine precision ceiling, independent of which mp_real_<P> wrappers are bound.
  m.attr("MAX_PRECISION") = static_cast<unsigned>(max_precision);

  // Storage precision for a request: smallest supported >= it, 0 if over the max.
  m.def("round_up_precision", &round_up_precision, py::arg("precision"),
        "Smallest supported precision >= the request; what a value is stored at.");

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
      .def("get_original_expression", &Formula::get_original_expression)
      .def_property_readonly("original_expression",
                             &Formula::get_original_expression,
                             "User input as received (preserves whitespace "
                             "and original case).")
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
          "evaluate",
          [](const Formula &self,
             const std::map<std::string, std::string> &variables_to_values) {
            GetValueVisitor visitor;
            visitor.variables_to_values = &variables_to_values;
            return self.visit_value(visitor);
          },
          "Evaluate the expression and return its value as an mp_real_<P> or \
mp_complex_<P> object; the returned type reflects whether the expression is \
real or complex.",
          py::arg("variables_to_values") = std::map<std::string, std::string>())
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

  register_mp_real(m);
  register_mp_complex(m);
  register_trace(m);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cseval.hpp>
#include <csformula.cpp>

int add(int i, int j)
{
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(formula, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: formula

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    py::enum_<std::ios_base::fmtflags>(m, "ios_base_fmtflags")
    .value("boolalpha", std::ios_base::boolalpha)
    .value("dec", std::ios_base::dec)
    .value("fixed", std::ios_base::fixed)
    .value("hex", std::ios_base::hex)
    .value("internal", std::ios_base::internal)
    .value("left", std::ios_base::left)
    .value("oct", std::ios_base::oct)
    .value("right", std::ios_base::right)
    .value("scientific", std::ios_base::scientific)
    .value("showbase", std::ios_base::showbase)
    .value("showpoint", std::ios_base::showpoint)
    .value("showpos", std::ios_base::showpos)
    .value("skipws", std::ios_base::skipws)
    .value("unitbuf", std::ios_base::unitbuf)
    .value("uppercase", std::ios_base::uppercase)
    .value("adjustfield", std::ios_base::adjustfield)
    .value("basefield", std::ios_base::basefield)
    .value("floatfield", std::ios_base::floatfield)
    .export_values();

    py::class_<csformula>(m, "csformula")
        .def(py::init<const std::string &, const size_t>())
        // .def("setName", &csformula::setName)
        .def("getExpression", &csformula::getExpression, "A function which adds two numbers")
        // .def_property("expression", &csformula<mp_real<1>>::getExpression, &csformula<mp_real<1>>::setExpression)
        // .def(py::init<const std::string &>());
        .def("get", (std::string(csformula::*)(const std::map<std::string, std::string> &) const) & csformula::get,
             "Calculate the value of the parsed string using the passed values of the variables.");
        // .def("get", (mp_real<1>(csformula<mp_real<1>>::*)(const std::map<std::string, mp_real<1>> &) const) & csformula<mp_real<1>>::get,
        //      "Calculate the value of the parsed string using the passed values of the variables.");
    // .def("set", (void (Pet::*)(int)) &Pet::set, "Set the pet's age")
    // .def("set", (void (Pet::*)(const std::string &)) &Pet::set, "Set the pet's name");
    // .def("getD", &csformula<mp_real<1>>::getD);

    py::class_<mp_real<1>>(m, "mp_real_1")
        .def(py::init<const std::string &>())
        .def("str", (std::string(mp_real<1>::*)(std::streamsize digits, std::ios_base::fmtflags) const) & mp_real<1>::str,
             "Returns the number formatted as a string, with at least precision digits, and in scientific format if scientific is true. ",
             py::arg("digits") = 0, py::arg("f") = std::ios_base::fmtflags(0) /*std::_Ios_Fmtflags::_S_scientific*/ /*1L << 8*/ /*std::ios_base::fmtflags(0)*/);

    //            std::string str(std::streamsize digits = 0, std::ios_base::fmtflags f = std::ios_base::fmtflags(0))const
    //    {
    //       return m_backend.str(digits, f);
    //    }

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

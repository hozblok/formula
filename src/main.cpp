#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cseval.h>
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

    py::enum_<std::_Ios_Fmtflags>(m, "_Ios_Fmtflags")
    .value("_S_boolalpha", std::_Ios_Fmtflags::_S_boolalpha)
    .value("_S_dec", std::_Ios_Fmtflags::_S_dec)
    .value("_S_fixed", std::_Ios_Fmtflags::_S_fixed)
    .value("_S_hex", std::_Ios_Fmtflags::_S_hex)
    .value("_S_internal", std::_Ios_Fmtflags::_S_internal)
    .value("_S_left", std::_Ios_Fmtflags::_S_left)
    .value("_S_oct", std::_Ios_Fmtflags::_S_oct)
    .value("_S_right", std::_Ios_Fmtflags::_S_right)
    .value("_S_scientific", std::_Ios_Fmtflags::_S_scientific)
    .value("_S_showbase", std::_Ios_Fmtflags::_S_showbase)
    .value("_S_showpoint", std::_Ios_Fmtflags::_S_showpoint)
    .value("_S_showpos", std::_Ios_Fmtflags::_S_showpos)
    .value("_S_skipws", std::_Ios_Fmtflags::_S_skipws)
    .value("_S_unitbuf", std::_Ios_Fmtflags::_S_unitbuf)
    .value("_S_uppercase", std::_Ios_Fmtflags::_S_uppercase)
    .value("_S_adjustfield", std::_Ios_Fmtflags::_S_adjustfield)
    .value("_S_basefield", std::_Ios_Fmtflags::_S_basefield)
    .value("_S_floatfield", std::_Ios_Fmtflags::_S_floatfield)
    .value("_S_ios_fmtflags_end", std::_Ios_Fmtflags::_S_ios_fmtflags_end)
    .value("_S_ios_fmtflags_max", std::_Ios_Fmtflags::_S_ios_fmtflags_max)
    .value("_S_ios_fmtflags_min", std::_Ios_Fmtflags::_S_ios_fmtflags_min)
    .export_values();

    py::class_<csformula<float100et>>(m, "csformula")
        .def(py::init<const std::string &>())
        // .def("setName", &csformula::setName)
        .def("getExpression", &csformula<float100et>::getExpression, "A function which adds two numbers")
        // .def_property("expression", &csformula<float100et>::getExpression, &csformula<float100et>::setExpression)
        // .def(py::init<const std::string &>());
        .def("get", (std::string(csformula<float100et>::*)(const std::map<std::string, std::string> &) const) & csformula<float100et>::get,
             "Calculate the value of the parsed string using the passed values of the variables.")
        .def("get", (float100et(csformula<float100et>::*)(const std::map<std::string, float100et> &) const) & csformula<float100et>::get,
             "Calculate the value of the parsed string using the passed values of the variables.");
    // .def("set", (void (Pet::*)(int)) &Pet::set, "Set the pet's age")
    // .def("set", (void (Pet::*)(const std::string &)) &Pet::set, "Set the pet's name");
    // .def("getD", &csformula<float100et>::getD);

    py::class_<float100et>(m, "float100et")
        .def(py::init<const std::string &>())
        .def("str", (std::string(float100et::*)(std::streamsize digits, std::ios_base::fmtflags) const) & float100et::str,
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

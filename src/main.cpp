#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cseval.h>
#include <csformula.cpp>

class Animal {
public:
    Animal() { std::cout << "constr"; }
    ~Animal() { std::cout << "destr"; }
    std::string go(int n_times) {
        std::string result;
        for (int i=0; i<n_times; ++i) {
            result += "krya! ";
        }
        return result;
    };
};

int add(int i, int j) {
    return i + j;
}

std::string call_go(Animal *animal) {
    return animal->go(3);
}

namespace py = pybind11;

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

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

    py::class_<Animal> animal(m, "Animal");
    animal.def("go", &Animal::go);
    animal.def(py::init<>());
    m.def("call_go", &call_go);

    // py::class_<Pet>(m, "Pet")
    //     .def(py::init<const std::string &>())
    //     .def("setName", &Pet::setName)
    //     .def("getName", &Pet::getName);

    py::class_<csformula<float100et>>(m, "csformula")
        .def(py::init<const std::string &>())
        // .def("setName", &csformula::setName)
        .def("getExpression", &csformula<float100et>::getExpression)
        // .def(py::init<const std::string &>());
        .def("get", &csformula<float100et>::get)
        .def("getD", &csformula<float100et>::getD);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

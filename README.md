# formula - Arbitrary-precision formula parser and solver for Python

[![PyPI](https://img.shields.io/pypi/v/formula.svg)](https://pypi.org/project/formula/)
[![downloads](https://img.shields.io/pypi/dm/formula.svg)](https://pypistats.org/packages/formula)
[![python](https://img.shields.io/badge/python-2.7%7C3.5%7C3.6%7C3.7%7C3.8-blue)](https://pypi.org/project/formula/)
[![Travis (.com)](https://img.shields.io/travis/com/hozblok/formula)](https://travis-ci.com/github/hozblok/formula/branches)
[![GitHub license](https://img.shields.io/github/license/hozblok/formula)](https://github.com/hozblok/formula/blob/master/LICENSE)
[![Gitter](https://badges.gitter.im/don_vanchos/Lobby.svg)](https://gitter.im/don_vanchos/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

![Usage example](preview.gif)

## Development status

Status: **Production/Stable**

Development plan:

+ Complex numbers support. (Character `i` are reserved for this by default.)

This project built with [pybind11](https://github.com/pybind/pybind11).

## Installation

### On Unix (Linux or OS X)

+ `pip install formula`

### On Windows (Requires Microsoft Visual Studio 2015+)

+ For Python 3.5+:
  + clone this repository
  + `pip install ./formula`
  + if you get an `error: Microsoft Visual C++ 14.0 is required.` install Microsoft
  Visual C++ Build Tools 14.0 from https://visualstudio.microsoft.com/visual-cpp-build-tools/
  and try again. Example of the correct selection to install:
  ![Microsoft Visual C++ Build Tools](winbuildtools.png)
+ For earlier versions of Python, including Python 2.7:

   Pybind11 requires a C++11 compliant compiler (i.e. Visual Studio 2015 on
   Windows). Running a regular `pip install` command will detect the version
   of the compiler used to build Python and attempt to build the extension
   with it. We must force the use of Visual Studio 2015.

  + clone this repository
  + `"%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64`
  + `set DISTUTILS_USE_SDK=1`
  + `set MSSdk=1`
  + `pip install ./formula`

   Note that this requires the user building `formula` to have registry edition
   rights on the machine, to be able to run the `vcvarsall.bat` script.

## Windows runtime requirements

On Windows, the Visual C++ 2015 redistributable packages are a runtime
requirement for this project. It can be found [here](https://www.microsoft.com/en-us/download/details.aspx?id=48145).

If you use the Anaconda python distribution, you may require the Visual Studio
runtime as a platform-dependent runtime requirement for you package:

```yaml
requirements:
  build:
    - python
    - setuptools
    - pybind11

  run:
   - python
   - vs2015_runtime  # [win]
```

## Documentation

**formula** contains case sensitive (by default) string parser.
Let's imagine that we have a string expression, e.g. `"(x^2+y)/sin(a*z)"`.
We want to calculate the value of this function in the following point:

```text
x=0.001, y=0.0000000000000000000000555, z=-2, a=-1,
```

So we pass the expression to the `formula` constructor.

```python
from formula import Formula
f5a = Formula("(x^2+y)/sin(a*z)")
```

And it is enough to call the `get(...)` method or the `get_derivative(...)` to calculate
the value of the expression or the derivative of the expression at this point.

```python
variables = {
  "x": "0.001",
  "y": "0.0000000000000000000000555",
  "z": "-2",
  "a": "-1",
}
value = f5a.get(variables)
x_derivative = f5a.get_derivative("x", variables)
z_derivative = f5a.get_derivative("z", variables)
```

## Test call

```python
from formula import Formula
pi = Formula("2*asin(x)", 64).get({"x": "1"})
```

## License

**formula** is provided under Apache license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.
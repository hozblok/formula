# formula
Python csformula wrapper.
==============

Development status: in developing

Thi project built with [pybind11](https://github.com/pybind/pybind11).

Installation
------------

**On Unix (Linux, OS X)**

 - `pip install formula`

**On Windows (Requires Visual Studio 2015)**

 - For Python 3.5+:
     - clone this repository
     - `pip install ./formula`
 - For earlier versions of Python, including Python 2.7:

   Pybind11 requires a C++11 compliant compiler (i.e. Visual Studio 2015 on
   Windows). Running a regular `pip install` command will detect the version
   of the compiler used to build Python and attempt to build the extension
   with it. We must force the use of Visual Studio 2015.

     - clone this repository
     - `"%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64`
     - `set DISTUTILS_USE_SDK=1`
     - `set MSSdk=1`
     - `pip install ./formula`

   Note that this requires the user building `formula` to have registry edition
   rights on the machine, to be able to run the `vcvarsall.bat` script.


Windows runtime requirements
----------------------------

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


Documentation
--------------------------

In progress...

License
-------

formula is provided under a Apache license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

Test call
---------

```python
from formula import csformula
pi = csformula("2*asin(x)", 64).get({'x': '1'})
```

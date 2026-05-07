"""Setup script is the centre of all activity in building,
distributing, and installing formula using the Distutils."""

import os
import sys
import sysconfig

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

__version__ = "4.0.3"

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


class GetPybindInclude:
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11  # pylint: disable=import-outside-toplevel

        return pybind11.get_include(self.user)


DEV_BOOST_HEADERS = os.path.join(CURRENT_DIR, "boost", "boost")
if __version__.startswith("dev") and os.path.exists(DEV_BOOST_HEADERS):
    BOOST_HEADERS = "boost/"
else:
    BOOST_HEADERS = "boost_headers/"


EXTRA_COMPILE_ARGS = []
if sys.platform == "darwin":
    # macOS libc++ removes std::unary_function in C++17; enable compatibility
    # for bundled Boost headers.
    EXTRA_COMPILE_ARGS.append("-D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION")

EXT_MODULES = [
    Pybind11Extension(
        "formula._formula",
        ["src/main.cpp"],
        include_dirs=[
            # Path to boost headers.
            BOOST_HEADERS,
            # Path to Formula headers.
            "src/",
            # Path to pybind11 headers.
            GetPybindInclude(),
            GetPybindInclude(user=True),
        ],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        language="c++",
    )
]

TEST_DEPS = ["pytest"]

# PyPy 3.8 on Ubuntu may not define LDSHARED, which breaks linking during
# extension builds. Provide a safe default for Linux/PyPy environments.
if sys.implementation.name == "pypy" and sys.platform.startswith("linux"):
    ldshared = sysconfig.get_config_var("LDSHARED")
    if not ldshared:
        cxx = os.environ.get("CXX", "g++")
        ldshared = f"{cxx} -shared"
        os.environ.setdefault("LDSHARED", ldshared)
        config_vars = sysconfig.get_config_vars()
        if config_vars is not None:
            config_vars["LDSHARED"] = ldshared
        if getattr(sysconfig, "_CONFIG_VARS", None) is not None:
            sysconfig._CONFIG_VARS["LDSHARED"] = ldshared

# Read the contents of the README file.
with open(os.path.join(CURRENT_DIR, "README.md")) as readme_file:
    LONG_DESCRIPTION = readme_file.read()


setup(
    author_email="hozblok@gmail.com",
    author="Ivan Ergunov",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    cmdclass={"build_ext": build_ext},
    description="Arbitrary-precision formula parser and solver.",
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    ext_modules=EXT_MODULES,
    extras_require={
        "test": TEST_DEPS,
        "dev": TEST_DEPS,
    },
    install_requires=["pybind11>=2.6,<3"],
    license="Apache-2.0",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    name="formula",
    packages=find_packages(),
    python_requires=">=3.6, <4",
    url="https://github.com/hozblok/formula",
    version=__version__,
    zip_safe=False,
)

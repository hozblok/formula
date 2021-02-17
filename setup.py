"""Setup script is the centre of all activity in building,
distributing, and installing formula using the Distutils."""

import os
import platform
import sys

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

# This `setup_helpers` file was copied from pybind11 ver. 2.6.2.
from setup_helpers import Pybind11Extension, build_ext

__version__ = "2.0.1"

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
elif platform.system() == "Windows":
    BOOST_HEADERS = "boost_win_headers/"
else:
    BOOST_HEADERS = "boost_lin_headers/"


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
        language="c++",
    )
]


class PyTest(TestCommand):
    """Setuptools Test command for invoking pytest."""

    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""  # pylint: disable=attribute-defined-outside-init

    def run_tests(self):
        import shlex  # pylint: disable=import-outside-toplevel

        # import here, cause outside the eggs aren't loaded
        import pytest  # pylint: disable=import-outside-toplevel

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


# Read the contents of the README file.
with open(os.path.join(CURRENT_DIR, "README.md")) as readme_file:
    LONG_DESCRIPTION = readme_file.read()


setup(
    author_email="hozblok@gmail.com",
    author="Ivan Ergunov",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    cmdclass={"build_ext": build_ext, "pytest": PyTest},
    description="Arbitrary-precision formula parser and solver.",
    ext_modules=EXT_MODULES,
    install_requires=["pybind11>=2.4"],
    license="Apache-2.0",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    name="formula",
    packages=find_packages(),
    python_requires=">3.5.*, <4",
    setup_requires=["pybind11>=2.6"],
    tests_require=["pytest>=2.1"],
    url="https://github.com/hozblok/formula",
    version=__version__,
    zip_safe=False,
)

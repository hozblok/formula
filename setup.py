import os
import platform
import sys

import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand

__version__ = "1.0.2"

this_directory = os.path.abspath(os.path.dirname(__file__))


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


dev_boost_headers = os.path.join(this_directory, "boost", "boost")
if __version__.startswith("dev") and os.path.exists(dev_boost_headers):
    boost_headers = "boost/"
elif platform.system() == "Windows":
    boost_headers = "boost_win_headers/"
else:
    boost_headers = "boost_lin_headers/"


ext_modules = [
    setuptools.Extension(
        "formula",
        ["src/main.cpp"],
        include_dirs=[
            # Path to boost headers.
            boost_headers,
            # Path to Formula headers.
            "src/",
            # Path to pybind11 headers.
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language="c++",
    )
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is preferred over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++11 support " "is needed!"
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex

        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


# Read the contents of the README file.
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setuptools.setup(
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
    cmdclass={"build_ext": BuildExt, "pytest": PyTest},
    description="Arbitrary-precision formula parser and solver.",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.2"],
    long_description_content_type="text/markdown",
    long_description=long_description,
    name="formula",
    packages=setuptools.find_packages(),
    python_requires=">2.6, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4",
    setup_requires=["pybind11>=2.2"],
    tests_require=["pytest>=2.1"],
    url="https://github.com/hozblok/formula",
    version=__version__,
    zip_safe=False,
)

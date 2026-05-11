"""Build script for the formula C++/Python extension.

Static metadata (name, description, classifiers, dependencies, etc.) lives
in `pyproject.toml`'s `[project]` table; this file is the imperative
shim that setuptools still needs to build the pybind11 C++ extension. It
also supplies the dynamic version (sourced from
`src/formula/__init__.py`) and the VERSION_INFO macro on the C++ side so
the version string remains a single source of truth.
"""

import os
import re
import sys

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def _read_version():
    init_path = os.path.join(CURRENT_DIR, "src", "formula", "__init__.py")
    with open(init_path, encoding="utf-8") as fh:
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', fh.read(), re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to find __version__ in src/formula/__init__.py")
    return match.group(1)


__version__ = _read_version()


DEV_BOOST_HEADERS = os.path.join(CURRENT_DIR, "boost", "boost")
if __version__.startswith("dev") and os.path.exists(DEV_BOOST_HEADERS):
    BOOST_HEADERS = "boost/"
else:
    BOOST_HEADERS = "boost_headers/"


EXTRA_COMPILE_ARGS = []
if sys.platform == "darwin":
    # Default deployment target for wheels built outside cibuildwheel; CI
    # overrides this via env. 10.15 covers all currently supported macOS
    # releases on Intel; arm64 builds bump to 11.0 automatically.
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "10.15")
    # macOS libc++ removes std::unary_function in C++17; enable compatibility
    # for bundled Boost headers.
    EXTRA_COMPILE_ARGS.append("-D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION")


# Strip symbol tables from the shared object (~25% smaller, no functional loss).
# .dynsym (incl. PyInit__formula) is kept; only .symtab/debug symbols go.
EXTRA_LINK_ARGS = []
if sys.platform == "linux":
    EXTRA_LINK_ARGS.append("-Wl,-s")
elif sys.platform == "darwin":
    # ld64 dropped -s; -x strips local symbols while keeping exported ones.
    EXTRA_LINK_ARGS.append("-Wl,-x")
# Windows/MSVC needs nothing: debug symbols go to a separate .pdb that wheels
# don't ship, so the .pyd carries no embedded symbol table to strip.


EXT_MODULES = [
    Pybind11Extension(
        "formula._formula",
        # Bindings split across translation units to bound per-compile memory.
        [
            "src/cpp/main.cpp",
            "src/cpp/bindings_mp_real.cpp",
            "src/cpp/bindings_mp_complex.cpp",
        ],
        include_dirs=[
            BOOST_HEADERS,
            "src/cpp/",
        ],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        language="c++",
    )
]

setup(
    version=__version__,
    cmdclass={"build_ext": build_ext},
    ext_modules=EXT_MODULES,
)

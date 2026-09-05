"""Setup script for the formula C++/Python extension.

The Python package lives in ``src/formula/`` and the C++ extension sources
live in ``src/cpp/`` (src-layout). The version string is sourced from
``src/formula/__init__.py`` so there is a single source of truth.
"""

import os
import re
import sys

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

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
if sys.platform != "win32":
    # No FMA contraction: the native tracer's double guards must take the
    # same branches as the Python reference (parity tests compare exactly).
    EXTRA_COMPILE_ARGS.append("-ffp-contract=off")
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
            "src/cpp/bindings_trace.cpp",
            "src/cpp/bindings_stage14.cpp",
        ],
        include_dirs=[
            BOOST_HEADERS,
            "src/cpp/",
        ],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        language="c++",
        cxx_std=17,
    )
]

CAPSYSRED_DEPS = ["PyYAML"]
TEST_DEPS = ["pytest"] + CAPSYSRED_DEPS

README_PATH = os.path.join(CURRENT_DIR, "README.md")
if os.path.exists(README_PATH):
    with open(README_PATH, encoding="utf-8") as readme_file:
        LONG_DESCRIPTION = readme_file.read()
else:
    # Don't fail the build if README.md is missing (partial checkout, tampered
    # sdist, custom build root). The long_description is metadata-only.
    LONG_DESCRIPTION = ""


setup(
    author_email="hozblok@gmail.com",
    author="Ivan Ergunov",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    cmdclass={"build_ext": build_ext},
    description="Multiprecision formula parser and solver.",
    ext_modules=EXT_MODULES,
    extras_require={
        "capsysred": CAPSYSRED_DEPS,
        "test": TEST_DEPS,
        "dev": TEST_DEPS,
    },
    license="Apache-2.0",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    name="formula",
    package_dir={"": "src"},
    package_data={"formula.henke": ["*.nff"]},
    packages=find_packages(where="src"),
    python_requires=">=3.11, <4",
    url="https://github.com/hozblok/formula",
    version=__version__,
    zip_safe=False,
)

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


# Header-tree selection.
#
# - "boost_headers/" is the vendored, pruned tree that ships in every sdist
#   and is the default for release builds.
# - "boost/" is the freshly downloaded tree produced by
#   boost_headers/update_command.py, used while bumping Boost versions
#   locally.
#
# Opt into the dev tree explicitly by setting FORMULA_USE_DEV_BOOST=1 in the
# environment. The previous heuristic (any version string starting with
# "dev") missed PEP 440 pre-release identifiers (e.g. "4.1.0.dev0") and
# couldn't be exercised without editing __version__ in source.
DEV_BOOST_HEADERS = os.path.join(CURRENT_DIR, "boost", "boost")
USE_DEV_BOOST = bool(os.environ.get("FORMULA_USE_DEV_BOOST")) and os.path.exists(
    DEV_BOOST_HEADERS
)
BOOST_HEADERS = "boost/" if USE_DEV_BOOST else "boost_headers/"
print(f"setup.py: BOOST_HEADERS={BOOST_HEADERS!r}")


EXTRA_COMPILE_ARGS = []
if sys.platform == "darwin":
    # Default deployment target for wheels built outside cibuildwheel; CI
    # overrides this via env. 10.15 covers all currently supported macOS
    # releases on Intel; arm64 builds bump to 11.0 automatically.
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "10.15")
    # macOS libc++ removes std::unary_function in C++17; enable compatibility
    # for bundled Boost headers.
    EXTRA_COMPILE_ARGS.append("-D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION")


EXT_MODULES = [
    Pybind11Extension(
        "formula._formula",
        ["src/cpp/main.cpp"],
        include_dirs=[
            BOOST_HEADERS,
            "src/cpp/",
        ],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        language="c++",
    )
]

TEST_DEPS = ["pytest"]

with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as readme_file:
    LONG_DESCRIPTION = readme_file.read()


setup(
    author_email="hozblok@gmail.com",
    author="Ivan Ergunov",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    cmdclass={"build_ext": build_ext},
    description="Arbitrary-precision formula parser and solver.",
    ext_modules=EXT_MODULES,
    extras_require={
        "test": TEST_DEPS,
        "dev": TEST_DEPS,
    },
    license="Apache-2.0",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    name="formula",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9, <4",
    url="https://github.com/hozblok/formula",
    version=__version__,
    zip_safe=False,
)

## Running tests

This project ships a compiled Python extension (`formula._formula`). The
extension must be built before tests can import the package. Because the
project uses **src-layout** (the Python package lives at
`src/formula/`, not at the repo root), a regular `pip install` works
fine — there is no source-tree directory to accidentally shadow the
installed package.

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

python -m pip install -U pip setuptools wheel
python -m pip install .[test]

python -m pytest -q
```

For active development, prefer an editable install so changes to the
Python wrapper take effect without reinstalling:

```bash
python -m pip install -e .[test]
```

### Run a subset of tests

```bash
# A single file
python -m pytest -v tests/test_simple.py

# Only the per-function tests added in the latest release
python -m pytest -v tests/functions/

# A single function's tests
python -m pytest -v tests/functions/one_arg/test_sin.py

# By keyword across the whole suite
python -m pytest -v -k "sin or cos"
```

### Troubleshooting

#### Compilation fails

Ensure a working C++ toolchain is on PATH:

| OS       | Toolchain                                            |
|----------|------------------------------------------------------|
| Linux    | `g++` (or `clang++`); `python3-dev`                  |
| macOS    | Xcode Command Line Tools (`xcode-select --install`)  |
| Windows  | Visual Studio Build Tools (C++ workload)             |

#### Stale extension after pulling new C++ changes

A compiled `.so`/`.pyd` from a previous editable build can mask a fresh
build failure. Before re-installing, clean any stray artifacts:

```bash
rm -f src/formula/_formula*.so src/formula/_formula*.pyd
rm -rf src/formula/__pycache__ build *.egg-info
python -m pip install .[test]
```

#### macOS: `ld: library not found` or wheel won't install on older macOS

Set the deployment target before installing so the resulting `.so` is
ABI-compatible with the macOS version you intend to run on:

```bash
export MACOSX_DEPLOYMENT_TARGET=10.15   # Intel; use 11.0+ for Apple Silicon
python -m pip install .[test]
```

#### PyPy on Linux: linker fails with empty/missing `LDSHARED`

Some PyPy builds on Linux ship without a default `LDSHARED`, which breaks
linking of C++ extensions:

```bash
export LDSHARED="g++ -shared"
python -m pip install .[test]
```

(This is harmless for CPython, which always defines `LDSHARED` from
`sysconfig`.)

#### `python setup.py build_ext --inplace` is deprecated

Setuptools >= 58 prints a deprecation warning; future versions will
remove it. Use `pip install .[test]` (or `pip install -e .[test]` for
editable mode) instead — they do the same thing through the supported
PEP 517/660 paths.

### Project layout (src-layout)

```
formula/                  ← repo root
├── src/
│   ├── formula/          ← Python package (import target)
│   │   ├── __init__.py
│   │   ├── formula.py
│   │   └── _formula.so   ← built C++ extension (after build)
│   └── cpp/              ← C++ extension sources
│       ├── main.cpp
│       ├── csconstants.hpp
│       ├── cseval/
│       └── csformula/
├── tests/
│   ├── functions/        ← per-function tests
│   ├── test_operations.py
│   └── test_simple.py
├── boost_headers/
├── setup.py
└── pyproject.toml
```

Running pytest from the repo root resolves `import formula` from the
installed wheel in `site-packages`, not from the source tree — this is
what makes the layout robust against the `ModuleNotFoundError:
No module named 'formula._formula'` shadowing trap that flat layouts
suffer from.

### What CI does

The GitHub Actions test workflows
([test-mac.yml](../.github/workflows/test-mac.yml),
[test-ubu.yml](../.github/workflows/test-ubu.yml),
[test-win.yml](../.github/workflows/test-win.yml)) run
`pip install .[test]` (non-editable) and then `python -m pytest -q`.
The publish workflow's wheel tests use `cibuildwheel`, which installs
the built wheel into a fresh venv and runs `pytest {project}/tests` —
see [pyproject.toml](../pyproject.toml) `[tool.cibuildwheel]`.

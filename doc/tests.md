## Running tests

This project ships a compiled Python extension (`formula._formula`). The
extension must be built before tests can import the package. The path of
least resistance is an **editable install** — it builds the extension and
places the resulting `.so` (or `.pyd` on Windows) directly into the source
tree where `pytest` can find it.

### Quick start (recommended)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

python -m pip install -U pip setuptools wheel

# Editable install — builds the C++ extension and installs test deps.
python -m pip install -e .[test]

python -m pytest -q
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

#### `ModuleNotFoundError: No module named 'formula._formula'`

Most common cause is **source-tree shadowing**. When `pytest` runs from
the repo root, Python finds the source `./formula/` directory before
`site-packages`. If that directory has no compiled `_formula.so`/`.pyd`,
the import of the extension fails — even when the package is installed
in `site-packages`.

Fixes (any one of these):

1. **Use editable install** (preferred):
   ```bash
   python -m pip uninstall -y formula
   python -m pip install -e .[test]
   ```
   Editable installs of packages with `ext_modules` build the extension
   in place. After this, `ls formula/_formula*` should show a compiled
   binary in the source tree.

2. **Use importlib import mode** (when you cannot reinstall, e.g. testing
   a built wheel against the source repo):
   ```bash
   python -m pytest --import-mode=importlib -q
   ```
   This stops `pytest` from prepending the project root to `sys.path`,
   so the installed wheel is resolved instead of the source tree.

3. **Run from outside the repo**:
   ```bash
   cd /tmp && python -m pytest /path/to/formula/tests
   ```

#### Stale extension after pulling new C++ changes

A compiled `.so` from a previous build can mask the failure of a fresh
build. Before re-installing, clean the tree:

```bash
rm -f formula/_formula*.so formula/_formula*.pyd
rm -rf formula/__pycache__ build *.egg-info
python -m pip install -e .[test]
```

#### Compilation fails

Ensure a working C++ toolchain is on PATH:

| OS       | Toolchain                                   |
|----------|---------------------------------------------|
| Linux    | `g++` (or `clang++`); `python3-dev`         |
| macOS    | Xcode Command Line Tools (`xcode-select --install`) |
| Windows  | Visual Studio Build Tools (C++ workload)    |

#### macOS: `ld: library not found` or wheel won't install on older macOS

Set the deployment target before installing so the resulting `.so` is
ABI-compatible with the macOS version you intend to run on:

```bash
export MACOSX_DEPLOYMENT_TARGET=10.15   # Intel; use 11.0+ for Apple Silicon
python -m pip install -e .[test]
```

#### PyPy on Linux: linker fails with empty/missing `LDSHARED`

Some PyPy builds on Linux ship without a default `LDSHARED`, which breaks
linking of C++ extensions:

```bash
export LDSHARED="g++ -shared"
python -m pip install -e .[test]
```

(This is harmless for CPython, which always defines `LDSHARED` from
`sysconfig`.)

#### `python setup.py build_ext --inplace` is deprecated

Setuptools >= 58 prints a deprecation warning; future versions will
remove it. Use `pip install -e .[test]` instead — it does the same
thing through the supported PEP 660 path.

### What CI does

The GitHub Actions test workflows
([test-mac.yml](../.github/workflows/test-mac.yml),
[test-ubu.yml](../.github/workflows/test-ubu.yml),
[test-win.yml](../.github/workflows/test-win.yml)) follow exactly this
pattern: `pip install -e .[test]` then `python -m pytest -q`. The
publish workflow's wheel tests use `cibuildwheel`, which installs the
built wheel into a fresh venv and runs
`pytest --import-mode=importlib {project}/tests` — see
[pyproject.toml](../pyproject.toml) `[tool.cibuildwheel]`.

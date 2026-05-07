## Running tests

This project ships a compiled Python extension (`formula._formula`). You must build it before running tests.

### Quick start (recommended)

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -U pytest pybind11

# Build the extension in-place so `import formula` works from this repo checkout
python setup.py build_ext --inplace

# Run tests
python -m pytest -q
```

### Run a single file

```bash
python setup.py build_ext --inplace
python -m pytest -v tests/test_simple.py
```

### Notes

- `python setup.py pytest` is deprecated in modern setuptools; use `python -m pytest` instead.

### Troubleshooting

- If you see `ModuleNotFoundError: No module named 'formula._formula'`:
  - you didn’t build the extension (or you built it with a different Python interpreter)
  - rerun: `python setup.py build_ext --inplace`
- If compilation fails:
  - ensure you have a working C++ toolchain installed for your OS (e.g. `g++`)

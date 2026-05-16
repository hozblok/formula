# Engineering guidance for AI-assisted changes

## Project context

`formula` is an arbitrary-precision formula parser and solver, distributed
as a Python package on PyPI. The Python surface (in `src/formula/`) is a
thin wrapper around a C++ extension built with **pybind11** and
**Boost.Multiprecision**. The extension lives in `src/cpp/` and is built
via the `Pybind11Extension` declared in `setup.py`. Prebuilt wheels are
shipped for CPython 3.9–3.12 and selected PyPy versions across Linux,
macOS, and Windows.

This file documents the engineering principles that should guide
AI-assisted changes to the Python surface. The C++ side has its own
constraints (Boost templates, ABI compatibility) and isn't covered here.

## Coding principles for AI-assisted changes

### 1. Prefer obvious failures over silent acceptance of wrong-shaped input

This is the lead principle. If a caller passes the wrong type or shape to
a public API, the wrapper must raise a clear, named exception whose
message identifies both the parameter and the offending value's type.

- **Do not coerce** unfamiliar types into something that "looks close."
- **Do not fall back** to a default that hides the mismatch.
- **Do not return early** with a sentinel that callers may not check.

The rationale: silent acceptance turns a one-line user mistake into a
latent bug that surfaces minutes or days later, deep inside the C++
extension or a downstream calculation, with no clear trail back to the
original misuse. A loud, well-named exception at the boundary is cheap
to fix and easy to grep for.

Concretely, prefer:

```python
if not isinstance(values, Mapping):
    raise TypeError(
        f"Solver.__call__ values must be a Mapping; got {type(values).__name__}"
    )
```

over:

```python
if not isinstance(values, Mapping):
    values = {}  # silently drop the misuse
```

### 2. Use the right exception type

- `TypeError` for wrong type (`int` where a `Mapping` is expected).
- `ValueError` for the right type but a bad value (`precision=-1`, an
  empty expression string).
- `RuntimeError` for unexpected runtime state (a vendored file missing
  where the build script needed it).
- **Never** raise a bare `Exception`.

The message should name the parameter and the offending value's type so
the caller can fix it without reading the traceback frame.

### 3. Public API contracts are sticky

Removing a parameter, tightening a type contract, or changing what an
exception signals counts as a breaking change. Mark it as such in the
commit body and call out the version bump implication.

## Test discipline

- Every behavior change comes with a regression test in `tests/`.
- The test must **fail on the pre-change branch and pass on the
  post-change branch.** If you can't construct such a test, the change
  is probably either trivial (style/docs) or its behavior contract
  isn't pinned down enough to land.
- Tests live alongside their behavior. New behavior in
  `src/formula/formula.py` → new file `tests/test_<feature>.py`.
- Use `pytest.raises(<ExceptionType>, match="<regex>")` for failure
  paths; assert the message names the right parameter.
- Do not delete or weaken existing tests to make a new fix land. If an
  existing test legitimately needs to change (e.g. because the API
  contract is changing), call that out explicitly in the commit body.

## Don'ts

- **No drive-by reformatting.** Don't reflow imports, switch quote
  styles, or "while I'm here" rewrites. Each PR's diff should be
  reviewable for the stated change alone.
- **Don't touch unrelated files.** If you spot another issue, open a
  separate PR or an issue.
- **Don't loosen type contracts for ergonomics.** "Accept anything,
  coerce later" is a recipe for the silent-failure mode this document
  is designed to prevent.
- **Don't bypass safety checks** (e.g. `--no-verify` on git hooks,
  `--force` on protected branches, disabling permission prompts) to
  make an obstacle go away. If a check is firing, investigate the
  underlying cause.

## Practical pointers

- The Python wrapper is in `src/formula/formula.py`; the package init
  is `src/formula/__init__.py` and pins `MAX_PRECISION = 8192`.
- Build the extension locally with `pip install -e .` from a venv. The
  pybind11 helpers in `setup.py` auto-discover MSVC on Windows and
  clang/gcc on macOS/Linux.
- Run the suite with `python -m pytest -q tests/`. It should be fast
  (well under a second) since the C++ side does the heavy lifting and
  the Python tests are mostly boundary checks.

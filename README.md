# formula - Multiprecision formula parser and solver

[![PyPI](https://img.shields.io/pypi/v/formula.svg)](https://pypi.org/project/formula/)
[![PyPI - Format](https://img.shields.io/pypi/format/formula)](https://pypi.org/project/formula/)
[![python](https://img.shields.io/badge/python-3.11%7C3.12%7C3.13-blue)](https://pypi.org/project/formula/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/formula)](https://pypistats.org/packages/formula)
[![Test on Ubuntu](https://github.com/hozblok/formula/actions/workflows/test-ubu.yml/badge.svg)](https://github.com/hozblok/formula/actions/workflows/test-ubu.yml)
[![Test on macOS](https://github.com/hozblok/formula/actions/workflows/test-mac.yml/badge.svg)](https://github.com/hozblok/formula/actions/workflows/test-mac.yml)
[![Test on Windows](https://github.com/hozblok/formula/actions/workflows/test-win.yml/badge.svg)](https://github.com/hozblok/formula/actions/workflows/test-win.yml)
[![GitHub license](https://img.shields.io/github/license/hozblok/formula)](https://github.com/hozblok/formula/blob/master/LICENSE)
[![Gitter](https://badges.gitter.im/don_vanchos/Lobby.svg)](https://gitter.im/don_vanchos/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

![Usage example](doc/img/preview.gif)

## Development status

[![PyPI - Status](https://img.shields.io/pypi/status/formula)](https://pypi.org/project/formula/)

Development plan:

-   [x] Deploy new version with complex numbers support. (v4.0)
-   [x] Deploy prebuilt wheels for Linux, macOS (Intel + Apple Silicon), and Windows via [cibuildwheel](https://cibuildwheel.pypa.io/).
-   [x] Add per-function regression tests covering all built-in math functions (see [tests/functions/](tests/functions/)).
-   [x] Support `sign()`, `abs()` and a Pythonic `Number` wrapper class.
-   [ ] Complex numbers tests and examples. (Character `i` is reserved for this by default.)
-   [ ] Formatting documentation. Limits and caveats.
-   [ ] Support `rand` for random number in [0.0 - 1.0].

This project is built with [pybind11](https://github.com/pybind/pybind11) and [boost](https://www.boost.org/doc/libs/1_83_0/libs/multiprecision/doc/html/index.html).

## Installation

```bash
pip install formula
```

Prebuilt wheels are published to PyPI for **Python 3.11 – 3.13** on:

-   **Linux** — `manylinux_2_28` x86_64 (glibc ≥ 2.28)
-   **macOS** — x86_64 and arm64 (Apple Silicon)
-   **Windows** — AMD64 and x86

If a wheel is unavailable for your platform, `pip` will fall back to
building from the source distribution; in that case a working C++
toolchain is required. See [Development → Building from source](#building-from-source).

### On Windows

-   `pip install formula`
-   <details><summary>if you get an `error: Microsoft Visual C++ 14.0 is required.`</summary>
      <p>

    Install Microsoft Visual C++ Build Tools 14.0 from https://visualstudio.microsoft.com/visual-cpp-build-tools/
    and try again. Example of the correct selection to install:
    ![Microsoft Visual C++ Build Tools](doc/img/winbuildtools.png)

    #### Windows runtime requirements

    On Windows, the Visual C++ 2015 redistributable packages are a runtime
    requirement for this project. It can be found [here](https://www.microsoft.com/en-us/download/details.aspx?id=48145).

    If you use the Anaconda python distribution, you may require the Visual Studio
    runtime as a platform-dependent runtime requirement for you package:

    ```yaml
    requirements:
        build:
            - python
            - setuptools
            - pybind11

        run:
            - python
            - vs2015_runtime # [win]
    ```

      </p>
    </details>

## Documentation

**formula** contains case sensitive (by default) string parser.
Let's imagine that we have a string expression: `"(x^2+y)/sin(a*pi)"`:

```python
>>> from formula import Solver
>>> formula = Solver("(x^2+y)/sin(a*pi)", precision=32)
```

Then we want to calculate the value of this function in the following point:

```python
>>> point = {"x": "3", "y": "3e-20", "a": "-0.5"}
```

And it is enough to call the `formula` object to calculate the value of the expression or the derivative of the expression at this point:

```python
>>> formula(point) # (3^2 + 3e-20)/sin(-pi/2)
'-9.00000000000000000003'
>>> formula(point, derivative="x") # 2*3/sin(-pi/2)
'-6'
>>> formula(point, derivative=("y", "a")) # [1/sin(-pi/2),- (3^2 + 3e-20) * cos(-pi/2) / sin(-pi/2)]
['-1', '-1.5633175729821453046351823925394e-47']  # cos(-pi/2) is an epsilon, not exact 0
```

## Simple examples

### One plus one =)

```python
>>> from formula import Solver, FmtFlags
>>> Solver("1+1", precision=32)()
'2'
>>> Solver("1+1", 32)(format_digits=20, format_flags=FmtFlags.showpos)
'+2'
>>> Solver("1+1", 32)(format_digits=20, format_flags=FmtFlags.fixed | FmtFlags.showpos)
'+2.00000000000000000000'
>>> Solver("1+1", 32)(format_digits=20, format_flags=FmtFlags.scientific | FmtFlags.showpos)
'+2.00000000000000000000e+00'
```

### Find the number of PI using arcsin

#### Precision = 32

```python
>>> from formula import Solver, FmtFlags
>>> Solver("2*asin(x)", precision=32)({"x": "1"})
# just 32 digits:
'3.1415926535897932384626433832795'
>>> Solver("2*asin(x)", 32)({"x": "1"}, format_digits=32)
# by default format_digits is equal to precision:
'3.1415926535897932384626433832795'
>>> Solver("2*asin(x)", 32)({"x": "1"}, format_digits=31)
# let's round in accordance with format_digits:
'3.14159265358979323846264338328'
>>> Solver("2*asin(x)", 32)({"x": "1"}, format_digits=30)
'3.14159265358979323846264338328'
>>> Solver("2*asin(x)", 32)({"x": "1"}, format_digits=29)
'3.1415926535897932384626433833'
>>> Solver("2*asin(x)", 32)(1, format_digits=28)
'3.141592653589793238462643383'
>>> Solver("2*asin(x)", 32)(1, format_digits=2)
'3.1'
>>> Solver("2*asin(x)", 32)(1, format_digits=1)
'3'
>>> Solver("2*asin(x)", 32)(1, format_digits=0)
# show the entire chunk of memory, including insignificant digits:
'3.1415926535897932384626433832795028841971'
```

#### Precision = 4096

```python
>>> from formula import Solver, FmtFlags
>>> Solver("2*asin(x)", precision=4096)(1) # 4095 digits of pi after the point ;-)
'3.141592653589793238462643383279502884197169399375105820974944592307816406286
208998628034825342117067982148086513282306647093844609550582231725359408128481
117450284102701938521105559644622948954930381964428810975665933446128475648233
786783165271201909145648566923460348610454326648213393607260249141273724587006
606315588174881520920962829254091715364367892590360011330530548820466521384146
951941511609433057270365759591953092186117381932611793105118548074462379962749
567351885752724891227938183011949129833673362440656643086021394946395224737190
702179860943702770539217176293176752384674818467669405132000568127145263560827
785771342757789609173637178721468440901224953430146549585371050792279689258923
542019956112129021960864034418159813629774771309960518707211349999998372978049
951059731732816096318595024459455346908302642522308253344685035261931188171010
003137838752886587533208381420617177669147303598253490428755468731159562863882
353787593751957781857780532171226806613001927876611195909216420198938095257201
065485863278865936153381827968230301952035301852968995773622599413891249721775
283479131515574857242454150695950829533116861727855889075098381754637464939319
255060400927701671139009848824012858361603563707660104710181942955596198946767
837449448255379774726847104047534646208046684259069491293313677028989152104752
162056966024058038150193511253382430035587640247496473263914199272604269922796
782354781636009341721641219924586315030286182974555706749838505494588586926995
690927210797509302955321165344987202755960236480665499119881834797753566369807
426542527862551818417574672890977772793800081647060016145249192173217214772350
141441973568548161361157352552133475741849468438523323907394143334547762416862
518983569485562099219222184272550254256887671790494601653466804988627232791786
085784383827967976681454100953883786360950680064225125205117392984896084128488
626945604241965285022210661186306744278622039194945047123713786960956364371917
287467764657573962413890865832645995813390478027590099465764078951269468398352
595709825822620522489407726719478268482601476990902640136394437455305068203496
252451749399651431429809190659250937221696461515709858387410597885959772975498
930161753928468138268683868942774155991855925245953959431049972524680845987273
644695848653836736222626099124608051243884390451244136549762780797715691435997
700129616089441694868555848406353422072225828488648158456028506016842739452267
467678895252138522549954666727823986456596116354886230577456498035593634568174
324112515076069479451096596094025228879710893145669136867228748940560101503308
617928680920874760917824938589009714909675985261365549781893129784821682998948
722658804857564014270477555132379641451523746234364542858444795265867821051141
354735739523113427166102135969536231442952484937187110145765403590279934403742
007310578539062198387447808478489683321445713868751943506430218453191048481005
370614680674919278191197939952061419663428754440643745123718192179998391015919
561814675142691239748940907186494231961567945208095146550225231603881930142093
762137855956638937787083039069792077346722182562599661501421503068038447734549
202605414665925201497442850732518666002132434088190710486331734649651453905796
268561005508106658796998163574736384052571459102897064140110971206280439039759
515677157700420337869936007230558763176359421873125147120532928191826186125867
321579198414848829164470609575270695722091756711672291098169091528017350671274
858322287183520935396572512108357915136988209144421006751033467110314126711136
990865851639831501970165151168517143765761835155650884909989859982387345528331
635507647918535893226185489632132933089857064204675259070915481416549859461637
180270981994309924488957571282890592323326097299712084433573265489382391193259
746366730583604142813883032038249037589852437441702913276561809377344403070746
921120191302033038019762110110044929321516084244485963766983895228684783123552
658213144957685726243344189303968642624341077322697802807318915441101044682325
271620105265227211166039666557309254711055785376346682065310989652691862056476
931257058635662018558100729360659876486118'
```

### Other examples

```python
>>> from formula import Solver, FmtFlags
>>> Solver("9.99 + 9e-20 + 9e-51", precision=64)(None, None, 50, FmtFlags.scientific)
'9.99000000000000000009000000000000000000000000000001e+00'
>>> Solver("9.99 + 9e-20 + 9e-51", 64)(None, None, 51, FmtFlags.scientific)
'9.990000000000000000090000000000000000000000000000009e+00'
>>> Solver("0 + 9e-20 + 9e-51", 64)(None, None, 31, FmtFlags.scientific)
'9.0000000000000000000000000000009e-20'
```

### Using `Number` as a multiprecision Python value

`Number` is a small Pythonic wrapper around `Solver` that lets you compose
expressions with the standard arithmetic and comparison operators while
keeping multiprecision throughout:

```python
>>> from formula import Number
>>> a = Number("1.0000000000000000000000000001", precision=64)
>>> b = Number("2", precision=64)
>>> str(a + b)
'3.0000000000000000000000000001'
>>> str(a * b)
'2.0000000000000000000000000002'
>>> str(abs(Number("-3.14", precision=32)))
'3.14'
>>> a > b
False
>>> a == Number("1.0000000000000000000000000001", precision=64)
True
```

Equality compares the `(real, imag)` strings from `pair` — exact complex
identities compare equal, but expressions with drift do not:

```python
>>> Number("i*i") == Number("-1")   # exact — no drift
True
>>> Number("i^4") == Number("1")    # drift in complex ^ → False
False
```

`Solver.pair(values, format_digits, format_flags)` returns a
`(real_str, imag_str)` tuple formatted consistently — real-only results
get `(value, "0")` so pairs are byte-comparable. Used internally
by `Number.__eq__` and `Number.__hash__`.

Supported operators: `+`, `-`, `*`, `/`, `**` (also written as `^` inside
expressions), `abs()`, `==`, `<`, `<=`, `>`, `>=`.

### Finding all ray–surface intersections

`RaySurface` finds **every** intersection of a ray `r(t) = O + t·d` with an
implicit surface `F(x, y, z) = 0`. Substituting the ray reduces the surface to a
single-variable `g(t) = F(O + t·d)`, so the intersections are exactly the real
roots of `g` on `[t_min, t_max]`. `t` is measured in units of `|d|` (the
direction is normalized internally), so for a unit `d` it is the distance along
the ray.

```python
>>> from formula import RaySurface
>>> rs = RaySurface("x*x + y*y - 1", precision=24)      # unit cylinder
>>> ts = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10)  # two crossings: t≈1, t≈3
>>> len(ts)
2
>>> pts = rs.points((-2, 0, 0), (1, 0, 0), t_max=10)    # (x, y, z) on the surface
>>> len(pts)
2
```

Pick the root-finder with `method=`:

| `method`      | Best for                                   | Guarantee |
|---------------|--------------------------------------------|-----------|
| `auto`        | default — picks per surface                | Sturm for polynomials, else Chebyshev ∪ subdivision |
| `sturm`       | algebraic surfaces (quadrics, tori, …)     | exact, complete real-root count |
| `chebyshev`   | smooth, low-oscillation analytic (`sin`/`exp`/…) | self-validating *fit*; captures tangencies |
| `subdivision` | general/oscillatory surfaces               | derivative-bound exclusion (practically reliable) |
| `sampling`    | quick, well-separated roots                | none — may miss thin/tangent features |

`auto` routes algebraic surfaces to `sturm` (exact) and other real surfaces to
`chebyshev` reconciled with `subdivision` as a safety net. Even-multiplicity
(tangent) roots that plain sampling steps over are recovered:

```python
>>> rs = RaySurface("x*x + y*y + z*z - 1", precision=24)   # sphere
>>> len(rs.intersect((-2, 1, 0), (1, 0, 0), t_max=10, method="sturm"))  # grazing
1
```

Keyword options: `t_min` (default `0`), plus per-method knobs such as
`max_degree` (sturm), `cheb_degree` (chebyshev) and `m2_samples` (subdivision).
Complex-valued surfaces are supported by `sturm` (real intersections are the
common roots of `Re g` and `Im g`); the other backends are real-only.

See [doc/ray-surface-intersections.md](doc/ray-surface-intersections.md) for the
full guide and limits, and [doc/ray-surface-design.md](doc/ray-surface-design.md)
for the design notes. Caveats in brief:

- `subdivision` is *practically* reliable rather than formally rigorous — its
  exclusion test is only as good as the estimated bound on `g''`.
- `chebyshev` self-validates the *fit*, not the root isolation. Its
  Chebyshev→monomial step is ill-conditioned at high degree, so it can silently
  miss roots on densely oscillatory surfaces; prefer `subdivision`/`auto` there.
- `sturm` is exact only up to a moderate algebraic degree (default cap 16; raise
  `max_degree` with care — equally-spaced interpolation degrades at high degree).

## Supported functions

These built-in functions are recognized inside expression strings and
covered by the per-function test suite under
[tests/functions/](tests/functions/):

| Arity | Functions                                                            |
|-------|----------------------------------------------------------------------|
| 1     | `abs`, `sign`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| 2     | `+` (add), `-` (sub), `*` (mul), `/` (div), `^` / `**` (pow)         |

Constants: `pi`, `e`. Imaginary unit: `i` (configurable via the
`imaginary_unit` argument).

## Numerical accuracy and drift

`precision=N` guarantees N decimal digits in the **representation** of a
value, not that every operation lands on a mathematically exact result.
Some expressions accumulate floating-point error at the last digits even
though the math identity is exact.

### Operations that are exact at the configured precision

- `+`, `-`, `*`, `/` on real numbers.
- `+`, `-`, `*`, `/` on complex numbers (including long `*` chains —
  `i*i*i*i*i*i*i*i` evaluates to exactly `1`).
- `^` with **real** base and integer exponent (uses square-and-multiply
  internally): `2^10`, `1.1^7`, etc.
- `abs(z)` when `re² + im²` is a perfect square (`abs(3+4*i) == 5`,
  `abs(i) == 1`).
- Transcendentals at points with exact algebraic results: `sin(0)`,
  `cos(0)`, `exp(0)`, `log(1)`, etc.
- Pythagorean identity in real space: `sin(pi/4)^2 + cos(pi/4)^2 == 1`.

### Operations that drift (visible at last digit or beyond)

- **`^` with a complex base, even for trivial integer exponents.**
  Complex `pow(a, b)` is computed as `exp(b·log(a))`, a transcendental
  chain — there is no integer-exponent specialization for complex values
  in boost.multiprecision. Drift grows with the magnitude of the
  exponent and the result:

  | Expression          | Mathematical value | Computed at `precision=24`                                          |
  |---------------------|--------------------|---------------------------------------------------------------------|
  | `i^4`               | `1`                | `1 + i·1e-24`                                                       |
  | `i^64`              | `1`                | `1 + i·1.2e-23`                                                     |
  | `(1+i)^16`          | `256`              | `256 + i·3.78e-21`                                                  |
  | `(1+i)^64`          | `4294967296`       | `4294967296 + i·2.5e-13` (≈ half of the available digits lost)      |
  | `exp(2*i*pi)`       | `1`                | `1 + i·1e-24`                                                       |

  **Workaround:** if the exponent is a small integer, replace `z^n`
  with the explicit product `z*z*…*z` — multiplication is exact for
  complex values.

- **Transcendental functions** (`sin`, `cos`, `exp`, `log`, `asin`,
  `acos`, `atan`, `tan`) when the mathematical result is irrational.
  `log(exp(1))` rounds clean to `1` at the configured precision; the full
  memory chunk (`format_digits=0`) exposes the round-trip drift:
  `1.0000000000000000000000018…`. The same expression with `+i*0` takes the
  complex path and drifts the other way: `0.99999999999999999999999959…`.

## Development

### Building from source

A working C++ toolchain is required when no wheel matches your platform
or when you check the project out for development:

| OS       | Toolchain                                                       |
|----------|-----------------------------------------------------------------|
| Linux    | `g++` (or `clang++`); `python3-dev`                             |
| macOS    | Xcode Command Line Tools (`xcode-select --install`)             |
| Windows  | Visual Studio Build Tools (C++ workload)                        |

```bash
git clone https://github.com/hozblok/formula.git
cd formula
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .[test]
```

The editable install builds the C++ extension and places the resulting
`_formula.so` (or `.pyd`) directly into the source tree, so `pytest` from
the repo root resolves `formula._formula` correctly.

### Running tests

See [doc/tests.md](doc/tests.md).

## License

**formula** is provided under Apache license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

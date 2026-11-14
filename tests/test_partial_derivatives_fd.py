"""Partial derivatives of orders 1..5 against mixed central differences of values only."""

from decimal import MAX_EMAX, MIN_EMIN, Decimal, localcontext
from fractions import Fraction

import pytest

from formula import Formula
from tests.decimal_pairs import parse
from tests.derivative_oracles import central_difference, poly_partial, to_decimal

H = Decimal("1e-25")
TOL = Decimal("1e-38")


@pytest.fixture(autouse=True)
def _decimal_context():
    with localcontext() as ctx:
        ctx.prec, ctx.Emax, ctx.Emin = 650, MAX_EMAX, MIN_EMIN
        yield


def _values(f):
    return lambda at: parse(f.get(at))


def _compare(f, point, names, h=H, tol=TOL):
    """None if D agrees with the central difference of values, else a description."""
    ref = central_difference(_values(f), point, names, h)
    got = parse(f.get_partial_derivative(list(names), point))
    finite = all(v.is_finite() for v in got + ref)
    if finite and all(abs(g - r) <= tol * max(1, abs(r)) for g, r in zip(got, ref)):
        return None
    return f"{''.join(names)}: got {got}, expected {ref}"


# 3.1: nested functions, variable powers, division, real and complex.
EXPRESSIONS = [
    "exp(sin(x*y)+cos(z*t))+w*x", "sin((x+y)*(z-w))+cos(t*x)",
    "log(2+x*x+y*y+z*z)", "sqrt(1+x*x+y*y+z*z+t*t+w*w)",
    "(x+2)^((y+1)*z)*exp(t*w)", "((x+2)/(y+3))^(z+t*w)",
    "sin(x*y*z*t*w)", "atan(x*y+z*t+w)", "atanh(sin(x*y)/2)",
    "asinh(x*y*z)+acosh(2+t*t+w*w)", "asin(sin(x*y)/2)+acos(cos(z*t)/2)",
    "(x*y+z*t)/(2+w*w+x*x)", "abs(x-y)*z*z*t*w",
    "exp(sin(x+i*y))*cos(z+i*t)+w", "log(2+(x+i*y)*(x+i*y))*z*t*w",
    "(x+2+i*y)^(z+i*t)*w", "atan(x+i*y)*sin(z*t*w)",
    "asinh(x+i*y)*acosh(2+z+i*t)*w", "sqrt(2+x+i*y)/(3+z+i*t)*w",
    "tanh(x+i*y)*atanh(z+i*t)*w", "asin(x+i*y)*acos(z+i*t)*w",
]
POINTS = [dict(zip("xyztw", ["0.2", "-0.3", "0.4", "-0.1", "0.5"])),
          dict(zip("xyztw", ["-0.4", "0.2", "-0.3", "0.5", "-0.1"]))]
NAMESETS = ["x", "xx", "xy", "xyz", "xxy", "xxyy", "xyzt", "xxxxx", "xxxyy", "xyztw", "xxyzw"]


@pytest.mark.parametrize("point", POINTS, ids=["p0", "p1"])
@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_against_central_differences_of_values(expr, point):
    f = Formula(expr, precision=256)
    failures = [m for m in (_compare(f, point, names) for names in NAMESETS) if m]
    assert not failures, f"{expr} at {point}:\n" + "\n".join(failures)


# 3.2: complex mixed block, z = x + iy, w = t + is.
TEMPLATES = [
    "exp(sin(z*w))", "tan(z)*tanh(w)", "log(z)*exp(w)", "sqrt(z)*sin(w)",
    "asin(z)*cos(w)", "acos(z)*sinh(w)", "atan(z*w)", "asinh(z)*cosh(w)",
    "acosh(z)*w", "atanh(z)*exp(w)", "z^(w+0.5)", "z^(-1.5)*w+(2-i)^(z*w)",
]
# Im z, Im w != 0: the h = 1e-25 stencil stays off the real-axis cuts and the poles.
COMPLEX_POINTS = [dict(zip("xyts", p)) for p in (["0.4", "0.3", "0.2", "-0.1"],
                                                  ["-0.7", "0.4", "0.2", "-0.1"],
                                                  ["-0.6", "-0.5", "0.2", "-0.1"])]
COMPLEX_NAMESETS = ["x", "xy", "xts", "xxyt", "xyyts", "xxtts"]


@pytest.mark.parametrize("point", COMPLEX_POINTS, ids=["q0", "q1", "q2"])
@pytest.mark.parametrize("template", TEMPLATES)
def test_complex_mixed_block_against_central_differences(template, point):
    expr = template.replace("z", "(x+i*y)").replace("w", "(t+i*s)")
    f = Formula(expr, precision=256)
    failures = [m for m in (_compare(f, point, names) for names in COMPLEX_NAMESETS) if m]
    assert not failures, f"{expr} at {point}:\n" + "\n".join(failures)


# 3.3: the oracle itself on polynomials with known derivatives.
ORACLE_POLYNOMIALS = [
    ("x^5", {(5, 0, 0, 0, 0): Fraction(1)}, {}),
    ("x^2*y^3", {(2, 3, 0, 0, 0): Fraction(1)}, {}),
    ("x*y*z*t*w", {(1, 1, 1, 1, 1): Fraction(1)}, {}),
    # (x + iy)^3 z = (x^3 - 3xy^2) z + i (3x^2 y - y^3) z
    ("(x+i*y)^3*z", {(3, 0, 1, 0, 0): Fraction(1), (1, 2, 1, 0, 0): Fraction(-3)},
     {(2, 1, 1, 0, 0): Fraction(3), (0, 3, 1, 0, 0): Fraction(-1)}),
]


@pytest.mark.parametrize("expr,real,imaginary", ORACLE_POLYNOMIALS)
def test_the_difference_oracle_on_polynomials(expr, real, imaginary):
    f = Formula(expr, precision=256)
    for point in POINTS:
        for names in NAMESETS:
            ref = central_difference(_values(f), point, names, H)
            exact = to_decimal(tuple(poly_partial(p, "xyztw", names, point)
                                     for p in (real, imaginary)))
            assert all(abs(r - e) <= TOL * max(1, abs(e)) for r, e in zip(ref, exact)), names


@pytest.mark.parametrize("expr", [EXPRESSIONS[0], EXPRESSIONS[4], EXPRESSIONS[15], EXPRESSIONS[19]])
def test_the_difference_oracle_is_stable_under_halving_the_step(expr):
    f = Formula(expr, precision=256)
    for names in NAMESETS:
        at_h = central_difference(_values(f), POINTS[0], names, H)
        at_half = central_difference(_values(f), POINTS[0], names, H / 2)
        assert all(abs(a - b) <= TOL * max(1, abs(b)) for a, b in zip(at_h, at_half)), names


# 4.1: both sides of the acosh cut, differences along x stay on the side.
@pytest.mark.parametrize("c", ["1e-3", "-1e-3"])
@pytest.mark.parametrize("x", ["-0.5", "0.25"])
def test_acosh_on_both_sides_of_the_cut(x, c):
    f = Formula(f"acosh(x+i*({c}))", precision=256)
    failures = [m for m in (_compare(f, {"x": x}, "x" * k) for k in range(1, 6)) if m]
    assert not failures, "\n".join(failures)

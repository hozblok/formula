"""Partial derivatives of orders 1..5 against exact rational references."""

from decimal import MAX_EMAX, MIN_EMIN, Decimal, localcontext
from fractions import Fraction
from itertools import combinations_with_replacement
from math import factorial
from random import Random

import pytest

from formula import Formula
from tests.decimal_pairs import parse
from tests.derivative_oracles import exact_point, poly_partial, poly_plus, poly_times, to_decimal

VARIABLES = "xyztw"
ZERO = (0,) * 5


@pytest.fixture(autouse=True)
def _decimal_context():
    with localcontext() as ctx:
        ctx.prec, ctx.Emax, ctx.Emin = 650, MAX_EMAX, MIN_EMIN
        yield


def _finite(z):
    return all(v.is_finite() for v in z)


def _agrees(got, expected, tol):
    """Each component within tol * max(1, |expected component|)."""
    return _finite(got) and all(abs(g - e) <= tol * max(1, abs(e)) for g, e in zip(got, expected))


# 2.1: random polynomial trees, exact monomial derivatives.

def _tree(rng, depth):
    """Random expression over xyztw and its exact polynomial."""
    if depth == 0 or rng.random() < .12:
        if rng.random() < .8:
            index = rng.randrange(5)
            return VARIABLES[index], {tuple(int(i == index) for i in range(5)): Fraction(1)}
        value = rng.randrange(-3, 4)
        return "(" + str(value) + ")", {ZERO: Fraction(value)}
    op = rng.choice(["+", "-", "*", "^", "/"])
    text, a = _tree(rng, depth - 1)
    if op == "^":
        exponent = rng.choice([2, 3])
        out = {ZERO: Fraction(1)}
        for _ in range(exponent):
            out = poly_times(out, a)
        return "(" + text + ")^" + str(exponent), out
    if op == "/":
        divisor = rng.choice([2, 3, 5])
        return "(" + text + ")/" + str(divisor), {p: c / divisor for p, c in a.items()}
    other, b = _tree(rng, depth - 1)
    out = poly_times(a, b) if op == "*" else poly_plus(a, b, -1 if op == "-" else 1)
    return "(" + text + op + other + ")", out


def _polynomial_cases():
    """The 14 cases of seed 713579: 10 real, the last 4 with an imaginary part."""
    rng, cases = Random(713579), []
    for case in range(14):
        expr, poly = _tree(rng, 4)
        expr = "(" + expr + ")+x*y*z*t*w"
        poly = poly_plus(poly, {(1, 1, 1, 1, 1): Fraction(1)})
        imaginary = {}
        if case >= 10:
            imag_expr, imaginary = _tree(rng, 3)
            expr += "+i*(" + imag_expr + ")"
        cases.append((expr, poly, imaginary))
    return cases


POLYNOMIALS = _polynomial_cases()
POLYNOMIAL_POINTS = [dict(zip(VARIABLES, ["0", "-1", "2", "1", "-2"])),
                     dict(zip(VARIABLES, ["2", "1", "-1", "-2", "0"]))]
# All multisets of xyztw of sizes 1..5: 251 per point.
NAMESETS = [names for k in range(1, 6) for names in combinations_with_replacement(VARIABLES, k)]


@pytest.mark.parametrize("point", POLYNOMIAL_POINTS, ids=["p0", "p1"])
@pytest.mark.parametrize("case", range(len(POLYNOMIALS)))
def test_random_polynomials_against_exact_monomial_derivatives(case, point):
    expr, poly, imaginary = POLYNOMIALS[case]
    f = Formula(expr, precision=64)
    failures = []
    for names in NAMESETS:
        expected = tuple(to_decimal((poly_partial(p, VARIABLES, names, point),))[0]
                         for p in (poly, imaginary))
        got = parse(f.get_partial_derivative(list(names), point))
        if not _agrees(got, expected, Decimal("1e-50")):
            failures.append(f"{''.join(names)}: got {got}, expected {expected}")
    assert not failures, f"{expr} at {point}:\n" + "\n".join(failures[:10])


# 2.2: x^y at x = 1, D_x^a D_y^b x^y = d^b/dy^b [y (y - 1) ... (y - a + 1)].

def _power_at_one(y, a, b):
    coefficients = [Fraction(1)]  # ascending powers of y
    for j in range(a):
        coefficients = [(coefficients[i - 1] if i else 0)
                        - j * (coefficients[i] if i < len(coefficients) else 0)
                        for i in range(len(coefficients) + 1)]
    for _ in range(b):
        coefficients = [i * c for i, c in enumerate(coefficients)][1:]
    return sum(c * y ** i for i, c in enumerate(coefficients))


XY_NAMESETS = [names for k in range(1, 6) for names in combinations_with_replacement("xy", k)]


@pytest.mark.parametrize("y", ["-3", "-1", "0", "0.5", "1", "2", "3", "5"])
@pytest.mark.parametrize("precision", [16, 32, 64, 256])
@pytest.mark.parametrize("expr", ["x^y", "(x+0*i)^y"])
def test_variable_power_at_base_one_against_falling_factorials(expr, precision, y):
    f = Formula(expr, precision=precision)
    tol = Decimal(10) ** (6 - precision)
    failures = []
    for names in XY_NAMESETS:
        a, b = names.count("x"), names.count("y")
        expected = to_decimal((_power_at_one(exact_point(y), a, b),))[0]
        got = parse(f.get_partial_derivative(list(names), {"x": "1", "y": y}))
        bound = tol * max(1, abs(expected))
        if not (_finite(got) and abs(got[0] - expected) <= bound and abs(got[1]) <= bound):
            failures.append(f"{''.join(names)}: got {got}, expected {expected}")
    assert not failures, f"{expr}, precision {precision}, y = {y}:\n" + "\n".join(failures)


# 1.2: the quotient rule of higher order.

@pytest.mark.parametrize("x,y", [("2", "3"), ("0.3", "-2"), ("-0.3", "2")])
@pytest.mark.parametrize("expr", ["x/y", "x/y+0*i"])
def test_quotient_rule_of_higher_order(expr, x, y):
    # D_y^k (x/y) = (-1)^k k! x / y^(k+1), D_x D_y^k (x/y) = (-1)^k k! / y^(k+1).
    f, at = Formula(expr, precision=64), {"x": x, "y": y}
    X, Y = exact_point(x), exact_point(y)
    for k in range(1, 6):
        expected = to_decimal(((-1) ** k * factorial(k) * X / Y ** (k + 1), 0))
        assert _agrees(parse(f.get_partial_derivative(["y"] * k, at)), expected,
                       Decimal("1e-55")), ("y" * k)
    for k in range(1, 5):
        expected = to_decimal(((-1) ** k * factorial(k) / Y ** (k + 1), 0))
        assert _agrees(parse(f.get_partial_derivative(["x"] + ["y"] * k, at)), expected,
                       Decimal("1e-55")), ("x" + "y" * k)

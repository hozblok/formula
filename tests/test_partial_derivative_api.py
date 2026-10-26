"""get_partial_derivative: order range, seeds between calls, permutations, structure."""

from decimal import MAX_EMAX, MIN_EMIN, Decimal, localcontext
from fractions import Fraction
from itertools import permutations
from math import factorial

import pytest

from formula import Formula
from tests.decimal_pairs import parse
from tests.derivative_oracles import central_difference, poly_partial, to_decimal


@pytest.fixture(autouse=True)
def _decimal_context():
    with localcontext() as ctx:
        ctx.prec, ctx.Emax, ctx.Emin = 650, MAX_EMAX, MIN_EMIN
        yield


def D(expr, names, point, precision=64):
    return parse(Formula(expr, precision=precision).get_partial_derivative(list(names), point))


# 5.1: orders 1..5 only; an error leaves the object usable; no seeds kept between calls.

@pytest.mark.parametrize("expr", ["x^3*y", "x^3*y+i*x"])
def test_order_out_of_range_raises_and_the_object_stays_usable(expr):
    f, at = Formula(expr, precision=64), {"x": "2", "y": "3"}
    for names in ([], ["x"] * 6, ["x", "y"] * 3):
        with pytest.raises(ValueError):
            f.get_partial_derivative(names, at)
        assert f.get_partial_derivative(["x", "x", "y"], at) == \
            Formula(expr, precision=64).get_partial_derivative(["x", "x", "y"], at)


@pytest.mark.parametrize("expr", ["x^3*y^2+sin(x*y)", "x^3*y^2+sin(x*y)+i*x*y"])
def test_alternating_calls_on_one_object_match_fresh_objects(expr):
    f = Formula(expr, precision=64)
    p, q = {"x": "0.7", "y": "-1.3"}, {"x": "2", "y": "0.25"}
    calls = [(["x"], p), (["x", "y", "y"], q), (["y"], p), (["x"] * 5, q),
             (["y", "x"], p), (["x"], q), (["y"] * 4, p), (["x", "x"], p)]
    for names, point in calls:
        assert f.get_partial_derivative(names, point) == \
            Formula(expr, precision=64).get_partial_derivative(names, point), names


# 5.2: permutations against independent oracles, structural zeros, long names.

def test_permutations_of_a_monomial_against_the_exact_polynomial():
    point = dict(zip("xyztw", ["0.5", "-2", "3", "0.25", "-1"]))
    poly = {(2, 1, 1, 1, 1): Fraction(1)}
    for names in ("xxyzw", "xyztw"):
        exact = to_decimal((poly_partial(poly, "xyztw", names, point), 0))
        for order in set(permutations(names)):
            got = D("x*x*y*z*t*w", order, point)
            assert all(abs(g - e) <= Decimal("1e-55") * max(1, abs(e))
                       for g, e in zip(got, exact)), order


@pytest.mark.parametrize("expr", ["exp(sin(x*y)+cos(z*t))+w*x", "sin(x*y*z*t*w)",
                                  "(x+2+i*y)^(z+i*t)*w"])
def test_permutations_of_nonlinear_expressions_against_differences(expr):
    f = Formula(expr, precision=256)
    point = dict(zip("xyztw", ["0.2", "-0.3", "0.4", "-0.1", "0.5"]))
    values = lambda at: parse(f.get(at))
    for names in ("xxyzw", "xyztw"):
        ref = central_difference(values, point, names, Decimal("1e-25"))
        for order in set(permutations(names)):
            got = parse(f.get_partial_derivative(list(order), point))
            assert all(abs(g - r) <= Decimal("1e-38") * max(1, abs(r))
                       for g, r in zip(got, ref)), order


@pytest.mark.parametrize("expr", ["x^2*y", "x^2*y+i*y"])
@pytest.mark.parametrize("names", ["z", "xxz", "xyz", "zzzzz", "xxyzt"])
def test_a_variable_absent_from_the_formula_gives_zero(expr, names):
    assert D(expr, names, {"x": "2", "y": "3"}) == (0, 0)


@pytest.mark.parametrize("expr,derivatives", [
    # At x = 0 the first derivative vanishes, the higher ones do not.
    ("x^2", [0, 2, 0, 0, 0]),
    ("x^3", [0, 0, 6, 0, 0]),
    ("x^5", [0, 0, 0, 0, 120]),
    ("exp(x^2)", [0, 2, 0, 12, 0]),
    ("sin(x^2)", [0, 2, 0, 0, 0]),
])
def test_a_zero_first_derivative_does_not_switch_off_the_higher_orders(expr, derivatives):
    for k, expected in enumerate(derivatives, 1):
        assert D(expr, "x" * k, {"x": "0"}) == (expected, 0), k


def test_mixed_derivative_through_a_zero_first_derivative():
    # x^2 y^3 at x = 0: d^2/dx^2 d^3/dy^3 = 2 * 6.
    assert D("x^2*y^3", "xxyyy", {"x": "0", "y": "1"}) == (12, 0)


@pytest.mark.parametrize("name", ["alpha", "Beta_2", "_x", "x_", "e10"])
def test_multi_character_names(name):
    # name^5 at 2: 5 * 16, 20 * 8, 60 * 4, 120 * 2, 120.
    for k, expected in enumerate([80, 160, 240, 240, 120], 1):
        assert D(f"{name}^5", [name] * k, {name: "2"}) == (expected, 0), k


# 5.6: infinite derivatives at a zero base raise; polynomials at 0 are finite.

@pytest.mark.parametrize("expr,k", [("x^0.5", 2), ("x^2.5", 3), ("x^4.5", 5),
                                    ("x^0.5+0*i", 2), ("x^2.5+0*i", 3), ("x^4.5+0*i", 5)])
def test_an_infinite_derivative_at_a_zero_base_raises(expr, k):
    with pytest.raises(ValueError):
        Formula(expr, precision=64).get_partial_derivative(["x"] * k, {"x": "0"})


@pytest.mark.parametrize("n", range(6))
def test_powers_at_zero_have_finite_polynomial_derivatives(n):
    for k in range(1, 6):
        assert D(f"x^{n}", "x" * k, {"x": "0"}) == (factorial(n) if k == n else 0, 0), k

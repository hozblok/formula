"""Formula.get_partial_derivative of orders 2..5 on real and complex formulas."""

from decimal import Decimal, localcontext
from itertools import combinations_with_replacement, permutations
from math import comb, factorial, perm

import pytest

from formula import Formula
from tests.decimal_pairs import add, near, parse, rel_near, scale, times_i
from tests.derivatives import first_derivative

PRECISION = 64
H = Decimal("1e-20")
TOL = Decimal("1e-30")
POINT = {"x": "0.3", "y": "0.7", "z": "1.1"}
MULTISETS = [names for order in (2, 3, 4, 5)
             for names in combinations_with_replacement("xyz", order)]

FUNCTIONS = ["sin", "asin", "cos", "acos", "tan", "atan", "log", "sqrt", "exp",
             "log2", "log10", "sinh", "cosh", "tanh", "asinh", "atanh"]
REAL = [f"{name}(x*y*z)" for name in FUNCTIONS] + [
    "acosh(1+x*y*z)",
    "x^y*z",
    "x/(y+z)",
    "(x+y)^3*z",
    "x^2*y^3+sin(x*z)",
    "abs(x-y)*z^2",
    "sign(x)*y*z",
    "exp(x)^y/z",
]
COMPLEX = [f"{name}(x*y*z+i*x)" for name in FUNCTIONS] + [
    "acosh(1+x*y*z+i*x)",
    "x^(y+i*z)",
    "(2+i)^(x*y*z)",
    "(x+i*y)^3*z",
    "x/(y+i*z)",
    "exp(i*x*y)*z^2",
]


def partial(expr, names, values=POINT, precision=PRECISION):
    f = Formula(expr, precision=precision)
    if len(names) == 1:
        return first_derivative(f, names[0], values)
    return f.get_partial_derivative(list(names), values)


def close(expr, names, precision=PRECISION, h=H, tol=TOL):
    """Partial over names against d/d names[-1] of the partial over names[:-1]."""
    with localcontext() as ctx:
        ctx.prec = 2 * precision + 50
        v = names[-1]
        hi = dict(POINT, **{v: str(Decimal(POINT[v]) + h)})
        lo = dict(POINT, **{v: str(Decimal(POINT[v]) - h)})
        upper = parse(partial(expr, names[:-1], hi, precision))
        lower = parse(partial(expr, names[:-1], lo, precision))
        got = parse(partial(expr, names, precision=precision))
        ref = [(a - b) / (2 * h) for a, b in zip(upper, lower)]
        return all(abs(g - r) < tol * max(1, abs(r)) for g, r in zip(got, ref))


@pytest.mark.parametrize("names", MULTISETS, ids="".join)
@pytest.mark.parametrize("expr", REAL + COMPLEX)
def test_matches_central_difference_of_lower_order(expr, names):
    assert close(expr, names)


@pytest.mark.parametrize("names", ["xy", "xyz", "xxx", "xxyyz"])
@pytest.mark.parametrize("expr", ["exp(x*y*z)", "x^y*z", "sin(x+i*y)*z"])
def test_precision_1024(expr, names):
    assert close(expr, names, precision=1024, h=Decimal("1e-300"), tol=Decimal("1e-500"))


@pytest.mark.parametrize("expr", REAL + COMPLEX)
def test_symmetric_in_the_order_of_variables(expr):
    for names in [("x", "y"), ("x", "y", "z"), ("x", "x", "y"), ("x", "x", "y", "z", "z")]:
        ref = parse(partial(expr, names))
        for perm in set(permutations(names)):
            got = parse(partial(expr, perm))
            assert all(abs(g - r) < Decimal("1e-50") * max(1, abs(r)) for g, r in zip(got, ref))


def test_polynomial_values():
    at = {"x": "2", "y": "3", "z": "5"}
    expected = {"xy": 360, "xxx": 270, "xxy": 360, "xyz": 72, "yyz": 16, "zzz": 0,
                "xxxy": 180, "xxyz": 72, "xyyz": 24, "xxxyy": 60, "xxxyz": 36,
                "xxyyz": 24, "xxxxy": 0}
    for names, value in expected.items():
        assert float(partial("x^3*y^2*z", names, at)) == value


def test_trivial():
    assert partial("x*y*z", "xyz") == "1"
    assert partial("x*y", "xyz") == "0"
    assert partial("5", "xyz", {}) == "0"
    assert partial("x^3", "xxx", {"x": "0"}) == "6"
    assert partial("x^2", "xxx", {"x": "0"}) == "0"
    assert partial("x^2", "xx", {"x": "0"}) == "2"
    assert partial("x^1.5*y", "xy", {"x": "0", "y": "1"}) == "0"
    assert partial("x^5", "xxxxx", {"x": "0"}) == "120"
    assert partial("x^4", "xxxxx", {"x": "0"}) == "0"
    assert partial("x*y*z*x*y", "xxyyz") == "4"


@pytest.mark.parametrize(
    "expr,names",
    [("x^0.5", "xx"), ("x^2.5", "xxx"), ("x^1.5*y", "xxy"), ("x^4.5", "xxxxx"),
     ("x^0.5+i", "xx")],
)
def test_infinite_derivative_raises(expr, names):
    with pytest.raises(ValueError):
        partial(expr, names, {"x": "0", "y": "1"})


def test_complex_power_at_zero_base():
    assert parse(partial("x^2+i", "xx", {"x": "0"})) == (2, 0)
    assert parse(partial("x^3*i", "xxx", {"x": "0"})) == (0, 6)
    assert parse(partial("x^5*i", "xxxxx", {"x": "0"})) == (0, 120)
    assert parse(Formula("x^0+i").get_derivative("x", {"x": "0"})) == (0, 0)


def test_order_one_matches_get_derivative():
    for expr in REAL + COMPLEX:
        f = Formula(expr, precision=PRECISION)
        assert f.get_partial_derivative(["y"], POINT) == f.get_derivative("y", POINT)


@pytest.mark.parametrize("names", [[], ["x"] * 6])
def test_order_out_of_range_raises(names):
    with pytest.raises(ValueError):
        Formula("x").get_partial_derivative(names, {"x": "1"})


def test_complex_abs_not_implemented():
    with pytest.raises(ValueError):
        partial("abs(x+i)", "xx", {"x": "1"})


# Identities: the derivatives against exact mathematics, not against each other.

AT = {"x": "0.3", "y": "0.7", "z": "1.1", "t": "0.4"}


@pytest.fixture(autouse=True)
def _decimal_precision():
    with localcontext() as ctx:
        ctx.prec = 600
        yield


def d(expr, names, values=AT, precision=PRECISION):
    """(re, im) of the partial derivative over names."""
    return parse(partial(expr, names, values, precision))


def value(expr, values=AT, precision=PRECISION):
    return parse(Formula(expr, precision=precision).get(values))


@pytest.mark.parametrize("precision,h,eps", [(64, "1e-10", "1e-55"), (256, "1e-40", "1e-230")])
@pytest.mark.parametrize("expr", REAL + COMPLEX + ["x^x", "x^(1/x)*sin(x)"])
def test_taylor_polynomial_of_order_five_reproduces_the_function(expr, precision, h, eps):
    # f(x0 + h) = sum_k f^(k)(x0) h^k / k! + O(h^6).
    h = Decimal(h)
    series = value(expr, precision=precision)
    for k in range(1, 6):
        series = add(series, scale(h**k / factorial(k), d(expr, "x" * k, precision=precision)))
    shifted = dict(AT, x=str(Decimal(AT["x"]) + h))
    assert near(series, value(expr, shifted, precision), Decimal(eps))


@pytest.mark.parametrize("expr,x", [
    ("asin(sin(x))", "0.3"), ("sin(asin(x))", "0.3"), ("acos(cos(x))", "0.3"),
    ("cos(acos(x))", "0.3"), ("atan(tan(x))", "0.3"), ("tan(atan(x))", "0.3"),
    ("exp(log(x))", "0.3"), ("log(exp(x))", "0.3"), ("sinh(asinh(x))", "0.3"),
    ("asinh(sinh(x))", "0.3"), ("cosh(acosh(x))", "1.3"), ("acosh(cosh(x))", "1.3"),
    ("tanh(atanh(x))", "0.3"), ("atanh(tanh(x))", "0.3"), ("sqrt(x)^2", "0.3"),
    ("2^log2(x)", "0.3"), ("10^log10(x)", "0.3"), ("exp(log(x+i))-i", "0.3"),
])
def test_inverse_compositions_are_the_identity(expr, x):
    values = {"x": x}
    assert near(d(expr, "x", values), (1, 0))
    for k in range(2, 6):
        assert near(d(expr, "x" * k, values), (0, 0))


@pytest.mark.parametrize("expr", [
    "sin(x*y)^2 + cos(x*y)^2",
    "cosh(x+y^2)^2 - sinh(x+y^2)^2",
    "(1 + tan(x*y)^2)*cos(x*y)^2",
    "exp(x*y)*exp(-x*y)",
    "log(x*y) - log(x) - log(y)",
    "sin(x+i*y)^2 + cos(x+i*y)^2",
])
def test_constant_identities_have_vanishing_derivatives(expr):
    for order in range(1, 6):
        for names in combinations_with_replacement("xy", order):
            assert near(d(expr, names), (0, 0)), names


@pytest.mark.parametrize("expr", [
    "exp(x)*sin(y)",
    "log(x^2+y^2)",
    "x^5 - 10*x^3*y^2 + 5*x*y^4",
    "atan(y/x)",
    "cosh(2*x)*cos(2*y)",
])
def test_laplacian_of_harmonic_functions_vanishes(expr):
    assert near(add(d(expr, "xx"), d(expr, "yy")), (0, 0))


def test_biharmonic_and_newtonian_potentials():
    # Away from 0: Delta^2 (r^2 log r^2) = 0 in the plane; Delta^2 r = 0 and
    # Delta (1/r) = 0 in space.
    plane = "(x^2+y^2)*log(x^2+y^2)"
    assert near(add(d(plane, "xxxx"), scale(2, d(plane, "xxyy")), d(plane, "yyyy")), (0, 0))
    space = "sqrt(x^2+y^2+z^2)"
    assert near(add(*[d(space, a + a + b + b) for a in "xyz" for b in "xyz"]), (0, 0))
    newton = "1/sqrt(x^2+y^2+z^2)"
    assert near(add(d(newton, "xx"), d(newton, "yy"), d(newton, "zz")), (0, 0))


@pytest.mark.parametrize("expr", [
    "exp(sin(x+i*y))",
    "log(1+(x+i*y)^2)",
    "(x+i*y)^5/(2+x+i*y)",
    "atan(x+i*y)",
    "sqrt(x+i*y)",
])
def test_holomorphic_functions_satisfy_cauchy_riemann_to_order_five(expr):
    # F = g(x + i y): d^(a+b) F / dx^a dy^b = i^b g^(a+b).
    for order in range(1, 6):
        along_x = d(expr, "x" * order)
        for b in range(1, order + 1):
            assert near(d(expr, "x" * (order - b) + "y" * b), times_i(along_x, b))


def test_heat_equation_to_order_five():
    # u = exp(-4t) sin(2x): u_t = u_xx, u_tt = u_xxxx, u_ttx = u_xxxxx.
    u = "exp(-4*t)*sin(2*x)"
    assert near(d(u, "t"), d(u, "xx"))
    assert near(d(u, "tt"), d(u, "xxxx"))
    assert near(d(u, "ttx"), d(u, "xxxxx"))


def test_euler_identities_for_a_homogeneous_function():
    # Degree 4: sum_b C(k, b) x^(k-b) y^b F_(x^(k-b) y^b) = 4!/(4-k)! F, 0 for k = 5.
    f = "x^3*y^2/(x+y) + x^2*sqrt(x^4+y^4)"
    x, y = Decimal(AT["x"]), Decimal(AT["y"])
    for k in range(1, 6):
        lhs = add(*[scale(comb(k, b) * x ** (k - b) * y**b, d(f, "x" * (k - b) + "y" * b))
                    for b in range(k + 1)])
        assert near(lhs, scale(perm(4, k), value(f)))


def test_leibniz_rule_of_order_five():
    # (sin(x) exp(2x))^(5) = sum_k C(5, k) sin^(k)(x) (exp(2x))^(5-k).
    def nth(expr, k):
        return value(expr) if k == 0 else d(expr, "x" * k)

    terms = [scale(comb(5, k) * nth("sin(x)", k)[0], nth("exp(2*x)", 5 - k)) for k in range(6)]
    assert near(d("sin(x)*exp(2*x)", "xxxxx"), add(*terms))


def test_infinite_derivative_behind_a_zero_factor_is_refused():
    # (x^2)^(5/2) = |x|^5 at 0: orders 1, 2 vanish; order 3 needs f'''(0) = inf times 0.
    at0 = {"x": "0"}
    assert partial("(x^2)^(5/2)", "x", at0) == "0"
    assert partial("(x^2)^(5/2)", "xx", at0) == "0"
    with pytest.raises(ValueError):
        partial("(x^2)^(5/2)", "xxx", at0)


def test_piecewise_factor_from_a_comparison():
    assert float(partial("(x>0)*x^3", "xxx", {"x": "1"})) == 6
    assert float(partial("(x>0)*x^3", "xxx", {"x": "-1"})) == 0


@pytest.mark.parametrize("names", ["xy", "xxz", "xyyz", "xxyzz"])
@pytest.mark.parametrize("expr", ["tan(x*y*z+i*x)", "x^y*z", "acosh(1+x*y*z)"])
def test_precision_64_agrees_with_precision_256(expr, names):
    assert near(d(expr, names, precision=64), d(expr, names, precision=256), Decimal("1e-60"))


def test_variable_absent_from_the_formula_gives_zero_at_every_order():
    for order in range(1, 6):
        assert partial("x^2*y", "z" * order, AT) == "0"


# tanh^(k) by T = tanh(x), S = 1/cosh(x)^2, free of the 1 - T^2 cancellation.
TANH_DERIVATIVES = {
    "x": "S",
    "xx": "-2*T*S",
    "xxx": "4*T^2*S - 2*S^2",
    "xxxx": "16*T*S^2 - 8*T^3*S",
    "xxxxx": "16*S^3 - 88*T^2*S^2 + 16*T^4*S",
}


@pytest.mark.parametrize("x,y", [("0.3", "0"), ("100", "100"), ("-100", "100"), ("3000", "3000")])
@pytest.mark.parametrize("names", list(TANH_DERIVATIVES))
def test_tanh_and_tan_derivatives_where_tanh_rounds_to_one(names, x, y):
    # exp(2y) keeps the derivatives near 1 where tanh(x) is +-1 to the precision.
    closed = TANH_DERIVATIVES[names].replace("S", "(1/cosh(x)^2)").replace("T", "tanh(x)")
    at = {"x": x, "y": y}
    expected = value(f"exp(2*y)*({closed})", at)
    assert near(d("exp(2*y)*tanh(x)", names, at), expected)
    # tan(i x) = i tanh(x)
    assert near(d("exp(2*y)*tan(i*x)", names, at), times_i(expected, 1))


# Points in all four quadrants, off every branch cut and pole.
PLANE = [("-2", "1"), ("-2", "-1"), ("-0.5", "0.3"), ("-0.5", "-0.3"),
         ("0.4", "-1.7"), ("2.5", "-0.6"), ("1.5", "0.8"), ("-1.2", "2.1")]
BRANCHED = [f"{name}(z)" for name in FUNCTIONS + ["acosh"]] + [
    "z^2.5", "z^(-1.5)", "z^(0.5+i)", "(2-3*i)^z", "z^z"]


@pytest.mark.parametrize("x,c", PLANE)
@pytest.mark.parametrize("template", BRANCHED)
def test_complex_derivatives_follow_the_branch_of_the_function(template, x, c):
    # F(x) = f(x + i c): order 1 against a central difference of the values of f;
    # each order k against the Taylor remainder of the values,
    # (f(z + h) - sum_{j<k} f^(j) h^j / j!) k! / h^k = f^(k) + O(h).
    expr = template.replace("z", f"(x+i*({c}))")
    at = {"x": x}
    h = Decimal("1e-20")
    upper = value(expr, {"x": str(Decimal(x) + h)})
    lower = value(expr, {"x": str(Decimal(x) - h)})
    slope = [(a - b) / (2 * h) for a, b in zip(upper, lower)]
    assert near(d(expr, "x", at), slope, Decimal("1e-30"))
    h = Decimal("1e-10")
    shifted = value(expr, {"x": str(Decimal(x) + h)})
    partial_sum = value(expr, at)
    for k in range(1, 6):
        dk = d(expr, "x" * k, at)
        remainder = [(s - p) * factorial(k) / h**k for s, p in zip(shifted, partial_sum)]
        assert near(remainder, dk, Decimal("1e-7")), k
        partial_sum = add(partial_sum, scale(h**k / factorial(k), dk))


@pytest.mark.parametrize("expr,x", [
    ("acosh(x+i*1e-35)", "1"), ("acosh(x+i*1e-35)", "-1"),
    ("asin(x+i*1e-35)", "1"), ("asin(x+i*1e-35)", "-1"),
    ("acos(x+i*1e-35)", "1"), ("acos(x+i*1e-35)", "-1"),
    ("atanh(x+i*1e-35)", "1"), ("atanh(x+i*1e-35)", "-1"),
    ("asinh(1e-35+i*x)", "1"), ("asinh(1e-35+i*x)", "-1"),
    ("atan(1e-35+i*x)", "1"), ("atan(1e-35+i*x)", "-1"),
    ("acosh(x)", "1.00000000000000000000000000000000001"),
    ("asin(x)", "0.99999999999999999999999999999999999"),
    ("atanh(x)", "0.99999999999999999999999999999999999"),
])
def test_inverse_functions_keep_the_precision_next_to_their_singular_points(expr, x):
    # 1e-35 from a branch point or pole: 1 -+ a^2 keeps its 1e-70 component.
    at = {"x": x}
    for k in range(1, 6):
        assert near(d(expr, "x" * k, at), d(expr, "x" * k, at, precision=256), Decimal("1e-60")), k


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_terms_with_conjugate_poles_cancel(k):
    # Each term's k-th derivative is ~(k-1)!/(2 x^k) at x = 1e-35 from its pole.
    at = {"x": "1e-35"}
    # atan(x+i) + atan(x-i) = atan(x/2) - pi/2 for x > 0.
    assert near(d("atan(x+i)+atan(x-i)", "x" * k, at), d("atan(x/2)", "x" * k, at))
    # atanh(1+ix) + atanh(1-ix) = log(4 + x^2)/2 - log(x) for x > 0.
    assert near(d("atanh(1+i*x)+atanh(1-i*x)", "x" * k, at), d("log(4+x^2)/2-log(x)", "x" * k, at))


@pytest.mark.parametrize("expr,x,names,closed", [
    ("1e100*atanh(x)", "1e-100", "xx", "1e100*2*x/(1-x^2)^2"),
    ("1e100*atanh(x)", "1e-100", "xxxx", "1e100*24*x*(1+x^2)/(1-x^2)^4"),
    ("1e200*atanh(x+i)", "1e100", "x", "1e200/(1-(x+i)^2)"),
    ("1e200*atan(1+i*x)", "1e100", "x", "1e200*i/(1+(1+i*x)^2)"),
])
def test_atan_atanh_near_zero_and_far_from_the_poles(expr, x, names, closed):
    at = {"x": x}
    assert rel_near(d(expr, names, at), value(closed, at))


@pytest.mark.parametrize("expr,x", [
    ("atanh(x)", "1e-100"), ("atan(x)", "1e-100"), ("atan(x)", "1e100"),
    ("asin(x)", "1e-100"), ("asinh(x)", "1e100"), ("acosh(x)", "1e100"),
    ("atanh(x+i*1e-100)", "1e-100"), ("atan(x+i*1e-100)", "1e-100"),
    ("atanh(x+i*1e100)", "1e100"), ("atan(x+i*1e100)", "1e100"),
    ("atanh(x+i)", "1e100"), ("atan(1+i*x)", "1e100"),
    ("asin(x+i*1e100)", "1e100"), ("asinh(x+i*1e100)", "1e100"), ("acosh(x+i*1e100)", "1e100"),
    ("atanh(x+i*1e-35)", "1"), ("atan(1e-35+i*x)", "1"),
])
def test_inverse_functions_near_zero_far_away_and_next_to_the_poles(expr, x):
    at = {"x": x}
    for k in range(1, 6):
        assert rel_near(d(expr, "x" * k, at), d(expr, "x" * k, at, precision=256)), k

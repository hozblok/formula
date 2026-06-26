"""Comprehensive audit of Formula.get_derivative and Solver derivative path.

Tests probe each dispatched derivative rule (cseval::functionsTwoArgsDLeft /
DRight in src/cpp/cseval/cseval.{hpp,cpp} and the complex twin), the
chain-rule wiring, special points, missing dispatch entries, and the
Solver(...)(values, derivative=...) pass-through.
"""

import math

import pytest

from formula import Formula, Solver


# Compact helpers --------------------------------------------------------

def deriv(expr, var, values=None, *, precision=24, digits=0):
    """get_derivative on a fresh Formula. values defaults to {}."""
    return Formula(expr, precision=precision).get_derivative(
        var, values or {}, digits
    )


def approx_eq(s, expected, tol=1e-10):
    """Parse the formatted decimal output and compare to expected float."""
    return abs(float(s) - expected) < tol


# A. Trivial baseline -----------------------------------------------------

def test_deriv_of_variable_is_one():
    assert deriv("x", "x", {"x": "0"}) == "1"


def test_deriv_of_numeric_constant_is_zero():
    assert deriv("5", "x") == "0"


def test_deriv_of_x_squared():
    assert deriv("x^2", "x", {"x": "3"}) == "6"


def test_deriv_of_x_cubed():
    assert deriv("x^3", "x", {"x": "2"}) == "12"


# B. Each unary function's derivative at a regular point ------------------

def test_deriv_sin_at_zero():
    assert deriv("sin(x)", "x", {"x": "0"}) == "1"


def test_deriv_cos_at_zero():
    assert deriv("cos(x)", "x", {"x": "0"}) == "0"


def test_deriv_tan_at_zero():
    assert deriv("tan(x)", "x", {"x": "0"}) == "1"


def test_deriv_asin_at_zero():
    assert deriv("asin(x)", "x", {"x": "0"}) == "1"


def test_deriv_acos_at_zero():
    assert deriv("acos(x)", "x", {"x": "0"}) == "-1"


def test_deriv_atan_at_zero():
    assert deriv("atan(x)", "x", {"x": "0"}) == "1"


def test_deriv_exp_at_zero():
    assert deriv("exp(x)", "x", {"x": "0"}) == "1"


def test_deriv_log_at_one():
    assert deriv("log(x)", "x", {"x": "1"}) == "1"


def test_deriv_sqrt_at_one():
    # d/dx sqrt(x) at x=1 = 1/2
    assert approx_eq(deriv("sqrt(x)", "x", {"x": "1"}, digits=12), 0.5)


def test_deriv_abs_positive():
    assert deriv("abs(x)", "x", {"x": "3"}) == "1"


def test_deriv_abs_negative():
    assert deriv("abs(x)", "x", {"x": "-3"}) == "-1"


# C. Chain rule, one level ------------------------------------------------

def test_chain_sin_of_2x():
    # d/dx sin(2x) at x=0 = 2*cos(0) = 2
    assert deriv("sin(2*x)", "x", {"x": "0"}) == "2"


def test_chain_exp_of_x_squared():
    # d/dx exp(x^2) at x=1 = 2*exp(1)
    expected = 2 * math.exp(1)
    assert approx_eq(deriv("exp(x^2)", "x", {"x": "1"}, digits=15), expected)


def test_chain_log_of_x_squared():
    # d/dx log(x^2) at x=2 = 2x/x^2 = 1
    assert approx_eq(deriv("log(x^2)", "x", {"x": "2"}, digits=15), 1.0)


# D. Chain rule, nested ---------------------------------------------------

def test_chain_log_of_exp_is_one():
    # d/dx log(exp(x)) at x=5 = 1
    assert approx_eq(deriv("log(exp(x))", "x", {"x": "5"}, digits=15), 1.0)


def test_chain_sin_of_cos_at_zero():
    # d/dx sin(cos(x)) at x=0 = cos(cos(0))*(-sin(0)) = cos(1)*0 = 0
    assert deriv("sin(cos(x))", "x", {"x": "0"}) == "0"


def test_chain_sin_of_cos_at_pi_over_two():
    # cos(pi/2)=0, sin(0)=0, chained: cos(cos(pi/2))*(-sin(pi/2)) = 1*(-1) = -1
    pi_over_two = "1.5707963267948966192313216916398"
    assert approx_eq(
        deriv("sin(cos(x))", "x", {"x": pi_over_two}, digits=15), -1.0
    )


# E. Sum / diff / prod / quot ---------------------------------------------

def test_deriv_sum():
    # d/dx (x + x^2) at x=3 = 1 + 6 = 7
    assert deriv("x + x^2", "x", {"x": "3"}) == "7"


def test_deriv_diff():
    # d/dx (x - x^2) at x=1 = 1 - 2 = -1
    assert deriv("x - x^2", "x", {"x": "1"}) == "-1"


def test_deriv_product_same_var():
    # d/dx (x*x) at x=4 = 8
    assert deriv("x*x", "x", {"x": "4"}) == "8"


def test_deriv_quotient_constant_numerator():
    # d/dx (1/x) at x=2 = -1/4
    assert approx_eq(deriv("1/x", "x", {"x": "2"}, digits=12), -0.25)


def test_deriv_quotient_same_var():
    # d/dx (x/x) at any non-zero x = 0
    assert deriv("x/x", "x", {"x": "3"}) == "0"


def test_deriv_complex_polynomial():
    # d/dx (3*x^2 + 2*x + 7) at x=5 = 30 + 2 = 32
    assert deriv("3*x^2 + 2*x + 7", "x", {"x": "5"}) == "32"


# F. Power corner cases ---------------------------------------------------

def test_deriv_x_to_the_x_at_one():
    # d/dx x^x at x=1: x^x*(log(x)+1) = 1*(0+1) = 1
    assert deriv("x^x", "x", {"x": "1"}) == "1"


def test_deriv_x_to_the_x_at_two():
    # d/dx x^x at x=2: 4 * (log(2) + 1)
    expected = 4 * (math.log(2) + 1)
    assert approx_eq(deriv("x^x", "x", {"x": "2"}, digits=15), expected)


def test_deriv_x_to_the_x_at_zero_raises():
    # 0^0-type: both base and exponent depend on x, so neither term is dropped;
    # _pow2(0, 0) hits the log(0) guard and raises -- a loud error, not a silent
    # NaN. (The derivative of x^x as x->0+ is -inf.)
    with pytest.raises((ValueError, RuntimeError)):
        deriv("x^x", "x", {"x": "0"})


def test_deriv_two_to_the_x_at_zero():
    # d/dx 2^x at x=0 = log(2)
    assert approx_eq(deriv("2^x", "x", {"x": "0"}, digits=15), math.log(2))


def test_deriv_x_to_zero():
    # d/dx x^0 at x=5 = d/dx 1 = 0
    assert deriv("x^0", "x", {"x": "5"}) == "0"


def test_deriv_x_to_negative_one():
    # d/dx x^(0-1) at x=2 = -1/4
    assert approx_eq(deriv("x^(0-1)", "x", {"x": "2"}, digits=12), -0.25)


def test_deriv_x_to_half():
    # d/dx x^(1/2) at x=4 = 1/(2*sqrt(4)) = 0.25
    assert approx_eq(deriv("x^(1/2)", "x", {"x": "4"}, digits=12), 0.25)


# G. Special / singular points -------------------------------------------

def test_deriv_log_at_zero_errors():
    # log'(0) divides by zero; csformula raises invalid_argument -> ValueError
    with pytest.raises((ValueError, RuntimeError)):
        deriv("log(x)", "x", {"x": "0"})


def test_deriv_sqrt_at_zero_errors():
    with pytest.raises((ValueError, RuntimeError)):
        deriv("sqrt(x)", "x", {"x": "0"})


def test_deriv_asin_at_one_errors():
    with pytest.raises((ValueError, RuntimeError)):
        deriv("asin(x)", "x", {"x": "1"})


def test_deriv_acos_at_one_errors():
    with pytest.raises((ValueError, RuntimeError)):
        deriv("acos(x)", "x", {"x": "1"})


def test_deriv_asin_at_negative_one_errors():
    with pytest.raises((ValueError, RuntimeError)):
        deriv("asin(x)", "x", {"x": "-1"})


def test_deriv_abs_at_zero_returns_zero():
    # sign(0) is 0 in Boost; document the behavior at the cusp.
    assert deriv("abs(x)", "x", {"x": "0"}) == "0"


def test_deriv_pow_base_zero_does_not_silently_produce_nan():
    # d/dx 0^x hits _pow2(0, b) = log(0) * 0 = NaN; the engine now raises
    # (consistent with _log_d's guard) instead of leaking a literal "nan".
    with pytest.raises((ValueError, RuntimeError)):
        Formula("0^x").get_derivative("x", {"x": "2"})


# G2. Structural-zero short-circuit --------------------------------------
# A term is dropped only when its operand is structurally constant (the
# variable does not occur in it) — not when the operand's derivative merely
# happens to vanish at a point. So base^const is differentiable at base=0
# (the constant exponent is dropped, never multiplying log(base)), while a
# genuine cusp reached with nonzero speed is still computed honestly.
# See doc/derivative-structural-zero-shortcircuit.md.

def test_deriv_pow_constant_exponent_at_zero_base():
    assert deriv("x^2", "x", {"x": "0"}) == "0"
    assert deriv("x^3", "x", {"x": "0"}) == "0"


def test_deriv_pow_at_vertex_is_finite():
    # (x-2)^2 at the vertex x=2 — the case test_intersect_surfaces works around.
    assert deriv("(x-2)^2", "x", {"x": "2"}) == "0"


def test_deriv_zero_base_variable_exponent_still_raises():
    # Compat: a variable exponent over a zero base (0^x) keeps the loud error.
    with pytest.raises((ValueError, RuntimeError)):
        deriv("0^x", "x", {"x": "2"})


def test_deriv_sqrt_of_square_at_zero_is_honest():
    # sqrt(x^2) = |x|: a real cusp at 0. The inner derivative vanishes there
    # (point zero), but x^2 still contains x, so the term is computed and
    # sqrt's guard fires — an honest error, not a fabricated 0.
    with pytest.raises((ValueError, RuntimeError)):
        deriv("sqrt(x^2)", "x", {"x": "0"})


def test_deriv_abs_of_square_at_zero_is_zero():
    # abs(x^2) = x^2 is smooth at 0. abs' is the finite sign(0)=0 (not a
    # singular partial), so the point-zero inner derivative gives 0 — no
    # false cusp error. Contrast sqrt(x^2) above.
    assert deriv("abs(x^2)", "x", {"x": "0"}) == "0"


def test_deriv_nested_even_power_at_root_is_zero():
    # ((x-1)^2)^2 = (x-1)^4 at x=1: nested constant exponents, each dropped by
    # the structural-zero gate, so the derivative is finite (=0) at the root.
    assert deriv("((x-1)^2)^2", "x", {"x": "1"}) == "0"


# G3. Known residual — _pow1 has no zero guard ---------------------------
# When the base contains the variable and reaches 0 with a fractional
# exponent, the structural gate correctly keeps the left term (the base
# depends on x), but _pow1(0, v<1) = v*0^(v-1) = +inf, times the point-zero
# inner derivative = NaN. Geometrically these are a cusp (derivative does
# not exist) or a vertical tangent (derivative +-inf), so the engine should
# raise — as sqrt/log do, and as _pow2 does after plan.md item 2. But _pow1
# has no zero guard yet, so it leaks NaN (point-zero inner) or +inf (non-zero
# inner). A clean guard would be real-side only: _pow1's singular region is
# u=0, v<1, and "v<1" is not expressible
# for Complex (no order). These xfail until _pow1 is guarded.
# See doc/derivative-structural-zero-shortcircuit.md (residual section).

@pytest.mark.xfail(
    strict=True,
    reason="_pow1 has no zero guard: (x^2)^(1/2) at 0 leaks NaN instead of raising",
)
def test_deriv_pow_half_of_square_at_zero_should_raise():
    # (x^2)^(1/2) = |x| at 0 — a cusp; the derivative does not exist.
    with pytest.raises((ValueError, RuntimeError)):
        deriv("(x^2)^(1/2)", "x", {"x": "0"})


@pytest.mark.xfail(
    strict=True,
    reason="_pow1 has no zero guard: (x^2)^(1/3) at 0 leaks NaN instead of raising",
)
def test_deriv_pow_third_of_square_at_zero_should_raise():
    # (x^2)^(1/3) = |x|^(2/3) at 0 — a vertical tangent; the true derivative
    # is +-inf. A fabricated finite value (the old numeric gate returned 0)
    # would be plainly wrong; the engine should raise.
    with pytest.raises((ValueError, RuntimeError)):
        deriv("(x^2)^(1/3)", "x", {"x": "0"})


@pytest.mark.xfail(
    strict=True,
    reason="_pow1 has no zero guard: x^(1/2) at 0 leaks +inf instead of raising",
)
def test_deriv_pow_half_at_zero_should_raise():
    # x^(1/2) = sqrt(x) at 0: vertical tangent, true derivative +inf. The inner
    # derivative is 1 (not a point zero), so _pow1(0, 0.5)=+inf leaks as "inf"
    # rather than NaN. The dedicated sqrt(x) raises here; ^ should match once
    # _pow1 is guarded.
    with pytest.raises((ValueError, RuntimeError)):
        deriv("x^(1/2)", "x", {"x": "0"})


# H. Derivative wrt non-existent variable --------------------------------

def test_deriv_wrt_missing_variable_is_zero():
    # d/dy (x^2) = 0; y is not in the expression at all
    assert deriv("x^2", "y", {"x": "3"}) == "0"


def test_deriv_wrt_unused_variable_in_multivar():
    # d/dz (x+y) = 0; expression has x,y but not z
    assert deriv("x+y", "z", {"x": "1", "y": "2"}) == "0"


# I. sign() derivative ---------------------------------------------------
# sign is constant a.e., so d/dx sign(f(x)) = 0. cseval.cpp maps "sign"
# to _zero in functionsTwoArgsDLeft (was _one; see plan.md item 1).

def test_deriv_of_sign_is_zero():
    # sign is constant a.e.; derivative must be 0.
    assert deriv("sign(x)", "x", {"x": "5"}) == "0"


def test_deriv_of_sign_of_polynomial():
    # d/dx sign(x^2) at x=3 = 0 (sign of any nonzero is constant nearby)
    assert deriv("sign(x^2)", "x", {"x": "3"}) == "0"


def test_deriv_of_x_times_sign_x():
    # d/dx (x*sign(x)) at x=5 = sign(5) + x*0 = 1.
    # With the bug, returns 1 + 5*1 = 6.
    assert deriv("x*sign(x)", "x", {"x": "5"}) == "1"


# J. Derivative dispatch for relational/logical operators ----------------
# | & = > < are piecewise-constant, so their derivative is 0 a.e. They are
# now registered to _zero in functionsTwoArgsDLeft / DRight (see plan.md
# item 3). Complex side carries only | & =, since > < are undefined on C.

def test_deriv_of_greater_than_is_zero():
    assert deriv("x > 0", "x", {"x": "5"}) == "0"


def test_deriv_of_less_than_is_zero():
    assert deriv("x < 5", "x", {"x": "3"}) == "0"


def test_deriv_of_equality_is_zero():
    assert deriv("x = 1", "x", {"x": "2"}) == "0"


def test_deriv_of_logical_or_is_zero():
    assert deriv("x | 0", "x", {"x": "3"}) == "0"


def test_deriv_of_logical_and_is_zero():
    assert deriv("x & 1", "x", {"x": "3"}) == "0"


# K. Mixed real / complex -------------------------------------------------

def test_deriv_of_imaginary_unit_times_x_squared():
    # Already in test_derivative_formatting; restated as part of the audit.
    assert Solver("i*x^2")({"x": "2"}, derivative="x") == "0+i*(4)"


def test_deriv_of_i_times_sin_at_zero():
    # d/dx (i*sin(x)) at x=0 = i*cos(0) = i. Real part 0; imag part starts with "1".
    out = Solver("i*sin(x)")({"x": "0"}, derivative="x", format_digits=10)
    assert out.startswith("0+i*(1")


def test_deriv_complex_polynomial_real_var():
    # d/dx (x^2 + i*x) at x=3 = 6 + i. Use small digits to suppress noise.
    out = Solver("x^2 + i*x")({"x": "3"}, derivative="x", format_digits=10)
    # Real part rounds to 6.000000000; imag part rounds to 1.000000000.
    assert out.startswith("6") and "+i*(1" in out


# L. Multi-variable -------------------------------------------------------

def test_partial_deriv_x_of_product():
    # ∂/∂x (x*y) at x=2,y=3 = 3
    assert deriv("x*y", "x", {"x": "2", "y": "3"}) == "3"


def test_partial_deriv_y_of_product():
    # ∂/∂y (x*y) at x=2,y=3 = 2
    assert deriv("x*y", "y", {"x": "2", "y": "3"}) == "2"


def test_partial_deriv_of_polynomial_with_both():
    # ∂/∂x (x^2 + y^2 + x*y) at x=1,y=2 = 2 + 2 = 4
    assert deriv("x^2 + y^2 + x*y", "x", {"x": "1", "y": "2"}) == "4"


# M. Unary minus ----------------------------------------------------------

def test_deriv_unary_minus_variable():
    # d/dx (0-x) = -1
    assert deriv("0-x", "x", {"x": "5"}) == "-1"


def test_deriv_unary_minus_polynomial():
    # d/dx (0-x^2) at x=2 = -4
    assert deriv("0-x^2", "x", {"x": "2"}) == "-4"


# N. Numeric exactness at chosen precision -------------------------------

def test_deriv_sin_at_zero_is_exact():
    # cos(0) is 1 exactly; output must contain no spurious digits.
    assert deriv("sin(x)", "x", {"x": "0"}, digits=20) == "1"


def test_deriv_exp_at_zero_is_exact():
    assert deriv("exp(x)", "x", {"x": "0"}, digits=20) == "1"


# O. Solver wrapper pass-through -----------------------------------------

def test_solver_derivative_matches_raw_formula():
    raw = Formula("x^3").get_derivative("x", {"x": "4"})
    via = Solver("x^3")({"x": "4"}, derivative="x")
    assert raw == via == "48"


def test_solver_one_var_shortcut_with_derivative():
    # Solver supports positional scalar for the single-variable case.
    assert Solver("x^2")(3, derivative="x") == "6"


# P. Derivative variable validity ----------------------------------------

def test_deriv_empty_variable_name_rejected_or_zero():
    # Asking for derivative wrt "" is meaningless; should not crash silently.
    try:
        out = deriv("x^2", "", {"x": "3"})
    except (ValueError, RuntimeError):
        return
    assert out == "0", f"empty-string variable name must be 0 or error, got {out!r}"


# Q. Composition with built-in constants ---------------------------------

def test_deriv_of_pi_times_x():
    # d/dx (pi*x) = pi
    out = deriv("pi*x", "x", {"x": "1"}, digits=15)
    assert out.startswith("3.1415926535")


# R. Missing values when derivative needs them ---------------------------

def test_deriv_missing_required_value_raises():
    # Expression uses x; derivative wrt x at no value should fail clearly.
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        Formula("x^2").get_derivative("x", {})


def test_deriv_extra_values_ignored():
    # Extra entries in values must not break the call.
    assert deriv("x^2", "x", {"x": "3", "junk": "999"}) == "6"

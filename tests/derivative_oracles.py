"""Independent derivative references: exact rationals, closed forms, values-only differences."""

from collections import Counter
from decimal import Decimal
from fractions import Fraction
from itertools import product
from math import factorial

# Complex numbers are (re, im) pairs of Fractions (exact) or Decimals.


def c_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def c_scale(c, a):
    return (c * a[0], c * a[1])


def c_mul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def c_div(a, b):
    d = b[0] * b[0] + b[1] * b[1]
    return ((a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d)


def c_pow(a, n):
    """a^n for an integer n by repeated multiplication."""
    one = (a[0] * 0 + 1, a[0] * 0)
    r = one
    for _ in range(abs(n)):
        r = c_mul(r, a)
    return r if n >= 0 else c_div(one, r)


def c_poly(coefficients, z):
    """sum_j coefficients[j] z^j."""
    total, power = (z[0] * 0, z[0] * 0), (z[0] * 0 + 1, z[0] * 0)
    for c in coefficients:
        total = c_add(total, c_scale(c, power))
        power = c_mul(power, z)
    return total


def falling(n, k):
    """n (n - 1) ... (n - k + 1)."""
    r = n * 0 + 1
    for j in range(k):
        r *= n - j
    return r


def to_decimal(z):
    return tuple(Decimal(v.numerator) / Decimal(v.denominator) if isinstance(v, Fraction)
                 else Decimal(v) for v in z)


def exact_point(value):
    """Fraction of a decimal string, exactly as written."""
    return Fraction(Decimal(value))


# Polynomials in several variables: {powers tuple: coefficient}.

def poly_plus(a, b, sign=1):
    out = dict(a)
    for powers, coefficient in b.items():
        out[powers] = out.get(powers, Fraction(0)) + sign * coefficient
    return {p: c for p, c in out.items() if c}


def poly_times(a, b):
    out = {}
    for p, c in a.items():
        for q, d in b.items():
            r = tuple(x + y for x, y in zip(p, q))
            out[r] = out.get(r, Fraction(0)) + c * d
    return {p: c for p, c in out.items() if c}


def poly_partial(poly, variables, names, point):
    """Exact mixed partial over names (a multiset of variables) at point."""
    requested = Counter(names)
    total = Fraction(0)
    for powers, coefficient in poly.items():
        term = coefficient
        for name, power in zip(variables, powers):
            n = requested[name]
            if n > power:
                term = 0
                break
            term *= falling(power, n) * exact_point(point[name]) ** (power - n)
        total += term
    return total


# Rational closed forms: atan^(k) = (-1)^(k-1) (k-1)! M_k / (1 + z^2)^k,
# atanh^(k) = (k-1)! N_k / (1 - z^2)^k; coefficients in ascending powers.
ATAN_M = [None, [1], [0, 2], [-1, 0, 3], [0, -4, 0, 4], [1, 0, -10, 0, 5]]
ATANH_N = [None, [1], [0, 2], [1, 0, 3], [0, 4, 0, 4], [1, 0, 10, 0, 5]]


def atan_derivative(z, k):
    one = (Fraction(1), Fraction(0))
    s = c_add(one, c_mul(z, z))
    value = c_div(c_poly(ATAN_M[k], z), c_pow(s, k))
    return c_scale((-1) ** (k - 1) * factorial(k - 1), value)


def atanh_derivative(z, k):
    one = (Fraction(1), Fraction(0))
    s = c_add(one, c_scale(-1, c_mul(z, z)))
    return c_scale(factorial(k - 1), c_div(c_poly(ATANH_N[k], z), c_pow(s, k)))


# Inverse-sqrt closed forms: f^(k) = P_k(z) W^(2k-1), W = f' on the branch of f.
ASIN_P = [None, [1], [0, 1], [1, 0, 2], [0, 9, 0, 6], [9, 0, 72, 0, 24]]
ASINH_P = [None, [1], [0, -1], [-1, 0, 2], [0, 9, 0, -6], [9, 0, -72, 0, 24]]
ACOSH_P = [None, [1], [0, -1], [1, 0, 2], [0, -9, 0, -6], [9, 0, 72, 0, 24]]


def d_sqrt(z):
    """Principal square root of a Decimal pair, componentwise accurate."""
    a, b = z
    r = (a * a + b * b).sqrt()
    if a >= 0:
        t = ((r + a) / 2).sqrt()
        return (t, b / (2 * t)) if t else (Decimal(0), Decimal(0))
    u = ((r - a) / 2).sqrt()
    return (abs(b) / (2 * u), u if b >= 0 else -u)


def _inverse_sqrt_like(polynomials, w, z, k):
    w = to_decimal(w)
    return c_mul(to_decimal(c_poly(polynomials[k], z)), c_pow(w, 2 * k - 1))


def asin_derivative(z, k):
    one = (Fraction(1), Fraction(0))
    w = c_div((Decimal(1), Decimal(0)), d_sqrt(to_decimal(c_add(one, c_scale(-1, c_mul(z, z))))))
    return _inverse_sqrt_like(ASIN_P, w, z, k)


def acos_derivative(z, k):
    return c_scale(-1, asin_derivative(z, k))


def asinh_derivative(z, k):
    one = (Fraction(1), Fraction(0))
    w = c_div((Decimal(1), Decimal(0)), d_sqrt(to_decimal(c_add(one, c_mul(z, z)))))
    return _inverse_sqrt_like(ASINH_P, w, z, k)


def acosh_derivative(z, k):
    one = (Fraction(1), Fraction(0))
    roots = c_mul(d_sqrt(to_decimal(c_add(z, c_scale(-1, one)))), d_sqrt(to_decimal(c_add(z, one))))
    return _inverse_sqrt_like(ACOSH_P, c_div((Decimal(1), Decimal(0)), roots), z, k)


CLOSED_FORMS = {"atan": atan_derivative, "atanh": atanh_derivative, "asin": asin_derivative,
                "acos": acos_derivative, "asinh": asinh_derivative, "acosh": acosh_derivative}


# Values only: the mixed difference over names with step h.

def central_difference(value, point, names, h):
    """sum over s in {-1, 1}^k of prod(s) f(p + h/2 sum_j s_j e_(names_j)) / h^k."""
    total = (Decimal(0), Decimal(0))
    for signs in product((-1, 1), repeat=len(names)):
        at = {name: Decimal(v) for name, v in point.items()}
        sign = 1
        for name, side in zip(names, signs):
            at[name] += side * h / 2
            sign *= side
        total = c_add(total, c_scale(sign, value({n: str(v) for n, v in at.items()})))
    return c_scale(1 / h ** len(names), total)

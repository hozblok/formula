"""Regression: derivative-of-^ uses log(), derivative-of-sqrt fires.

Closes TODOs:
  src/cpp/cseval/cseval.hpp:212          (// TODO test log())
  src/cpp/cseval/cseval.hpp:285          (// TODO test _sqrt_d())
  src/cpp/cseval/cseval_complex.hpp:213  (// TODO test log())
  src/cpp/cseval/cseval_complex.hpp:286  (// TODO test _sqrt_d())

The `^` derivative's right path is `_pow2(a, b) = log(a) * pow(a, b)`,
which is only reached when the exponent is itself a function of the
differentiation variable. The `_sqrt_d` path is reached for d(sqrt(x))/dx.
Both fire on the real and complex evaluation paths.
"""

from formula import Solver


def test_pow_derivative_via_log_real():
    # d(x^y)/dy = log(x) * x^y.  At x=2, y=3:  log(2)*8 = 5.5451774444795625...
    result = Solver("x^y", precision=24)(
        {"x": "2", "y": "3"}, derivative="y"
    )
    assert result.startswith("5.545177444479562475337856")


def test_pow_derivative_via_log_complex():
    # Same identity through the complex path. The presence of 'i' in the
    # expression forces complex evaluation, so _pow2 in cseval_complex.hpp
    # runs instead of the real twin.
    result = Solver("(2+i*0)^y", precision=24)(
        {"y": "3"}, derivative="y"
    )
    assert result.startswith("5.5451774444795624753378")


def test_sqrt_derivative_real():
    # d(sqrt(x))/dx = 1/(2*sqrt(x)).  At x=4:  1/(2*2) = 0.25.
    result = Solver("sqrt(x)", precision=24)({"x": "4"}, derivative="x")
    assert result == "0.25"


def test_sqrt_derivative_complex():
    # Same identity through the complex path (forced by 'i' in the expression).
    result = Solver("sqrt(x+i*0)", precision=24)({"x": "4"}, derivative="x")
    assert result.startswith("0.25")

#ifndef EVAL_COMPLEX_MPF_H
#define EVAL_COMPLEX_MPF_H

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/multiprecision/cpp_complex.hpp>
#include <boost/variant.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../csconstants.hpp"

/**
 * Arbitrary-precision arithmetic with complex numbers.
 */
template <unsigned N>
using mp_complex =
    boost::multiprecision::number<boost::multiprecision::cpp_complex_backend<N>,
                                  boost::multiprecision::et_off>;

/**
 * Class for evaluating formula specified by the string
 * typename Complex (de facto mp_complex<NUMBER>).
 */
template <typename Complex>
class cseval_complex {
  /**
   * Kind of formula element:
   * 'n' - number, 'v' - variable, 'f' - function, 'e' - error.
   */
  char kind_;

  /**
   * The function name or variable name for current node. e.g.
   * "+","-","/","x","a","sin"
   * (!) "i" - by default reserved for complex numbers. (!)
   */
  std::string id_;

  /**
   * Value which was parsed from expression, value makes sense if kind is 'n'.
   */
  Complex value_;

  /** Child elements of the formula tree. */
  cseval_complex *left_eval_, *right_eval_;

  /**
   * Symbol of the imaginary unit (by default - 'i').
   * Only one Latin character (!).
   */
  const char imaginary_unit_;

  /** Try to find each symbol of {symbols} string in the {str} string. */
  std::unordered_map<char, size_t> operations_outside_parentheses(
      const std::string &str, const std::string &symbols) const;

  /**  Try to find symbols outside parentheses: '(' and ')'. */
  bool isThereSymbolsOutsideParentheses(const std::string &str) const;

 public:
  cseval_complex(std::string expression, char imaginary_unit = 'i');

  cseval_complex(const cseval_complex &other);

  ~cseval_complex();

  cseval_complex &operator=(const cseval_complex &other) {
    if (this != &other) {
      kind_ = other.kind_;
      id_ = std::string(other.id_);
      value_ = Complex(other.value_);
      // imaginary_unit_ is const, fixed at construction; not reassigned.
      delete left_eval_;
      left_eval_ = other.left_eval_
                       ? new cseval_complex<Complex>(*other.left_eval_)
                       : nullptr;
      delete right_eval_;
      right_eval_ = other.right_eval_
                        ? new cseval_complex<Complex>(*other.right_eval_)
                        : nullptr;
    }
    return *this;
  }

  void collect_variables(std::unordered_set<std::string> *variables) {
    if (left_eval_) {
      left_eval_->collect_variables(variables);
    }
    if (right_eval_) {
      right_eval_->collect_variables(variables);
    }
    if (kind_ == 'v') {
#ifdef CSDEBUG
      std::cout << "collect_variables, kind:" << kind_ << ", id:" << id_
                << std::endl;
#endif
      variables->insert(std::string(id_));
    }
  }

  // Whether the differentiation variable occurs anywhere in this subtree.
  // A subtree without it is constant w.r.t. the variable: its derivative is
  // identically zero, regardless of the value its partials take at a point.
  bool depends_on(const std::string &variable) const {
    if (kind_ == 'v') return id_ == variable;
    if (left_eval_ && left_eval_->depends_on(variable)) return true;
    if (right_eval_ && right_eval_->depends_on(variable)) return true;
    return false;
  }

  // Evaluation of subformula.
  Complex calculate(const std::map<std::string, Complex> &variables_to_values,
                    const std::map<std::string, Complex (*)(Complex, Complex)>
                        &mapFunctionTwoArgsValue = functionsTwoArgs,
                    const std::map<std::string, Complex (*)(Complex)>
                        &mapFunctionOneArgValue = functionsOneArg) const;
  Complex calculate(
      const std::map<std::string, std::string> &variables_to_values,
      const std::map<std::string, Complex (*)(Complex, Complex)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Complex (*)(Complex)>
          &mapFunctionOneArgValue = functionsOneArg) const;
  Complex calculate(const std::map<std::string, double> &variables_to_values,
                    const std::map<std::string, Complex (*)(Complex, Complex)>
                        &mapFunctionTwoArgsValue = functionsTwoArgs,
                    const std::map<std::string, Complex (*)(Complex)>
                        &mapFunctionOneArgValue = functionsOneArg) const;
  // Evaluation derivative of subformula.
  Complex calculate_derivative(
      const std::string &variable,
      const std::map<std::string, Complex> &variables_to_values,
      const std::map<std::string, Complex (*)(Complex, Complex)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Complex (*)(Complex)>
          &mapFunctionOneArgValue = functionsOneArg,
      const std::map<std::string, Complex (*)(Complex, Complex)>
          &mapFunctionDerivLeft = functionsTwoArgsDLeft,
      const std::map<std::string, Complex (*)(Complex, Complex)>
          &mapFunctionDerivRight = functionsTwoArgsDRight) const;

  Complex calculate_derivative(
      const std::string &variable,
      const std::map<std::string, std::string> &variables_to_values,
      const std::map<std::string, Complex (*)(Complex, Complex)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Complex (*)(Complex)>
          &mapFunctionOneArgValue = functionsOneArg,
      const std::map<std::string, Complex (*)(Complex, Complex)>
          &mapFunctionDerivLeft = functionsTwoArgsDLeft,
      const std::map<std::string, Complex (*)(Complex, Complex)>
          &mapFunctionDerivRight = functionsTwoArgsDRight) const;

  // Useful constants.
  const static Complex ZERO;
  const static Complex ONE;

  //+ general static methods that have a one-to-one correspondence
  // with the operations specified in the expression string
  // "+" - addition
  static Complex _add(Complex a, Complex b) { return a + b; }
  // "-" - subtraction
  static Complex _sub(Complex a, Complex b) { return a - b; }
  // "&" - and
  static Complex _and(Complex a, Complex b) {
    return a != ZERO && b != ZERO ? ONE : ZERO;
  }
  // "|" - or
  static Complex _or(Complex a, Complex b) {
    return a != ZERO || b != ZERO ? ONE : ZERO;
  }
  // "=" - is equal to
  static Complex _eq(Complex a, Complex b) { return a == b ? ONE : ZERO; }
  // "/" - division
  static Complex _truediv(Complex a, Complex b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the \'/\' operation");
    }
    return a / b;
  }
  // division for the computation of the derivative (left path)
  static Complex _truediv1(Complex, Complex b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation the left path \
of the derivative");
    }
    return 1 / b;
  }
  // division for the computation of the derivative (right path)
  static Complex _truediv2(Complex a, Complex b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
right path of the derivative");
    }
    return ZERO - a / (b * b);
  }
  // "*" - multiplication
  static Complex _mul(Complex a, Complex b) { return a * b; }
  // multiplication for the computation of the derivative (left path)
  static Complex _mul1(Complex, Complex b) { return b; }
  // multiplication for the computation of the derivative (right path)
  static Complex _mul2(Complex a, Complex) { return a; }
  // "^" - exponentiation
  static Complex _pow(Complex a, Complex b) { return pow(a, b); }
  // exponentiation for the computation of the derivative (left path)
  static Complex _pow1(Complex a, Complex b) { return (b * _pow(a, b - ONE)); }
  // exponentiation for the computation of the derivative (right path)
  static Complex _pow2(Complex a, Complex b) {
    if (a == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the power derivative (log(0) of the base)");
    }
    return _log(a) * _pow(a, b);
  }
  //- general static methods

  //+ trigonometric functions, exp, log, sqrt and methods for the computation of
  // the derivative:
  // "sin"
  static Complex _sin(Complex a) { return sin(a); }
  // "sin" for the derivative
  static Complex _sin_d(Complex a, Complex) { return cos(a); }
  // "asin"
  static Complex _asin(Complex a) { return asin(a); }
  // "asin" for the derivative
  static Complex _asin_d(Complex a, Complex) {
    if (a * a == ONE) {
      // TODO may be inf?
      throw std::invalid_argument(
          "Division by zero during the computation of the arcsin derivative");
    }
    return ONE / _sqrt(ONE - a * a);
  }
  // "cos"
  static Complex _cos(Complex a) { return cos(a); }
  // "cos" for the derivative
  static Complex _cos_d(Complex a, Complex) { return ZERO - _sin(a); }
  // "acos"
  static Complex _acos(Complex a) { return acos(a); }
  static Complex _acos_d(Complex a, Complex) {
    if (a * a == ONE) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the arccos derivative");
    }
    return ZERO - (ONE / _sqrt(ONE - a * a));
  }
  // "tan"
  static Complex _tan(Complex a) { return tan(a); }
  // "tan" for the derivative
  static Complex _tan_d(Complex a, Complex) {
    if (_cos(a) == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the tangent derivative");
    }
    return ONE / (_cos(a) * _cos(a));
  }
  // "atan"
  static Complex _atan(Complex a) { return atan(a); }
  // "atan" for the derivative
  static Complex _atan_d(Complex a, Complex) { return ONE / (ONE + a * a); }
  // "exp"
  static Complex _exp(Complex a) { return exp(a); }
  // "exp" for the derivative
  static Complex _exp_d(Complex a, Complex) { return exp(a); }
  // "abs"
  static Complex _abs(Complex a) { return abs(a); }
  // TODO: "abs" for the derivative
  static Complex _abs_d(Complex a, Complex) {
    throw std::invalid_argument("Not implemented: abs() derivative");
  }
  // "log" - (!) Natural logarithm
  static Complex _log(Complex a) { return log(a); }
  // "log" - (!) Natural logarithm for the derivative
  static Complex _log_d(Complex a, Complex) {
    if (a == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the natural logarithm derivative");
    }
    return ONE / a;
  }
  // "sqrt" - square root
  static Complex _sqrt(Complex a) { return sqrt(a); }
  // TODO test _sqrt_d()
  // "sqrt" for the derivative
  static Complex _sqrt_d(Complex a, Complex) {
    if (sqrt(a) == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the sqrt derivative");
    }
    return ONE / (2 * sqrt(a));
  }
  // "log2" - base-2 logarithm (boost complex has no native log2/log10;
  // express via natural log: log_b(z) = ln(z) / ln(b))
  static Complex _log2(Complex a) { return _log(a) / _log(Complex("2", "0.0")); }
  static Complex _log2_d(Complex a, Complex) {
    if (a == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of the log2 derivative");
    }
    static const Complex LN2 = _log(Complex("2", "0.0"));
    return ONE / (a * LN2);
  }
  // "log10" - base-10 logarithm
  static Complex _log10(Complex a) {
    return _log(a) / _log(Complex("10", "0.0"));
  }
  static Complex _log10_d(Complex a, Complex) {
    if (a == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of the log10 derivative");
    }
    static const Complex LN10 = _log(Complex("10", "0.0"));
    return ONE / (a * LN10);
  }
  // "sinh", "cosh", "tanh"
  static Complex _sinh(Complex a) { return sinh(a); }
  static Complex _sinh_d(Complex a, Complex) { return cosh(a); }
  static Complex _cosh(Complex a) { return cosh(a); }
  static Complex _cosh_d(Complex a, Complex) { return sinh(a); }
  static Complex _tanh(Complex a) { return tanh(a); }
  static Complex _tanh_d(Complex a, Complex) {
    Complex c = cosh(a);
    return ONE / (c * c);
  }
  // "asinh", "acosh", "atanh"
  static Complex _asinh(Complex a) { return asinh(a); }
  static Complex _asinh_d(Complex a, Complex) {
    // Branch points at z = +-i (where 1 + z^2 = 0); real side never hits this.
    if (ONE + a * a == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of the asinh derivative");
    }
    return ONE / _sqrt(ONE + a * a);
  }
  static Complex _acosh(Complex a) { return acosh(a); }
  static Complex _acosh_d(Complex a, Complex) {
    if (a * a == ONE) {
      throw std::invalid_argument(
          "Division by zero during the computation of the acosh derivative");
    }
    return ONE / _sqrt(a * a - ONE);
  }
  static Complex _atanh(Complex a) { return atanh(a); }
  static Complex _atanh_d(Complex a, Complex) {
    if (a * a == ONE) {
      throw std::invalid_argument(
          "Division by zero during the computation of the atanh derivative");
    }
    return ONE / (ONE - a * a);
  }
  //- trigonometric functions, exp, log, sqrt

  //+ auxiliary methods for the computation of the derivative:
  static Complex _zero(Complex, Complex) { return ZERO; }
  static Complex _one(Complex, Complex) { return ONE; }
  static Complex _m_one(Complex, Complex) { return ZERO - ONE; }
  //- auxiliary methods

  // dictionaries contain the appropriate names of operations and static methods
  // for evaluating, e.g.
  // "+" -> _add
  // "sin" -> _sin
  static const std::map<std::string, Complex (*)(Complex, Complex)>
      functionsTwoArgs;
  static const std::map<std::string, Complex (*)(Complex)> functionsOneArg;

  // dictionaries contain references to derivatives of basic functions and their
  // names:
  static const std::map<std::string, Complex (*)(Complex, Complex)>
      functionsTwoArgsDLeft;
  static const std::map<std::string, Complex (*)(Complex, Complex)>
      functionsTwoArgsDRight;
};

#endif  // EVAL_COMPLEX_MPF_H

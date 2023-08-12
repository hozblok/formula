#ifndef EVAL_MPF_H
#define EVAL_MPF_H

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/variant.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../csconstants.hpp"

/**
 * Arbitrary-precision arithmetic.
 */
template <unsigned N>
using mp_real =
    boost::multiprecision::number<boost::multiprecision::cpp_dec_float<N>,
                                  boost::multiprecision::et_off>;

/**
 * Class for evaluating formula specified by the string
 * typename Real (not mp_real<NUMBER>) for double support.
 */
template <typename Real>
class cseval {
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
  Real value_;

  /** Child elements of the formula tree. */
  cseval *left_eval_, *right_eval_;

  /**
   * Symbol of the imaginary unit (by default - 'i').
   * Only one Latin character (!).
   * TODO: delete me.
   */
  const char imaginary_unit_;

  /** Try to find each symbol of {symbols} string in the {str} string. */
  std::unordered_map<char, size_t> operations_outside_parentheses(
      const std::string &str, const std::string &symbols) const;

  /**  Try to find symbols outside parentheses: '(' and ')'. */
  bool isThereSymbolsOutsideParentheses(const std::string &str) const;

 public:
  cseval(std::string expression, char imaginary_unit = 'i');

  cseval(const cseval &other);

  ~cseval();

  cseval &operator=(const cseval &other) {
    if (this != &other) {
      kind_ = other.kind_;
      id_ = std::string(other.id_);
      value_ = Real(other.value_);
      imaginary_unit_ = other.imaginary_unit_;
      if (left_eval_) {
        delete left_eval_;
      }
      if (other.left_eval_) {
        left_eval_ = new cseval<Real>(other.left_eval_);
      } else {
        left_eval_ = nullptr;
      }
      if (right_eval_) {
        delete right_eval_;
      }
      if (other.right_eval_) {
        right_eval_ = new cseval<Real>(other.right_eval_);
      } else {
        right_eval_ = nullptr;
      }
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

  // Evaluation of subformula.
  Real calculate(
      const std::map<std::string, Real> &variables_to_values,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          cseval<Real>::functionsOneArg) const;
  Real calculate(
      const std::map<std::string, std::string> &variables_to_values,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          cseval<Real>::functionsOneArg) const;
  Real calculate(
      const std::map<std::string, double> &variables_to_values,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          cseval<Real>::functionsOneArg) const;
  // Evaluation derivative of subformula.
  Real calculate_derivative(
      const std::string &variable,
      const std::map<std::string, Real> &variables_to_values,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          cseval<Real>::functionsOneArg,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft =
          cseval<Real>::functionsTwoArgsDLeft,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight =
          cseval<Real>::functionsTwoArgsDRight) const;

  Real calculate_derivative(
      const std::string &variable,
      const std::map<std::string, std::string> &variables_to_values,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          cseval<Real>::functionsOneArg,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft =
          cseval<Real>::functionsTwoArgsDLeft,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight =
          cseval<Real>::functionsTwoArgsDRight) const;

  // Usefull constants.
  const static Real ZERO;
  const static Real ONE;

  //+ general static methods that have a one-to-one correspondence
  // with the operations specified in the expression string
  // "+" - addition
  static Real _add(Real a, Real b) { return a + b; }
  // "-" - subtraction
  static Real _sub(Real a, Real b) { return a - b; }
  // "&" - and
  static Real _and(Real a, Real b) {
    return a != ZERO && b != ZERO ? ONE : ZERO;
  }
  // "|" - or
  static Real _or(Real a, Real b) {
    return a != ZERO || b != ZERO ? ONE : ZERO;
  }
  // "=" - is equal to
  static Real _eq(Real a, Real b) { return a == b ? ONE : ZERO; }
  // "<"
  static Real _lt(Real a, Real b) { return a < b ? ONE : ZERO; }
  // ">"
  static Real _gt(Real a, Real b) { return a > b ? ONE : ZERO; }
  // "/" - division
  static Real _truediv(Real a, Real b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the \'/\' operation");
    }
    return a / b;
  }
  // division for the computation of the derivative (left path)
  static Real _truediv1(Real, Real b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation the left path \
of the derivative");
    }
    return 1 / b;
  }
  // division for the computation of the derivative (right path)
  static Real _truediv2(Real a, Real b) {
    if (b == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
right path of the derivative");
    }
    return ZERO - a / (b * b);
  }
  // "*" - multiplication
  static Real _mul(Real a, Real b) { return a * b; }
  // multiplication for the computation of the derivative (left path)
  static Real _mul1(Real, Real b) { return b; }
  // multiplication for the computation of the derivative (right path)
  static Real _mul2(Real a, Real) { return a; }
  // "^" - exponentiation
  static Real _pow(Real a, Real b) { return pow(a, b); }
  // exponentiation for the computation of the derivative (left path)
  static Real _pow1(Real a, Real b) { return (b * _pow(a, b - ONE)); }
  // exponentiation for the computation of the derivative (right path)
  // TODO test log()
  static Real _pow2(Real a, Real b) { return (_log(a) * _pow(a, b)); }
  //- general static methods

  //+ trigonometric functions, exp, log, sqrt and methods for the computation of
  // the derivative:
  // "sin"
  static Real _sin(Real a) { return sin(a); }
  // "sin" for the derivative
  static Real _sin_d(Real a, Real) { return cos(a); }
  // "asin"
  static Real _asin(Real a) { return asin(a); }
  // "asin" for the derivative
  static Real _asin_d(Real a, Real) {
    if (a * a == ONE) {
      // TODO may be inf?
      throw std::invalid_argument(
          "Division by zero during the computation of the arcsin derivative");
    }
    return ONE / _sqrt(ONE - a * a);
  }
  // "cos"
  static Real _cos(Real a) { return cos(a); }
  // "cos" for the derivative
  static Real _cos_d(Real a, Real) { return ZERO - _sin(a); }
  // "acos"
  static Real _acos(Real a) { return acos(a); }
  static Real _acos_d(Real a, Real) {
    if (a * a == ONE) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the arccos derivative");
    }
    return ZERO - (ONE / _sqrt(ONE - a * a));
  }
  // "tan"
  static Real _tan(Real a) { return tan(a); }
  // "tan" for the derivative
  static Real _tan_d(Real a, Real) {
    if (_cos(a) == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the tangent derivative");
    }
    return ONE / (_cos(a) * _cos(a));
  }
  // "atan"
  static Real _atan(Real a) { return atan(a); }
  // "atan" for the derivative
  static Real _atan_d(Real a, Real) { return ONE / (ONE + a * a); }
  // "exp"
  static Real _exp(Real a) { return exp(a); }
  // "exp" for the derivative
  static Real _exp_d(Real a, Real) { return exp(a); }
  // "log" - (!) Natural logarithm
  static Real _log(Real a) { return log(a); }
  // "log" - (!) Natural logarithm for the derivative
  static Real _log_d(Real a, Real) {
    if (a == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the natural logarithm derivative");
    }
    return ONE / a;
  }
  // "sqrt" - square root
  static Real _sqrt(Real a) { return sqrt(a); }
  // TODO test _sqrt_d()
  // "sqrt" for the derivative
  static Real _sqrt_d(Real a, Real) {
    if (sqrt(a) == ZERO) {
      throw std::invalid_argument(
          "Division by zero during the computation of \
the sqrt derivative");
    }
    return ONE / (2 * sqrt(a));
  }
  //- trigonometric functions, exp, log, sqrt

  //+ auxiliary methods for the computation of the derivative:
  static Real _zero(Real, Real) { return ZERO; }
  static Real _one(Real, Real) { return ONE; }
  static Real _m_one(Real, Real) { return ZERO - ONE; }
  //- auxiliary methods

  // dictionaries contain the appropriate names of operations and static methods
  // for evaluating, e.g.
  // "+" -> _add
  // "sin" -> _sin
  static const std::map<std::string, Real (*)(Real, Real)> functionsTwoArgs;
  static const std::map<std::string, Real (*)(Real)> functionsOneArg;

  // dictionaries contain references to derivatives of basic functions and their
  // names: TODO: delete me?
  static const std::map<std::string, Real (*)(Real, Real)>
      functionsTwoArgsDLeft;
  static const std::map<std::string, Real (*)(Real, Real)>
      functionsTwoArgsDRight;
  static const std::map<std::string, Real (*)(Real)> funcs1ArgD;
};

#endif  // EVAL_MPF_H

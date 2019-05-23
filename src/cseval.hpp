#ifndef EVAL_MPF_H
#define EVAL_MPF_H

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/variant.hpp>

#ifndef CSDEBUG
#define CSDEBUG
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// TODO
// 512 bit of pi number.
#ifndef M_PI_STR
#define M_PI_STR                                                               \
  "3."                                                                         \
  "14159265358979323846264338327950288419716939937510582097494459230781640628" \
  "62089986280348253421170679821480865132823066470938446095505822317253594081" \
  "28481117"
#endif

/**
 * Arbitrary-precision arithmetic.
 */
template <unsigned N>
using mp_real =
    boost::multiprecision::number<boost::multiprecision::cpp_dec_float<N>,
                                  boost::multiprecision::et_on>;

/**
 * All available precisions for high accuracy calculations.
 */
enum AllowedPrecisions : unsigned {
  p_16 = 16U,
  p_24 = 24U,
  p_32 = 32U,
  p_48 = 48U,
  p_64 = 64U,
  p_96 = 96U,
  p_128 = 128U,
  p_192 = 192U,
  p_256 = 256U,
  p_384 = 384U,
  p_512 = 512U,
  p_768 = 768U,
  p_1024 = 1024U,
  p_2048 = 2048U,
  p_3072 = 3072U,
  p_4096 = 4096U,
  p_6144 = 6144U,
  p_8192 = 8192U,
};


/**
 * Allowed precisions sorted in ascending order.
 */
static const AllowedPrecisions min_precision = p_16;
static const AllowedPrecisions max_precision = p_8192;
static const std::tuple<AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions,
                        AllowedPrecisions, AllowedPrecisions, AllowedPrecisions>
    precisions = std::make_tuple(p_16, p_24, p_32, p_48, p_64, p_96, p_128,
                                  p_192, p_256, p_384, p_512, p_768, p_1024,
                                  p_2048, p_3072, p_4096, p_6144, p_8192);
// TODO
struct DoSomething {
  template <typename T>
  bool operator()(T &t) const {
    std::cout << t << "\n";
    return true;
  }
};

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type for_each(
    const std::tuple<Tp...> &, FuncT)  // Unused arguments are given no names.
{}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type for_each(const std::tuple<Tp...> &t, FuncT f) {
  bool next = f(std::get<I>(t));
  if (next) {
    for_each<I + 1, FuncT, Tp...>(t, f);
  }
}

// TODO Complex numbers
// typedef std::complex<float100et> complexFloat100et;

using boost::math::constants::pi;
using boost::multiprecision::acos;
using boost::multiprecision::asin;
using boost::multiprecision::atan;
using boost::multiprecision::cos;
using boost::multiprecision::exp;
using boost::multiprecision::fabs;
using boost::multiprecision::log;
using boost::multiprecision::pow;
using boost::multiprecision::sin;
using boost::multiprecision::sqrt;
using boost::multiprecision::swap;
using boost::multiprecision::tan;

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
  char kind;

  /**
   * The function name or variable name for current node. e.g.
   * "+","-","/","x","a"
   * (!) all variable names must be represented by one Latin letter (!).
   * (!) i, j - reserved for complex numbers. (!)
   */
  std::string id;

  /**
   * Value which was parset from expression, value makes sense if kind is 'n'.
   */
  Real value;

  /** TODO Child elements of the formula tree. */
  cseval *leftEval, *rightEval;

  /** Symbol of the imaginary unit (by default - 'i'). */
  char imaginary_unit;

  /** Try to find each symbol of {symbols} string in the {str} string. */
  std::unordered_map<char, long> findSymbolsOutsideBrackets(
      const std::string &str, const std::string &symbols) const;

  /**  Try to find symbols outside parentheses: '(' and ')'. */
  bool isThereSymbolsOutsideParentheses(const std::string &str) const;

 public:
  cseval(std::string expression, char imaginary_unit = 'i');

  cseval(const cseval &other);

  ~cseval();

  cseval &operator=(const cseval &other) {
    if (this != &other) {
      kind = other.kind;
      id = std::string(other.id);
      value = Real(other.value);
      imaginary_unit = other.imaginary_unit;
      delete leftEval;
      if (other.leftEval) {
        leftEval = new cseval<Real>(other.leftEval);
      } else {
        leftEval = nullptr;
      }
      delete rightEval;
      if (other.rightEval) {
        rightEval = new cseval<Real>(other.rightEval);
      } else {
        rightEval = nullptr;
      }
    }
    return *this;
  }

  // Evaluation of subformula.
  Real calculate(const std::map<std::string, Real> &mapVariableValues,
                 const std::map<std::string, Real (*)(Real, Real)>
                     &mapFunctionTwoArgsValue = functionsTwoArgs,
                 const std::map<std::string, Real (*)(Real)>
                     &mapFunctionOneArgValue = functionsOneArg) const;
  Real calculate(const std::map<std::string, std::string> &mapVariableValues,
                 const std::map<std::string, Real (*)(Real, Real)>
                     &mapFunctionTwoArgsValue = functionsTwoArgs,
                 const std::map<std::string, Real (*)(Real)>
                     &mapFunctionOneArgValue = functionsOneArg) const;
  Real calculate(const std::map<std::string, double> &mapVariableValues,
                 const std::map<std::string, Real (*)(Real, Real)>
                     &mapFunctionTwoArgsValue = functionsTwoArgs,
                 const std::map<std::string, Real (*)(Real)>
                     &mapFunctionOneArgValue = functionsOneArg) const;
  // Evaluation derivative of subformula.
  Real calculateDerivative(
      const std::string &variable,
      const std::map<std::string, Real> &mapVariableValues,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          functionsOneArg,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft =
          functionsTwoArgsDLeft,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight =
          functionsTwoArgsDRight) const;

  Real calculateDerivative(
      const std::string &variable,
      const std::map<std::string, std::string> &mapVariableValues,
      const std::map<std::string, Real (*)(Real, Real)>
          &mapFunctionTwoArgsValue = functionsTwoArgs,
      const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue =
          functionsOneArg,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft =
          functionsTwoArgsDLeft,
      const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight =
          functionsTwoArgsDRight) const;

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
  // names:
  static const std::map<std::string, Real (*)(Real, Real)>
      functionsTwoArgsDLeft;
  static const std::map<std::string, Real (*)(Real, Real)>
      functionsTwoArgsDRight;
  static const std::map<std::string, Real (*)(Real)> functionsOneArgD;
};

#endif  // EVAL_MPF_H

#ifndef EVAL_MPF_H
#define EVAL_MPF_H
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <boost/algorithm/string.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/variant.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 512 bit of pi number
#ifndef M_PI_STR
#define M_PI_STR "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117"
#endif

// arbitrary-precision arithmetic
template <unsigned N>
using mp_real = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<N>, boost::multiprecision::et_on>;

/**
 * TODO 1<<0, 1<<1, 1<<2, 1<<3, 1<<4, ...
 * Sorted in ascending order.
 */
constexpr const unsigned precisions[] =
    {1, 2, 4,
     8, 16, 32,
     64, 128, 256,
     512, 1024, 2048,
     4096, 8192, 8193};

enum allowed_precisions
{
  power_of_two_0 = precisions[0],
  power_of_two_1 = precisions[1],
  power_of_two_2 = precisions[2],
  power_of_two_3 = precisions[3],
  power_of_two_4 = precisions[4],
  power_of_two_5 = precisions[5],
  power_of_two_6 = precisions[6],
  power_of_two_7 = precisions[7],
  power_of_two_8 = precisions[8],
  power_of_two_9 = precisions[9],
  power_of_two_10 = precisions[10],
  power_of_two_11 = precisions[11],
  power_of_two_12 = precisions[12],
  power_of_two_13 = precisions[13],
  power_of_two_14 = precisions[14],
};

typedef boost::multiprecision::cpp_dec_float<1> mp_float_2_0;
typedef boost::multiprecision::cpp_dec_float<2> mp_float_2_1;
typedef boost::multiprecision::cpp_dec_float<4> mp_float_2_2;
typedef boost::multiprecision::cpp_dec_float<8> mp_float_2_3;
typedef boost::multiprecision::cpp_dec_float<16> mp_float_2_4;
typedef boost::multiprecision::cpp_dec_float<32> mp_float_2_5;
typedef boost::multiprecision::cpp_dec_float<64> mp_float_2_6;
typedef boost::multiprecision::cpp_dec_float<128> mp_float_2_7;
typedef boost::multiprecision::cpp_dec_float<256> mp_float_2_8;
typedef boost::multiprecision::cpp_dec_float<512> mp_float_2_9;
typedef boost::multiprecision::cpp_dec_float<1024> mp_float_2_10;
typedef boost::multiprecision::cpp_dec_float<2048> mp_float_2_11;
typedef boost::multiprecision::cpp_dec_float<4096> mp_float_2_12;
typedef boost::multiprecision::cpp_dec_float<8192> mp_float_2_13;
typedef boost::multiprecision::cpp_dec_float<16384> mp_float_2_14;
typedef boost::multiprecision::cpp_dec_float<32768> mp_float_2_15;
typedef boost::multiprecision::cpp_dec_float<65536> mp_float_2_16;
typedef boost::multiprecision::cpp_dec_float<131072> mp_float_2_17;
typedef boost::multiprecision::cpp_dec_float<262144> mp_float_2_18;
typedef boost::multiprecision::cpp_dec_float<524288> mp_float_2_19;
typedef boost::multiprecision::cpp_dec_float<1048576> mp_float_2_20;
typedef boost::multiprecision::cpp_dec_float<2097152> mp_float_2_21;
typedef boost::multiprecision::cpp_dec_float<4194304> mp_float_2_22;
typedef boost::multiprecision::cpp_dec_float<8388608> mp_float_2_23;
typedef boost::multiprecision::cpp_dec_float<16777216> mp_float_2_24;
typedef boost::multiprecision::cpp_dec_float<33554432> mp_float_2_25;
typedef boost::multiprecision::cpp_dec_float<67108864> mp_float_2_26;
typedef boost::multiprecision::cpp_dec_float<134217728> mp_float_2_27;
typedef boost::multiprecision::cpp_dec_float<268435456> mp_float_2_28;
typedef boost::multiprecision::cpp_dec_float<536870912> mp_float_2_29;
typedef boost::multiprecision::cpp_dec_float<1073741824> mp_float_2_30;
typedef boost::multiprecision::cpp_dec_float<2147483647> mp_float_2_31_1;

// typedef boost::multiprecision::number<mp_float_2_0, boost::multiprecision::et_off> float100noet;
// by default using et_on everywhere
// TODO
typedef boost::multiprecision::number<mp_float_2_0, boost::multiprecision::et_on> float_2_0_et; // TODO
typedef boost::multiprecision::number<mp_float_2_1, boost::multiprecision::et_on> float_2_1_et;
typedef boost::multiprecision::number<mp_float_2_2, boost::multiprecision::et_on> float_2_2_et;
typedef boost::multiprecision::number<mp_float_2_3, boost::multiprecision::et_on> float_2_3_et;
typedef boost::multiprecision::number<mp_float_2_4, boost::multiprecision::et_on> float_2_4_et;
typedef boost::multiprecision::number<mp_float_2_5, boost::multiprecision::et_on> float_2_5_et;
typedef boost::multiprecision::number<mp_float_2_6, boost::multiprecision::et_on> float_2_6_et;
typedef boost::multiprecision::number<mp_float_2_7, boost::multiprecision::et_on> float_2_7_et;
typedef boost::multiprecision::number<mp_float_2_8, boost::multiprecision::et_on> float_2_8_et;
typedef boost::multiprecision::number<mp_float_2_9, boost::multiprecision::et_on> float_2_9_et;
typedef boost::multiprecision::number<mp_float_2_10, boost::multiprecision::et_on> float_2_10_et;
typedef boost::multiprecision::number<mp_float_2_11, boost::multiprecision::et_on> float_2_11_et;
typedef boost::multiprecision::number<mp_float_2_12, boost::multiprecision::et_on> float_2_12_et;
typedef boost::multiprecision::number<mp_float_2_13, boost::multiprecision::et_on> float_2_13_et;
typedef boost::multiprecision::number<mp_float_2_14, boost::multiprecision::et_on> float_2_14_et;
typedef boost::multiprecision::number<mp_float_2_15, boost::multiprecision::et_on> float_2_15_et;
typedef boost::multiprecision::number<mp_float_2_16, boost::multiprecision::et_on> float_2_16_et;
typedef boost::multiprecision::number<mp_float_2_17, boost::multiprecision::et_on> float_2_17_et;
typedef boost::multiprecision::number<mp_float_2_18, boost::multiprecision::et_on> float_2_18_et;
typedef boost::multiprecision::number<mp_float_2_19, boost::multiprecision::et_on> float_2_19_et;
typedef boost::multiprecision::number<mp_float_2_20, boost::multiprecision::et_on> float_2_20_et;
typedef boost::multiprecision::number<mp_float_2_21, boost::multiprecision::et_on> float_2_21_et;
typedef boost::multiprecision::number<mp_float_2_22, boost::multiprecision::et_on> float_2_22_et;
typedef boost::multiprecision::number<mp_float_2_23, boost::multiprecision::et_on> float_2_23_et;
typedef boost::multiprecision::number<mp_float_2_24, boost::multiprecision::et_on> float_2_24_et;
typedef boost::multiprecision::number<mp_float_2_25, boost::multiprecision::et_on> float_2_25_et;
typedef boost::multiprecision::number<mp_float_2_26, boost::multiprecision::et_on> float_2_26_et;
typedef boost::multiprecision::number<mp_float_2_27, boost::multiprecision::et_on> float_2_27_et;
typedef boost::multiprecision::number<mp_float_2_28, boost::multiprecision::et_on> float_2_28_et;
typedef boost::multiprecision::number<mp_float_2_29, boost::multiprecision::et_on> float_2_29_et;
typedef boost::multiprecision::number<mp_float_2_30, boost::multiprecision::et_on> float_2_30_et;
typedef boost::multiprecision::number<mp_float_2_31_1, boost::multiprecision::et_on> float_2_31_1_et;
// complex numbers
// TODO
// typedef std::complex<float100et> complexFloat100et;
// TODO delete me
// function with two arguments
// typedef float100et (*mathFunctionTwoArgs)(float100et, float100et);
// function with one argument
// typedef float100et (*mathFunctionOneArg)(float100et);

// error codes (?)
#define WRONG_BRACKETS 10000
#define IMPOSIBLE_VALUE_CONVERSION 10001
#define EMPTY_EXPRESSION 10002
#define UNKNOWN_VARIABLE_OR_FUNCTION 10003
#define DIVISION_BY_ZERO 10004
#define UNKNOWN_VARIABLE_OR_FUNCTION_D 10005
#define UNKNOWN_CALCULATE_D_ERROR 10006
#define UNKNOWN_CALCULATE_ERROR 10099

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
class cseval
{
private:
  /** 
   * Kind of formula element:
   * 'n' - number, 'v' - variable, 'f' - function, 'e' - error.
   */
  char kind;
  /**
   * The function name or variable name for current node. e.g. "+","-","/","x","a"
   * (!) all variable names must be represented by one Latin letter (!).
   * (!) i, j - reserved for complex numbers. (!)
   */
  std::string id;
  /**
   * Value which was parset from expression, value makes sense if kind is 'n'.
   */
  Real value;
  /**
   * Child elements of the formula tree.
   */
  cseval *leftEval, *rightEval;
  /**
   * Try to find each symbol of {symbols} string in the {str} string.
   */
  std::unordered_map<char, int> findSymbolsOutsideBrackets(const std::string &str, const std::string &symbols) const;
  /**
   *  Try to find symbols outside parentheses: '(' and ')'.
   */
  bool isThereSymbolsOutsideParentheses(const std::string &str) const;

public:
  cseval(std::string expression);
  ~cseval();
  // cseval & operator = (const cseval & other) {
  //     if (this != &other) {
  //         delete leftEval;
  //         delete rightEval;
  //         leftEval = other.leftEval;
  //         rightEval = other.rightEval;
  //         kind = other.kind;
  //         id = other.id;
  //         value = other.value;
  //     }
  //     return *this;
  // }
  // evaluation of subformula
  Real calculate(const std::map<std::string, Real> &mapVariableValues,
                 const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue = functionsTwoArgs,
                 const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue = functionsOneArg) const;
  Real calculate(const std::map<std::string, std::string> &mapVariableValues,
                 const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue = functionsTwoArgs,
                 const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue = functionsOneArg) const;
  // evaluation derivative of subformula
  Real calculateDerivative(const std::string &variable,
                           const std::map<std::string, Real> &mapVariableValues,
                           const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue = functionsTwoArgs,
                           const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue = functionsOneArg,
                           const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft = functionsTwoArgsDLeft,
                           const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight = functionsTwoArgsDRight) const;
  Real calculateDerivative(const std::string &variable,
                           const std::map<std::string, std::string> &mapVariableValues,
                           const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue = functionsTwoArgs,
                           const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue = functionsOneArg,
                           const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft = functionsTwoArgsDLeft,
                           const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight = functionsTwoArgsDRight) const;

  // usefull constants
  const static Real ZERO;
  const static Real ONE;

  //+ general static methods that have a one-to-one correspondence
  // with the operations specified in the expression string
  // "+" - addition
  static Real _add(Real a, Real b) { return a + b; }
  // "-" - subtraction
  static Real _sub(Real a, Real b) { return a - b; }
  // "&" - and
  static Real _and(Real a, Real b) { return a != ZERO && b != ZERO ? ONE : ZERO; }
  // "|" - or
  static Real _or(Real a, Real b) { return a != ZERO || b != ZERO ? ONE : ZERO; }
  // "=" - is equal to
  static Real _eq(Real a, Real b) { return a == b ? ONE : ZERO; }
  // "<"
  static Real _lt(Real a, Real b) { return a < b ? ONE : ZERO; }
  // ">"
  static Real _gt(Real a, Real b) { return a > b ? ONE : ZERO; }
  // "/" - division
  static Real _truediv(Real a, Real b)
  {
    if (b == ZERO)
    {
      std::cerr << "ERROR: division by zero error!";
      throw DIVISION_BY_ZERO;
    }
    return a / b;
  }
  // division for the computation of the derivative (left path)
  static Real _truediv1(Real, Real b)
  {
    if (b == ZERO)
    {
      std::cerr << "ERROR: division by zero error!";
      throw DIVISION_BY_ZERO;
    }
    return 1 / b;
  }
  // division for the computation of the derivative (right path)
  static Real _truediv2(Real a, Real b)
  {
    if (b == ZERO)
    {
      std::cerr << "ERROR: division by zero error!";
      throw DIVISION_BY_ZERO;
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
  //TODO test log()
  static Real _pow2(Real a, Real b) { return (_log(a) * _pow(a, b)); }
  //- general static methods

  //+ trigonometric functions, exp, log, sqrt and methods for the computation of the derivative:
  // "sin"
  static Real _sin(Real a) { return sin(a); }
  // "sin" for the derivative
  static Real _sin_d(Real a, Real) { return cos(a); }
  // "asin"
  static Real _asin(Real a) { return asin(a); }
  // "asin" for the derivative
  static Real _asin_d(Real a, Real)
  {
    if (a * a == ONE)
    {
      std::cerr << "ERROR: division by zero error!";
      throw DIVISION_BY_ZERO;
    }
    return ONE / _sqrt(ONE - a * a);
  }
  // "cos"
  static Real _cos(Real a) { return cos(a); }
  // "cos" for the derivative
  static Real _cos_d(Real a, Real) { return ZERO - _sin(a); }
  // "acos"
  static Real _acos(Real a) { return acos(a); }
  static Real _acos_d(Real a, Real)
  {
    if (a * a == ONE)
    {
      std::cerr << "ERROR: division by zero error!";
      throw DIVISION_BY_ZERO;
    }
    return ZERO - (ONE / _sqrt(ONE - a * a));
  }
  // "tan"
  static Real _tan(Real a) { return tan(a); }
  // "tan" for the derivative
  static Real _tan_d(Real a, Real)
  {
    if (_cos(a) == ZERO)
    {
      std::cerr << "ERROR: division by zero error!";
      throw DIVISION_BY_ZERO;
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
  static Real _log_d(Real a, Real)
  {
    if (a == ZERO)
    {
      std::cerr << "ERROR: division by zero error!";
      throw DIVISION_BY_ZERO;
    }
    return ONE / a;
  }
  // "sqrt" - square root
  static Real _sqrt(Real a) { return sqrt(a); }
  // TODO test _sqrt_d()
  // "sqrt" for the derivative
  static Real _sqrt_d(Real a, Real)
  {
    if (sqrt(a) == ZERO)
    {
      std::cerr << "ERROR: division by zero error!";
      throw DIVISION_BY_ZERO;
    }
    return ONE / (2 * sqrt(a));
  }
  //- trigonometric functions, exp, log, sqrt

  //+ auxiliary methods for the computation of the derivative:
  static Real _zero(Real, Real) { return ZERO; }
  static Real _one(Real, Real) { return ONE; }
  static Real _m_one(Real, Real) { return ZERO - ONE; }
  //- auxiliary methods

  // dictionaries contain the appropriate names of operations and static methods for evaluating, e.g.
  // "+" -> _add
  // "sin" -> _sin
  static const std::map<std::string, Real (*)(Real, Real)> functionsTwoArgs;
  static const std::map<std::string, Real (*)(Real)> functionsOneArg;

  // dictionaries contain references to derivatives of basic functions and their names:
  static const std::map<std::string, Real (*)(Real, Real)> functionsTwoArgsDLeft;
  static const std::map<std::string, Real (*)(Real, Real)> functionsTwoArgsDRight;
  static const std::map<std::string, Real (*)(Real)> functionsOneArgD;
};

#endif // EVAL_MPF_H

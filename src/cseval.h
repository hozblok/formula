#ifndef EVAL_MPF_H
#define EVAL_MPF_H
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 512 bit of pi number
#ifndef M_PI_STR
#define M_PI_STR "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117"
#endif

// arbitrary-precision arithmetic
typedef boost::multiprecision::cpp_dec_float<16> mp_float_100;
typedef boost::multiprecision::number<mp_float_100, boost::multiprecision::et_off> float100noet;
// by default using float_100_et everywhere
typedef boost::multiprecision::number<mp_float_100, boost::multiprecision::et_on> float100et;
// complex numbers
typedef std::complex<float100et> complexFloat100et;
// function with two arguments
typedef float100et (*mathFunctionTwoArgs)(float100et, float100et);
// function with one argument
typedef float100et (*mathFunctionOneArg)(float100et);

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

// Class for evaluating formula specified by the string
template <typename Real>
class cseval
{
private:
  // kind of formula element:
  // 'n' - number, 'v' - variable, 'f' - function, 'e' - error
  char kind;
  // the function name or variable name for current node. e.g. "+","-","/","x","a"
  // (!) all variable names must be represented by one Latin letter (!)
  std::string id;
  // value which was parset from expression, value makes sense if kind is 'n'
  Real value;
  // child elements in the formula tree
  cseval *leftEval, *rightEval;
  // try to find each symbol of {symbols} string in the {str} string
  std::unordered_map<char, int> findSymbolsOutsideBrackets(const std::string &str, const std::string &symbols) const;
  bool isThereSymbolsOutsideBrackets(const std::string &str) const;
  
public:
  cseval(std::string expression);
  ~cseval();
  // evaluation of subformula
  Real calculate(const std::map<std::string, Real> &mapVariableValues,
                       const std::map<std::string, mathFunctionTwoArgs> &mapFunctionTwoArgsValue = functionsTwoArgs,
                       const std::map<std::string, mathFunctionOneArg> &mapFunctionOneArgValue = functionsOneArg) const;
  Real calculate(const std::map<std::string, std::string> &mapVariableValues,
                       const std::map<std::string, mathFunctionTwoArgs> &mapFunctionTwoArgsValue = functionsTwoArgs,
                       const std::map<std::string, mathFunctionOneArg> &mapFunctionOneArgValue = functionsOneArg) const;
  // evaluation derivative of subformula
  Real calculateDerivative(const std::string &variable,
                                 const std::map<std::string, Real> &mapVariableValues,
                                 const std::map<std::string, mathFunctionTwoArgs> &mapFunctionTwoArgsValue = functionsTwoArgs,
                                 const std::map<std::string, mathFunctionOneArg> &mapFunctionOneArgValue = functionsOneArg,
                                 const std::map<std::string, mathFunctionTwoArgs> &mapFunctionDerivLeft = functionsTwoArgsDLeft,
                                 const std::map<std::string, mathFunctionTwoArgs> &mapFunctionDerivRight = functionsTwoArgsDRight) const;
  Real calculateDerivative(const std::string &variable,
                                 const std::map<std::string, std::string> &mapVariableValues,
                                 const std::map<std::string, mathFunctionTwoArgs> &mapFunctionTwoArgsValue = functionsTwoArgs,
                                 const std::map<std::string, mathFunctionOneArg> &mapFunctionOneArgValue = functionsOneArg,
                                 const std::map<std::string, mathFunctionTwoArgs> &mapFunctionDerivLeft = functionsTwoArgsDLeft,
                                 const std::map<std::string, mathFunctionTwoArgs> &mapFunctionDerivRight = functionsTwoArgsDRight) const;

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
  static const std::map<std::string, mathFunctionTwoArgs> functionsTwoArgs;
  static const std::map<std::string, mathFunctionOneArg> functionsOneArg;

  // dictionaries contain references to derivatives of basic functions and their names:
  static const std::map<std::string, mathFunctionTwoArgs> functionsTwoArgsDLeft;
  static const std::map<std::string, mathFunctionTwoArgs> functionsTwoArgsDRight;
  static const std::map<std::string, mathFunctionOneArg> functionsOneArgD;
};

#endif // EVAL_MPF_H

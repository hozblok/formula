#include "cseval.hpp"

template <typename Real>
const Real cseval<Real>::ZERO = Real("0");

template <typename Real>
const Real cseval<Real>::ONE = Real("1");

template <typename Real>
const std::map<std::string, Real (*)(Real, Real)>
    cseval<Real>::functionsTwoArgs = {
        {std::string("|"), _or},  {std::string("&"), _and},
        {std::string("="), _eq},  {std::string(">"), _gt},
        {std::string("<"), _lt},  {std::string("+"), _add},
        {std::string("-"), _sub}, {std::string("/"), _truediv},
        {std::string("*"), _mul}, {std::string("^"), _pow}};
template <typename Real>
const std::map<std::string, Real (*)(Real)> cseval<Real>::functionsOneArg = {
    {std::string("sin"), _sin}, {std::string("asin"), _asin},
    {std::string("cos"), _cos}, {std::string("acos"), _acos},
    {std::string("tan"), _tan}, {std::string("atan"), _atan},
    {std::string("log"), _log}, {std::string("sqrt"), _sqrt},
    {std::string("exp"), _exp}};
template <typename Real>
const std::map<std::string, Real (*)(Real, Real)>
    cseval<Real>::functionsTwoArgsDLeft = {
        {std::string("+"), _one},       {std::string("-"), _one},
        {std::string("*"), _mul1},      {std::string("/"), _truediv1},
        {std::string("^"), _pow1},      {std::string("sin"), _sin_d},
        {std::string("asin"), _asin_d}, {std::string("cos"), _cos_d},
        {std::string("acos"), _acos_d}, {std::string("tan"), _tan_d},
        {std::string("atan"), _atan_d}, {std::string("log"), _log_d},
        {std::string("sqrt"), _sqrt_d}, {std::string("exp"), _exp_d}};
template <typename Real>
const std::map<std::string, Real (*)(Real, Real)>
    cseval<Real>::functionsTwoArgsDRight = {{std::string("+"), _one},
                                            {std::string("-"), _m_one},
                                            {std::string("*"), _mul2},
                                            {std::string("/"), _truediv2},
                                            {std::string("^"), _pow2}};

template <typename Real>
cseval<Real>::cseval(const cseval<Real> &other)
    : kind(other.kind),
      id(std::string(other.id)),
      value(Real(other.value)),
      leftEval(nullptr),
      rightEval(nullptr),
      imaginary_unit(other.imaginary_unit) {
#ifdef CSDEBUG
  std::cout << "copy constructor cseval" << std::endl;
#endif
  if (other.leftEval) {
    leftEval = new cseval<Real>(*other.leftEval);
  }
  if (other.rightEval) {
    rightEval = new cseval<Real>(*other.rightEval);
  }
}

template <typename Real>
cseval<Real>::cseval(std::string expression, char imaginary_unit)
    : kind('e'),
      id(""),
      value("0"),
      leftEval(nullptr),
      rightEval(nullptr),
      imaginary_unit(imaginary_unit) {
#ifdef CSDEBUG
  std::cout << "constructor cseval, expression:" << expression << std::endl;
#endif
  if (expression.empty()) {
    throw std::invalid_argument(
        "Expression string is empty or \
some substring within brackets \
is empty");
  }

  // Remove braces.
  while (!isThereSymbolsOutsideParentheses(expression)) {
    expression.erase(expression.cbegin());
    expression.pop_back();
  }

  // Find operations: logical or, logical and, relational =, <, >, addition,
  // subtraction, multiplication, division, the construction of the power First,
  // we are looking for addition and subtraction, because division and
  // multiplication have a higher priority and should be performed first.
  std::string operations("|&=><+-*/^");
  std::unordered_map<char, long> foundedOperation =
      findSymbolsOutsideBrackets(expression, operations);
  for (std::string::const_iterator it = operations.cbegin();
       it != operations.cend(); ++it) {
    if (foundedOperation.at(*it) != -1) {
      kind = 'f';
      id = *it;
      if (*it == '-' || *it == '+') {
        // if '-' represents negative number or negative value of variable, not
        // a subtraction operation similarly for '+'
        if (foundedOperation.at(*it) == 0) {
          // allowed '-x', '--x', '---x*y', '-.01' etc.
          leftEval = new cseval<Real>(std::string("0"));
          rightEval = new cseval<Real>(expression.substr(1));
          return;
        } else if (operations.find(expression.at(foundedOperation.at(*it) -
                                                 1)) != std::string::npos) {
          // allowed 'x/-1' or 'y^-x' etc.
          // go on to the next operation
          continue;
        }
      }
      // split the string into two parts
      // separator: + - * / ^
      leftEval =
          new cseval<Real>(expression.substr(0, foundedOperation.at(*it)));
      rightEval =
          new cseval<Real>(expression.substr(foundedOperation.at(*it) + 1));
      return;
    }
  }

  size_t len = expression.size();
  size_t iLeftBracket = expression.find('(');
  if (iLeftBracket == len - 1) {
    throw std::invalid_argument(
        "The given expression contains the wrong \
location and / or number of brackets");
  }
  if (iLeftBracket == std::string::npos) {
    // there is no '(' => kind === 'v' or 'n'; or id === "pi"
    // variable or number or pi
    // TODO: support multisymbol variables
    if (expression.find("pi") == 0) {
      kind = 'n';
      id = "pi";
      value = Real(M_PI_STR);
    } else if (len == 1 && expression.at(0) == imaginary_unit) {
      throw std::invalid_argument(
          (boost::format("Complex number indicator was founded in \
the expression: %s. Not implemented \
at the moment") %
           expression)
              .str());
    } else if (len == 1 && expression.at(0) >= 'a' && expression.at(0) <= 'z') {
      // x or y or z or other available variables
      kind = 'v';
      id = expression;
    } else {
      // this is number
      kind = 'n';
      id = expression;
      try {
        value = Real(expression.data());
      } catch (...) {
        throw std::invalid_argument(
            (boost::format("Unknown number value: %s") % expression).str());
      }
    }
  } else if (iLeftBracket < len) {
    // this is a function
    std::string name_fun = expression.substr(0, iLeftBracket);
    expression = expression.substr(iLeftBracket + 1);
    expression.pop_back();
    kind = 'f';
    id = name_fun;
    size_t iComma = expression.find(',');
    if (iComma != std::string::npos) {
      leftEval = new cseval<Real>(expression.substr(0, iComma));
      rightEval = new cseval<Real>(expression.substr(iComma + 1));
    } else {
      leftEval = new cseval<Real>(expression);
    }
  }
}

template <typename Real>
cseval<Real>::~cseval() {
#ifdef CSDEBUG
  std::cout << "desctructor cseval, id:" << id << " kind:" << kind << std::endl;
#endif
  if (leftEval) {
    delete leftEval;
    leftEval = nullptr;
  }
  if (rightEval) {
    delete rightEval;
    rightEval = nullptr;
  }
}

template <typename Real>
Real cseval<Real>::calculate(
    const std::map<std::string, Real> &mapVariableValues,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue) const {
  if (kind == 'n') {
    return value;
  } else if (kind == 'v') {
    typename std::map<std::string, Real>::const_iterator it;
    for (it = mapVariableValues.cbegin(); it != mapVariableValues.cend();
         ++it) {
      if (it->first == id) {
        return it->second;
      }
    }
    throw std::invalid_argument(
        (boost::format("The required value is not found during the \
calculation of the expression, id: %s") %
         id)
            .str());
  } else if (kind == 'f') {
    if (leftEval && rightEval) {
      // function with two arguments
      Real left("0");
      Real right("0");
      left = leftEval->calculate(mapVariableValues, mapFunctionTwoArgsValue,
                                 mapFunctionOneArgValue);
      right = rightEval->calculate(mapVariableValues, mapFunctionTwoArgsValue,
                                   mapFunctionOneArgValue);
      typename std::map<std::string, Real (*)(Real, Real)>::const_iterator
          itFunction;
      itFunction = mapFunctionTwoArgsValue.find(id);
      if (itFunction != mapFunctionTwoArgsValue.cend()) {
        return itFunction->second(left, right);
      }
    } else if (leftEval) {
      // function with one argument
      Real left("0");
      left = leftEval->calculate(mapVariableValues, mapFunctionTwoArgsValue,
                                 mapFunctionOneArgValue);
      typename std::map<std::string, Real (*)(Real)>::const_iterator itFunction;
      itFunction = mapFunctionOneArgValue.find(id);
      if (itFunction != mapFunctionOneArgValue.cend()) {
        return itFunction->second(left);
      }
    }
    throw std::invalid_argument(
        (boost::format("The required function is not found during \
the calculation of the expression, id: %s") %
         id)
            .str());
  }
  throw std::runtime_error(
      (boost::format("Unknown error during the calculation of the expression, \
id: %s, kind: %s") %
       id % kind)
          .str());
}

template <typename Real>
Real cseval<Real>::calculate(
    const std::map<std::string, std::string> &mapVariableValues,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue) const {
  typename std::map<std::string, Real> values;
  typename std::map<std::string, std::string>::const_iterator it;
  for (it = mapVariableValues.cbegin(); it != mapVariableValues.cend(); ++it) {
    values[it->first] = Real(it->second);
  }
  return calculate(values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
}

template <typename Real>
Real cseval<Real>::calculate(
    const std::map<std::string, double> &mapVariableValues,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue) const {
  typename std::map<std::string, Real> values;
  typename std::map<std::string, double>::const_iterator it;
  for (it = mapVariableValues.cbegin(); it != mapVariableValues.cend(); ++it) {
    values[it->first] = Real(it->second);
  }
  return calculate(values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
}

template <typename Real>
Real cseval<Real>::calculateDerivative(
    const std::string &variable,
    const std::map<std::string, Real> &mapVariableValues,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight)
    const {
  if (kind == 'n') {
    return ZERO;
  } else if (kind == 'v') {
    if (id == variable) {
      return ONE;
    }
    return ZERO;
  } else if (kind == 'f') {
    if (leftEval && rightEval) {
      // (u+v)'=u'+v'=1*d+1*c
      // (u-v)'=u'-v'=1*d-1*c
      // (u*v)'=vu'+uv'=b*d+a*c
      // (u/v)'=(vu'-uv')/v^2=(1/v)*u'-(u/v^2)v'=(1/b)*d-(a/b^2)*c
      // (u^v)'=(e^(v*ln(u)))'=e^(v*ln(u)) * (v*ln(u))'=v*u^(v-1)*u' +
      // (u^v)*ln(u)*v'=b*a^(b-1)*d + a^b*ln(a)*c a===u, b===v, d===u', c===v'
      Real a = leftEval->calculate(mapVariableValues, mapFunctionTwoArgsValue,
                                   mapFunctionOneArgValue);
      Real d = leftEval->calculateDerivative(
          variable, mapVariableValues, mapFunctionTwoArgsValue,
          mapFunctionOneArgValue, mapFunctionDerivLeft, mapFunctionDerivRight);
      Real b = rightEval->calculate(mapVariableValues, mapFunctionTwoArgsValue,
                                    mapFunctionOneArgValue);
      Real c = rightEval->calculateDerivative(
          variable, mapVariableValues, mapFunctionTwoArgsValue,
          mapFunctionOneArgValue, mapFunctionDerivLeft, mapFunctionDerivRight);
      typename std::map<std::string, Real (*)(Real, Real)>::const_iterator
          itFunction_1;
      itFunction_1 = mapFunctionDerivLeft.find(id);
      typename std::map<std::string, Real (*)(Real, Real)>::const_iterator
          itFunction_2;
      itFunction_2 = mapFunctionDerivRight.find(id);
      if (itFunction_1 != mapFunctionDerivLeft.cend() &&
          itFunction_2 != mapFunctionDerivRight.cend()) {
        return itFunction_1->second(a, b) * d + itFunction_2->second(a, b) * c;
      }
    } else if (leftEval) {
      // the same, but b === 0 and c === 0
      Real a = leftEval->calculate(mapVariableValues, mapFunctionTwoArgsValue,
                                   mapFunctionOneArgValue);
      Real d = leftEval->calculateDerivative(
          variable, mapVariableValues, mapFunctionTwoArgsValue,
          mapFunctionOneArgValue, mapFunctionDerivLeft, mapFunctionDerivRight);
      typename std::map<std::string, Real (*)(Real, Real)>::const_iterator
          itFunction;
      itFunction = mapFunctionDerivLeft.find(id);
      if (itFunction != mapFunctionDerivLeft.cend()) {
        return itFunction->second(a, ZERO) * d;
      }
    }
    throw std::invalid_argument(
        (boost::format("The required function is not found \
during the calculation of the derivative, id: %s") %
         id)
            .str());
  }
  throw std::runtime_error(
      (boost::format("Unknown error during the calculation of the derivative, \
id: %s, kind: %s") %
       id % kind)
          .str());
}

template <typename Real>
Real cseval<Real>::calculateDerivative(
    const std::string &variable,
    const std::map<std::string, std::string> &mapVariableValues,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight)
    const {
  typename std::map<std::string, Real> values;
  typename std::map<std::string, std::string>::const_iterator it;
  for (it = mapVariableValues.cbegin(); it != mapVariableValues.cend(); ++it) {
    values[it->first] = Real(it->second);
  }
  return calculateDerivative(variable, values, mapFunctionTwoArgsValue,
                             mapFunctionOneArgValue, mapFunctionDerivLeft,
                             mapFunctionDerivRight);
}

template <typename Real>
std::unordered_map<char, long> cseval<Real>::findSymbolsOutsideBrackets(
    const std::string &str, const std::string &symbols) const {
  unsigned int countBraces = 0;

  std::unordered_map<char, long> operationIndex;
  for (std::string::const_iterator it = symbols.cbegin(); it < symbols.cend();
       ++it) {
    operationIndex[*it] = -1;
  }
  for (std::string::const_iterator it = str.cbegin(); it != str.cend(); ++it) {
    if (*it == '(') {
      countBraces++;
    } else if (*it == ')') {
      countBraces--;
    }
    // we get only the first match in str, so we need the last condition
    if (countBraces == 0 && symbols.find(*it) != std::string::npos &&
        operationIndex.at(*it) == -1) {
      operationIndex[*it] = it - str.cbegin();
    }
  }
  return operationIndex;
}

template <typename Real>
bool cseval<Real>::isThereSymbolsOutsideParentheses(
    const std::string &str) const {
  unsigned int countBraces = 0;
  std::string::const_iterator it = str.cbegin();
  if (*it != '(') {
    return true;
  }
  countBraces++;
  it++;

  for (; it != str.cend(); ++it) {
    if (countBraces == 0) {
      if (*it == '(') {
        // '(x+1)(y+1)'? There is no
        // mathematical operation between brackets.
        throw std::invalid_argument(
            "Expression cannot be parsed: \
there may be no mathematical operation between brackets");
      }
      return true;
    }
    if (*it == '(') {
      countBraces++;
    } else if (*it == ')') {
      countBraces--;
    }
  }
  return false;
}
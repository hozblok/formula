#include "./cseval.hpp"

template <typename Real>
const Real cseval<Real>::ZERO = Real("0");

template <typename Real>
const Real cseval<Real>::ONE = Real("1");

template <typename Real>
const std::map<std::string, Real (*)(Real, Real)>
    cseval<Real>::functionsTwoArgs = {
        {std::string("|"), cseval<Real>::_or},
        {std::string("&"), cseval<Real>::_and},
        {std::string("="), cseval<Real>::_eq},
        {std::string(">"), cseval<Real>::_gt},
        {std::string("<"), cseval<Real>::_lt},
        {std::string("+"), cseval<Real>::_add},
        {std::string("-"), cseval<Real>::_sub},
        {std::string("/"), cseval<Real>::_truediv},
        {std::string("*"), cseval<Real>::_mul},
        {std::string("^"), cseval<Real>::_pow}};

template <typename Real>
const std::map<std::string, Real (*)(Real)> cseval<Real>::functionsOneArg = {
    {std::string("sin"), cseval<Real>::_sin},
    {std::string("asin"), cseval<Real>::_asin},
    {std::string("cos"), cseval<Real>::_cos},
    {std::string("acos"), cseval<Real>::_acos},
    {std::string("tan"), cseval<Real>::_tan},
    {std::string("atan"), cseval<Real>::_atan},
    {std::string("log"), cseval<Real>::_log},
    {std::string("sqrt"), cseval<Real>::_sqrt},
    {std::string("exp"), cseval<Real>::_exp}};

template <typename Real>
const std::map<std::string, Real (*)(Real, Real)>
    cseval<Real>::functionsTwoArgsDLeft = {
        {std::string("+"), cseval<Real>::_one},
        {std::string("-"), cseval<Real>::_one},
        {std::string("*"), cseval<Real>::_mul1},
        {std::string("/"), cseval<Real>::_truediv1},
        {std::string("^"), cseval<Real>::_pow1},
        {std::string("sin"), cseval<Real>::_sin_d},
        {std::string("asin"), cseval<Real>::_asin_d},
        {std::string("cos"), cseval<Real>::_cos_d},
        {std::string("acos"), cseval<Real>::_acos_d},
        {std::string("tan"), cseval<Real>::_tan_d},
        {std::string("atan"), cseval<Real>::_atan_d},
        {std::string("log"), cseval<Real>::_log_d},
        {std::string("sqrt"), cseval<Real>::_sqrt_d},
        {std::string("exp"), cseval<Real>::_exp_d}};

template <typename Real>
const std::map<std::string, Real (*)(Real, Real)>
    cseval<Real>::functionsTwoArgsDRight = {
        {std::string("+"), cseval<Real>::_one},
        {std::string("-"), cseval<Real>::_m_one},
        {std::string("*"), cseval<Real>::_mul2},
        {std::string("/"), cseval<Real>::_truediv2},
        {std::string("^"), cseval<Real>::_pow2}};

template <typename Real>
cseval<Real>::cseval(const cseval<Real> &other)
    : kind_(other.kind_),
      id_(std::string(other.id_)),
      value_(Real(other.value_)),
      left_eval_(nullptr),
      right_eval_(nullptr),
      imaginary_unit_(other.imaginary_unit_) {
#ifdef CSDEBUG
  std::cout << "copy constructor cseval+, kind: " << kind_ << ", id: " << id_
            << std::endl;
#endif
  if (other.left_eval_) {
    left_eval_ = new cseval<Real>(*other.left_eval_);
  }
  if (other.right_eval_) {
    right_eval_ = new cseval<Real>(*other.right_eval_);
  }
#ifdef CSDEBUG
  std::cout << "copy constructor cseval-" << std::endl;
#endif
}

template <typename Real>
cseval<Real>::cseval(std::string expression, char imaginary_unit)
    : kind_('e'),
      id_(""),
      value_("0"),
      left_eval_(nullptr),
      right_eval_(nullptr),
      imaginary_unit_(imaginary_unit) {
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
    expression.erase(expression.begin());
    expression.pop_back();
  }

  // First, we are looking for addition and subtraction, because division and
  // multiplication have a higher priority and should be performed first.
  std::unordered_map<char, size_t> op_to_index =
      operations_outside_parentheses(expression, kOperations);
  for (std::string::const_iterator it = kOperations.cbegin();
       it != kOperations.cend(); ++it) {
    auto op_index = op_to_index.at(*it);
    if (op_index != kNpos) {
      if (*it == '-' || *it == '+') {
        // if '-' represents negative number or negative value of variable, not
        // a subtraction operation similarly for '+'
        if (op_index == 0) {
          // allowed '-x', '--x', '---x*y', '-.01' etc.
          kind_ = 'f';
          id_ = *it;
          left_eval_ = new cseval<Real>(std::string("0"));
          right_eval_ = new cseval<Real>(expression.substr(1));
          return;
        } else if (kOperations.find(expression.at(op_index - 1)) !=
                   std::string::npos) {
          // allowed 'x/-1' or 'y^-x' etc.
          // go on to the next operation
          continue;
        }
      }
      kind_ = 'f';
      id_ = *it;
      // Split the string into two parts by the separator (operation).
      left_eval_ = new cseval<Real>(expression.substr(0, op_to_index.at(*it)));
      right_eval_ =
          new cseval<Real>(expression.substr(op_to_index.at(*it) + 1));
      return;
    }
  }

  size_t iLeftBracket = expression.find('(');
  if (expression.back() == '(') {
    throw std::invalid_argument(
        "The given expression contains the wrong \
location and / or number of brackets");
  }
  if (iLeftBracket == std::string::npos) {
    // There is no '(' => kind === 'v' or 'n'; or id === "pi".
    // Variable or number or pi.
    if (expression == std::string("pi")) {
      kind_ = 'n';
      id_ = "pi";
      value_ = Real(M_PI_STR);
    } else if (expression == std::string(1, imaginary_unit_)) {
      throw std::invalid_argument(
          (boost::format("Complex number indicator was founded in \
the expression: %s. Not implemented at the class for Real numbers") %
           expression)
              .str());
    } else if (std::regex_match(expression, kIsNumberRegex)) {
      kind_ = 'n';
      id_ = expression;
      try {
        value_ = Real(expression.data());
      } catch (...) {
        throw std::invalid_argument(
            (boost::format("Unknown number value: %s") % expression).str());
      }
    } else {
      // Validate:
      if (kForbiddenStartVariableSymbols.find(expression.at(0)) !=
          std::string::npos) {
        throw std::invalid_argument(
            (boost::format(
                 "Variable name '%s' starts with the forbidden character "
                 "'%s', "
                 "a variable name cannot begin with a number or a point") %
             expression % std::string(1, expression.at(0)))
                .str());
      }
      std::string var_name(expression);
      std::string invalid_characters("");
      std::smatch m;
      while (std::regex_search(var_name, m, kWrongVariableRegex)) {
        if (!invalid_characters.empty()) invalid_characters += ", ";
        for (auto x : m) invalid_characters += x;
        var_name = m.suffix().str();
      }
      if (!invalid_characters.empty()) {
        throw std::invalid_argument(
            (boost::format(
                 "Variable name '%s' contains forbidden characters: %s") %
             expression % invalid_characters)
                .str());
      }
      kind_ = 'v';
      id_ = expression;
    }
  } else {
    // There is '(' in the expression. This is a function.
    std::string name_fun = expression.substr(0, iLeftBracket);
    expression = expression.substr(iLeftBracket + 1);
    expression.pop_back();
    kind_ = 'f';
    id_ = name_fun;
    size_t index_comma = expression.find(',');
    if (index_comma != std::string::npos) {
      left_eval_ = new cseval<Real>(expression.substr(0, index_comma));
      right_eval_ = new cseval<Real>(expression.substr(index_comma + 1));
    } else {
      left_eval_ = new cseval<Real>(expression);
    }
  }
}

template <typename Real>
cseval<Real>::~cseval() {
#ifdef CSDEBUG
  std::cout << "desctructor cseval, id:" << id_ << " kind:" << kind_
            << std::endl;
#endif
  if (left_eval_) {
    delete left_eval_;
    left_eval_ = nullptr;
  }
  if (right_eval_) {
    delete right_eval_;
    right_eval_ = nullptr;
  }
}

template <typename Real>
Real cseval<Real>::calculate(
    const std::map<std::string, Real> &variables_to_values,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue) const {
  if (kind_ == 'n') {
    return value_;
  } else if (kind_ == 'v') {
    typename std::map<std::string, Real>::const_iterator it;
    for (it = variables_to_values.cbegin(); it != variables_to_values.cend();
         ++it) {
      if (it->first == id_) {
        return it->second;
      }
    }
    throw std::invalid_argument(
        (boost::format("The required value is not found during the \
calculation of the expression, variable name: '%s'") %
         id_)
            .str());
  } else if (kind_ == 'f') {
    if (left_eval_ && right_eval_) {
      // function with two arguments
      Real left("0");
      Real right("0");
      left = left_eval_->calculate(variables_to_values, mapFunctionTwoArgsValue,
                                   mapFunctionOneArgValue);
      right = right_eval_->calculate(
          variables_to_values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
      typename std::map<std::string, Real (*)(Real, Real)>::const_iterator
          itFunction;
      itFunction = mapFunctionTwoArgsValue.find(id_);
      if (itFunction != mapFunctionTwoArgsValue.cend()) {
        return itFunction->second(left, right);
      }
    } else if (left_eval_) {
      // function with one argument
      Real left("0");
      left = left_eval_->calculate(variables_to_values, mapFunctionTwoArgsValue,
                                   mapFunctionOneArgValue);
      typename std::map<std::string, Real (*)(Real)>::const_iterator itFunction;
      itFunction = mapFunctionOneArgValue.find(id_);
      if (itFunction != mapFunctionOneArgValue.cend()) {
        return itFunction->second(left);
      }
    }
    throw std::invalid_argument(
        (boost::format("The required function is not found during \
the calculation of the expression, id: %s") %
         id_)
            .str());
  }
  throw std::runtime_error(
      (boost::format("Unknown error during the calculation of the expression, \
id: %s, kind: %s") %
       id_ % kind_)
          .str());
}

template <typename Real>
Real cseval<Real>::calculate(
    const std::map<std::string, std::string> &variables_to_values,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue) const {
  typename std::map<std::string, Real> values;
  typename std::map<std::string, std::string>::const_iterator it;
  for (it = variables_to_values.cbegin(); it != variables_to_values.cend();
       ++it) {
    values[it->first] = Real(it->second);
  }
  return calculate(values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
}

template <typename Real>
Real cseval<Real>::calculate(
    const std::map<std::string, double> &variables_to_values,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue) const {
  typename std::map<std::string, Real> values;
  typename std::map<std::string, double>::const_iterator it;
  for (it = variables_to_values.cbegin(); it != variables_to_values.cend();
       ++it) {
    values[it->first] = Real(it->second);
  }
  return calculate(values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
}

template <typename Real>
Real cseval<Real>::calculate_derivative(
    const std::string &variable,
    const std::map<std::string, Real> &variables_to_values,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight)
    const {
  if (kind_ == 'n') {
    return ZERO;
  } else if (kind_ == 'v') {
    if (id_ == variable) {
      return ONE;
    }
    return ZERO;
  } else if (kind_ == 'f') {
    if (left_eval_ && right_eval_) {
      // (u+v)'=u'+v'=1*d+1*c
      // (u-v)'=u'-v'=1*d-1*c
      // (u*v)'=vu'+uv'=b*d+a*c
      // (u/v)'=(vu'-uv')/v^2=(1/v)*u'-(u/v^2)v'=(1/b)*d-(a/b^2)*c
      // (u^v)'=(e^(v*ln(u)))'=e^(v*ln(u)) * (v*ln(u))'=v*u^(v-1)*u' +
      // (u^v)*ln(u)*v'=b*a^(b-1)*d + a^b*ln(a)*c a===u, b===v, d===u', c===v'
      Real a = left_eval_->calculate(
          variables_to_values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
      Real d = left_eval_->calculate_derivative(
          variable, variables_to_values, mapFunctionTwoArgsValue,
          mapFunctionOneArgValue, mapFunctionDerivLeft, mapFunctionDerivRight);
      Real b = right_eval_->calculate(
          variables_to_values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
      Real c = right_eval_->calculate_derivative(
          variable, variables_to_values, mapFunctionTwoArgsValue,
          mapFunctionOneArgValue, mapFunctionDerivLeft, mapFunctionDerivRight);
      typename std::map<std::string, Real (*)(Real, Real)>::const_iterator
          itFunction_1;
      itFunction_1 = mapFunctionDerivLeft.find(id_);
      typename std::map<std::string, Real (*)(Real, Real)>::const_iterator
          itFunction_2;
      itFunction_2 = mapFunctionDerivRight.find(id_);
      if (itFunction_1 != mapFunctionDerivLeft.cend() &&
          itFunction_2 != mapFunctionDerivRight.cend()) {
        return itFunction_1->second(a, b) * d + itFunction_2->second(a, b) * c;
      }
    } else if (left_eval_) {
      // the same, but b === 0 and c === 0
      Real a = left_eval_->calculate(
          variables_to_values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
      Real d = left_eval_->calculate_derivative(
          variable, variables_to_values, mapFunctionTwoArgsValue,
          mapFunctionOneArgValue, mapFunctionDerivLeft, mapFunctionDerivRight);
      typename std::map<std::string, Real (*)(Real, Real)>::const_iterator
          itFunction;
      itFunction = mapFunctionDerivLeft.find(id_);
      if (itFunction != mapFunctionDerivLeft.cend()) {
        return itFunction->second(a, ZERO) * d;
      }
    }
    throw std::invalid_argument(
        (boost::format("The required function is not found \
during the calculation of the derivative, id: %s") %
         id_)
            .str());
  }
  throw std::runtime_error(
      (boost::format("Unknown error during the calculation of the derivative, \
id: %s, kind: %s") %
       id_ % kind_)
          .str());
}

template <typename Real>
Real cseval<Real>::calculate_derivative(
    const std::string &variable,
    const std::map<std::string, std::string> &variables_to_values,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
    const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft,
    const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight)
    const {
  typename std::map<std::string, Real> values;
  typename std::map<std::string, std::string>::const_iterator it;
  for (it = variables_to_values.cbegin(); it != variables_to_values.cend();
       ++it) {
    values[it->first] = Real(it->second);
  }
  return calculate_derivative(variable, values, mapFunctionTwoArgsValue,
                              mapFunctionOneArgValue, mapFunctionDerivLeft,
                              mapFunctionDerivRight);
}

template <typename Real>
std::unordered_map<char, size_t> cseval<Real>::operations_outside_parentheses(
    const std::string &str, const std::string &symbols) const {
  size_t countBraces = 0;

  std::unordered_map<char, size_t> op_to_ind;
  for (std::string::const_iterator cit = symbols.cbegin(); cit < symbols.cend();
       ++cit) {
    op_to_ind[*cit] = kNpos;
  }
  for (std::string::const_reverse_iterator crit = str.crbegin();
       crit != str.crend(); ++crit) {
    char op = *crit;
    if (op == ')') {
      countBraces++;
    } else if (op == '(') {
      countBraces--;
    }
    // We get only the last match in str, so we need the last condition.
    // For '-' & '/' we get the last match for correct ordering in e.g. "1-0-1".
    // "2^5/2^2/2^2" != 32
    if (countBraces == 0 && symbols.find(op) != std::string::npos &&
        op_to_ind.at(op) == kNpos) {
      bool is_number = false;
      if (crit != str.crend() && (op == '+' || op == '-')) {
        char prev = *(crit + 1);
        if (prev == 'e' || prev == 'E') {
          // Check that '-' or '+' is not the path of number.
          // e.g. support Formula("-002.e-0")
          for (auto index_check_number = crit + 2;
               index_check_number != str.crend(); ++index_check_number) {
            std::string check_symbol = std::string(1, *index_check_number);
            if (kOperationsWithParentheses.find(check_symbol) !=
                std::string::npos) {
              break;
            }
            if (kNumberSymbols.find(check_symbol) != std::string::npos) {
              is_number = true;
            } else if (is_number) {
              is_number = false;
              break;
            }
          }
        } else {
          auto _crit = crit;
          auto _prev = *(crit + 1);
          while (_crit != str.crend() && (_prev == '+' || _prev == '-') &&
                 op_to_ind.at(_prev) == kNpos) {
            _crit++;
            // 'op' mustn't be changed.
            if (*_crit == op) {
              // update position of 'op'. e.g Formula("0-+-+1")
              crit = _crit;
            }
            _prev = *(_crit + 1);
          }
        }
      }
      if (!is_number) {
        // It is not the path of the number.
        op_to_ind[op] = std::distance(crit, str.crend()) - 1;
      }
    }
  }
  return op_to_ind;
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
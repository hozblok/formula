#include "csformula.hpp"

csformula::csformula(const std::string &_expression, const unsigned _precision,
                     const char _imaginary_unit, const bool _case_insensitive)
    : origin_precision(0),
      precision(AllowedPrecisions::p_16),
      expression(""),
      imaginary_unit(_imaginary_unit),
      case_insensitive(_case_insensitive) {
  preparePrecision(_precision);
  prepareExpression(_expression);
  initEval();
}

csformula::csformula(const csformula &other)
    : origin_precision(other.origin_precision),
      precision(other.precision),
      expression(std::string(other.expression)),
      imaginary_unit(other.imaginary_unit),
      case_insensitive(other.case_insensitive) {
  auto visitor = std::bind(visitor_init_eval_from_copy(), &this->eval,
                           std::placeholders::_1);
  boost::apply_visitor(visitor, other.eval);
}

void csformula::setPrecision(const unsigned _precision) {
  preparePrecision(_precision);
  initEval();
}

void csformula::setExpression(const std::string &_expression) {
  prepareExpression(_expression);
  initEval();
}

bool csformula::validateBrackets(const std::string &str) {
  int count = 0;
  for (std::string::const_iterator it = str.cbegin(); it != str.cend(); ++it) {
    if (*it == '(') {
      count++;
    } else if (*it == ')') {
      count--;
    }
    if (count < 0) {
      return false;
    }
  }
  return (count == 0);
}

template <typename Real>
Real csformula::get(
    const std::map<std::string, Real> &mapVariableValues) const {
  visitor_real_value_getter<Real> visitor = visitor_real_value_getter<Real>();
  visitor.mapVariableValues = &mapVariableValues;
  return boost::apply_visitor(visitor, this->eval);
}

std::string csformula::get(
    const std::map<std::string, std::string> &mapVariableValues,
    std::streamsize digits, std::ios_base::fmtflags format) const {
  visitor_value_getter<std::string> visitor =
      visitor_value_getter<std::string>();
  visitor.mapVariableValues = &mapVariableValues;
  visitor.digits = digits;
  visitor.format = format;
  return boost::apply_visitor(visitor, this->eval);
}

std::string csformula::get(
    const std::map<std::string, double> &mapVariableValues,
    std::streamsize digits, std::ios_base::fmtflags format) const {
  visitor_value_getter<double> visitor = visitor_value_getter<double>();
  visitor.mapVariableValues = &mapVariableValues;
  visitor.digits = digits;
  visitor.format = format;
  return boost::apply_visitor(visitor, this->eval);
}

// template <typename Real>
// Real csformula<Real>::getD(const std::string variable, const
// std::map<std::string, Real> &mapVariableValues) const
// {
//     return eval->calculateDerivative(variable, mapVariableValues);
// }

std::string csformula::getD(
    const std::string variable,
    const std::map<std::string, std::string> &mapVariableValues) const {
  visitor_derivative_getter visitor = visitor_derivative_getter();
  visitor.variable = &variable;
  visitor.mapVariableValues = &mapVariableValues;
  return boost::apply_visitor(visitor, this->eval);
}

void csformula::preparePrecision(const unsigned precision) {
  AllowedPrecisions result = min_precision;
  for_each(precisions,
           [&precision, &result](const AllowedPrecisions &prec) -> bool {
             if (precision <= prec) {
               result = prec;
               return false;
             }
             return true;
           });
  this->origin_precision = precision;
  this->precision = result;
  if (this->precision == min_precision &&
      this->origin_precision > static_cast<unsigned>(min_precision)) {
    throw std::invalid_argument(
        (boost::format("The selected precision value %s exceeds the \
allowed maximum %s") %
         origin_precision % static_cast<unsigned>(max_precision))
            .str());
  }
}

void csformula::prepareExpression(const std::string &_expression) {
  if (_expression.empty()) {
    throw std::invalid_argument(
        "Cannot set the expression, \
the string is empty");
  }
  if (!validateBrackets(_expression)) {
    throw std::invalid_argument(
        "The given expression contains the wrong \
location and / or number of brackets");
  }
  expression = _expression;
  // Remove spaces, tabs, \n, \r, \t, \v.
  boost::algorithm::erase_all(expression, " ");
  boost::algorithm::erase_all(expression, "\n");
  boost::algorithm::erase_all(expression, "\r");
  boost::algorithm::erase_all(expression, "\t");
  boost::algorithm::erase_all(expression, "\v");

  if (case_insensitive) {
    boost::algorithm::to_lower(expression);
  }
}

void csformula::initEval() {
  switch (precision) {
    case AllowedPrecisions::p_16:
      _initEval<AllowedPrecisions::p_16>();
      break;
    case AllowedPrecisions::p_24:
      _initEval<AllowedPrecisions::p_24>();
      break;
    case AllowedPrecisions::p_32:
      _initEval<AllowedPrecisions::p_32>();
      break;
    case AllowedPrecisions::p_48:
      _initEval<AllowedPrecisions::p_48>();
      break;
    case AllowedPrecisions::p_64:
      _initEval<AllowedPrecisions::p_64>();
      break;
    case AllowedPrecisions::p_96:
      _initEval<AllowedPrecisions::p_96>();
      break;
    case AllowedPrecisions::p_128:
      _initEval<AllowedPrecisions::p_128>();
      break;
    case AllowedPrecisions::p_192:
      _initEval<AllowedPrecisions::p_192>();
      break;
    case AllowedPrecisions::p_256:
      _initEval<AllowedPrecisions::p_256>();
      break;
    case AllowedPrecisions::p_384:
      _initEval<AllowedPrecisions::p_384>();
      break;
    case AllowedPrecisions::p_512:
      _initEval<AllowedPrecisions::p_512>();
      break;
    case AllowedPrecisions::p_768:
      _initEval<AllowedPrecisions::p_768>();
      break;
    case AllowedPrecisions::p_1024:
      _initEval<AllowedPrecisions::p_1024>();
      break;
    case AllowedPrecisions::p_2048:
      _initEval<AllowedPrecisions::p_2048>();
      break;
    case AllowedPrecisions::p_3072:
      _initEval<AllowedPrecisions::p_3072>();
      break;
    case AllowedPrecisions::p_4096:
      _initEval<AllowedPrecisions::p_4096>();
      break;
    case AllowedPrecisions::p_6144:
      _initEval<AllowedPrecisions::p_6144>();
      break;
    case AllowedPrecisions::p_8192:
      _initEval<AllowedPrecisions::p_8192>();
      break;
  }
}

#include "csformula.hpp"

csformula::csformula(const std::string &_expression, const unsigned _precision,
                     const char _imaginary_unit, const bool _case_insensitive)
    : origin_precision(0),
      precision(AllowedPrecisions::p_16),
      expression(""),
      imaginary_unit(_imaginary_unit),
      case_insensitive(_case_insensitive) {
  setPrecision(_precision);
  setExpression(_expression);
}

void csformula::setPrecision(const unsigned _precision) {
  origin_precision = _precision;

  AllowedPrecisions last;
  for (const AllowedPrecisions &prec : precisions) {
    if (origin_precision <= prec) {
      precision = prec;
      break;
    }
    last = prec;
  }
  if (precision == AllowedPrecisions::p_16 &&
      origin_precision > AllowedPrecisions::p_16) {
    throw std::invalid_argument(
        (boost::format("The selected precision value %s exceeds the \
allowed maximum %s") %
         origin_precision % static_cast<unsigned>(last))
            .str());
  }
}

void csformula::setExpression(const std::string &_expression) {
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

  // TODO test memory leaks.
  // delete eval;
  switch (precision) {
    case AllowedPrecisions::p_16:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_16>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_24:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_24>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_32:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_32>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_48:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_48>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_64:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_64>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_96:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_96>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_128:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_128>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_192:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_192>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_256:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_256>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_384:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_384>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_512:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_512>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_768:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_768>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_1024:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_1024>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_2048:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_2048>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_3072:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_3072>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_4096:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_4096>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_6144:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_6144>>>(
          expression, imaginary_unit);
      break;
    case AllowedPrecisions::p_8192:
      eval = std::make_shared<cseval<mp_real<AllowedPrecisions::p_8192>>>(
          expression, imaginary_unit);
      break;
  }
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

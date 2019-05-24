#include "csformula.hpp"

Formula::Formula(const std::string &_expression, const unsigned _precision,
                     const char _imaginary_unit, const bool _case_insensitive)
    : origin_precision(0),
      precision(AllowedPrecisions::p_16),
      expression(""),
      imaginary_unit(_imaginary_unit),
      case_insensitive(_case_insensitive) {
  prepare_precision(_precision);
  prepare_expression(_expression);
  init_eval();
}

Formula::Formula(const Formula &other)
    : origin_precision(other.origin_precision),
      precision(other.precision),
      expression(std::string(other.expression)),
      imaginary_unit(other.imaginary_unit),
      case_insensitive(other.case_insensitive) {
  auto visitor = std::bind(InitEvalFromCopyVisitor(), &this->eval,
                           std::placeholders::_1);
  boost::apply_visitor(visitor, other.eval);
}

void Formula::set_precision(const unsigned _precision) {
  prepare_precision(_precision);
  init_eval();
}

void Formula::set_expression(const std::string &_expression) {
  prepare_expression(_expression);
  init_eval();
}

bool Formula::validate_brackets(const std::string &str) {
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
Real Formula::get(
    const std::map<std::string, Real> &mapVariableValues) const {
  GetCalculatedRealVisitor<Real> visitor = GetCalculatedRealVisitor<Real>();
  visitor.mapVariableValues = &mapVariableValues;
  return boost::apply_visitor(visitor, this->eval);
}

std::string Formula::get(
    const std::map<std::string, std::string> &mapVariableValues,
    std::streamsize digits, std::ios_base::fmtflags format) const {
  GetCalculatedStringVisitor<std::string> visitor =
      GetCalculatedStringVisitor<std::string>();
  visitor.mapVariableValues = &mapVariableValues;
  visitor.digits = digits;
  visitor.format = format;
  return boost::apply_visitor(visitor, this->eval);
}

std::string Formula::get(
    const std::map<std::string, double> &mapVariableValues,
    std::streamsize digits, std::ios_base::fmtflags format) const {
  GetCalculatedStringVisitor<double> visitor = GetCalculatedStringVisitor<double>();
  visitor.mapVariableValues = &mapVariableValues;
  visitor.digits = digits;
  visitor.format = format;
  return boost::apply_visitor(visitor, this->eval);
}

// template <typename Real>
// Real Formula<Real>::get_derivative(const std::string variable, const
// std::map<std::string, Real> &mapVariableValues) const
// {
//     return eval->calculateDerivative(variable, mapVariableValues);
// }

std::string Formula::get_derivative(
    const std::string variable,
    const std::map<std::string, std::string> &mapVariableValues) const {
  GetCalculatedDerivativeStringVisitor visitor = GetCalculatedDerivativeStringVisitor();
  visitor.variable = &variable;
  visitor.mapVariableValues = &mapVariableValues;
  return boost::apply_visitor(visitor, this->eval);
}

void Formula::prepare_precision(const unsigned &precision) {
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

void Formula::prepare_expression(const std::string &_expression) {
  if (_expression.empty()) {
    throw std::invalid_argument(
        "Cannot set the expression, \
the string is empty");
  }
  if (!validate_brackets(_expression)) {
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

#include "./csformula.hpp"

Formula::Formula(const std::string &expression, const unsigned precision,
                 const char imaginary_unit, const bool case_insensitive)
    : origin_precision_(0),
      precision_(AllowedPrecisions::p_16),
      expression_(""),
      imaginary_unit_(imaginary_unit),
      is_complex_(false),
      case_insensitive_(case_insensitive) {
#ifdef CSDEBUG
  std::cout << "constructor Formula +" << std::endl;
#endif
  prepare_precision(precision);
  prepare_expression(expression);
  init_eval();
#ifdef CSDEBUG
  std::cout << "constructor Formula -" << std::endl;
#endif
}

Formula::Formula(const Formula &other)
    : origin_precision_(other.origin_precision_),
      precision_(other.precision_),
      expression_(std::string(other.expression_)),
      imaginary_unit_(other.imaginary_unit_),
      is_complex_(other.is_complex_),
      case_insensitive_(other.case_insensitive_) {
#ifdef CSDEBUG
  std::cout << "copy constructor Formula+" << std::endl;
#endif
  if (is_complex_) {
    auto visitor_complex =
        std::bind(InitEvalFromCopyVisitor<CSEvalComplexVariant>(),
                  &eval_complex_, std::placeholders::_1);
    boost::apply_visitor(visitor_complex, other.eval_complex_);
  } else {
    auto visitor = std::bind(InitEvalFromCopyVisitor<CSEvalVariant>(), &eval_,
                             std::placeholders::_1);
    boost::apply_visitor(visitor, other.eval_);
  }
#ifdef CSDEBUG
  std::cout << "copy constructor Formula-" << std::endl;
#endif
}

void Formula::set_precision(const unsigned precision) {
  prepare_precision(precision);
  init_eval();
}

void Formula::set_expression(const std::string &expression) {
  prepare_expression(expression);
  init_eval();
}

bool Formula::validate_brackets(const std::string &str) const {
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

template <typename RealOrComplex>
RealOrComplex Formula::get(
    const std::map<std::string, RealOrComplex> &variables_to_values) const {
  GetCalculatedVisitor<RealOrComplex> visitor =
      GetCalculatedVisitor<RealOrComplex>();
  visitor.variables_to_values = &variables_to_values;
  if (is_complex_) {
    return boost::apply_visitor(visitor, eval_complex_);
  } else {
    return boost::apply_visitor(visitor, eval_);
  }
}

std::string Formula::get(
    const std::map<std::string, std::string> &variables_to_values,
    std::streamsize digits, std::ios_base::fmtflags format) const {
  auto visitor = GetCalculatedStringVisitor<std::string>();
  visitor.variables_to_values = &variables_to_values;
  visitor.digits = digits;
  visitor.format = format;
  visitor.is_complex = is_complex_;
  if (is_complex_) {
    return boost::apply_visitor(visitor, eval_complex_);
  } else {
    return boost::apply_visitor(visitor, eval_);
  }
}

std::string Formula::get(
    const std::map<std::string, double> &variables_to_values,
    std::streamsize digits, std::ios_base::fmtflags format) const {
  auto visitor = GetCalculatedStringVisitor<double>();
  visitor.variables_to_values = &variables_to_values;
  visitor.digits = digits;
  visitor.format = format;
  visitor.is_complex = is_complex_;
  if (is_complex_) {
    return boost::apply_visitor(visitor, eval_complex_);
  } else {
    return boost::apply_visitor(visitor, eval_);
  }
}

// TODO support Real and double and ... values.
// template <typename Real>
// Real Formula<Real>::get_derivative(const std::string variable, const
// std::map<std::string, Real> &variables_to_values) const
// {
//     return eval->calculate_derivative(variable, variables_to_values);
// }

std::string Formula::get_derivative(
    const std::string variable,
    const std::map<std::string, std::string> &variables_to_values,
    std::streamsize digits, std::ios_base::fmtflags format) const {
  auto visitor =
      std::bind(GetCalculatedDerivativeStringVisitor(), std::placeholders::_1,
                variable, variables_to_values, digits, format);
  if (is_complex_) {
    return boost::apply_visitor(visitor, eval_complex_);
  } else {
    return boost::apply_visitor(visitor, eval_);
  }
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
  origin_precision_ = precision;
  precision_ = result;
  if (precision_ == min_precision &&
      origin_precision_ > static_cast<unsigned>(min_precision)) {
    throw std::invalid_argument(
        (boost::format("The selected precision value %s exceeds the \
allowed maximum %s") %
         origin_precision_ % static_cast<unsigned>(max_precision))
            .str());
  }
}

void Formula::prepare_expression(const std::string &expression) {
  if (expression.empty()) {
    throw std::invalid_argument(
        "Cannot set the expression, \
the string is empty");
  }
  if (!validate_brackets(expression)) {
    throw std::invalid_argument(
        "The given expression contains the wrong \
location and / or number of brackets");
  }
  // TODO check for forbidden symbols for complex: < or >, <= or >=.
  expression_ = expression;
  // Remove spaces, tabs, \n, \r, \t, \v.
  boost::algorithm::erase_all(expression_, " ");
  boost::algorithm::erase_all(expression_, "\n");
  boost::algorithm::erase_all(expression_, "\r");
  boost::algorithm::erase_all(expression_, "\t");
  boost::algorithm::erase_all(expression_, "\v");

  // Check whether expression_ contains complex numbers or not.
  if (case_insensitive_) {
    std::regex regexp_check_complex_numbers(
        "\\b(" + std::string(1, imaginary_unit_) + ")\\b",
        std::regex_constants::icase);
    is_complex_ = std::regex_search(expression_, regexp_check_complex_numbers);
  } else {
    std::regex regexp_check_complex_numbers(
        "\\b(" + std::string(1, imaginary_unit_) + ")\\b");
    is_complex_ = std::regex_search(expression_, regexp_check_complex_numbers);
  }

  if (case_insensitive_) {
    // TODO cannot give back to user the original expression?
    boost::algorithm::to_lower(expression_);
    // TODO cannot give back to user the original expression?
    // boost::algorithm::to_lower(imaginary_unit_);
  }
}

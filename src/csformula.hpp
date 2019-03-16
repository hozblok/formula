#ifndef CSFORMULA_H
#define CSFORMULA_H
#include "cseval.cpp"

// TODO use namespace!
// {csformula} - wrapper for a {cseval} class that will parse a string
// {csformula} contains case insensitive string parser.
// Each variable can only be set as a single Latin character in a string
//
// to calculate the value of the derivative of a function, we use the following simple transformations:
// (u + v)' = u' + v' = 1 * d + 1 * c
// (u - v)' = u' - v' = 1 * d - 1 * c
// (u * v)' = v * u' + u * v'= b * d + a * c
// (u / v)' = (v * u' - u * v') / v^2 = (1 / v) * u' - (u / v^2) * v' = (1 / b) * d - (a / b^2) * c
// (u ^ v)' = (e^(v * ln(u)))' = e^(v * ln(u)) * (v * ln(u))'=v * u^(v - 1) * u' + (u^v) * ln(u) * v' =
// = b * a^(b - 1) * d + a^b * ln(a) * c

class csformula
{
private:
  /**
   * Precision that was selected by user.
   */
  unsigned origin_precision;
  /**
   * Current precision of the formula expression (should be >= 1).
   * 0 means that this precision has not calculated yet.
   */
  unsigned precision;
  /**
   * Expression to evaluate, for example "x|y" or "(x+1)*(y-0.004)*(sin(x))^2"
   * Equals to user-supplied excluding whitespace symbols (' ', '\n', '\t', '\v').
   */
  std::string expression;
  /**
   * Pointer to a recursive, smart formula string parser
   */
  boost::variant<
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_0>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_1>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_2>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_3>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_4>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_5>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_6>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_7>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_8>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_9>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_10>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_11>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_12>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_13>>>,
      std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_14>>>>
      eval;
  /**
   * Checks the order of parentheses in the string expression.
   */
  bool validateBrackets(const std::string &str);

public:
  csformula(const std::string &_expression, const unsigned _precision = 0);
  ~csformula() {}
  /**
   * Get current string formula expression.
   */
  std::string getExpression() const
  {
    return expression;
  }
  /**
   * Initialize the csformula instance by the string expression.
   */
  void setExpression(const std::string &_expression);
  /**
   * Get current precision.
   */
  unsigned getPrecision() const
  {
    return precision;
  }
  /**
   * Set new precision.
   */
  void setPrecision(const unsigned _precision = 0);

  // Get the calculated value of the formula in accordance with the dictionary {mapVariableValues}
  // contains values of the variables
  // NOTE: Variables can be only letters of the Latin alphabet ('a','b',...,'z')
  // and 'i' and 'j' reserved for complex number.
  // template< typename Real >
  // Real get(const std::map<std::string, Real> &mapVariableValues = {}) const;
  std::string get(const std::map<std::string, std::string> &mapVariableValues = {});
  // get the calculated value of partial derivative with respect to a variable {variable} and
  // the {mapVariableValues} dictionary contains values of the variables
  // Real getD(const std::string variable, const std::map<std::string, Real> &mapVariableValues = {}) const;
  std::string getD(const std::string variable, const std::map<std::string, std::string> &mapVariableValues = {}) const;
};

struct visitor_value_getter : public boost::static_visitor<std::string>
{
  template <typename T>
  std::string operator()(const T &eval) const
  {
    return eval->calculate(*mapVariableValues).str();
  }
  const std::map<std::string, std::string> *mapVariableValues;
};

struct visitor_derivative_getter : public boost::static_visitor<std::string>
{
  template <typename T>
  std::string operator()(const T &eval) const
  {
    return eval->calculateDerivative(*variable, *mapVariableValues).str();
  }
  const std::string *variable;
  const std::map<std::string, std::string> *mapVariableValues;
};

#endif // CSFORMULA_H

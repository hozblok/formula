#ifndef CSFORMULA_H
#define CSFORMULA_H
#include "cseval.cpp"

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
template< typename Real >
class csformula
{
private:
  // expression to evaluate, for example "(x+1)*(y-0.004)*(sin(x))^2"
  std::string expression;
  // pointer to a recursive, smart formula string parser
  cseval<Real> *eval;
  // simple checker the correct order of parentheses
  bool validateBrackets(const std::string &str);

public:
  csformula(const std::string &texpression);
  ~csformula();
  // Initialize formula by the expression.
  void setExpression(const std::string &texpression);
  // Get current expression.
  std::string getExpression() const { return expression; }

  // Get the calculated value of the formula in accordance with the dictionary {mapVariableValues}
  // contains values of the variables
  // NOTE: Variables can be only letters of the Latin alphabet ('a','b',...,'z')
  Real get(const std::map<std::string, Real> &mapVariableValues = {}) const;
  std::string get(const std::map<std::string, std::string> &mapVariableValues = {}) const;
  // get the calculated value of partial derivative with respect to a variable {variable} and
  // the {mapVariableValues} dictionary contains values of the variables
  // Real getD(const std::string variable, const std::map<std::string, Real> &mapVariableValues = {}) const;
  std::string getD(const std::string variable, const std::map<std::string, std::string> &mapVariableValues = {}) const;
};

#endif // CSFORMULA_H

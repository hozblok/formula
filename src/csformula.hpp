#ifndef CSFORMULA_H
#define CSFORMULA_H
#include "cseval.cpp"

// TODO use namespace!
// {csformula} - wrapper for a {cseval} class that will parse a string
// {csformula} contains case sensitive (by default) string parser.
// Each variable can only be set as a single Latin character in a string
//
// to calculate the value of the derivative of a function, we use the following
// simple transformations: (u + v)' = u' + v' = 1 * d + 1 * c (u - v)' = u' - v'
// = 1 * d - 1 * c (u * v)' = v * u' + u * v'= b * d + a * c (u / v)' = (v * u'
// - u * v') / v^2 = (1 / v) * u' - (u / v^2) * v' = (1 / b) * d - (a / b^2) * c
// (u ^ v)' = (e^(v * ln(u)))' = e^(v * ln(u)) * (v * ln(u))'=v * u^(v - 1) * u'
// + (u^v) * ln(u) * v' = = b * a^(b - 1) * d + a^b * ln(a) * c

class csformula {
 private:
  /**
   * Precision that was selected by user.
   */
  unsigned origin_precision;
  /**
   * Current precision of the formula expression (should be >= 1).
   * 0 means that this precision has not calculated yet.
   */
  AllowedPrecisions precision;
  /**
   * Expression to evaluate, for example "x|y" or "(x+1)*(y-0.004)*(sin(x))^2"
   * Equals to user-supplied excluding whitespace symbols (' ', '\n', '\t',
   * '\v').
   */
  std::string expression;
  /**
   * Symbol indicator of the imaginary unit. (dy default - 'i')
   */
  char imaginary_unit;
  /**
   * Whether parse expression in case insensitive style or not.
   */
  bool case_insensitive;
  /**
   * Pointer to a recursive, smart formula string parser
   */
  boost::variant<std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_16>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_24>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_32>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_48>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_64>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_96>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_128>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_192>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_256>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_384>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_512>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_768>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_1024>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_2048>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_3072>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_4096>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_6144>>>,
                 std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_8192>>>>
      eval;
  /**
   * Checks the order of parentheses in the string expression.
   */
  bool validateBrackets(const std::string &str);

 public:
  csformula(const std::string &_expression, const unsigned _precision = 0,
            const char _imaginary_unit = 'i',
            const bool _case_insensitive = false);
  ~csformula() {}
  /**
   * Get current string formula expression.
   */
  std::string getExpression() const { return expression; }
  /**
   * Initialize the csformula instance by the string expression.
   */
  void setExpression(const std::string &_expression);
  /**
   * Get current precision.
   */
  unsigned getPrecision() const { return precision; }
  /**
   * Set new precision.
   */
  void setPrecision(const unsigned _precision = 0);

  /**
   * Get the calculated value of the formula in accordance with the dictionary
   * {mapVariableValues} contains values of the variables.
   * NOTE: Variables can be
   * only letters of the Latin alphabet ('a','b',...,'z') and 'i' and 'j'
   * reserved for complex number. template< typename Real > Real get(const
   * std::map<std::string, Real> &mapVariableValues = {}) const;
   */
  template <typename Real>
  Real get(const std::map<std::string, Real> &mapVariableValues = {}) const;
  std::string get(
      const std::map<std::string, std::string> &mapVariableValues = {},
      std::streamsize digits = 0,
      std::ios_base::fmtflags format = std::ios_base::fmtflags(0)) const;
  std::string get(
      const std::map<std::string, double> &mapVariableValues = {},
      std::streamsize digits = 0,
      std::ios_base::fmtflags format = std::ios_base::fmtflags(0)) const;

  // get the calculated value of partial derivative with respect to a variable
  // {variable} and the {mapVariableValues} dictionary contains values of the
  // variables Real getD(const std::string variable, const std::map<std::string,
  // Real> &mapVariableValues = {}) const;
  std::string getD(
      const std::string variable,
      const std::map<std::string, std::string> &mapVariableValues = {}) const;
};

/**
 * Visitor to get string value after calculating the formula.
 * ArgType: string or double.
 */
template <typename ArgType>
struct visitor_value_getter : public boost::static_visitor<std::string> {
  template <typename T>
  std::string operator()(const T &eval) const {
    return eval->calculate(*mapVariableValues).str(digits, format);
  }
  const std::map<std::string, ArgType> *mapVariableValues;
  std::streamsize digits;
  std::ios_base::fmtflags format;
};

/**
 * Visitor to get Real value after calculating the formula.
 */
template <typename Real>
struct visitor_real_value_getter : public boost::static_visitor<Real> {
  template <typename T>
  Real operator()(const T &eval) const {
    return eval->calculate(*mapVariableValues);
  }
  const std::map<std::string, Real> *mapVariableValues;
};

struct visitor_derivative_getter : public boost::static_visitor<std::string> {
  template <typename T>
  std::string operator()(const T &eval) const {
    return eval->calculateDerivative(*variable, *mapVariableValues).str();
  }
  const std::string *variable;
  const std::map<std::string, std::string> *mapVariableValues;
};

#endif  // CSFORMULA_H
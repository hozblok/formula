#ifndef CSFORMULA_H
#define CSFORMULA_H
#include "cseval.cpp"

typedef boost::variant<
    std::shared_ptr<cseval<mp_real<AllowedPrecisions::p_16>>>,
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
    EvalVariantSmall;

/**
 * Visitor to get type of current Eval object and create the
 * copy of it.
 */
struct InitEvalFromCopyVisitor : public boost::static_visitor<void> {
  template <typename SHARED_PTR_CSEVAL_T>
  void operator()(EvalVariantSmall *eval,
                  const SHARED_PTR_CSEVAL_T &other_eval) {
    this->set_eval(eval, other_eval.get());
  }
  template <typename CSEVAL_T>
  void set_eval(EvalVariantSmall *eval, const CSEVAL_T *other_eval_raw) {
    *eval = std::make_shared<CSEVAL_T>(*other_eval_raw);
  }
};

/**
 * Visitor to get string value after calculating the formula.
 * ArgType: string or double.
 */
template <typename ArgType>
struct GetCalculatedStringVisitor : public boost::static_visitor<std::string> {
  template <typename T>
  std::string operator()(const T &eval) const {
    return eval->calculate(*variables_to_values).str(digits, format);
  }
  const std::map<std::string, ArgType> *variables_to_values;
  std::streamsize digits;
  std::ios_base::fmtflags format;
};

/** Visitor to get Real value after calculating the formula. */
template <typename Real>
struct GetCalculatedRealVisitor : public boost::static_visitor<Real> {
  template <typename T>
  Real operator()(const T &eval) const {
    return eval->calculate(*variables_to_values);
  }
  const std::map<std::string, Real> *variables_to_values;
};

/** Visitor to calculate partial derivative of the formula. */
struct GetCalculatedDerivativeStringVisitor
    : public boost::static_visitor<std::string> {
  template <typename T>
  std::string operator()(
      const T &eval, const std::string &variable,
      const std::map<std::string, std::string> &variables_to_values,
      std::streamsize &digits,
      std::ios_base::fmtflags &format) const {
    return eval->calculate_derivative(variable, variables_to_values).str();
  }
};

struct CollectVariablesVisitor : public boost::static_visitor<void> {
  template <typename T>
  void operator()(std::unordered_set<std::string> *variables, const T &eval) {
    eval->collect_variables(variables);
  }
};

//  * To calculate the value of the derivative of a function, we use
//  * the following simple transformations:
//  * a = u; b = v; c = v'; d = u'
//  * (u + v)' = u' + v' = 1 * d + 1 * c
//  * (u - v)' = u' - v' = 1 * d - 1 * c
//  * (u * v)' = v * u' + u * v'= b * d + a * c
//  * (u / v)' = (v * u' - u * v') / v^2
//  * = (1 / v) * u' - (u / v^2) * v' = (1 / b) * d - (a / b^2) * c
//  * (u ^ v)' = (e^(v * ln(u)))' = e^(v * ln(u)) * (v * ln(u))'
//  * = v * u^(v - 1) * u' + (u^v) * ln(u) * v'
//  * = b * a^(b - 1) * d + a^b * ln(a) * c

/**
 * Formula - wrapper for Eval class to parse and evaluate the
 * formula expressions with any required accuracy. The accuracy
 * (precision) of the calculations can be changed on the fly.
 * Formula contains case sensitive (by default) string parser.
 * Each variable can only be set as a string of any characters
 * except .*()[]{}~!@#$%\?,;`'\"|&=><+/^- characters.
 */
class Formula {
  /** User-selected calculation precision. */
  unsigned origin_precision_;

  /**
   * Current precision of the formula expression.
   * Valid value >= 0. 0 means that this precision has not
   * calculated yet. Selected from the available options
   * close to the user ones.
   */
  AllowedPrecisions precision_;

  /**
   * Expression to evaluate, e.g. "x|y" or
   * "(x+1)*(y-0.004)*(sin(x))^2"
   * May differ from user input in the following:
   * Cannot contain whitespace symbols (' ', '\n', '\t', '\v')
   * and mathes the case-sensitivity according to the
   * 'case_insensitive_' option.
   */
  std::string expression_;

  /** Symbol - indicator of the imaginary unit. (dy default - 'i') */
  char imaginary_unit_;

  /** Whether parse expression in case insensitive style or not. */
  bool case_insensitive_;

  /** Pointer to a recursive, smart formula string parser object. */
  EvalVariantSmall eval_;

  /** Initialize СSEval object. */
  template <std::size_t I = 0>
  inline typename std::enable_if<I == kPrecisionsLength>::type init_eval() {}
  template <std::size_t I = 0>
      inline typename std::enable_if < I<kPrecisionsLength>::type init_eval() {
    if (static_cast<unsigned>(precision_) == precisions_array[I]) {
      eval_ = std::make_shared<cseval<mp_real<precisions_array[I]>>>(
          expression_, imaginary_unit_);
    } else if (I + 1 < kPrecisionsLength) {
      init_eval<I + 1>();
    }
  }

  /** Validate, remove spaces, change case-sensitivity, remember expression. */
  void prepare_expression(const std::string &expression);

  /** Validate, select the appropiate precision, remember origin value. */
  void prepare_precision(const unsigned &precision);

  /** Checks the order of parentheses in the string expression. */
  bool validate_brackets(const std::string &str) const;

 public:
  Formula(const std::string &expression, const unsigned precision = 0,
          const char imaginary_unit = 'i', const bool case_insensitive = false);

  Formula(const Formula &other);

  ~Formula() {
#ifdef CSDEBUG
    std::cout << "destructor Formula" << std::endl;
#endif
  }

  /** Get current string formula expression. */
  std::string get_expression() const { return expression_; }

  /** Initialize the Formula instance by the string expression. */
  void set_expression(const std::string &expression);

  /** Get current precision. */
  unsigned get_precision() const { return static_cast<unsigned>(precision_); }

  /** Set new precision. */
  void set_precision(const unsigned precision = 0);

  /** Create copy of the Formula object. */
  Formula *copy() { return new Formula(*this); }

  /** Parse all variables from the formula expression. */
  std::unordered_set<std::string> variables() const {
    std::unordered_set<std::string> variables;
    auto visitor = std::bind(CollectVariablesVisitor(), &variables, std::placeholders::_1);
    boost::apply_visitor(visitor, eval_);
    return variables;
  }

  /**
   * Get the calculated value of the formula in accordance with
   * the 'map_variable_values' dictionary contains values of
   * the variables.
   * NOTE: Variables can be only letters of the Latin alphabet
   * ('a','b',...,'z') and 'i' (by default) reserved for complex
   * number.
   */
  template <typename Real>
  Real get(const std::map<std::string, Real> &variables_to_values = {}) const;

  std::string get(
      const std::map<std::string, std::string> &variables_to_values = {},
      std::streamsize digits = 0,
      std::ios_base::fmtflags format = std::ios_base::fmtflags(0)) const;

  std::string get(
      const std::map<std::string, double> &variables_to_values = {},
      std::streamsize digits = 0,
      std::ios_base::fmtflags format = std::ios_base::fmtflags(0)) const;

  // Get the calculated value of partial derivative with respect to a
  // variable {variable} and the {variables_to_values} dictionary contains
  // values of the variables Real get_derivative(const std::string variable,
  // const std::map<std::string, Real> &variables_to_values = {}) const;
  std::string get_derivative(
      const std::string variable,
      const std::map<std::string, std::string> &variables_to_values = {},
      std::streamsize digits = 0,
      std::ios_base::fmtflags format = std::ios_base::fmtflags(0)) const;
};

#endif  // CSFORMULA_H
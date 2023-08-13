#ifndef CSVISITORS_H
#define CSVISITORS_H

#include <map>
#include <string>
#include <unordered_set>

/**
 * Visitor to get string value after calculating the formula.
 * ARG_T: string or double.
 */
template <typename ARG_T>
struct GetCalculatedStringVisitor : public boost::static_visitor<std::string> {
  template <typename CSEVAL_T>
  std::string operator()(const CSEVAL_T &eval_any) const {
    if (is_complex) {
      auto value = eval_any->calculate(*variables_to_values);
      std::string real_part = value.real().str(digits, format);
      std::string imag_part = value.imag().str(digits, format);
      return real_part + std::string("+i*(") + imag_part + std::string(")");
    } else {
      return eval_any->calculate(*variables_to_values).str(digits, format);
    }
  }
  const std::map<std::string, ARG_T> *variables_to_values;
  std::streamsize digits;
  std::ios_base::fmtflags format;
  bool is_complex;
};

/**
 * Visitor to get type of current Eval object and create the copy of it.
 */
template <typename VARIANT_T>
struct InitEvalFromCopyVisitor : public boost::static_visitor<void> {
  template <typename SHARED_PTR_CSEVAL_T>
  void operator()(VARIANT_T *eval, const SHARED_PTR_CSEVAL_T &other_eval) {
    this->set_eval(eval, other_eval.get());
  }
  template <typename CSEVAL_T>
  void set_eval(VARIANT_T *eval, const CSEVAL_T *other_eval_raw) {
    *eval = std::make_shared<CSEVAL_T>(*other_eval_raw);
  }
};

/** Visitor to get Real or Complex value after calculating the formula. */
template <typename RealOrComplex>
struct GetCalculatedVisitor : public boost::static_visitor<RealOrComplex> {
  template <typename T>
  RealOrComplex operator()(const T &eval) const {
    return eval->calculate(*variables_to_values);
  }
  const std::map<std::string, RealOrComplex> *variables_to_values;
};

/** Visitor to calculate partial derivative of the formula. */
struct GetCalculatedDerivativeStringVisitor
    : public boost::static_visitor<std::string> {
  template <typename T>
  std::string operator()(
      const T &eval, const std::string &variable,
      const std::map<std::string, std::string> &variables_to_values,
      std::streamsize &digits, std::ios_base::fmtflags &format) const {
    return eval->calculate_derivative(variable, variables_to_values).str();
  }
};

struct CollectVariablesVisitor : public boost::static_visitor<void> {
  template <typename T>
  void operator()(std::unordered_set<std::string> *variables, const T &eval) {
    eval->collect_variables(variables);
  }
};

#endif  // CSVISITORS_H
#ifndef CSVISITORS_H
#define CSVISITORS_H

#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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
      return real_part + "+" + std::string(1, imaginary_unit) + "*(" +
             imag_part + ")";
    } else {
      return eval_any->calculate(*variables_to_values).str(digits, format);
    }
  }
  const std::map<std::string, ARG_T> *variables_to_values;
  std::streamsize digits;
  std::ios_base::fmtflags format;
  bool is_complex;
  char imaginary_unit;
};

/**
 * Visitor to get (real, imag) parts as a pair of strings. For real-only
 * expressions the imaginary part is zero formatted with the same digits/format
 * as the real part, so the resulting pair has consistent string shape and can
 * be compared byte-for-byte against the pair of a complex expression whose
 * imaginary part computes to exactly zero.
 */
template <typename ARG_T>
struct GetCalculatedPairVisitor
    : public boost::static_visitor<std::pair<std::string, std::string>> {
  template <typename CSEVAL_T>
  std::pair<std::string, std::string> operator()(
      const CSEVAL_T &eval_any) const {
    auto value = eval_any->calculate(*variables_to_values);
    if (is_complex) {
      return {value.real().str(digits, format),
              value.imag().str(digits, format)};
    }
    decltype(value) zero(0);
    return {value.str(digits, format), zero.str(digits, format)};
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

/** Visitor to calculate first or mixed partial derivatives. Same output
 * shape as GetCalculatedStringVisitor. */
struct GetCalculatedDerivativeStringVisitor
    : public boost::static_visitor<std::string> {
  template <typename T>
  std::string operator()(const T &eval) const {
    auto value = variable
                     ? eval->calculate_derivative(*variable, *variables_to_values)
                     : eval->calculate_partial_derivative(*variables,
                                                          *variables_to_values);
    if (is_complex) {
      std::string real_part = value.real().str(digits, format);
      std::string imag_part = value.imag().str(digits, format);
      return real_part + "+" + std::string(1, imaginary_unit) + "*(" +
             imag_part + ")";
    }
    return value.str(digits, format);
  }
  const std::string *variable = nullptr;
  const std::vector<std::string> *variables = nullptr;
  const std::map<std::string, std::string> *variables_to_values;
  std::streamsize digits;
  std::ios_base::fmtflags format;
  bool is_complex;
  char imaginary_unit;
};

struct GetCalculatedJetStringVisitor
    : public boost::static_visitor<
          std::pair<std::vector<std::string>, uint32_t>> {
  template <typename T>
  std::pair<std::vector<std::string>, uint32_t> operator()(
      const T &eval) const {
    using Value = decltype(eval->calculate(std::map<std::string, std::string>()));
    std::map<std::string, csjet<Value>> jets;
    for (const auto &item : *variables_to_coefficients) {
      csjet<Value> jet(order, Value(0));
      jet.support = supports->at(item.first);
      for (unsigned m = 0; m < jet.c.size(); ++m) {
        const auto &text = item.second[m];
        if constexpr (is_complex_number<Value>) {
          const std::string marker = "+" + std::string(1, imaginary_unit) + "*(";
          auto split = text.find(marker);
          if (split != std::string::npos && text.back() == ')') {
            auto real = text.substr(0, split);
            auto imag = text.substr(split + marker.size(),
                                    text.size() - split - marker.size() - 1);
            validate_variable_value(item.first, real);
            validate_variable_value(item.first, imag);
            jet.c[m] = Value(real, imag);
          } else {
            validate_variable_value(item.first, text);
            jet.c[m] = Value(text, "0");
          }
        } else {
          validate_variable_value(item.first, text);
          jet.c[m] = Value(text);
        }
        const auto &value = jet.c[m];
        if (isinf(value.real()) || isnan(value.real()) ||
            isinf(value.imag()) || isnan(value.imag())) {
          throw std::invalid_argument("Jet coefficients must be finite");
        }
        if (!jet.has(m) && value != Value(0)) {
          throw std::invalid_argument("A nonzero jet coefficient needs structural support");
        }
      }
      jets.emplace(item.first, std::move(jet));
    }
    auto result = eval->calculate_jet(order, jets);
    std::vector<std::string> values;
    values.reserve(result.c.size());
    for (const auto &value : result.c) {
      if constexpr (is_complex_number<Value>) {
        values.push_back(value.real().str() + "+" + imaginary_unit + "*(" +
                         value.imag().str() + ")");
      } else {
        values.push_back(value.str());
      }
    }
    return {std::move(values), result.support};
  }
  unsigned order;
  const std::map<std::string, std::vector<std::string>> *variables_to_coefficients;
  const std::map<std::string, uint32_t> *supports;
  char imaginary_unit;
};

struct CollectVariablesVisitor : public boost::static_visitor<void> {
  template <typename T>
  void operator()(std::unordered_set<std::string> *variables, const T &eval) {
    eval->collect_variables(variables);
  }
};

#endif  // CSVISITORS_H

#include "csformula.h"

csformula::csformula(const std::string &texpression, const unsigned tprecision) : precision(tprecision),
                                                                                expression("")
{
    setExpression(texpression, tprecision);
}

csformula::~csformula()
{
    // delete eval;
}

// constexpr size_t degrees_of_two[9] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
constexpr const unsigned resolve_precision(const unsigned prec) {
  for (const unsigned *ptr=precisions; ptr<=&precisions[15]; ptr++) {
    if (prec <= *ptr) {
      return *ptr;
    }
  }
  return 256;
}

const unsigned testt(const unsigned prec) {
    if (0 <= prec) {
        return allowed_precisions::power_of_two_4;
    }
    return allowed_precisions::power_of_two_1;
}

void csformula::setExpression(const std::string &texpression, const unsigned tprecision)
{
    if (texpression.empty())
    {
        std::cerr << "ERROR: Empty expression\n";
        throw EMPTY_EXPRESSION;
    }
    if (!validateBrackets(texpression))
    {
        std::cerr << "ERROR: wrong brackets in expression\n";
        throw WRONG_BRACKETS;
    }
    expression = texpression;
    // TODO -> tests.
    // boost::algorithm::to_lower(expression);

    // Remove spaces, tabs, \n, \r, \t, \v.
    boost::algorithm::erase_all(expression, " ");
    boost::algorithm::erase_all(expression, "\n");
    boost::algorithm::erase_all(expression, "\r");
    boost::algorithm::erase_all(expression, "\t");
    boost::algorithm::erase_all(expression, "\v");

    // delete eval;
    //cseval<mp_real<prepare_precision(precision)>> * test = new cseval<mp_real<prepare_precision(precision)>>(expression);
    // const size_t prec = precision;
    // constexpr size_t degrees_of_two[9] = {1,2,4,8,16,32,64,128,256};
    // const size_t *ptr=degrees_of_two;
    // const qw a = A;
    // const unsigned prec = allowed_precisions::power_of_two_4;
    // const unsigned prec = testt(tprecision);
    if (tprecision <= 1<<1) {
        precision = allowed_precisions::power_of_two_1;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_1>>>(expression);
    } else if (tprecision <= 1<<2) {
        precision = allowed_precisions::power_of_two_2;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_2>>>(expression);
    } else if (tprecision <= 1<<3) {
        precision = allowed_precisions::power_of_two_3;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_3>>>(expression);
    } else if (tprecision <= 1<<4) {
        precision = allowed_precisions::power_of_two_4;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_4>>>(expression);
    } else if (tprecision <= 1<<5) {
        precision = allowed_precisions::power_of_two_5;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_5>>>(expression);
    } else if (tprecision <= 1<<6) {
        precision = allowed_precisions::power_of_two_6;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_6>>>(expression);
    } else if (tprecision <= 1<<7) {
        precision = allowed_precisions::power_of_two_7;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_7>>>(expression);
    } else if (tprecision <= 1<<8) {
        precision = allowed_precisions::power_of_two_8;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_8>>>(expression);
    } else if (tprecision <= 1<<9) {
        precision = allowed_precisions::power_of_two_9;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_9>>>(expression);
    } else if (tprecision <= 1<<10) {
        precision = allowed_precisions::power_of_two_10;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_10>>>(expression);
    } else if (tprecision <= 1<<11) {
        precision = allowed_precisions::power_of_two_11;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_11>>>(expression);
    } else if (tprecision <= 1<<12) {
        precision = allowed_precisions::power_of_two_12;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_12>>>(expression);
    } else if (tprecision <= 1<<13) {
        precision = allowed_precisions::power_of_two_13;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_13>>>(expression);
    } else if (tprecision <= 1<<14) {
        precision = allowed_precisions::power_of_two_14;
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_14>>>(expression);
    }
    // if (0 <= tprecison) {
    //     const unsigned prec = allowed_precisions::power_of_two_4;//resolve_precision(tprecision);
    // }
    // std::cout << "TODO: resolved prec:" << prec << "\n";
    
    // for (int i = 0; i < 9; ++i) {
    //     if (tprecision <= degrees_of_two[i]) {
    //         eval = std::make_shared<cseval<mp_real<precisions[i]>>>(expression);
    //     }
    // }
    // eval = std::make_shared<cseval<mp_real<prec>>>(expression);
}

bool csformula::validateBrackets(const std::string &str)
{
    int count = 0;
    for (std::string::const_iterator it = str.cbegin(); it != str.cend(); ++it)
    {
        if (*it == '(')
        {
            count++;
        }
        else if (*it == ')')
        {
            count--;
        }
        if (count < 0)
        {
            return false;
        }
    }
    return (count == 0);
}

// template <typename Real>
// Real csformula<Real>::get(const std::map<std::string, Real> &mapVariableValues) const
// {
//     return eval->calculate(mapVariableValues);
// }

std::string csformula::get(const std::map<std::string, std::string> &mapVariableValues) const
{
    if (precision <= 1<<1) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_1>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<2) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_2>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<3) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_3>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<4) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_4>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<5) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_5>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<6) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_6>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<7) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_7>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<8) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_8>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<9) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_9>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<10) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_10>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<11) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_11>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<12) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_12>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<13) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_13>>>>(eval)->calculate(mapVariableValues).str();
    } else if (precision <= 1<<14) {
        return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_14>>>>(eval)->calculate(mapVariableValues).str();
    }
    // TODO delete me.
    // return std::string("asdff");
    // const unsigned prec = allowed_precisions::power_of_two_4;
    // return boost::get<std::shared_ptr<cseval<mp_real<prec>>>>(eval)->calculate(mapVariableValues).str();
}

// template <typename Real>
// Real csformula<Real>::getD(const std::string variable, const std::map<std::string, Real> &mapVariableValues) const
// {
//     return eval->calculateDerivative(variable, mapVariableValues);
// }

std::string csformula::getD(const std::string variable, const std::map<std::string, std::string> &mapVariableValues) const
{
    // TODO delete me.
    return std::string("asdff");
    // const unsigned prec = allowed_precisions::power_of_two_4;
    // return boost::get<std::shared_ptr<cseval<mp_real<prec>>>>(eval)->calculateDerivative(variable, mapVariableValues).str();
}

#include "csformula.hpp"

csformula::csformula(const std::string &_expression, const unsigned _precision) : origin_precision(0),
                                                                                  precision(0),
                                                                                  expression("")
{
    setPrecision(_precision);
    setExpression(_expression);
}

void csformula::setPrecision(const unsigned _precision)
{
    precision = 0;
    origin_precision = _precision;
    for (const unsigned &prec : precisions)
    {
        if (_precision <= prec)
        {
            precision = prec;
            break;
        }
    }
    if (precision == 0)
    {
        std::cerr << "Cannot use '" << _precision << "' precision. This is too much value!\n";
    }
    std::cout << "TODO: resolved prec:" << precision << "\n";
}

void csformula::setExpression(const std::string &_expression)
{
    if (_expression.empty())
    {
        std::cerr << "ERROR: Empty expression\n";
        throw EMPTY_EXPRESSION;
    }
    if (!validateBrackets(_expression))
    {
        std::cerr << "ERROR: wrong brackets in expression\n";
        throw WRONG_BRACKETS;
    }
    expression = _expression;
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

    if (precision == precisions[0])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_0>>>(expression);
    }
    else if (precision == precisions[1])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_1>>>(expression);
    }
    else if (precision == precisions[2])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_2>>>(expression);
    }
    else if (precision == precisions[3])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_3>>>(expression);
    }
    else if (precision == precisions[4])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_4>>>(expression);
    }
    else if (precision == precisions[5])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_5>>>(expression);
    }
    else if (precision == precisions[6])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_6>>>(expression);
    }
    else if (precision == precisions[7])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_7>>>(expression);
    }
    else if (precision == precisions[8])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_8>>>(expression);
    }
    else if (precision == precisions[9])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_9>>>(expression);
    }
    else if (precision == precisions[10])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_10>>>(expression);
    }
    else if (precision == precisions[11])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_11>>>(expression);
    }
    else if (precision == precisions[12])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_12>>>(expression);
    }
    else if (precision == precisions[13])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_13>>>(expression);
    }
    else if (precision == precisions[14])
    {
        eval = std::make_shared<cseval<mp_real<allowed_precisions::power_of_two_14>>>(expression);
    }

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

std::string csformula::get(const std::map<std::string, std::string> &mapVariableValues)
{
    visitor_value_getter visitor = visitor_value_getter();
    visitor.mapVariableValues = &mapVariableValues;
    return boost::apply_visitor(visitor, eval);
    // if (precision <= 1 << 1)
    // {
    // return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_1>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 2)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_2>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 3)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_3>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 4)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_4>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 5)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_5>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 6)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_6>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 7)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_7>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 8)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_8>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 9)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_9>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 10)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_10>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 11)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_11>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 12)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_12>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 13)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_13>>>>(eval)->calculate(mapVariableValues).str();
    // }
    // else if (precision <= 1 << 14)
    // {
    //     return boost::get<std::shared_ptr<cseval<mp_real<allowed_precisions::power_of_two_14>>>>(eval)->calculate(mapVariableValues).str();
    // }
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
    visitor_derivative_getter visitor = visitor_derivative_getter();
    visitor.variable = &variable;
    visitor.mapVariableValues = &mapVariableValues;
    return boost::apply_visitor(visitor, eval);
    // const unsigned prec = allowed_precisions::power_of_two_4;
    // return boost::get<std::shared_ptr<cseval<mp_real<prec>>>>(eval)->calculateDerivative(variable, mapVariableValues).str();
}

#include "csformula.h"

csformula::csformula(const std::string &texpression) : expression(""),
                                                       eval_2_0(0),
                                                       eval_2_1(0),
                                                       eval_2_2(0),
                                                       eval_2_3(0),
                                                       eval_2_4(0)
{
    setExpression(texpression);
}

csformula::~csformula()
{
    delete eval_2_1;
    eval_2_1 = 0;
}

void csformula::setExpression(const std::string &texpression)
{
    expression = texpression;
    if (expression.empty())
    {
        std::cerr << "ERROR: Empty expression\n";
        throw EMPTY_EXPRESSION;
    }
    if (!validateBrackets(expression))
    {
        std::cerr << "ERROR: wrong brackets in expression\n";
        throw WRONG_BRACKETS;
    }
    boost::algorithm::to_lower(expression);

    // remove spaces, tabs, \n, \r, \t, \v
    boost::algorithm::erase_all(expression, " ");
    boost::algorithm::erase_all(expression, "\n");
    boost::algorithm::erase_all(expression, "\r");
    boost::algorithm::erase_all(expression, "\t");
    boost::algorithm::erase_all(expression, "\v");

    delete eval_2_1;
    eval_2_1 = new cseval<mp_real<2>>(expression);
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
    return eval_2_1->calculate(mapVariableValues).str();
}

// template <typename Real>
// Real csformula<Real>::getD(const std::string variable, const std::map<std::string, Real> &mapVariableValues) const
// {
//     return eval->calculateDerivative(variable, mapVariableValues);
// }

std::string csformula::getD(const std::string variable, const std::map<std::string, std::string> &mapVariableValues) const
{
    return eval_2_1->calculateDerivative(variable, mapVariableValues).str();
}

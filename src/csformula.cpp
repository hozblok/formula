#include "csformula.h"

template <typename Real>
csformula<Real>::csformula(const std::string &texpression) : expression(""),
                                                       eval(0)
{
    setExpression(texpression);
}

template <typename Real>
csformula<Real>::~csformula()
{
    delete eval;
    eval = 0;
}

template <typename Real>
void csformula<Real>::setExpression(const std::string &texpression)
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

    delete eval;
    eval = new cseval<float100et>(expression);
}

template <typename Real>
bool csformula<Real>::validateBrackets(const std::string &str)
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
// float100et csformula<Real>::get(const std::map<std::string, float100et> &mapVariableValues) const
// {
//     return eval->calculate(mapVariableValues, functionsTwoArgs, functionsOneArg);
// }

template <typename Real>
std::string csformula<Real>::get(const std::map<std::string, std::string> &mapVariableValues) const
{
    return eval->calculate(mapVariableValues).str();
}

// template <typename Real>
// float100et csformula<Real>::getD(const std::string variable, const std::map<std::string, float100et> &mapVariableValues) const
// {
//     return eval->calculateDerivative(variable,
//                                      mapVariableValues,
//                                      functionsTwoArgs,
//                                      functionsOneArg,
//                                      functionsTwoArgsDLeft,
//                                      functionsTwoArgsDRight);
// }

template <typename Real>
std::string csformula<Real>::getD(const std::string variable, const std::map<std::string, std::string> &mapVariableValues) const
{
    return eval->calculateDerivative(variable, mapVariableValues).str();
}

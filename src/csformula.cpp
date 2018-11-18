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
    eval = new cseval<Real>(expression);
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

template <typename Real>
Real csformula<Real>::get(const std::map<std::string, Real> &mapVariableValues) const
{
    return eval->calculate(mapVariableValues);
}

template <typename Real>
std::string csformula<Real>::get(const std::map<std::string, std::string> &mapVariableValues) const
{
    return eval->calculate(mapVariableValues).str();
}

// template <typename Real>
// Real csformula<Real>::getD(const std::string variable, const std::map<std::string, Real> &mapVariableValues) const
// {
//     return eval->calculateDerivative(variable, mapVariableValues);
// }

template <typename Real>
std::string csformula<Real>::getD(const std::string variable, const std::map<std::string, std::string> &mapVariableValues) const
{
    return eval->calculateDerivative(variable, mapVariableValues).str();
}

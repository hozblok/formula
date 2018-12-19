#include "csformula.h"

csformula::csformula(const std::string &texpression, const size_t tprecision) :
    precision(tprecision),
    expression(""),
    eval(nullptr)
{
    setExpression(texpression);
}

csformula::~csformula()
{
    // delete eval;
}

constexpr size_t degrees_of_two[9] = {1,2,4,8,16,32,64,128,256};
// const size_t prepare_precision(size_t prec) {
//   for (const size_t *ptr=degrees_of_two; ptr<=&degrees_of_two[9]; ptr++) {
//     if (prec <= *ptr) {
//       return *ptr;
//     }
//   }
//   return 256;
// }


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

    // delete eval;
    //cseval<mp_real<prepare_precision(precision)>> * test = new cseval<mp_real<prepare_precision(precision)>>(expression);
    // const size_t prec = precision;
    // constexpr size_t degrees_of_two[9] = {1,2,4,8,16,32,64,128,256};
    // const size_t *ptr=degrees_of_two;
    // const qw a = A;
    eval = std::make_shared<cseval<mp_real<1>>>(expression);
    // for (int i = 0; i < 9; ++i) {
    //     if (precision <= degrees_of_two[i]) {
    //         eval = cseval<mp_real<a>>(expression);
    //     }
    // }
    
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
    return std::string("asdff");
    // return eval->calculate(mapVariableValues).str();
}

// template <typename Real>
// Real csformula<Real>::getD(const std::string variable, const std::map<std::string, Real> &mapVariableValues) const
// {
//     return eval->calculateDerivative(variable, mapVariableValues);
// }

std::string csformula::getD(const std::string variable, const std::map<std::string, std::string> &mapVariableValues) const
{
    return std::string("asdff");
    // return eval->calculateDerivative(variable, mapVariableValues).str();
}

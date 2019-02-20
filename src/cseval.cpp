#include "cseval.h"

template <typename Real>
const Real cseval<Real>::ZERO = Real("0");

template <typename Real>
const Real cseval<Real>::ONE = Real("1");

template <typename Real>
const std::map<std::string, Real (*)(Real, Real)> cseval<Real>::functionsTwoArgs = {{std::string("|"), _or},
                                                                                    {std::string("&"), _and},
                                                                                    {std::string("="), _eq},
                                                                                    {std::string(">"), _gt},
                                                                                    {std::string("<"), _lt},
                                                                                    {std::string("+"), _add},
                                                                                    {std::string("-"), _sub},
                                                                                    {std::string("/"), _truediv},
                                                                                    {std::string("*"), _mul},
                                                                                    {std::string("^"), _pow}};
template <typename Real>
const std::map<std::string, Real (*)(Real)> cseval<Real>::functionsOneArg = {{std::string("sin"), _sin},
                                                                             {std::string("asin"), _asin},
                                                                             {std::string("cos"), _cos},
                                                                             {std::string("acos"), _acos},
                                                                             {std::string("tan"), _tan},
                                                                             {std::string("atan"), _atan},
                                                                             {std::string("log"), _log},
                                                                             {std::string("sqrt"), _sqrt},
                                                                             {std::string("exp"), _exp}};
template <typename Real>
const std::map<std::string, Real (*)(Real, Real)> cseval<Real>::functionsTwoArgsDLeft = {{std::string("+"), _one},
                                                                                         {std::string("-"), _one},
                                                                                         {std::string("*"), _mul1},
                                                                                         {std::string("/"), _truediv1},
                                                                                         {std::string("^"), _pow1},
                                                                                         {std::string("sin"), _sin_d},
                                                                                         {std::string("asin"), _asin_d},
                                                                                         {std::string("cos"), _cos_d},
                                                                                         {std::string("acos"), _acos_d},
                                                                                         {std::string("tan"), _tan_d},
                                                                                         {std::string("atan"), _atan_d},
                                                                                         {std::string("log"), _log_d},
                                                                                         {std::string("sqrt"), _sqrt_d},
                                                                                         {std::string("exp"), _exp_d}};
template <typename Real>
const std::map<std::string, Real (*)(Real, Real)> cseval<Real>::functionsTwoArgsDRight = {{std::string("+"), _one},
                                                                                          {std::string("-"), _m_one},
                                                                                          {std::string("*"), _mul2},
                                                                                          {std::string("/"), _truediv2},
                                                                                          {std::string("^"), _pow2}};

template <typename Real>
cseval<Real>::cseval(std::string expression) : kind('e'),
                                               id(""),
                                               value(0),
                                               leftEval(0),
                                               rightEval(0)
{
    if (expression.empty())
    {
        std::cerr << "ERROR: Empty expression! Check string, brackets and values within brackets!\n";
        throw EMPTY_EXPRESSION;
    }

    // remove braces
    while (!isThereSymbolsOutsideParentheses(expression))
    {
        expression.erase(expression.cbegin());
        expression.pop_back();
    }

    // find operations: logical or, logical and, relational =, <, >, addition, subtraction, multiplication, division, the construction of the power
    // First, we are looking for addition and subtraction,
    // because division and multiplication have a higher priority
    // and should be performed first.
    std::string operations("|&=><+-*/^");
    std::unordered_map<char, int> foundedOperation = findSymbolsOutsideBrackets(expression, operations);
    for (std::string::const_iterator it = operations.cbegin(); it != operations.cend(); ++it)
    {
        if (foundedOperation.at(*it) != -1)
        {
            kind = 'f';
            id = *it;
            if (*it == '-' || *it == '+')
            {
                // if '-' represents negative number or negative value of variable, not a subtraction operation
                // similarly for '+'
                if (foundedOperation.at(*it) == 0)
                {
                    // allowed '-x', '--x', '---x*y', '-.01' etc.
                    leftEval = new cseval(std::string("0"));
                    rightEval = new cseval(expression.substr(1));
                    return;
                }
                else if (operations.find(expression.at(foundedOperation.at(*it) - 1)) != std::string::npos)
                {
                    // allowed 'x/-1' or 'y^-x' etc.
                    // go on to the next operation
                    continue;
                }
            }
            // split the string into two parts
            // separator: + - * / ^
            leftEval = new cseval(expression.substr(0, foundedOperation.at(*it)));
            rightEval = new cseval(expression.substr(foundedOperation.at(*it) + 1));
            return;
        }
    }

    size_t len = expression.size();
    size_t iLeftBracket = expression.find('(');
    if (iLeftBracket == len - 1)
    {
        std::cerr << "ERROR: cannot parse expression, wrong brackets, last symbol is '('.\n";
        throw WRONG_BRACKETS;
    }
    if (iLeftBracket == std::string::npos)
    {
        // there is no '(' => kind === 'v' or 'n'; or id === "pi" //variable or number or pi
        if (expression.find("pi") == 0)
        {
            kind = 'n';
            id = "pi";
            value = Real(M_PI_STR);
        }
        else if (len == 1 && expression.at(0) >= 'a' && expression.at(0) <= 'z')
        {
            // x or y or z or other available variables
            kind = 'v';
            id = expression;
        }
        else
        {
            // this is number
            kind = 'n';
            id = expression;
            try
            {
                value = Real(expression.data());
            }
            catch (...)
            {
                std::cerr << "ERROR: unknown number value: \'" << expression << "\'\n";
                throw IMPOSIBLE_VALUE_CONVERSION;
            }
        }
    }
    else if (iLeftBracket < len)
    { // this is a function:
        std::string name_fun = expression.substr(0, iLeftBracket);
        expression = expression.substr(iLeftBracket + 1);
        expression.pop_back();
        kind = 'f';
        id = name_fun;
        size_t iComma = expression.find(',');
        if (iComma != std::string::npos)
        {
            leftEval = new cseval(expression.substr(0, iComma));
            rightEval = new cseval(expression.substr(iComma + 1));
        }
        else
        {
            leftEval = new cseval(expression);
        }
    }
}

template <typename Real>
cseval<Real>::~cseval()
{
    if (leftEval != 0)
    {
        delete leftEval;
        leftEval = 0;
    }
    if (rightEval != 0)
    {
        delete rightEval;
        rightEval = 0;
    }
}

template <typename Real>
Real cseval<Real>::calculate(const std::map<std::string, Real> &mapVariableValues,
                             const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
                             const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue) const
{
    if (kind == 'n')
    {
        return value;
    }
    else if (kind == 'v')
    {
        typename std::map<std::string, Real>::const_iterator it;
        for (it = mapVariableValues.cbegin(); it != mapVariableValues.cend(); ++it)
        {
            if (it->first == id)
            {
                return it->second;
            }
        }
    }
    else if (kind == 'f')
    {
        if (leftEval != 0 && rightEval != 0)
        {
            // function with two arguments
            Real left("0");
            Real right("0");
            left = leftEval->calculate(mapVariableValues, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
            right = rightEval->calculate(mapVariableValues, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
            typename std::map<std::string, Real (*)(Real, Real)>::const_iterator itFunction;
            itFunction = mapFunctionTwoArgsValue.find(id);
            if (itFunction != mapFunctionTwoArgsValue.cend())
            {
                return itFunction->second(left, right);
            }
            else
            {
                std::cerr << "ERROR: Unknown variable or function \'" << id << "\'\n";
                throw UNKNOWN_VARIABLE_OR_FUNCTION;
            }
        }
        else if (leftEval != 0)
        {
            // function with one argument
            Real left("0");
            left = leftEval->calculate(mapVariableValues, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
            typename std::map<std::string, Real (*)(Real)>::const_iterator itFunction;
            itFunction = mapFunctionOneArgValue.find(id);
            if (itFunction != mapFunctionOneArgValue.cend())
            {
                return itFunction->second(left);
            }
            else
            {
                std::cerr << "ERROR: Unknown variable or function \'" << id << "\'\n";
                throw UNKNOWN_VARIABLE_OR_FUNCTION;
            }
        }
    }
    std::cerr << "ERROR: Unknown error, unknown variable or function \'" << id << "\' kind'" << kind << "'\n";
    throw UNKNOWN_CALCULATE_ERROR;
}

template <typename Real>
Real cseval<Real>::calculate(const std::map<std::string, std::string> &mapVariableValues,
                             const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
                             const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue) const
{
    typename std::map<std::string, Real> values;
    typename std::map<std::string, std::string>::const_iterator it;
    for (it = mapVariableValues.cbegin(); it != mapVariableValues.cend(); ++it)
    {
        values[it->first] = Real(it->second);
    }
    return calculate(values, mapFunctionTwoArgsValue, mapFunctionOneArgValue);
}

template <typename Real>
Real cseval<Real>::calculateDerivative(const std::string &variable,
                                       const std::map<std::string, Real> &mapVariableValues,
                                       const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
                                       const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue,
                                       const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft,
                                       const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight) const
{
    if (kind == 'n')
    {
        return ZERO;
    }
    else if (kind == 'v')
    {
        if (id == variable)
        {
            return ONE;
        }
        else
        {
            return ZERO;
        }
    }
    else if (kind == 'f')
    {
        if (leftEval != 0 && rightEval != 0)
        {
            // (u+v)'=u'+v'=1*d+1*c
            // (u-v)'=u'-v'=1*d-1*c
            // (u*v)'=vu'+uv'=b*d+a*c
            // (u/v)'=(vu'-uv')/v^2=(1/v)*u'-(u/v^2)v'=(1/b)*d-(a/b^2)*c
            // (u^v)'=(e^(v*ln(u)))'=e^(v*ln(u)) * (v*ln(u))'=v*u^(v-1)*u' + (u^v)*ln(u)*v'=b*a^(b-1)*d + a^b*ln(a)*c
            // a===u, b===v, d===u', c===v'
            Real a = leftEval->calculate(mapVariableValues,
                                         mapFunctionTwoArgsValue,
                                         mapFunctionOneArgValue);
            Real d = leftEval->calculateDerivative(variable,
                                                   mapVariableValues,
                                                   mapFunctionTwoArgsValue,
                                                   mapFunctionOneArgValue,
                                                   mapFunctionDerivLeft,
                                                   mapFunctionDerivRight);
            Real b = rightEval->calculate(mapVariableValues,
                                          mapFunctionTwoArgsValue,
                                          mapFunctionOneArgValue);
            Real c = rightEval->calculateDerivative(variable,
                                                    mapVariableValues,
                                                    mapFunctionTwoArgsValue,
                                                    mapFunctionOneArgValue,
                                                    mapFunctionDerivLeft,
                                                    mapFunctionDerivRight);
            typename std::map<std::string, Real (*)(Real, Real)>::const_iterator itFunction_1;
            itFunction_1 = mapFunctionDerivLeft.find(id);
            typename std::map<std::string, Real (*)(Real, Real)>::const_iterator itFunction_2;
            itFunction_2 = mapFunctionDerivRight.find(id);
            if (itFunction_1 != mapFunctionDerivLeft.cend() && itFunction_2 != mapFunctionDerivRight.cend())
            {
                return itFunction_1->second(a, b) * d + itFunction_2->second(a, b) * c;
            }
            else
            {
                std::cerr << "ERROR: Unknown variable or function for derivative operation \'" << id << "\'\n";
                throw UNKNOWN_VARIABLE_OR_FUNCTION_D;
            }
        }
        else if (leftEval != 0)
        {
            // the same, but b === 0 and c === 0
            Real a = leftEval->calculate(mapVariableValues,
                                         mapFunctionTwoArgsValue,
                                         mapFunctionOneArgValue);
            Real d = leftEval->calculateDerivative(variable,
                                                   mapVariableValues,
                                                   mapFunctionTwoArgsValue,
                                                   mapFunctionOneArgValue,
                                                   mapFunctionDerivLeft,
                                                   mapFunctionDerivRight);
            typename std::map<std::string, Real (*)(Real, Real)>::const_iterator itFunction;
            itFunction = mapFunctionDerivLeft.find(id);
            if (itFunction != mapFunctionDerivLeft.cend())
            {
                return itFunction->second(a, ZERO) * d;
            }
            else
            {
                std::cerr << "ERROR: Unknown variable or function for derivative operation \'" << id << "\'\n";
                throw UNKNOWN_VARIABLE_OR_FUNCTION_D;
            }
        }
    }
    std::cerr << "ERROR: Unknown error, unknown variable or function for derivative operation \'" << id << "\' kind'" << kind << "'\n";
    throw UNKNOWN_CALCULATE_D_ERROR;
}

template <typename Real>
Real cseval<Real>::calculateDerivative(const std::string &variable,
                                       const std::map<std::string, std::string> &mapVariableValues,
                                       const std::map<std::string, Real (*)(Real, Real)> &mapFunctionTwoArgsValue,
                                       const std::map<std::string, Real (*)(Real)> &mapFunctionOneArgValue,
                                       const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivLeft,
                                       const std::map<std::string, Real (*)(Real, Real)> &mapFunctionDerivRight) const
{
    typename std::map<std::string, Real> values;
    typename std::map<std::string, std::string>::const_iterator it;
    for (it = mapVariableValues.cbegin(); it != mapVariableValues.cend(); ++it)
    {
        values[it->first] = Real(it->second);
    }
    return calculateDerivative(variable, values, mapFunctionTwoArgsValue, mapFunctionOneArgValue, mapFunctionDerivLeft, mapFunctionDerivRight);
}

template <typename Real>
std::unordered_map<char, int> cseval<Real>::findSymbolsOutsideBrackets(const std::string &str, const std::string &symbols) const
{
    unsigned int countBraces = 0;

    std::unordered_map<char, int> operationIndex;
    for (std::string::const_iterator it = symbols.cbegin(); it < symbols.cend(); ++it)
    {
        operationIndex[*it] = -1;
    }
    for (std::string::const_iterator it = str.cbegin(); it != str.cend(); ++it)
    {
        if (*it == '(')
        {
            countBraces++;
        }
        else if (*it == ')')
        {
            countBraces--;
        }
        // we get only the first match in str, so we need the last condition
        if (countBraces == 0 && symbols.find(*it) != std::string::npos && operationIndex.at(*it) == -1)
        {
            operationIndex[*it] = it - str.cbegin();
        }
    }
    return operationIndex;
}

template <typename Real>
bool cseval<Real>::isThereSymbolsOutsideParentheses(const std::string &str) const
{
    unsigned int countBraces = 0;
    std::string::const_iterator it = str.cbegin();
    if (*it != '(')
    {
        return true;
    }
    else
    {
        countBraces++;
        it++;
    }

    for (; it != str.cend(); ++it)
    {
        if (countBraces == 0)
        {
            if (*it == '(')
            {
                // '(x+1)(y+1)'? Wrong expression. There is no mathematical operation between brackets.
                throw WRONG_BRACKETS;
            }
            return true;
        }
        if (*it == '(')
        {
            countBraces++;
        }
        else if (*it == ')')
        {
            countBraces--;
        }
    }
    return false;
}
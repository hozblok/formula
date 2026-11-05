"""First derivative through both mechanisms: get_derivative and get_partial_derivative."""

import pytest

from formula import FmtFlags


def first_derivative(formula, variable, values=None, digits=0, format=FmtFlags.default):
    """formula.get_derivative; get_partial_derivative([variable]) must give the same."""
    values = {} if values is None else values
    try:
        result = formula.get_derivative(variable, values, digits, format)
    except Exception as ex:
        with pytest.raises(type(ex)):
            formula.get_partial_derivative([variable], values, digits, format)
        raise
    assert formula.get_partial_derivative([variable], values, digits, format) == result
    return result

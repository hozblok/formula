"""Regression: clear error message when values are missing or the wrong type.

Before, calling solver() (values=None) on a multi-variable formula raised
ValueError with "The value of the 'values' parameter is not a dict! Its
type is <class 'NoneType'>" — misleading, because the actual problem is
that the caller didn't pass any values at all.
"""

import pytest

from formula import Solver


def test_missing_values_dict_for_multi_variable_formula():
    solver = Solver("x + y")
    with pytest.raises(ValueError, match="Missing values for variables"):
        solver()


def test_wrong_type_for_values_on_multi_variable_formula():
    solver = Solver("x + y")
    with pytest.raises(ValueError, match="Expected a Mapping"):
        solver(123)


def test_wrong_type_error_names_the_actual_type():
    solver = Solver("x + y")
    with pytest.raises(ValueError, match="int"):
        solver(42)


def test_proper_values_still_work():
    solver = Solver("x + y")
    assert solver({"x": "1", "y": "2"}) == "3"


def test_scalar_values_for_no_variable_formula():
    # Used to crash with "'int' object is not iterable".
    solver = Solver("1 + 1")
    with pytest.raises(ValueError, match="no variables"):
        solver(5)


def test_none_values_for_no_variable_formula_still_work():
    assert Solver("1 + 1")() == "2"

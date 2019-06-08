from formula import formula, fmtflags
import pytest

def test_evaluation():
    """Evaluate several simple expressions."""
    assert True

def test_changing_precision():
    """Change precision on the fly."""
    assert True

    f = formula("2*asin(x)", 16)
    assert "3.1415926535897932384626433832795" == f.get({"x": "1"})
    f.set_precision(240)
    assert "3.14" == f.get({"x": "1"})

def test_changing_expression():
    """Change expression and get value."""

    f = formula("2*asin(x)", 16)
    assert "3.1415926535897932384626433832795" == f.get({"x": "1"})
    assert "2*asin(x)" == f.expression
    f.set_expression("asin(x)")
    assert "1.57079632679489661923132169163975" == f.get({"x": "1"})
    f.expression = "sin(x)"
    assert "0.8414709848078965066609974948209608755664" == f.get({"x": "1"})

@pytest.mark.xfail(reason="under development")
def test_formula_object_copy():
    """Compare formula objects."""

    f = formula("2*asin(x)", 16)
    f2 = formula("2*asin(x)", 24)
    assert f2 == f  

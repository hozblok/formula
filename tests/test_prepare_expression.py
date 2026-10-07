"""Regression: prepare_expression rejects ordering ops on complex; original
input is preserved alongside the parser-normalized form.

Closes TODOs:
  src/cpp/csformula/csformula.cpp:185  (// TODO check for forbidden symbols
                                         for complex: < or >, <= or >=.)
  src/cpp/csformula/csformula.cpp:195  (// TODO cannot give back to user
                                         the original expression?)

The forbidden-symbol check landed at some earlier point but the TODO
above it was never removed; these tests pin the behavior so the comment
deletion is anchored. The original-expression preservation is new: a
parallel `expression_original_` field is stashed before whitespace
stripping / case lowering, and exposed via `get_original_expression()`
(plus the `original_expression` property).
"""

import pytest

from formula import Formula


# --- comparison-on-complex rejection (TODO #3) ---

@pytest.mark.parametrize(
    "expr",
    ["x+i<y", "x+i>y", "x+i<=y", "x+i>=y", "x<y+i", "1+i*2>0"],
)
def test_complex_expression_rejects_comparison(expr):
    with pytest.raises(ValueError, match="complex numbers contains wrong"):
        Formula(expr)


def test_real_expression_allows_lt():
    # No 'i' → real path, '<' is a valid binary operator.
    f = Formula("x<y")
    assert f.get({"x": "1", "y": "2"}) == "1"


# --- original-expression preservation (TODO #4) ---

def test_get_original_expression_preserves_case():
    f = Formula("X + Y", case_insensitive=True)
    # parser-normalized form is lowercased and whitespace-stripped
    assert f.get_expression() == "x+y"
    # original input survives the lowering
    assert f.get_original_expression() == "X + Y"


def test_get_original_expression_preserves_whitespace():
    f = Formula("  x  +  y  ")
    assert f.get_expression() == "x+y"
    assert f.get_original_expression() == "  x  +  y  "


def test_original_expression_property_accessor():
    f = Formula("FoO + bAr", case_insensitive=True)
    assert f.original_expression == "FoO + bAr"


def test_get_original_expression_for_case_sensitive_equals_get_expression():
    # In the case-sensitive path with no whitespace, the two should match.
    f = Formula("x+y")
    assert f.get_expression() == "x+y"
    assert f.get_original_expression() == "x+y"


def test_get_original_expression_updates_on_set_expression():
    f = Formula("X + Y", case_insensitive=True)
    f.set_expression("NEW + EXPR")
    assert f.get_expression() == "new+expr"
    assert f.get_original_expression() == "NEW + EXPR"

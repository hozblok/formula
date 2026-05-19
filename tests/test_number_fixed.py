"""Tests for Number.fixed property.

`fixed` is the canonical fixed-point string at the configured precision.
It backs __hash__ and __prepare_comparison, so the property must:
  1. evaluate the expression (not return the raw text)
  2. collapse equivalent syntactic forms to one string
  3. never use scientific notation (otherwise "1e-5" and "0.00001" would
     compare unequal as strings even though they're the same value)
  4. extend its digit count with higher precision
"""

from formula import Number


def test_fixed_returns_str():
    assert isinstance(Number("1").fixed, str)


def test_fixed_is_evaluated_not_raw_expression():
    n = Number("2+3")
    assert n.expression == "2+3"
    assert n.fixed.startswith("5")


def test_equivalent_expressions_collapse_to_same_fixed():
    assert Number("2+3").fixed == Number("5").fixed
    assert Number("1/2").fixed == Number("0.5").fixed
    assert Number("2^3").fixed == Number("8").fixed
    assert Number("(1+2)*3").fixed == Number("9").fixed


def test_different_values_have_different_fixed():
    assert Number("1").fixed != Number("2").fixed
    assert Number("0.1").fixed != Number("0.2").fixed


def test_fixed_no_exponent_for_small_value():
    # 1e-8 would normally render in scientific notation; fixed must spell
    # it out so string equality matches value equality.
    assert "e" not in Number("1e-8").fixed.lower()


def test_fixed_no_exponent_for_large_value():
    assert "e" not in Number("1e22").fixed.lower()


def test_higher_precision_yields_more_fractional_digits():
    low = Number("1/3", precision=4).fixed
    high = Number("1/3", precision=48).fixed
    low_fractional = low.split(".", 1)[1] if "." in low else ""
    high_fractional = high.split(".", 1)[1] if "." in high else ""
    assert len(high_fractional) > len(low_fractional)


def test_fixed_is_deterministic_across_instances():
    assert Number("1/3").fixed == Number("1/3").fixed


def test_fixed_is_idempotent_on_same_instance():
    n = Number("(1+2)/(3+4)")
    assert n.fixed == n.fixed


def test_fixed_agrees_across_constructor_input_forms():
    assert (
        Number(7).fixed
        == Number("7").fixed
        == Number(7.0).fixed
        == Number(Number("7")).fixed
    )


def test_fixed_keeps_negative_sign():
    assert Number("-2.5").fixed.startswith("-")


def test_fixed_zero_has_no_sign():
    assert not Number("0").fixed.startswith("-")

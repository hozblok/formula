"""Shared formula surfaces preserve their evaluator and normalize jet inputs."""

import pytest

from formula import Formula, Number, Solver
from formula.capsysred.shared.implicit_surface import Surface
from formula.capsysred.shared.jets import _Algebra, _Jet


@pytest.mark.parametrize("formula_type", [Formula, Solver])
def test_existing_formula_is_shared_and_expression_changes_reach_both_paths(formula_type):
    formula = formula_type("x*y+z", precision=64)
    surface = Surface(formula, precision=128, coordinate_scale=Number("2", 64))
    assert surface.formula is formula
    assert surface.precision == formula.precision == 64
    alg = _Algebra(2, 64)
    scalar_point = (1, 2, 3)
    jet_point = (alg.variable("1", (0, 0), 0), 2, 3)

    assert float(surface.value(scalar_point)) == 14
    assert tuple(map(float, surface.gradient(scalar_point))) == (8, 4, 2)
    assert tuple(map(float, surface.value(jet_point).c)) == (14, 8, 8, 0)
    assert tuple(g.value for g in surface.gradient(jet_point)) == (8, 4, 2)

    formula.expression = "x^2+3*y-2*z"
    assert surface.expression == formula.expression
    assert float(surface.value(scalar_point)) == 4
    assert tuple(map(float, surface.gradient(scalar_point))) == (8, 6, -4)
    assert tuple(map(float, surface.value(jet_point).c)) == (4, 8, 8, 8)
    gradient = surface.gradient(jet_point)
    assert tuple(g.value for g in gradient) == (8, 6, -4)
    assert tuple(float(g.c[1]) for g in gradient) == (8, 0, 0)


@pytest.mark.parametrize("axis, first, second, gradient_derivative", [
    (1, 11, 4, (3, 4, 1)), (2, 20, 6, (2, 1, 6)),
])
def test_jet_in_later_coordinate_lifts_scalar_and_number_coordinates(axis, first, second, gradient_derivative):
    alg = _Algebra(2, 64)
    surface = Surface("x^2+2*y^2+3*z^2+x*y*z")
    point = [Number("1", 64), 2.0, 3]
    point[axis] = alg.variable(point[axis], (0, 0), 0)

    value = surface.value(point)
    assert value.alg is alg
    assert tuple(map(float, value.c)) == (42, first, first, second)
    gradient = surface.gradient(point)
    assert all(component.alg is alg for component in gradient)
    assert tuple(component.value for component in gradient) == (8, 11, 20)
    assert tuple(float(component.c[1]) for component in gradient) == gradient_derivative


def test_value_and_gradient_reject_different_algebras_even_for_an_unused_axis():
    first, second = _Algebra(2, 64), _Algebra(2, 64)
    surface = Surface("x")
    point = (first.constant(1), second.constant(2), 3)
    for evaluate in (surface.value, surface.gradient):
        with pytest.raises(ValueError, match="same algebra"):
            evaluate(point)


def test_value_and_gradient_reject_a_different_jet_precision():
    surface = Surface("z^2", precision=64)
    alg = _Algebra(2, 128)
    point = (1, 2, alg.variable(3, (0, 0), 0))
    for evaluate in (surface.value, surface.gradient):
        with pytest.raises(ValueError, match="precision must agree"):
            evaluate(point)


def test_rounded_jet_precision_matches_the_formula_precision():
    surface = Surface("y^2", precision=64)
    alg = _Algebra(2, 50)
    point = (1, alg.variable(3, (0, 0), 0), 4)
    value = surface.value(point)
    assert value.alg is alg
    assert tuple(map(float, value.c)) == (9, 6, 6, 2)
    gradient = surface.gradient(point)
    assert tuple(component.value for component in gradient) == (0, 6, 0)
    assert tuple(map(float, gradient[1].c)) == (6, 2, 2, 0)


def test_ray_jets_reexports_the_shared_classes():
    from formula.capsysred.stages import _ray_jets

    assert _ray_jets.Surface is Surface
    assert _ray_jets._Algebra is _Algebra
    assert _ray_jets._Jet is _Jet


@pytest.mark.parametrize("existing_formula", [False, True])
def test_coordinate_scale_must_match_effective_surface_precision(existing_formula):
    expression = Formula("x", precision=64) if existing_formula else "x"
    with pytest.raises(ValueError, match="coordinate_scale precision"):
        Surface(expression, precision=128 if existing_formula else 64,
                coordinate_scale=Number("2", 128))

"""Shared boundaries retain unequal complex fields and actual phase anchors."""

from copy import deepcopy
from decimal import Decimal as D, localcontext
import math

import pytest

from formula import Number
from formula.capsysred.stages._s19_domain import domain_indicator


def pair(amplitudes=((1, 0), (1, 0)), *, folded=False, offset="0"):
    with localcontext() as context:
        context.prec = 64
        shift = D(offset)
        points = [(D(0), D(0)), (D(1), D(0)), (D(0), D(1)), (D(1), D(1))]
        exits = list(points)
        if folded:
            exits[3] = (D(".2"), D(".2"))
        result = []
        for index, face in enumerate(((0, 1, 2), (1, 3, 2))):
            entrance = [[str(points[i][j]+shift) for j in range(2)] for i in face]
            output = [[str(exits[i][j]+shift) for j in range(2)] for i in face]
            center = [str(sum(D(p[j]) for p in output)/3) for j in range(2)]
            result.append(dict(input=entrance, exit=output, center_exit=center,
                               direction=["0", "0", "1"], opl="0", amplitude=list(map(str, amplitudes[index])),
                               bore=0, reflections=2, maslov=0, orientation=-1 if folded and index else 1))
        return result


def midpoint_callback(records, *, bow=(".6", ".6"), offset="0"):
    mapping = {}
    with localcontext() as context:
        context.prec = 64
        for row in records:
            for i in range(3):
                j = (i+1) % 3
                q = tuple((D(row["input"][i][n])+D(row["input"][j][n]))/2 for n in range(2))
                x = tuple((D(row["exit"][i][n])+D(row["exit"][j][n]))/2 for n in range(2))
                mapping[(row["bore"], q)] = x
        shift = D(offset)
        mapping[(0, (D(".5")+shift,)*2)] = tuple(D(value)+shift for value in bow)
    return lambda bore, q: mapping[(bore, q)]


def test_matching_fields_remove_only_the_internal_boundary_term():
    records = pair(((1, ".4"), (1, ".4")))
    result = domain_indicator(records, midpoint_callback(records), k="1", precision=64)
    assert result["passed"]
    assert D(result["weighted_domain_indicator"]) == 0
    assert D(result["independent_triangle_amplitude_relative"]) == D(".4")
    assert result["counts"]["internal_same_branch"] == 1
    assert result["p6_fan_checks"]["simple"] == 2


def test_unequal_complex_amplitudes_leave_the_exact_jump_integral():
    records = pair(((1, ".3"), (2, "-.4")))
    result = domain_indicator(records, midpoint_callback(records), k="1", precision=64, tolerance="1")
    # The only curved edge has ear area 0.1; safety=2 and a constant complex jump.
    expected = .2*abs((1+.3j)-(2-.4j))
    assert float(result["weighted_domain_indicator"]) == pytest.approx(expected, rel=1e-14)
    assert result["passed"]


def test_opposite_output_orientations_use_sum_of_amplitudes():
    records = pair(((1, ".3"), (2, "-.4")), folded=True)
    result = domain_indicator(records, midpoint_callback(records), k="1", precision=64, tolerance="10")
    expected = .2*(abs(1+.3j)+abs(2-.4j))
    assert float(result["weighted_domain_indicator"]) == pytest.approx(expected, rel=1e-14)
    assert result["counts"]["nonmatching_branch"] == 1
    assert result["exit_vertex_disagreements"] == 0
    assert result["passed"]


def test_unresolved_neighbor_keeps_full_amplitude_term():
    records = pair(((1, ".3"), (2, "-.4")))
    result = domain_indicator(records[:1], midpoint_callback(records), k="1", precision=64,
                              unresolved=[records[1]], tolerance="1")
    assert float(result["weighted_domain_indicator"]) == pytest.approx(.2*abs(1+.3j))
    assert result["counts"]["unresolved_neighbor"] == 1


@pytest.mark.parametrize("offset", ["0", "1000000.125"])
def test_full_opl_and_distinct_anchors_preserve_one_physical_field(offset):
    records = pair(((1, ".4"), (1, ".4")), offset=offset)
    with localcontext() as context:
        context.prec = 64
        turn = D(str(Number("2*pi", 64)))
        common = D("12345678901234567890123456789.123456789")
        for index, row in enumerate(records):
            row["direction"] = [".3", "-.2", "1"]
            row["opl"] = str(common+D(".3")*D(row["center_exit"][0])
                             - D(".2")*D(row["center_exit"][1])+index*turn/D(10))
    result = domain_indicator(records, midpoint_callback(records, offset=offset), k="10", precision=64)
    assert result["passed"]
    assert abs(D(result["weighted_domain_indicator"])) < D("1e-30")


def test_unequal_phase_gradients_bound_every_sample_on_the_ear():
    records = pair(((1, ".3"), (2, "-.4")))
    records[0].update(direction=[".7", "-.2", "1"], opl=".24")
    records[1].update(direction=["-.1", ".4", "1"], opl="-.19")
    result = domain_indicator(records, midpoint_callback(records), k="3", precision=64, tolerance="10")
    bound = float(result["weighted_domain_indicator"])/.2
    points = [(1., 0.), (.6, .6), (0., 1.)]
    for i in range(21):
        for j in range(21-i):
            weights = (i/20, j/20, 1-(i+j)/20)
            x = [sum(w*p[n] for w, p in zip(weights, points)) for n in range(2)]
            fields = []
            for row in records:
                phase = 3*(float(row["opl"])+sum(float(row["direction"][n])
                          * (x[n]-float(row["center_exit"][n])) for n in range(2)))
                fields.append(complex(*map(float, row["amplitude"]))*complex(math.cos(phase), math.sin(phase)))
            assert abs(fields[0]-fields[1]) <= bound+1e-13


def test_p6_or_fan_failure_rejects_even_when_equal_field_jump_is_zero():
    records = pair()
    result = domain_indicator(records, midpoint_callback(records, bow=("-1", "-1")),
                              k="1", precision=64, tolerance="1000")
    assert D(result["weighted_domain_indicator"]) == 0
    assert not result["passed"]
    assert result["p6_fan_checks"]["non_simple"] or result["p6_fan_checks"]["bad_fan_orientation"]


def test_disagreeing_shared_vertex_is_explicitly_rejected():
    records = pair()
    records[1]["exit"][0][0] = "1.01"
    result = domain_indicator(records, midpoint_callback(records), k="1", precision=64, tolerance="1000")
    assert not result["passed"]
    assert result["exit_vertex_disagreements"] == 1


def test_domain_tolerance_is_an_enforced_gate():
    records = pair(((1, ".3"), (2, "-.4")))
    result = domain_indicator(records, midpoint_callback(records), k="1", precision=64, tolerance=".001")
    assert not result["passed"]
    assert D(result["weighted_domain_relative"]) > D(result["tolerance"])

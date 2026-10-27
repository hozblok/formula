"""Amplitude calculators share the caller's numeric context."""

import pytest

from formula import NumberContext, xray
from formula.capsysred import Simulation
from formula.capsysred.fresnel import FresnelAmplitude, FloatLineAmplitudes
from formula.capsysred.spectrum import spectral_lines
from formula.capsysred.symbolic import LineAmplitudes


@pytest.mark.parametrize("precision", [30, 64])
def test_amplitudes_share_context_and_agree(precision):
    N = NumberContext(precision)
    energy = N(8)
    lines = spectral_lines({"mode": "monochromatic"}, energy)
    single = FresnelAmplitude(xray.FUSED_SILICA, energy, N)
    exact = LineAmplitudes(xray.FUSED_SILICA, lines, N)
    floating = FloatLineAmplitudes(xray.FUSED_SILICA, lines, N)
    assert single.numbers is exact.numbers is floating.numbers is N
    sin_theta = N("0.001")
    expected = single(sin_theta)
    actual = exact([sin_theta])[0]
    assert actual.precision == expected.precision == N.precision
    assert float(abs(actual - expected)) < 1e-25
    assert floating([float(sin_theta)])[0] == pytest.approx(complex(expected), abs=1e-12)
    assert single.product([]).precision == exact([])[0].precision == N.precision
    assert single.product([]) == exact([])[0] == N(1)
    assert floating([]) == [1 + 0j]


@pytest.mark.parametrize("calculator", [FresnelAmplitude, LineAmplitudes, FloatLineAmplitudes])
def test_context_rejects_energy_with_different_precision(calculator):
    energy = NumberContext(256)(8)
    values = energy if calculator is FresnelAmplitude else spectral_lines(
        {"mode": "monochromatic"}, energy
    )
    with pytest.raises(ValueError, match="precision mismatch"):
        calculator(xray.FUSED_SILICA, values, NumberContext(64))


def test_simulations_own_independent_contexts():
    low = Simulation.from_dict({"precision": 30})
    high = Simulation.from_dict({"precision": 64})
    for sim in (low, high):
        assert sim.fresnel.numbers is sim.line_amps.numbers is sim.numbers
        assert sim.fresnel.product([]).precision == sim.numbers.precision
    assert low.numbers.precision != high.numbers.precision

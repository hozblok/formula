"""Independent checks of the reduced 2D unfolded B5 operator."""

import pytest

np = pytest.importorskip("numpy")

from formula.capsysred.stages._b5_slab import UnfoldedSlab


def test_straight_slab_single_dirichlet_mode():
    slab = UnfoldedSlab(k=7.3, half_width=1.2, length=0.79,
                        curvature=0, dz=0.13, n_inner=63)
    frequency = 5 * np.pi / slab.width
    field = np.sin(frequency * (slab.x + slab.half_width))
    expected = field * np.exp(-1j * frequency**2 * slab.length / (2 * slab.k))
    np.testing.assert_allclose(slab.propagate(field), expected, atol=3e-14)
    assert slab.steps * slab.step == slab.length


def test_bent_slab_preserves_oddness_norm_and_psd_factors():
    slab = UnfoldedSlab(9.1, 1.0, 0.8, -0.17, 0.03, 47)
    rng = np.random.default_rng(17)
    factors = rng.normal(size=(47, 3)) + 1j*rng.normal(size=(47, 3))
    original = factors.copy()
    unfolded = slab.propagate_unfolded(slab.unfold(factors))
    result = unfolded[1:48]
    np.testing.assert_array_equal(factors, original)
    np.testing.assert_allclose(unfolded[1:48], -unfolded[49:][::-1], atol=5e-14)
    np.testing.assert_allclose(unfolded[[0, 48]], 0, atol=5e-14)
    np.testing.assert_allclose(result.conj().T @ result,
                               factors.conj().T @ factors, rtol=2e-14, atol=1e-12)
    density = result @ result.conj().T
    assert np.linalg.eigvalsh(density)[0] > -1e-12*np.trace(density).real


def test_bent_slab_matches_independent_dst_reference():
    scipy_fft = pytest.importorskip("scipy.fft")
    k = 2*np.pi / (1.23984198e-6 / 8048)
    half_width = 24e-6 + 1/(k*np.sqrt(2*7.125763840840872e-6))
    slab = UnfoldedSlab(k, half_width, 0.23, -1/2156.25, 0.001, 127)
    source = np.array([-.525e-6, .2e-6, .525e-6])
    fields = np.exp(1j*k*(slab.x[:, None] + 72e-6 - source)**2/(2*.675))
    reference = fields.copy()
    modes = np.arange(1, slab.n_inner + 1)*np.pi/(2*half_width)
    kinetic = np.exp(-1j*modes**2*slab.step/(2*k))[:, None]
    kick = np.exp(-1j*k*slab.curvature*slab.x*slab.step/2)[:, None]
    for _ in range(slab.steps):
        reference *= kick
        reference = scipy_fft.idst(
            kinetic*scipy_fft.dst(reference, type=1, axis=0), type=1, axis=0)
        reference *= kick
    actual = slab.propagate(fields)
    assert np.linalg.norm(actual-reference)/np.linalg.norm(reference) < 5e-13


def test_chord_correction_is_nontrivial_only_across_folds():
    slab = UnfoldedSlab(7.0, 1.0, 0.2, -0.3, 0.1, 31)
    left = np.array([.2, .8, 2.2, 2.8])
    right = np.array([.4, 1.4, 2.4, 3.4])
    np.testing.assert_allclose(slab.chord_residual(left, right), 0, atol=1e-15)
    left, right = .8, 2.5
    residual = slab.chord_residual(left, right)
    assert abs(residual) > .1
    midpoint = (left + right)/2
    classical = np.exp(-1j*slab.step*(left-right)*slab.potential_gradient(midpoint))
    exact = np.exp(-1j*slab.step*(slab.potential(left)-slab.potential(right)))
    np.testing.assert_allclose(classical*np.exp(1j*residual), exact, atol=1e-15)


def test_zero_length_is_identity_and_invalid_input_fails():
    slab = UnfoldedSlab(7, 1, 0, -.2, .1, 7)
    field = np.arange(7, dtype=complex)
    np.testing.assert_array_equal(slab.propagate(field), field)
    with pytest.raises(ValueError, match="expected shape"):
        slab.propagate(np.ones((8, 2)))
    with pytest.raises(ValueError, match="finite"):
        slab.propagate(np.full(7, np.nan))
    with pytest.raises(ValueError, match="positive"):
        UnfoldedSlab(7, 1, 1, .2, 0, 7)

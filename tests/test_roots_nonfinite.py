"""Non-finite values in the root finders: a kink's undefined derivative takes the
bisection step; a non-finite g, M2 sample or root candidate is an error, never a
nan root."""

import pytest

from formula import NumberContext, RaySurface
from formula._roots import _general, _union, sampling, subdivision
from formula._roots.utils import finite, finite_sign, merge_close_roots, rtsafe, sign

N = NumberContext(64)
NAN, INF = N(0) / N(0), N(1) / N(0)


class Kink:
    """|t| + 2t + 0.5: the root -0.5; g' undefined at t = 0."""

    def g(self, t):
        return abs(t) + 2 * t + N("0.5")

    def gprime(self, t):
        return t / abs(t) + 2


class Pole:
    """1/t: a sign change across a pole, no root."""

    def g(self, t):
        return 1 / t

    def gprime(self, t):
        return -1 / (t * t)


def _is_half(root):
    return abs(root + N("0.5")) < N("1e-60")


def test_finite_and_sign_classify_inf_and_nan():
    assert finite(N(1)) and not finite(INF) and not finite(NAN)
    assert sign(INF) == 1 and finite_sign(INF) is None and finite_sign(NAN) is None
    with pytest.raises(ValueError, match="non-finite"):
        sign(NAN)


def test_a_kink_at_the_bracket_midpoint_bisects_to_the_root():
    # One sample: the bracket [-1, 1] has its first midpoint at the kink.
    assert _is_half(rtsafe(Kink(), N(-1), N(1), N("1e-62")))
    root, = sampling.find_all(Kink(), N(-1), N(1), 64, samples=1)
    assert _is_half(root)


def test_a_pole_inside_the_bracket_is_an_error_not_a_root():
    with pytest.raises(ValueError, match="non-finite g"):
        rtsafe(Pole(), N(-1), N(1), N("1e-62"))
    with pytest.raises(ValueError, match="non-finite g"):
        sampling.find_all(Pole(), N(-1), N(1), 64, samples=1)


def test_non_finite_root_candidates_are_refused():
    for lists in ([[NAN], [N(1), N(2)]], [[N(1), N(2)], [NAN]], [[INF]]):
        with pytest.raises(ValueError, match="non-finite root"):
            _union(lists, 64)
    with pytest.raises(ValueError, match="non-finite root"):
        merge_close_roots([N(1), NAN], N("1e-10"))
    assert _union([[N(1), N(2)], [N(2), N(3)]], 64) == [1, 2, 3]


def test_subdivision_refuses_a_non_finite_m2_sample_and_the_root_survives():
    # 200 samples on [-1, 1] put a node on the kink: the M2 estimate is not a number.
    with pytest.raises(ValueError, match="non-finite g'"):
        subdivision.find_all(Kink(), N(-1), N(1), 64)
    root, = subdivision.find_all(Kink(), N(-1), N(1), 64, m2_samples=3)
    assert _is_half(root)
    roots = _general(Kink(), N(-1), N(1), 64)
    assert any(_is_half(r) for r in roots)


class Pole03:
    """1/(t - 0.3): a sign change across a pole between the samples, no root."""

    def g(self, t):
        return 1 / (t - N("0.3"))

    def gprime(self, t):
        return -1 / ((t - N("0.3")) * (t - N("0.3")))


class Jump:
    """sign(t - 0.3): a jump through zero, no root."""

    def g(self, t):
        return N(1) if t > N("0.3") else N(-1)

    def gprime(self, t):
        return N(0)


def test_a_pole_or_a_jump_between_the_samples_is_no_root():
    # A root's residual vanishes with the bracket; a pole's grows and a jump's stays.
    for func in (Pole03(), Jump()):
        assert rtsafe(func, N(0), N(1), N("1e-62")) is None
        assert sampling.find_all(func, N(0), N(1), 64, samples=1) == []
    assert RaySurface("1/(x-0.3)", 64).intersect((0, 0, 0), (1, 0, 0), 1, method="sampling", samples=1) == []
    root, = sampling.find_all(Kink(), N(-1), N(1), 64, samples=1)
    assert _is_half(root)

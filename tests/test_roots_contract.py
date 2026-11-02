"""The degenerate-case contract of the root finders (2026-11-02):
an identically zero g raises ZeroFunctionError, an unresolvable candidate raises
UnresolvedRootError, and neither verdict is swallowed by the auto/general fallbacks.
Everything decidable keeps a plain list answer."""

import pytest

from formula import (NumberContext, RaySurface, UnresolvedRootError,
                     ZeroFunctionError)
from formula._roots import chebyshev, sampling, sturm, subdivision
from formula._roots.utils import rtsafe, tol_full

N = NumberContext(64)
O, D = (0, 0, 0), (1, 0, 0)
METHODS = ("sampling", "sturm", "chebyshev", "subdivision", "auto")


def ray(expression, precision=64):
    surface = RaySurface(expression, precision)
    return surface, surface.function(O, D)


def options(method):
    if method == "sampling":
        return {"samples": 4}
    if method == "subdivision":
        return {"region_tol": N("0.1")}
    return {}


@pytest.mark.parametrize("method", METHODS)
def test_zero_function_raises(method):
    # The ray lies in the surface: no finite list of isolated roots exists.
    surface, _ = ray("y")
    with pytest.raises(ZeroFunctionError):
        surface.intersect(O, D, 1, method=method, **options(method))


def test_zero_function_raises_in_direct_backends():
    for module, kw in ((sampling, {"samples": 4}), (sturm, {}), (chebyshev, {}),
                       (subdivision, {"region_tol": N("0.1")})):
        _, func = ray("y")
        with pytest.raises(ZeroFunctionError):
            module.find_all(func, N(0), N(1), 64, **kw)


@pytest.mark.parametrize("method", METHODS)
def test_nonzero_constant_is_rootless(method):
    surface, _ = ray("y + 3")
    assert surface.intersect(O, D, 1, method=method, **options(method)) == []


@pytest.mark.parametrize("method", ("sturm", "chebyshev", "subdivision", "auto"))
def test_tangency_is_unresolved(method):
    # A finite-precision evaluation cannot tell (x-.3)^2 from (x-.3)^2 +- 1e-70.
    surface, _ = ray("(x-.3)^2")
    with pytest.raises(UnresolvedRootError):
        surface.intersect(O, D, 1, method=method)


@pytest.mark.parametrize("method", ("sturm", "chebyshev"))
def test_positive_dip_inside_the_noise_band_is_unresolved(method):
    surface, _ = ray("(x-.3)^2 + 1e-64")
    with pytest.raises(UnresolvedRootError):
        surface.intersect(O, D, 1, method=method)


@pytest.mark.parametrize("expression", ("(x-.3)^2 + 1e-40", "(x-.3)^2 + 1e-25"))
@pytest.mark.parametrize("method", ("sturm", "chebyshev", "subdivision"))
def test_positive_dip_above_the_noise_band_is_rootless(expression, method):
    surface, _ = ray(expression)
    assert surface.intersect(O, D, 1, method=method) == []


@pytest.mark.parametrize("method", ("sturm", "chebyshev"))
def test_close_pair_splits_on_the_original_polynomial(method):
    surface, _ = ray("(x-.3)*(x-.300000000000000000001)")
    roots = surface.intersect(O, D, 1, method=method)
    assert len(roots) == 2
    assert abs(roots[0] - N("0.3")) < N("1e-40")
    assert abs(roots[1] - N("0.300000000000000000001")) < N("1e-40")


def test_four_root_cluster_resolved_by_sturm():
    # Four simple roots 1e-12 apart; the approximate gcd used to fuse them.
    surface, _ = ray("(x-.3)*(x-.300000000001)*(x-.300000000002)*(x-.300000000003)")
    roots = surface.intersect(O, D, 1, method="sturm")
    assert len(roots) == 4
    for k, root in enumerate(roots):
        assert abs(root - (N("0.3") + N("1e-12") * k)) < N("1e-25")


@pytest.mark.parametrize("method", ("sturm", "chebyshev"))
def test_one_endpoint_root_reported_once(method):
    # The endpoint used to come back twice from chebyshev.
    surface, _ = ray("x*(x-1)")
    roots = surface.intersect(O, D, 1, method=method)
    assert len(roots) == 2
    assert roots[0] == 0 and abs(roots[1] - 1) < N("1e-60")


def test_small_scale_subdivision_keeps_only_real_roots():
    # The absolute residual test accepted non-roots of a tiny-scale g.
    surface, _ = ray("1e-30*x*(x-1)")
    roots = surface.intersect(O, D, 1, method="subdivision")
    assert [float(t) for t in roots] == [0.0, 1.0]


def test_singleton_interval_judges_the_root_by_the_newton_step():
    # A point interval of a tiny-scale g is not a root just because |g| is small.
    _, func = ray("1e-30*(x-.3)")
    assert subdivision.find_all(func, N("0.4"), N("0.4"), 64) == []
    assert subdivision.find_all(func, N("0.3"), N("0.3"), 64) == [N("0.3")]


def test_rtsafe_stagnation_raises():
    # At |t| ~ 1e30 the bracket cannot shrink below an absolute 1e-62.
    _, func = ray("(x-1e30)^2 - 0.2")
    with pytest.raises(UnresolvedRootError):
        rtsafe(func, N("1e30"), N("1e30") + 1, tol_full(N))
    _, plain = ray("x^2 - 0.2")
    root = rtsafe(plain, N(0), N(1), tol_full(N))
    assert abs(root - N.eval("sqrt(0.2)")) < N("1e-60")


def test_verdicts_pass_through_the_auto_fallback():
    # ZeroFunctionError(ValueError) and UnresolvedRootError(ArithmeticError) must not be
    # swallowed as backend failures by the sturm -> general fallback chain.
    surface, _ = ray("y")
    with pytest.raises(ZeroFunctionError):
        surface.intersect(O, D, 1, method="auto", samples=4)
    surface, _ = ray("(x-.3)^2")
    with pytest.raises(UnresolvedRootError):
        surface.intersect(O, D, 1, method="auto")

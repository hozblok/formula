"""Diverse surface and geometry tests for RaySurface (generated set, all verified).

Complements test_intersect.py and test_intersect_robustness.py with a broad
sweep of surface families and ray geometries. Every case has a closed-form
expected answer.
"""

import pytest

from formula import Number, RaySurface, Solver
from formula._roots import is_polynomial


def near(value, expected, eps="1e-18"):
    """True if value is within eps of expected (a Number, number, or expression string)."""
    prec = value.precision
    return abs(value - Number(expected, prec)) < Number(eps, prec)


def roots_close(roots, expected, eps="1e-18"):
    """True if roots matches expected one-for-one within eps."""
    return len(roots) == len(expected) and all(
        near(r, e, eps) for r, e in zip(roots, expected)
    )


# ---------------------------------------------------------------------------
# Quadrics & conics (cone, ellipsoid, hyperboloids, paraboloid, plane)
# ---------------------------------------------------------------------------

def test_div_conics_double_cone_two_roots():
    rs = RaySurface("x*x + y*y - z*z", precision=48)
    roots = rs.intersect((1, 0, -3), (0, 0, 1), t_max=10, method="sturm")
    assert roots_close(roots, ["2", "4"], eps="1e-30")

def test_div_conics_ellipsoid_auto_two_roots():
    rs = RaySurface("x*x/4 + y*y/9 + z*z - 1", precision=48)
    roots = rs.intersect((0, -6, 0), (0, 1, 0), t_max=12, method="auto")
    assert roots_close(roots, ["3", "9"], eps="1e-28")

def test_div_conics_one_sheet_hyperboloid_two_roots():
    rs = RaySurface("x*x + y*y - z*z - 1", precision=48)
    roots = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=10, method="sturm")
    assert roots_close(roots, ["2", "4"], eps="1e-30")

def test_div_conics_two_sheet_hyperboloid_two_roots():
    rs = RaySurface("z*z - x*x - y*y - 1", precision=48)
    roots = rs.intersect((0, 0, -4), (0, 0, 1), t_max=10, method="sturm")
    assert roots_close(roots, ["3", "5"], eps="1e-28")

def test_div_conics_elliptic_paraboloid_two_roots():
    rs = RaySurface("x*x + y*y - z", precision=48)
    roots = rs.intersect((-5, 0, 4), (1, 0, 0), t_max=12, method="sturm")
    assert roots_close(roots, ["3", "7"], eps="1e-30")

def test_div_conics_plane_single_root_arclength():
    rs = RaySurface("x - 3", precision=48)
    roots = rs.intersect((0, 0, 0), (3, 4, 0), t_max=20, method="sturm")
    assert roots_close(roots, ["5"], eps="1e-28")

def test_div_conics_ellipsoid_tangent_double_root():
    rs = RaySurface("x*x/4 + y*y/9 + z*z - 1", precision=48)
    roots = rs.intersect((-5, 3, 0), (1, 0, 0), t_max=12, method="sturm")
    assert roots_close(roots, ["5"], eps="1e-12")
    missed = rs.intersect((-5, 3, 0), (1, 0, 0), t_max=12, method="sampling")
    assert missed == []


# ---------------------------------------------------------------------------
# Transcendental surfaces (exp, log, atan, Gaussian)
# ---------------------------------------------------------------------------

def test_div_transcendental_exp_minus_two_ln2_all_backends():
    # exp(x)-2 = 0 has the single root x = ln 2; chebyshev, subdivision and
    # auto (which routes this non-polynomial to the general backends) agree.
    rs = RaySurface("exp(x) - 2", precision=32)
    for method in ("chebyshev", "subdivision", "auto"):
        roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method=method)
        assert roots_close(roots, ["log(2)"], eps="1e-24")

def test_div_transcendental_log_minus_one_e():
    # log(x)-1 = 0 at x = e. Interval starts at t_min=1 (x=1) to stay clear of the
    # log domain edge at x=0; all general backends recover e.
    rs = RaySurface("log(x) - 1", precision=32)
    for method in ("chebyshev", "subdivision", "auto"):
        roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=4, t_min=1, method=method)
        assert roots_close(roots, ["exp(1)"], eps="1e-24")

def test_div_transcendental_exp_neg_half_chebyshev_matches_subdivision():
    # exp(-x)-0.5 = 0 at x = ln 2. The chebyshev fit and the subdivision search
    # must land on the same root.
    rs = RaySurface("exp(-x) - 0.5", precision=32)
    cheb = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="chebyshev")
    sub = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="subdivision")
    assert roots_close(cheb, ["log(2)"], eps="1e-24")
    assert roots_close(cheb, [str(r) for r in sub], eps="1e-20")

def test_div_transcendental_atan_minus_one_tan1():
    # atan(x)-1 = 0 at x = tan 1. subdivision and auto recover the same root.
    rs = RaySurface("atan(x) - 1", precision=32)
    sub = rs.intersect((0, 0, 0), (1, 0, 0), t_max=5, method="subdivision")
    auto = rs.intersect((0, 0, 0), (1, 0, 0), t_max=5, method="auto")
    assert roots_close(sub, ["tan(1)"], eps="1e-24")
    assert roots_close(auto, [str(r) for r in sub], eps="1e-26")

def test_div_transcendental_gaussian_two_roots():
    # exp(-(x-2)^2)-0.5 = 0 at x = 2 +/- sqrt(ln 2) (two roots).
    rs = RaySurface("exp(-(x-2)*(x-2)) - 0.5", precision=32)
    s = Number("sqrt(log(2))", 32)
    two = Number(2, 32)
    expected = [two - s, two + s]
    for method in ("chebyshev", "subdivision", "auto"):
        roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=4, method=method)
        assert roots_close(roots, expected, eps="1e-22")

def test_div_transcendental_gaussian_two_roots_pow_form():
    # Same Gaussian, written with ^2. Before the structural-zero fix this NaN'd:
    # the constant exponent's spurious log(base) term poisoned the derivative
    # both at the peak (base 0) and all along x<2 (base negative). With the fix
    # that term is dropped, so the ^2 form matches the product form.
    rs = RaySurface("exp(-(x-2)^2) - 0.5", precision=32)
    s = Number("sqrt(log(2))", 32)
    two = Number(2, 32)
    expected = [two - s, two + s]
    for method in ("chebyshev", "subdivision", "auto"):
        roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=4, method=method)
        assert roots_close(roots, expected, eps="1e-22")
    # Detailed x<2 coverage: where the base (x-2) is negative the ^2 surface
    # derivative must be finite and equal to the product form (it was NaN
    # pre-fix). This is the negative-base branch the original test sidesteps.
    pow_form = Solver("exp(-(x-2)^2)", precision=32)
    product = Solver("exp(-(x-2)*(x-2))", precision=32)
    for x in ("1.9", "1.5", "1.0", "0.4", "0"):
        d_pow = pow_form({"x": x}, derivative="x", format_digits=30)
        d_prd = product({"x": x}, derivative="x", format_digits=30)
        assert "nan" not in d_pow.lower()
        assert near(Number(d_pow, 32), d_prd, eps="1e-28")

def test_div_transcendental_points_lie_on_exp_surface():
    # The recovered intersection point of exp(x)-2 must satisfy F(x,y,z)=0:
    # exp(x)-2 == 0 to working precision.
    rs = RaySurface("exp(x) - 2", precision=32)
    pts = rs.points((0, 0, 0), (1, 0, 0), t_max=3, method="auto")
    assert len(pts) == 1
    x, y, z = pts[0]
    residual = Number("exp(1)", 32) ** x - Number(2, 32)
    assert abs(residual) < Number("1e-28", 32)

def test_div_transcendental_auto_is_general_not_sturm_for_exp():
    # A transcendental surface is not a polynomial, so auto must take the general
    # (chebyshev union subdivision) path and reproduce the chebyshev root.
    rs = RaySurface("exp(x) - 2", precision=32)
    assert not is_polynomial(rs.surface)
    auto = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="auto")
    cheb = rs.intersect((0, 0, 0), (1, 0, 0), t_max=3, method="chebyshev")
    assert roots_close(auto, [str(r) for r in cheb], eps="1e-24")


# ---------------------------------------------------------------------------
# Ray geometry (negative t_min, oblique, off-center, arc length)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['sturm', 'sampling', 'auto'])
def test_div_geometry_negative_tmin_straddles_origin(method):
    # Origin at the center of sphere r=2; with t_min<0 the hits are t=-2 and t=2.
    rs = RaySurface('x*x + y*y + z*z - 4', precision=48)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=10, t_min=-10, method=method)
    assert roots_close(roots, ['-2', '2'], eps='1e-13')

def test_div_geometry_oblique_3d_ray_through_center():
    # O=(-3,-3,-3), dir (1,1,1) passes through the center of sphere r=sqrt(3);
    # |O|=3*sqrt(3), so hits at t = 3sqrt3 -/+ sqrt3 = 2*sqrt(3) and 4*sqrt(3).
    rs = RaySurface('x*x + y*y + z*z - 3', precision=48)
    roots = rs.intersect((-3, -3, -3), (1, 1, 1), t_max=20, method='sturm')
    s3 = Number('sqrt(3)', 48)
    assert roots_close(roots, [s3 * Number(2, 48), s3 * Number(4, 48)], eps='1e-13')
    pts = rs.points((-3, -3, -3), (1, 1, 1), t_max=20, method='sturm')
    assert len(pts) == 2
    for x, y, z in pts:
        assert abs(x * x + y * y + z * z - Number(3, 48)) < Number('1e-30', 48)

def test_div_geometry_off_center_sphere():
    # Sphere centered at x=5, radius 2; ray from origin along +x enters at x=3, exits at x=7.
    rs = RaySurface('(x-5)*(x-5) + y*y + z*z - 4', precision=48)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=10, method='sturm')
    assert roots_close(roots, ['3', '7'], eps='1e-13')

@pytest.mark.parametrize('method', ['sturm', 'sampling', 'auto'])
def test_div_geometry_direction_away_no_hits(method):
    # Sphere centered at x=5; firing from the origin toward -x walks away from it forever.
    rs = RaySurface('(x-5)*(x-5) + y*y + z*z - 4', precision=48)
    assert rs.intersect((0, 0, 0), (-1, 0, 0), t_max=10, method=method) == []

def test_div_geometry_narrow_window_captures_subset():
    # Unit cylinder; full range gives t=1,3. A window around 3 must keep only that root.
    rs = RaySurface('x*x + y*y - 1', precision=48)
    full = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method='sturm')
    assert roots_close(full, ['1', '3'], eps='1e-13')
    narrow = rs.intersect((-2, 0, 0), (1, 0, 0), t_max='3.5', t_min='2.5', method='sturm')
    assert roots_close(narrow, ['3'], eps='1e-13')

def test_div_geometry_arc_length_under_nonunit_direction():
    # Direction is normalized internally so t measures arc length; a 3x-scaled
    # direction yields identical roots, equal to the +/- radius distances.
    rs = RaySurface('x*x + y*y + z*z - 16', precision=48)
    unit = rs.intersect((0, 0, 0), (0, 0, 1), t_max=10, t_min=-10, method='sturm')
    scaled = rs.intersect((0, 0, 0), (0, 0, 3), t_max=10, t_min=-10, method='sturm')
    assert roots_close(scaled, [str(u) for u in unit], eps='1e-30')
    assert roots_close(scaled, ['-4', '4'], eps='1e-13')

def test_div_geometry_nonunit_oblique_origin_on_surface():
    # O=(-2,-2,-2) lies on sphere r=sqrt(12) (|O|^2=12), dir (2,2,2) non-unit and
    # through the center: t=0 is a root, the far exit is t=4*sqrt(3).
    rs = RaySurface('x*x + y*y + z*z - 12', precision=48)
    roots = rs.intersect((-2, -2, -2), (2, 2, 2), t_max=20, t_min=-5, method='sturm')
    s3 = Number('sqrt(3)', 48)
    assert roots_close(roots, [Number(0, 48), Number(4, 48) * s3], eps='1e-13')
    pts = rs.points((-2, -2, -2), (2, 2, 2), t_max=20, t_min=-5, method='sturm')
    assert len(pts) == 2
    for x, y, z in pts:
        assert abs(x * x + y * y + z * z - Number(12, 48)) < Number('1e-30', 48)


# ---------------------------------------------------------------------------
# Invariants (determinism, sorting, points<->t, high precision)
# ---------------------------------------------------------------------------

def test_div_consistency_repeated_call_is_deterministic():
    rs = RaySurface("(x*x - 1) * (x*x - 4)", precision=48)
    o, d = (-3, 0, 0), (1, 0, 0)
    a = rs.intersect(o, d, t_max=6, method="sturm")
    b = rs.intersect(o, d, t_max=6, method="sturm")
    assert len(a) == len(b) == 4
    assert [str(t) for t in a] == [str(t) for t in b]

def test_div_consistency_points_match_ts_and_lie_on_surface():
    rs = RaySurface("x*x + y*y + z*z - 1", precision=48)
    o, d = (-2, 0, 0), (1, 0, 0)
    ts = rs.intersect(o, d, t_max=10, method="sturm")
    pts = rs.points(o, d, t_max=10, method="sturm")
    func = rs.function(o, d)
    assert len(pts) == len(ts) == 2
    for t, (x, y, z) in zip(ts, pts):
        assert tuple(str(c) for c in (x, y, z)) == tuple(str(c) for c in func.point_at(t))
        assert abs(x * x + y * y + z * z - Number(1, 48)) < Number("1e-40", 48)

def test_div_consistency_results_sorted_ascending():
    rs = RaySurface("(x*x - 1) * (x*x - 4)", precision=48)
    roots = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=6, method="sturm")
    assert len(roots) == 4
    assert all(roots[i] < roots[i + 1] for i in range(len(roots) - 1))

def test_div_consistency_high_precision_sqrt2():
    rs = RaySurface("x*x - 2", precision=128)
    assert is_polynomial(rs.surface)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=5, method="sturm")
    assert len(roots) == 1
    assert near(roots[0], "sqrt(2)", eps="1e-100")

def test_div_consistency_high_precision_pi():
    rs = RaySurface("sin(x)", precision=128)
    roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=4, t_min="0.5", method="subdivision")
    assert len(roots) == 1
    assert near(roots[0], "4*atan(1)", eps="1e-100")

def test_div_consistency_auto_and_subdivision_agree_on_count():
    rs = RaySurface("cos(x)", precision=32)
    o, d = (0, 0, 0), (1, 0, 0)
    auto = rs.intersect(o, d, t_max=6, t_min="0.1", method="auto")
    sub = rs.intersect(o, d, t_max=6, t_min="0.1", method="subdivision")
    assert len(auto) == len(sub) == 2
    assert roots_close(auto, [str(t) for t in sub], eps="1e-20")

def test_div_consistency_subdivision_repeated_call_is_deterministic():
    rs = RaySurface("cos(x) + 1", precision=32)
    o, d = (0, 0, 0), (1, 0, 0)
    a = rs.intersect(o, d, t_max=10, t_min="0.1", method="subdivision")
    b = rs.intersect(o, d, t_max=10, t_min="0.1", method="subdivision")
    assert len(a) == len(b) == 2
    assert [str(t) for t in a] == [str(t) for t in b]


# ---------------------------------------------------------------------------
# Multiplicity & tangency
# ---------------------------------------------------------------------------

def test_div_multiplicity_perfect_square_double_root_sturm():
    # (x-2)^2 + y^2 + z^2 = 0 ; ray along +x from x=-1 grazes at x=2 -> double root t=3.
    rs = RaySurface('(x-2)*(x-2) + y*y + z*z', precision=48)
    roots = rs.intersect((-1, 0, 0), (1, 0, 0), t_max=10, method='sturm')
    assert roots_close(roots, ['3'], eps='1e-12')

def test_div_multiplicity_double_root_sampling_misses_sturm_finds():
    rs = RaySurface('(x-2)*(x-2) + y*y + z*z', precision=48)
    sampled = rs.intersect((-1, 0, 0), (1, 0, 0), t_max=10, method='sampling')
    assert sampled == []  # sign never changes across an even-multiplicity root
    sturm = rs.intersect((-1, 0, 0), (1, 0, 0), t_max=10, method='sturm')
    assert roots_close(sturm, ['3'], eps='1e-12')

def test_div_multiplicity_quadruple_root_collapses_to_single():
    # (x-1)^4 ; quadruple root at x=1 -> square-free Sturm yields one root t=3.
    rs = RaySurface('(x-1)^4', precision=48)
    roots = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method='sturm')
    assert len(roots) == 1
    assert roots_close(roots, ['3'], eps='1e-12')

def test_div_multiplicity_sphere_tangent_to_plane():
    # Unit sphere at (0,1,0); ray in plane y=0 along +x from x=-2 is tangent at x=0 -> t=2.
    rs = RaySurface('x*x + (y-1)*(y-1) + z*z - 1', precision=48)
    sturm = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method='sturm')
    assert roots_close(sturm, ['2'], eps='1e-12')
    sampled = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method='sampling')
    assert sampled == []

def test_div_multiplicity_mixed_simple_and_double():
    # x*(x-3)^2 : simple root x=0 (t=1), double root x=3 (t=4) for origin x=-1.
    rs = RaySurface('x*(x-3)*(x-3)', precision=48)
    sturm = rs.intersect((-1, 0, 0), (1, 0, 0), t_max=10, method='sturm')
    assert roots_close(sturm, ['1', '4'], eps='1e-12')
    sampled = rs.intersect((-1, 0, 0), (1, 0, 0), t_max=10, method='sampling')
    assert len(sampled) == 1 and near(sampled[0], '1', eps='1e-12')

def test_div_multiplicity_auto_routes_polynomial_to_sturm_for_tangency():
    rs = RaySurface('x*x + (y-1)*(y-1) + z*z - 1', precision=48)
    assert is_polynomial(rs.surface)  # algebraic -> auto picks Sturm
    auto = rs.intersect((-2, 0, 0), (1, 0, 0), t_max=10, method='auto')
    assert roots_close(auto, ['2'], eps='1e-12')


# ---------------------------------------------------------------------------
# Higher-degree algebraic surfaces
# ---------------------------------------------------------------------------

def test_div_higher_degree_cubic_three_real_roots():
    rs = RaySurface("(x-1)*(x-2)*(x-3)", precision=48)
    assert is_polynomial(rs.surface)
    for method in ("sturm", "auto"):
        roots = rs.intersect((-4, 0, 0), (1, 0, 0), t_max=10, method=method)
        assert len(roots) == 3
        assert roots_close(roots, ["5", "6", "7"], eps="1e-25")

def test_div_higher_degree_quintic_five_real_roots():
    rs = RaySurface("x*(x-1)*(x+1)*(x-2)*(x+2)", precision=48)
    for method in ("sturm", "auto"):
        roots = rs.intersect((-3, 0, 0), (1, 0, 0), t_max=8, method=method)
        assert len(roots) == 5
        assert roots_close(roots, ["1", "2", "3", "4", "5"], eps="1e-22")

def test_div_higher_degree_sextic_six_real_roots():
    rs = RaySurface("(x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)", precision=48)
    for method in ("sturm", "auto"):
        roots = rs.intersect((0, 0, 0), (1, 0, 0), t_max=8, method=method)
        assert len(roots) == 6
        assert roots_close(roots, ["1", "2", "3", "4", "5", "6"], eps="1e-22")

def test_div_higher_degree_cassini_quartic_four_roots():
    rs = RaySurface("(x*x+y*y)^2 - 2*(x*x-y*y)", precision=48)
    assert is_polynomial(rs.surface)
    roots = rs.intersect((-3, "0.2", 0), (1, 0, 0), t_max=6, method="sturm")
    assert len(roots) == 4
    expected = [
        "1.6301404674233317577173806251660714444567677176",
        "2.7914697599655340187562181451685338788509835483",
        "3.2085302400344659812437818548314661211490164516",
        "4.3698595325766682422826193748339285555432322823",
    ]
    assert roots_close(roots, expected, eps="1e-12")
    mid = Number(3, 48)
    assert near(roots[0] + roots[3], str(2 * mid), eps="1e-20")
    assert near(roots[1] + roots[2], str(2 * mid), eps="1e-20")

def test_div_higher_degree_product_eight_roots():
    rs = RaySurface("(x*x-1)*(x*x-4)*(x*x-9)*(x*x-16)", precision=48)
    assert is_polynomial(rs.surface)
    for method in ("sturm", "auto"):
        roots = rs.intersect((-5, 0, 0), (1, 0, 0), t_max=11, method=method)
        assert len(roots) == 8
        assert roots_close(
            roots, ["1", "2", "3", "4", "6", "7", "8", "9"], eps="1e-18"
        )

def test_div_higher_degree_sextic_diagonal_arclength():
    rs = RaySurface("(x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)", precision=48)
    roots = rs.intersect((0, 0, 0), (3, 4, 0), t_max=12, method="sturm")
    assert len(roots) == 6
    expected = [str(Number(k, 48) / Number("0.6", 48)) for k in range(1, 7)]
    assert roots_close(roots, expected, eps="1e-22")

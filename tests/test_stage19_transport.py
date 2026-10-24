"""Independent Decimal checks: no array/scientific package is imported."""

from decimal import Decimal as D, localcontext
from types import SimpleNamespace

from formula.capsysred.stages._s19_transport import (
    count_free_caustics, dot, norm, reflect_variations, transport_ray,
    wall_normal_derivative,
)


def _diagonal(a, b):
    return [[D(str(a)), D(0)], [D(0), D(str(b))]]


def test_focal_multiplicity_and_endpoints():
    q = _diagonal(1, 1)
    cases = [(1, 1, 0, False), (-2, 1, 1, False), (-2, -3, 2, False),
             (-2, -2, 2, False), (-1, 1, 0, True), ("-1e-50", -2, 1, False)]
    for a, b, count, uncertain in cases:
        assert count_free_caustics(q, _diagonal(a, b), 1) == (count, uncertain)


def test_near_double_root_behind_entrance_is_not_uncertain():
    # The determinant discriminant is tiny and negative, but its extremum
    # lies at t=-1, safely outside the physical [0,1] segment.
    q = [[D(1), D("1e-30")], [D("-1e-30"), D(1)]]
    assert count_free_caustics(q, _diagonal(1, 1), 1) == (0, False)


def test_flat_reflection_orientation_is_not_maslov():
    outgoing, q, p = reflect_variations(
        [D(".6"), D(0), D(".8")],
        [[D(1), D(0)], [D(0), D(1)], [D(0), D(0)]],
        [[D(0), D(0)] for _ in range(3)],
        [D(1), D(0), D(0)], [[D(0)]*3 for _ in range(3)],
    )
    assert outgoing == [D("-.6"), D(0), D(".8")]
    assert q == [[D(-1), D(0)], [D(0), D(1)], [D(0), D(0)]]
    assert count_free_caustics(q[:2], p[:2], 1) == (0, False)


def test_normals_against_independent_finite_differences():
    cylinder = {"center": [0, 0], "radius": "2.4e-5"}
    torus = {"center": ["72e-6", 0], "radius": "24e-6", "kind": "torus",
             "bend": {"radius": "2156.25", "toward": [-1, 0]}}
    with localcontext() as context:
        context.prec = 64
        for bore, point in [(cylinder, ["18e-6", "16e-6", ".12"]),
                            (torus, ["80e-6", "20e-6", ".18"])]:
            point = list(map(D, point))
            normal, derivative, _ = wall_normal_derivative(point, bore, 0)
            h = D("1e-16")
            for j in range(3):
                plus, minus = point.copy(), point.copy()
                plus[j] += h
                minus[j] -= h
                np = wall_normal_derivative(plus, bore, 0)[0]
                nm = wall_normal_derivative(minus, bore, 0)[0]
                for i in range(3):
                    finite = (np[i]-nm[i])/(2*h)
                    assert abs(finite-derivative[i][j]) < D("1e-16")
                assert abs(dot(normal, [derivative[i][j] for i in range(3)])) < D("1e-55")


def _cylinder_ray(q, source, radius=D(1), exit_z=D(10)):
    """Independent Decimal analytic cylinder trace; no production tracer."""
    start = [*q, D(0)]
    difference = [start[i]-source[i] for i in range(3)]
    distance = norm(difference)
    u = [v/distance for v in difference]
    reflections = []
    for _ in range(20):
        a = dot(u[:2], u[:2])
        b = 2*dot(start[:2], u[:2])
        c = dot(start[:2], start[:2])-radius*radius
        roots = ([(-b+(b*b-4*a*c).sqrt())/(2*a),
                  (-b-(b*b-4*a*c).sqrt())/(2*a)] if a else [])
        future = [r for r in roots if r > D("1e-40")]
        time = min(future) if future else D("Infinity")
        if start[2]+time*u[2] > exit_z:
            time = (exit_z-start[2])/u[2]
            point = [start[i]+time*u[i] for i in range(3)]
            return SimpleNamespace(point=point, direction=u, reflections=reflections)
        start = [start[i]+time*u[i] for i in range(3)]
        n = [start[0]/radius, start[1]/radius, D(0)]
        incidence = dot(u, n)
        reflections.append((start.copy(), abs(incidence)))
        u = [u[i]-2*incidence*n[i] for i in range(3)]
    raise AssertionError("unexpected excessive reflections")


def test_cylinder_drt_matches_independent_neighbor_traces():
    with localcontext() as context:
        context.prec = 64
        source = list(map(D, [".08", "-.04", "-2"]))
        qs = [list(map(D, p)) for p in [(".1", ".1"), (".5", ".15"), (".6", "-.4")]]
        paths = [_cylinder_ray(q, source) for q in qs]
        assert [len(ray.reflections) for ray in paths] == [0, 1, 2]
        for q, ray in zip(qs, paths):
            result = transport_ray(source, q, ray, {"center": [0, 0], "radius": 1}, 0)
            assert result["valid"], result
            h = D("1e-14")
            for j in range(2):
                plus, minus = q.copy(), q.copy()
                plus[j] += h
                minus[j] -= h
                rp, rm = _cylinder_ray(plus, source), _cylinder_ray(minus, source)
                assert len(rp.reflections) == len(rm.reflections) == len(ray.reflections)
                for i in range(2):
                    dx = (rp.point[i]-rm.point[i])/(2*h)
                    dp = (rp.direction[i]-rm.direction[i])/(2*h)
                    assert abs(result["Q"][i][j]-dx) < D("1e-22")
                    assert abs(result["P"][i][j]-dp) < D("1e-22")
            assert result["lagrangian_defect"] < D("1e-55")
        assert transport_ray(source, qs[0], paths[0], {"center": [0, 0], "radius": 1}, 0)["maslov"] == 0


def test_free_jacobian_and_invalid_archived_direction():
    with localcontext() as context:
        context.prec = 64
        source, q = list(map(D, [".01", "-.02", "-2"])), list(map(D, [".1", ".2"]))
        ray = _cylinder_ray(q, source, radius=D(100), exit_z=D(5))
        result = transport_ray(source, q, ray, {"center": [0, 0], "radius": 100}, 0)
        assert result["valid"]
        for i in range(2):
            for j in range(2):
                assert abs(result["Q"][i][j]-(D("3.5") if i == j else 0)) < D("1e-55")
        ray.direction = [D(0), D(0), D(1)]
        assert not transport_ray(source, q, ray, {"center": [0, 0], "radius": 100}, 0)["valid"]


def test_real_c2_torus_transport_matches_neighbor_traces():
    from formula.formula import Number
    from formula.capsysred.native import make_tracer
    from formula.capsysred.shared.nums import vsub, vunit
    from formula.capsysred.surfaces import CapillaryBundle

    precision = 64
    bore = {"kind": "torus", "center": ["72e-6", "0"], "radius": "24e-6",
            "bend": {"radius": "2156.25", "toward": [-1, 0]}}
    typed = dict(kind="torus", center=tuple(Number(v, precision) for v in bore["center"]),
                 radius=Number(bore["radius"], precision),
                 bend=dict(radius=Number("2156.25", precision),
                           toward=(Number(-1, precision), Number(0, precision))))
    optic = CapillaryBundle([typed], Number(0, precision), Number(".23", precision))
    tracer = make_tracer(optic)
    source = (Number("1e-7", precision), Number("-2e-7", precision), Number("-.675", precision))

    def trace(q):
        target = tuple(Number(str(v), precision) for v in (*q, D(0)))
        ray = tracer(source, vunit(vsub(target, source)), optic, optic.z1, 200)
        assert ray.fate == "screen"
        return ray

    with localcontext() as context:
        context.prec = precision
        q = [D("79e-6"), D("8e-6")]
        ray = trace(q)
        assert len(ray.reflections) >= 1
        result = transport_ray(source, q, ray, bore, 0)
        assert result["valid"], result
        h = D("1e-13")
        for j in range(2):
            plus, minus = q.copy(), q.copy()
            plus[j] += h
            minus[j] -= h
            rp, rm = trace(plus), trace(minus)
            assert len(rp.reflections) == len(rm.reflections) == len(ray.reflections)
            for i in range(2):
                dx = (D(str(rp.point[i]))-D(str(rm.point[i])))/(2*h)
                dp = (D(str(rp.direction[i]))-D(str(rm.direction[i])))/(2*h)
                assert abs(result["Q"][i][j]-dx) < D("1e-12")
                assert abs(result["P"][i][j]-dp) < D("1e-10")

"""Topology, area/field representation, and bounded real tracing tests."""

from decimal import Decimal as D, localcontext
from itertools import permutations
from types import SimpleNamespace
import pytest

from formula.formula import Number
from formula.capsysred.stages._s19_mesh import (
    DEFAULTS, _Partition, _TraceCache, _center, _check, _check_eager, _children,
    _complex_close, _complex_enclosure, _model, _point_field_checks,
    _probe_points, _record, _well_shaped,
    build_source_mesh, disk_partition, signed_area,
)


def test_inscribed_partition_has_positive_area_and_reasonable_angles():
    with localcontext() as context:
        context.prec = 64
        points, faces = disk_partition((0, 0), 1, rings=8, inset=0)
        triangles = [[points[i] for i in face] for face in faces]
        assert len(points) == 289 and len(faces) == 512
        assert all(signed_area(triangle) > 0 for triangle in triangles)
        cosine = D(str(Number("cos(pi/9)", 64)))
        assert all(_well_shaped(triangle, cosine*cosine) for triangle in triangles)
        area = sum(map(signed_area, triangles), D(0))
        expected = D(str(Number("32*sin(pi/32)", 64)))
        assert abs(area-expected) < D("1e-60")
        # Correct disk deficit is about .16%, not an unresolved-area estimate.
        pi = D(str(Number("pi", 64)))
        assert D(".00160") < (pi-area)/pi < D(".00161")


def test_neighbor_bisection_preserves_conformity_and_area():
    points = [(D(0), D(0)), (D(1), D(0)), (D(1), D(1)), (D(0), D(1))]
    mesh = _Partition(points, [(0, 1, 2), (0, 2, 3)])
    removed, created = mesh.split(0, 20, 1000)[0]
    assert len(removed) == 2 and len(created) == 4
    for step in range(30):
        ident = sorted(mesh.faces)[step % len(mesh.faces)]
        assert mesh.split(ident, 20, 1000)[1] is None
    triangles = [[mesh.points[i] for i in face] for face in mesh.faces.values()]
    assert sum(map(signed_area, triangles), D(0)) == 1
    assert all(signed_area(triangle) > 0 for triangle in triangles)
    for edge, owners in mesh.edges.items():
        assert len(owners) in (1, 2)
        if len(owners) == 1:
            a, b = [mesh.points[i] for i in edge]
            # All single-owner edges must be on the original boundary.
            assert any(a[j] == b[j] and a[j] in (0, 1) for j in range(2))
    used_vertices = {v for face in mesh.faces.values() for v in face}
    assert len(used_vertices)-len(mesh.edges)+len(mesh.faces) == 1


def _incompatible_pair():
    points = [(D(0), D(0)), (D(1), D(0)), (D(".5"), D(".5")), (D(".5"), D(-1))]
    return _Partition(points, [(0, 1, 2), (1, 0, 3)])


def test_longest_edge_closure_reports_only_surviving_children():
    mesh = _incompatible_pair()
    area = sum((signed_area([mesh.points[i] for i in face]) for face in mesh.faces.values()), D(0))
    (removed, created), limit = mesh.split(0, 20, 100)
    assert limit is None and removed == [0, 1]
    assert set(created) == set(mesh.faces) and len(created) == 5
    assert 0 not in mesh.faces
    assert sum((signed_area([mesh.points[i] for i in face]) for face in mesh.faces.values()), D(0)) == area
    assert sorted(mesh.depth.values()) == [1, 1, 1, 2, 2]


@pytest.mark.parametrize("max_depth,max_triangles,reason", [(20, 3, "triangle_budget"), (1, 100, "max_depth")])
def test_longest_edge_closure_rolls_back_partial_work(max_depth, max_triangles, reason):
    mesh = _incompatible_pair()
    original = (list(mesh.points), dict(mesh.faces), dict(mesh.depth),
                {edge: set(owners) for edge, owners in mesh.edges.items()}, mesh.next_id)
    assert mesh.split(0, max_depth, max_triangles) == (None, reason)
    assert (mesh.points, mesh.faces, mesh.depth, dict(mesh.edges), mesh.next_id) == original
    fresh = _incompatible_pair()
    assert mesh.split(0, 20, 100) == fresh.split(0, 20, 100)
    assert (mesh.points, mesh.faces, mesh.depth, dict(mesh.edges)) == (
        fresh.points, fresh.faces, fresh.depth, dict(fresh.edges))


def test_local_disk_refinement_preserves_shape_conformity_and_area():
    with localcontext() as context:
        context.prec = 64
        points, faces = disk_partition((0, 0), 1, rings=8, inset=0)
        mesh = _Partition(points, faces)
        original_area = sum((signed_area([points[i] for i in face]) for face in faces), D(0))
        boundary = {edge for edge, owners in mesh.edges.items() if len(owners) == 1}
        cosine = D(str(Number("cos(pi/9)", 64)))
        for _ in range(160):
            local = [ident for ident, face in mesh.faces.items()
                     if sum((a-b)**2 for a, b in zip(_center([mesh.points[i] for i in face]),
                                                    (D(".35"), D(".1")))) < D(".08")]
            ident = max(local, key=lambda i: signed_area([mesh.points[v] for v in mesh.faces[i]]))
            before = set(mesh.faces)
            (removed, created), limit = mesh.split(ident, 30, 5000)
            assert limit is None and ident in removed
            assert set(removed) == before-set(mesh.faces)
            assert set(created) == set(mesh.faces)-before
        triangles = [[mesh.points[i] for i in face] for face in mesh.faces.values()]
        assert all(_well_shaped(triangle, cosine*cosine) for triangle in triangles)
        assert abs(sum(map(signed_area, triangles), D(0))-original_area) < D("1e-60")
        assert all(len(owners) in (1, 2) for owners in mesh.edges.values())
        assert {edge for edge, owners in mesh.edges.items() if len(owners) == 1} == boundary
        vertices = {v for face in mesh.faces.values() for v in face}
        assert len(vertices)-len(mesh.edges)+len(mesh.faces) == 1


def test_probe_cache_keys_are_canonical_without_coordinate_rounding():
    with localcontext() as context:
        context.prec = 64
        triangle = [(D("1e-5"), D(0)), (D("-1e-5"), D(1)), (D("1e-69"), D(2))]
        assert len({_center(order) for order in permutations(triangle)}) == 1
        for order in permutations(triangle):
            probes = _probe_points(order)
            assert probes[:3] == list(order)
            assert probes[3] == tuple((order[0][j]+order[1][j])/2 for j in range(2))
            assert probes[-1] == _center(triangle)


def test_central_actual_point_and_full_path_are_retained():
    class SyntheticCache:
        def get(self, q):
            x, y = q
            return dict(valid=True, reflections=0, maslov=0,
                        point=(x*(1+x/5), y, D(1)), direction=(D(0), D(0), D(1)),
                        source_distance=D(1), uz_in=D(1), fresnel=(D(1), D(0)),
                        determinant=1+2*x/5,
                        opl=D("1234567890123456789.1234567890123456789"))

    with localcontext() as context:
        context.prec = 64
        triangle = [(D(0), D(0)), (D(2), D(0)), (D(0), D(1))]
        model, reason = _model(triangle, SyntheticCache(), D("1e-16"))
        assert reason is None
        assert model["center"]["point"][:2] != _center(model["output"])
        assert model["jacobian"] == D("1.4")
        assert abs(model["amplitude"][0]-1/D("1.4").sqrt()) < D("1e-62")
        assert model["center"]["opl"] == D("1234567890123456789.1234567890123456789")
        children = [_model(child, SyntheticCache(), D("1e-16"))[0] for child in _children(triangle)]
        assert sum((child["input_area"] for child in children), D(0)) == model["input_area"]
        assert children[0]["jacobian"] != children[1]["jacobian"]
        # The parent's flux normalization must not be reused for each child.
        assert children[0]["amplitude"] != children[1]["amplitude"]


def test_complex_gate_checks_phase_and_remains_defined_at_zero():
    assert not _complex_close((D(1), D(0)), (D(-1), D(0)), D(".02"), D("1e-6"))
    assert _complex_close((D("1e-7"), D(0)), (D(0), D(0)), D(".02"), D("1e-6"))
    assert not _complex_close((D("1e-5"), D(0)), (D(0), D(0)), D(".02"), D("1e-6"))


def _small_sim():
    precision = 64
    cap = SimpleNamespace(z0=Number(0, precision), z1=Number(".0001", precision),
                          bores=[dict(center=[0, 0], radius="1e-5")])
    cfg = SimpleNamespace(precision=precision, max_bounces=20, capillary=cap)
    return SimpleNamespace(cfg=cfg, lines=[1], lam=Number("1e-4", precision),
                           fresnel=lambda sine: Number(1, precision))


def test_real_free_mesh_area_balance_and_explicit_budget():
    sim = _small_sim()
    options = dict(initial_rings=1, aperture_inset=".001", screens=[], max_nodes=100,
                   max_triangles=100, max_depth=8)
    result = build_source_mesh(sim, ["0", "0", "-1"], options)
    assert result["stats"]["accepted_triangles"] == 8
    assert not result["unresolved"]
    assert all(triangle["maslov"] == 0 for triangle in result["triangles"])
    stat = result["stats"]["bores"][0]
    assert abs(D(stat["partition_balance_defect_m2"])) < D("1e-65")
    assert D(stat["aperture_deficit_m2"]) > 0
    assert not result["stats"]["accuracy_validated"]
    partial = build_source_mesh(sim, ["0", "0", "-1"], {**options, "max_nodes": 2})
    stat = partial["stats"]["bores"][0]
    assert partial["stats"]["nodes"] == 2
    assert len(partial["unresolved"]) == 8
    assert not partial["triangles"]
    assert stat["unresolved_area_m2"] == stat["partition_area_m2"]
    assert stat["unresolved_reasons"] == {"node_budget": 8}
    without_nodes = build_source_mesh(sim, ["0", "0", "-1"], options, retain_nodes=False)
    assert without_nodes["triangles"] == result["triangles"]
    assert not without_nodes["nodes"]
    assert without_nodes["stats"]["nodes"] == result["stats"]["nodes"] > 0
    assert without_nodes["stats"]["retained_nodes"] == 0


def test_fast_transport_does_not_change_the_saved_field_model():
    pytest.importorskip("formula._stage19_transport")
    sim = _small_sim()
    options = dict(initial_rings=1, aperture_inset=".001", screens=[], max_nodes=100,
                   max_triangles=100, max_depth=8)
    slow = build_source_mesh(sim, ["2e-6", "-3e-6", "-1"], options, retain_nodes=False)
    fast = build_source_mesh(sim, ["2e-6", "-3e-6", "-1"],
                             {**options, "transport_backend": "interval"}, retain_nodes=False)
    assert slow["triangles"]
    assert fast["triangles"] == slow["triangles"]
    assert fast["unresolved"] == slow["unresolved"]
    assert fast["stats"]["bores"][0]["transport_backends"] == {
        "analytic_free": fast["stats"]["nodes"]}


def test_raw_trace_is_pending_until_full_get(monkeypatch):
    from formula.capsysred.stages import _s19_mesh as module
    original = module.transport_ray
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "transport_ray", counted)
    sim = _small_sim()
    with localcontext() as context:
        context.prec = 64
        cache = _TraceCache(sim, (D(0), D(0), D(-1)), sim.cfg.capillary.bores[0], DEFAULTS, 64)
        q = (D(0), D(0))
        raw = cache.get_raw(q)
        assert raw["valid"] is None
        assert raw["reason"] == "physics_pending"
        assert raw["physics_status"] == "pending"
        assert raw["maslov"] is None
        assert "fresnel" not in raw
        assert not calls and len(cache.pending_rays) == 1
        complete = cache.get(q)
        assert complete is raw and complete["valid"] is True
        assert complete["physics_status"] == "computed" and complete["reason"] == "ok"
        assert complete["maslov"] == 0
        assert cache.get(q) is complete and len(calls) == 1
        assert not cache.pending_rays and dict(cache.reasons) == {"ok": 1}


def test_lazy_phase_rejection_does_not_evaluate_transport(monkeypatch):
    from formula.capsysred.stages import _s19_mesh as module
    original = module.transport_ray
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "transport_ray", counted)
    sim = _small_sim()
    options = dict(DEFAULTS, lazy_physics=True)
    with localcontext() as context:
        context.prec = 64
        cache = _TraceCache(sim, (D(0), D(0), D(-1)), sim.cfg.capillary.bores[0], options, 64)
        points, faces = disk_partition((0, 0), "1e-5", rings=1, inset=".001")
        tri = tuple(points[v] for v in faces[0])
        cosine = D(str(Number("cos(pi/9)", 64)))
        lazy, reason = _check(tri, cache, options, D("1e20"), [], cosine*cosine)
        assert lazy is None and reason == "exit_phase"
        assert not calls
        assert all(row["valid"] is None for row in cache.records.values())
        eager, reason = _check_eager(tri, cache, options, D("1e20"), [], cosine*cosine)
        assert eager is None and reason == "exit_phase"
        assert calls


def test_lazy_survivor_still_checks_maslov():
    class SyntheticCache:
        z0, z1, origin = D(0), D(1), (D(0), D(0), D(-1))

        def get_raw(self, q):
            return dict(id=0, fate="screen", valid=None, reason="physics_pending",
                        q=q, point=(*q, D(1)), direction=(D(0), D(0), D(1)),
                        reflections=0, source_distance=D(1), uz_in=D(1), opl=D(1))

        def get(self, q):
            return dict(self.get_raw(q), valid=True, reason="ok", determinant=D(1),
                        fresnel=(D(1), D(0)), fresnel_each=[],
                        maslov=1 if q == (D(".5"), D(0)) else 0)

    triangle = ((D(0), D(0)), (D(1), D(0)), (D(0), D(1)))
    options = dict(DEFAULTS, lazy_physics=True)
    cosine = D(str(Number("cos(pi/9)", 64)))
    model, reason = _check(triangle, SyntheticCache(), options, D(1), [], cosine*cosine)
    assert model is None and reason == "mixed_history"


def test_completed_lazy_mesh_has_identical_physical_elements():
    sim = _small_sim()
    options = dict(initial_rings=1, aperture_inset=".001", screens=[], max_nodes=100,
                   max_triangles=100, max_depth=8)
    source = ["2e-6", "-3e-6", "-1"]
    eager = build_source_mesh(sim, source, options)
    lazy = build_source_mesh(sim, source, {**options, "lazy_physics": True})
    assert eager["triangles"] and not eager["unresolved"] and not lazy["unresolved"]
    assert lazy["triangles"] == eager["triangles"]
    assert all(node["valid"] is True for node in lazy["nodes"])
    assert lazy["stats"]["physics_pending_nodes"] == 0
    with pytest.raises(ValueError, match="lazy_physics must be boolean"):
        build_source_mesh(sim, source, {**options, "lazy_physics": "yes"})


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("check_individual", [False, True])
@pytest.mark.parametrize("changed_product", [False, True])
def test_fresnel_product_gate_with_unequal_individual_phases(lazy, check_individual, changed_product):
    class SyntheticCache:
        z0, z1, origin = D(0), D(1), (D(0), D(0), D(-1))

        def get_raw(self, q):
            return dict(id=0, fate="screen", valid=None, reason="physics_pending",
                        q=q, point=(*q, D(1)), direction=(D(0), D(0), D(1)),
                        reflections=2, source_distance=D(1), uz_in=D(1), opl=D(1))

        def get(self, q):
            coefficients = [(D(".8"), D(0)), (D(".8"), D(0))]
            if q == (D(".5"), D(0)):
                coefficients = [(D(0), D(".8")),
                                (D(0), D(".8") if changed_product else D("-.8"))]
            (a, b), (c, d) = coefficients
            return dict(self.get_raw(q), valid=True, reason="ok", determinant=D(1),
                        fresnel=(a*c-b*d, a*d+b*c), fresnel_each=coefficients, maslov=0)

    triangle = ((D(0), D(0)), (D(1), D(0)), (D(0), D(1)))
    options = dict(DEFAULTS, lazy_physics=lazy, check_individual_fresnel=check_individual)
    cosine = D(str(Number("cos(pi/9)", 64)))
    model, reason = _check(triangle, SyntheticCache(), options, D(1), [], cosine*cosine)
    if check_individual:
        assert model is None and reason == "fresnel_each"
    elif changed_product:
        assert model is None and reason == "fresnel_product"
    else:
        assert reason is None and model["amplitude"] == (D(".64"), D(0))
        assert _record(model, 0)["fresnel_diagnostics"] == {
            "individual_exceedances": 2, "individual_check_enforced": False}


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_individual_fresnel_option_requires_boolean(invalid):
    with pytest.raises(ValueError, match="check_individual_fresnel must be boolean"):
        build_source_mesh(_small_sim(), ["0", "0", "-1"],
                          {"check_individual_fresnel": invalid})


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_point_field_gate_detects_hidden_thin_map_variation(lazy, nonlinear):
    class SyntheticCache:
        z0, z1, origin = D(0), D(1), (D(0), D(0), D(-1))

        def get_raw(self, q):
            x, y = q
            shift = y*(y-D(".5"))*(y-1) if nonlinear else D(0)
            return dict(id=0, fate="screen", valid=None, reason="physics_pending", q=q,
                        point=(x, D(".01")*(y+shift), D(1)),
                        direction=(D(0), D(0), D(1)), reflections=0,
                        source_distance=D(1), uz_in=D(1), opl=D(1))

        def get(self, q):
            y = q[1]
            derivative = 1+3*y*y-3*y+D(".5") if nonlinear else D(1)
            return dict(self.get_raw(q), valid=True, reason="ok", maslov=0,
                        fresnel=(D(1), D(0)), fresnel_each=[], determinant=D(".01")*derivative)

    with localcontext() as context:
        context.prec = 64
        triangle = ((D(0), D(0)), (D(1), D(0)), (D(0), D(1)))
        cache = SyntheticCache()
        options = dict(DEFAULTS, lazy_physics=lazy, eps_A=".05", eps_rho=".05")
        model, reason = _model(triangle, cache, D("1e-16"))
        assert reason is None and model["jacobian"] == D(".01")
        for child in _children(triangle):
            small, reason = _model(child, cache, D("1e-16"))
            assert reason is None and small["jacobian"] == model["jacobian"]
            assert small["amplitude"] == model["amplitude"]
        cosine = D(str(Number("cos(pi/9)", 64)))
        checked, reason = _check(triangle, cache, options, D(1), [], cosine*cosine)
        if nonlinear:
            assert checked is None and reason == "point_amplitude"
        else:
            assert reason is None and checked["amplitude"] == (D(10), D(0))
            assert all(value == 0 for value in checked["point_field_max"].values())


def test_point_density_gate_is_independent_and_retains_complex_amplitude():
    options = dict(DEFAULTS, eps_A=".05", eps_rho=".05")
    model = dict(amplitude=(D(0), D(-1)), jacobian=D(1))
    node = dict(determinant=D("1.21"), fresnel=(D("1.1"), D(0)), maslov=1,
                source_distance=D(1), uz_in=D(1), direction=(D(0), D(0), D(1)))
    checks = _point_field_checks(node, model, options, D(1))
    assert checks["amplitude"][0] is True
    assert checks["density"][0] is False
    node["fresnel"] = (D(0), D("1.1"))
    assert _point_field_checks(node, model, options, D(1))["amplitude"][0] is False
    node["fresnel"] = (D("1e-9"), D(0))
    model["amplitude"] = (D(0), D(0))
    assert all(result[0] is True for result in _point_field_checks(node, model, options, D(1)).values())


def test_determinant_enclosure_does_not_accept_only_matching_endpoints():
    with localcontext() as context:
        context.prec = 64
        reference = (D(1), D(0))
        direction = (D(2499)/2501, D(100)/2501)
        low = tuple(D(".9995")*v for v in direction)
        high = tuple(D("1.0005")*v for v in direction)
        epsilon, absolute = D(".03999"), D("1e-12")
        assert _complex_close(low, reference, epsilon, absolute)
        assert _complex_close(high, reference, epsilon, absolute)
        assert not _complex_close(direction, reference, epsilon, absolute)
        assert _complex_enclosure(low, high, reference, epsilon, absolute)[0] is None


@pytest.mark.parametrize("exact_jacobian,expected", [("1", None), ("1.2", "point_amplitude")])
def test_ambiguous_point_determinant_requests_exact_transport(exact_jacobian, expected):
    class SyntheticCache:
        z0, z1, origin = D(0), D(1), (D(0), D(0), D(-1))

        def __init__(self):
            self.records, self.refined = {}, []

        def get(self, q):
            if q not in self.records:
                self.records[q] = dict(id=len(self.records), valid=True, reason="ok", q=q,
                    point=(*q, D(1)), direction=(D(0), D(0), D(1)), reflections=0,
                    source_distance=D(1), uz_in=D(1), opl=D(1), maslov=0,
                    fresnel=(D(1), D(0)), fresnel_each=[], determinant=D(1),
                    determinant_bounds=(D(".8"), D("1.2")))
            return self.records[q]

        def refine_determinant(self, q):
            self.refined.append(q)
            value = D(exact_jacobian)
            self.records[q].update(determinant=value, determinant_bounds=(value, value))
            return self.records[q]

    triangle = ((D(0), D(0)), (D(1), D(0)), (D(0), D(1)))
    cache = SyntheticCache()
    cosine = D(str(Number("cos(pi/9)", 64)))
    model, reason = _check(triangle, cache, DEFAULTS, D(1), [], cosine*cosine)
    assert reason == expected
    assert (model is None) == (expected is not None)
    assert cache.refined


def test_point_field_bounds_handle_orientation_and_invalid_certificates():
    node = dict(determinant=D(-1), determinant_bounds=(D("-1.001"), D("-.999")),
                fresnel=(D(1), D(0)), maslov=0, source_distance=D(1), uz_in=D(1),
                direction=(D(0), D(0), D(1)))
    model = dict(amplitude=(D(1), D(0)), jacobian=D(1))
    assert all(result[0] is True for result in _point_field_checks(node, model, DEFAULTS, D(1)).values())
    for bounds in ((D(-1), D(1)), (D(1), D(-1)), (D(0), D(1)),
                   (D("1e-17"), D(1)), (D("NaN"), D(1)), (None, None)):
        node["determinant_bounds"] = bounds
        assert all(result[0] is None for result in _point_field_checks(node, model, DEFAULTS, D(1)).values())


@pytest.mark.parametrize("mismatch", [None, "maslov", "determinant"])
def test_real_exact_field_fallback_retraces_without_adding_a_node(monkeypatch, mismatch):
    from formula.capsysred.stages import _s19_mesh as module
    sim = _small_sim()
    with localcontext() as context:
        context.prec = 64
        cache = _TraceCache(sim, (D(0), D(0), D(-1)), sim.cfg.capillary.bores[0], DEFAULTS, 64)
        q = (D(0), D(0))
        node = cache.get(q)
        determinant = node["determinant"]
        node["determinant_bounds"] = (determinant*D(".999"), determinant*D("1.001"))
        original = module.transport_ray

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if mismatch == "maslov":
                result["maslov"] += 1
            if mismatch == "determinant":
                result["determinant"] *= 2
            return result

        monkeypatch.setattr(module, "transport_ray", changed)
        refined = cache.refine_determinant(q)
        assert refined is node and len(cache.records) == 1
        assert cache.determinant_retraces == 1
        assert cache.transport_backends["decimal_field_fallback"] == 1
        if mismatch is None:
            assert refined["valid"] and refined["determinant"] == determinant
            assert refined["determinant_bounds"] == (determinant, determinant)
        else:
            assert refined["valid"] is False
            assert refined["reason"] == "determinant_refinement_failure"


@pytest.mark.parametrize("lazy", [False, True])
def test_physical_exit_phase_does_not_penalize_interior_reparameterization(lazy):
    class SyntheticCache:
        z0, z1, origin = D(0), D(1), (D(0), D(0), D(-1))

        def get_raw(self, q):
            x, y = q
            output = x+D(".1")*x*y*(1-x-y), y
            return dict(id=0, fate="screen", valid=None, reason="physics_pending", q=q,
                        point=(*output, D(1)), direction=(D(".01"), D(0), D(1)),
                        reflections=0, source_distance=D(1), uz_in=D(1), opl=1+D(".01")*output[0])

        def get(self, q):
            x, y = q
            return dict(self.get_raw(q), valid=True, reason="ok", maslov=0,
                        fresnel=(D(1), D(0)), fresnel_each=[], determinant=1+D(".1")*y*(1-2*x-y))

    with localcontext() as context:
        context.prec = 64
        triangle = ((D(0), D(0)), (D(1), D(0)), (D(0), D(1)))
        options = dict(DEFAULTS, lazy_physics=lazy, eps_A=".05", eps_rho=".05")
        cosine = D(str(Number("cos(pi/9)", 64)))
        cache = SyntheticCache()
        model, reason = _check(triangle, cache, options, D(10000), [], cosine*cosine)
        assert model is None and reason == "exit_phase"
        screens = [dict(z="2", edge_x="100", edge_y="100")]
        model, reason = _check(triangle, cache, dict(options, phase_coordinates="physical_exit"),
                               D(10000), screens, cosine*cosine)
        assert reason is None and model["phase_max"] < D("1e-55")
        assert model["geometry_max"] > 0
        record = _record(model, 0)
        assert record["phase_coordinates"] == "physical_exit"
        assert record["phase_diagnostics"]["screens"] == []
        assert record["domain_indicator_certified"] is False


@pytest.mark.parametrize("lazy", [False, True])
def test_physical_exit_rejects_collinear_midpoint_backtracking(lazy):
    class SyntheticCache:
        z0, z1, origin = D(0), D(1), (D(0), D(0), D(-1))

        def get_raw(self, q):
            x, y = q
            output = (D(0), D(".0011")) if q == (D(0), D(".5")) else (x, y*D(".001"))
            return dict(id=0, fate="screen", valid=None, reason="physics_pending", q=q,
                        point=(*output, D(1)), direction=(D(0), D(0), D(1)),
                        reflections=0, source_distance=D(1), uz_in=D(1), opl=D(1))

        def get(self, q):
            return dict(self.get_raw(q), valid=True, reason="ok", maslov=0,
                        fresnel=(D(1), D(0)), fresnel_each=[], determinant=D(".001"))

    triangle = ((D(0), D(0)), (D(1), D(0)), (D(0), D(1)))
    cosine = D(str(Number("cos(pi/9)", 64)))
    model, reason = _check(triangle, SyntheticCache(),
        dict(DEFAULTS, lazy_physics=lazy, phase_coordinates="physical_exit"), D(1), [], cosine*cosine)
    assert model is None and reason in ("domain_non_simple", "domain_fan_orientation")


def test_optional_target_preserves_records_and_area_balance():
    options = dict(initial_rings=1, aperture_inset=".001", screens=[], max_nodes=100,
                   max_triangles=100, max_depth=8, phase_coordinates="physical_exit")
    source = ["0", "0", "-1"]
    complete = build_source_mesh(_small_sim(), source, options)
    partial = build_source_mesh(_small_sim(), source, dict(options, target_accepted_fraction=".4"))
    stat = partial["stats"]["bores"][0]
    assert stat["target_reached"] and D(stat["accepted_fraction"]) >= D(".4")
    assert stat["domain_indicator"]["passed"]
    assert stat["unresolved_reasons"] == {"coverage_target": 4}
    assert partial["triangles"] == complete["triangles"][:4]
    assert partial["stats"]["nodes"] < complete["stats"]["nodes"]
    assert abs(D(stat["partition_balance_defect_m2"])) < D("1e-65")
    assert D(stat["incremental_area_defect_m2"]) == 0


def test_target_subtracts_removed_accepted_neighbor(monkeypatch):
    from formula.capsysred.stages import _s19_mesh as module
    points = [(D(0), D(0)), (D(1), D(0)), (D(1), D(1)), (D(0), D(1))]
    monkeypatch.setattr(module, "disk_partition", lambda *args, **kwargs: (points, [(0, 1, 2), (0, 2, 3)]))
    calls = []

    def check(triangle, *args):
        calls.append(triangle)
        return ((None, "synthetic_refinement") if len(calls) == 2
                else (dict(input=triangle, input_area=abs(signed_area(triangle))), None))

    monkeypatch.setattr(module, "_check", check)
    monkeypatch.setattr(module, "_record", lambda model, bore: model)
    sim = _small_sim()
    sim.cfg.capillary.bores[0]["radius"] = "1"
    result = build_source_mesh(sim, ["0", "0", "-1"], dict(screens=[], target_accepted_fraction=".2"))
    stat = result["stats"]["bores"][0]
    assert stat["target_reached"] and stat["accepted_triangles"] == 3
    assert D(stat["accepted_area_m2"]) == D(".75")
    assert D(stat["unresolved_area_m2"]) == D(".25")
    assert D(stat["incremental_area_defect_m2"]) == 0
    assert D(stat["partition_balance_defect_m2"]) == 0


def test_budget_drains_cached_faces_once_without_splitting(monkeypatch):
    from formula.capsysred.stages import _s19_mesh as module
    checked = []

    def check(triangle, cache, *args):
        checked.append(triangle)
        cache.get_raw((D(0), D(0)))
        if len(checked) == 1:
            cache.get_raw((D(".001"), D(0)))
        if len(checked) == 2:
            return dict(input=triangle, input_area=abs(signed_area(triangle))), None
        return None, "synthetic_physics_rejection"

    def forbidden_split(*args):
        pytest.fail("budget drain must never refine the partition")

    monkeypatch.setattr(module, "_check", check)
    monkeypatch.setattr(module, "_record", lambda model, bore: model)
    monkeypatch.setattr(module._Partition, "split", forbidden_split)
    result = build_source_mesh(_small_sim(), ["0", "0", "-1"],
                               dict(initial_rings=1, max_nodes=1, screens=[]))
    stat = result["stats"]["bores"][0]
    assert len(checked) == 8 and len(set(checked)) == 8
    assert stat["nodes"] == 1 and stat["splits"] == 0
    assert stat["budget_drain_checked"] == 7 and stat["budget_drain_accepted"] == 1
    assert stat["accepted_triangles"] == 1 and stat["unresolved_triangles"] == 7
    assert stat["unresolved_reasons"] == {"node_budget": 1, "node_budget:synthetic_physics_rejection": 6}
    assert abs(D(stat["partition_balance_defect_m2"])) < D("1e-65")


@pytest.mark.parametrize("option,value", [("phase_coordinates", "other"), ("target_accepted_fraction", True),
    ("target_accepted_fraction", "0"), ("target_accepted_fraction", "1.01"),
    ("target_accepted_fraction", "NaN"), ("domain_relative", "0"), ("domain_relative", "Infinity")])
def test_physical_exit_options_are_explicit_and_finite(option, value):
    with pytest.raises(ValueError, match=option):
        build_source_mesh(_small_sim(), ["0", "0", "-1"], {option: value})


@pytest.mark.parametrize("retain_nodes", [False, True])
def test_final_domain_check_releases_trace_cache_and_partition(monkeypatch, retain_nodes):
    import weakref
    from formula.capsysred.stages import _s19_mesh as module
    cache_type, partition_type = module._TraceCache, module._Partition
    indicator = module.domain_indicator
    references, calls = {}, []

    def new_cache(*args, **kwargs):
        cache = cache_type(*args, **kwargs)
        references["cache"] = weakref.ref(cache)
        return cache

    def new_partition(*args, **kwargs):
        partition = partition_type(*args, **kwargs)
        references["partition"] = weakref.ref(partition)
        return partition

    def check_domain(*args, **kwargs):
        assert references["cache"]() is None
        assert references["partition"]() is None
        calls.append(True)
        return indicator(*args, **kwargs)

    monkeypatch.setattr(module, "_TraceCache", new_cache)
    monkeypatch.setattr(module, "_Partition", new_partition)
    monkeypatch.setattr(module, "domain_indicator", check_domain)
    result = build_source_mesh(_small_sim(), ["0", "0", "-1"],
        dict(initial_rings=1, screens=[], phase_coordinates="physical_exit", progress=True),
        retain_nodes=retain_nodes)
    assert calls == [True]
    assert result["stats"]["bores"][0]["domain_indicator"]["passed"]
    assert result["stats"]["nodes"] > 0
    assert len(result["nodes"]) == (result["stats"]["nodes"] if retain_nodes else 0)
    assert result["stats"]["retained_nodes"] == len(result["nodes"])

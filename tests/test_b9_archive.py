from types import SimpleNamespace

import numpy as np
import pytest

from formula.capsysred.stages._b9_archive import apply_tube_flux, exit_nodes, prepare_exit_mesh, _p1_triangle_power
from formula.capsysred.stages._b5_transport import count_free_caustics, reflect_variations


def _synthetic_nodes(seed=8, count=300):
    rng = np.random.default_rng(seed)
    entrance = rng.uniform(-.9, .9, (count, 2))
    mapping = np.array([[1.2, .15], [-.1, .9]])
    points = entrance@mapping.T+np.array([.1, -.2])
    gradient = np.array([2.3, -.8])
    phase = points@gradient+1.7
    amplitude = 2.+.1*points[:, 0]+.2j*points[:, 1]
    nodes = dict(entrance=entrance, points=points, phase=phase, amplitude=amplitude,
                 directions=np.tile(gradient/10, (count, 1)), bore=np.zeros(count, int),
                 determinant=np.full(count, np.linalg.det(mapping)), maslov=np.zeros(count, int),
                 reflections=np.zeros(count, int), valid=np.ones(count, bool),
                 ray_ids=np.arange(count), k=10., determinant_floor=1e-12, transport={})
    cap = SimpleNamespace(bores=[dict(center=[0., 0.], radius=2.)])
    return nodes, cap


def test_exit_backprojection_and_absolute_amplitude_match_spherical_wave():
    source = np.array([.01, -.02, -2.])
    entrance = np.array([[.1, .2], [-.2, .3]])
    distance = np.sqrt(np.sum((entrance-source[:2])**2, axis=1)+4)
    u = np.column_stack([entrance-source[:2], np.full(2, 2.)])/distance[:, None]
    k, mode_z, exit_z = 100., 5., 1.
    mode_distance = (mode_z-source[2])/u[:, 2]
    mode = dict(origin=source, points=source[:2]+mode_distance[:, None]*u[:, :2],
                directions=u[:, :2], phase_opl=k*(mode_z-source[2])*(1/u[:, 2]-1),
                refl=[[], []], ray_ids=np.arange(2))
    cap = SimpleNamespace(z0=0., bores=[dict(center=[0., 0.], radius=1.)])
    result = exit_nodes(mode, cap, mode_z=mode_z, exit_z=exit_z, k=k, fresnel=[1., .7+.2j])
    true_distance = (exit_z-source[2])/u[:, 2]
    assert np.all(result["valid"])
    np.testing.assert_allclose(result["points"], source[:2]+true_distance[:, None]*u[:, :2], atol=2e-16)
    np.testing.assert_allclose(result["phase"], k*(exit_z-source[2])*(1/u[:, 2]-1), atol=8e-14)
    np.testing.assert_allclose(result["amplitude"], np.array([1., .7+.2j])/true_distance, rtol=4e-16)
    np.testing.assert_allclose(result["Q"], np.tile(1.5*np.eye(2), (2, 1, 1)), atol=8e-16)


def test_affine_exit_mesh_preserves_phase_and_unequal_complex_amplitudes():
    nodes, cap = _synthetic_nodes()
    result = prepare_exit_mesh(nodes, cap, holdout_stride=5)
    np.testing.assert_allclose(result["gradient"], np.tile([2.3, -.8], (len(result["triangles"]), 1)), atol=2e-12)
    np.testing.assert_allclose(result["vertex_amplitude"], nodes["amplitude"][result["ray_indices"]])
    holdout = result["metadata"]["holdout"]
    assert holdout["checked_count"] > 40
    assert holdout["relative_complex_field_rms"] < 3e-15
    assert holdout["phase_rad"]["max"] < 1e-14
    assert holdout["geometry_m"]["max"] < 1e-15
    assert 0 < result["metadata"]["accepted_entrance_area_fraction"] < 1


def test_mesh_does_not_join_reflection_families_or_fill_missing_area():
    nodes, cap = _synthetic_nodes()
    nodes["reflections"] = (nodes["entrance"][:, 0] > 0).astype(int)
    full = prepare_exit_mesh(nodes, cap, holdout_stride=0)
    history = nodes["reflections"][full["ray_indices"]]
    assert np.all(history == history[:, :1])
    metadata = full["metadata"]
    assert metadata["rejection_counts"]["mixed_reflections"] > 0
    assert metadata["accepted_entrance_area_fraction"] < metadata["convex_hull_area_fraction"]


def test_exit_fold_and_invalid_transport_are_explicitly_excluded():
    nodes, cap = _synthetic_nodes()
    nodes["determinant"] *= -1
    result = prepare_exit_mesh(nodes, cap, holdout_stride=0)
    assert not len(result["triangles"])
    assert result["metadata"]["rejection_counts"]["folded_exit"] > 0
    assert result["metadata"]["accepted_entrance_area_fraction"] == 0
    nodes["determinant"] *= -1
    nodes["valid"][:] = False
    result = prepare_exit_mesh(nodes, cap, holdout_stride=0)
    assert not len(result["triangles"])
    assert result["metadata"]["rejection_counts"]["invalid_vertex"] > 0


def test_holdout_detects_nonlinear_phase_and_respects_prefix_budget():
    nodes, cap = _synthetic_nodes(count=900)
    nodes["phase"] += 8*np.sum(nodes["points"]**2, axis=1)
    coarse = prepare_exit_mesh(nodes, cap, budget=150)
    fine = prepare_exit_mesh(nodes, cap, budget=900)
    assert coarse["ray_indices"].max() < 150
    assert fine["metadata"]["holdout"]["phase_rad"]["rms"] < coarse["metadata"]["holdout"]["phase_rad"]["rms"]
    assert coarse["metadata"]["holdout"]["relative_complex_field_rms"] > .02


def test_backprojection_cannot_skip_a_reflection_or_source_plane():
    cap = SimpleNamespace(z0=0., bores=[dict(center=[0., 0.], radius=1.)])
    mode = dict(origin=[0., 0., -1.], points=np.array([[0., 0.]]), directions=np.array([[0., 0.]]),
                phase_opl=np.array([0.]), refl=[[[1., 0., .8]]])
    with pytest.raises(ValueError, match="across an archived reflection"):
        exit_nodes(mode, cap, mode_z=1., exit_z=.5, k=10., fresnel=1.)
    with pytest.raises(ValueError, match="exit must follow"):
        exit_nodes(mode, cap, mode_z=1., exit_z=0., k=10., fresnel=1.)


def test_one_and_two_planar_reflections_match_unfolded_image_source(monkeypatch):
    source, exit_z, mode_z, k = np.array([.03, .04, -2.]), 12., 15., 7.
    entrance = np.array([[.3, .1], [.6, -.2]])
    paths, positions, directions, phases, initial_vectors = [], [], [], [], []
    for q in entrance:
        point = np.r_[q, 0.]
        initial = (point-source)/np.linalg.norm(point-source)
        initial_vectors.append(initial)
        u, path = initial.copy(), []
        while True:
            wall = np.copysign(1., u[0])
            travel = (wall-point[0])/u[0]
            if point[2]+travel*u[2] > exit_z:
                break
            point = point+travel*u
            path.append(point.copy().tolist())
            u[0] *= -1
        point = point+(mode_z-point[2])/u[2]*u
        paths.append(path)
        positions.append(point[:2])
        directions.append(u[:2])
        phases.append(k*(mode_z-source[2])*(1/u[2]-1))
    assert [len(path) for path in paths] == [1, 2]
    mode = dict(origin=source, points=np.array(positions), directions=np.array(directions),
                phase_opl=np.array(phases), refl=paths)
    cap = SimpleNamespace(z0=0., bores=[dict(center=[0., 0.], radius=1.)])

    def planar_transport(exit_mode, cap, z):
        qs, ps, maslov = [], [], []
        for i, path in enumerate(paths):
            u = initial_vectors[i].copy()[None]
            point = np.r_[entrance[i], 0.]
            r = np.eye(3)[:, :2][None].copy()
            distance = np.linalg.norm(point-source)
            du = (r-u[:, :, None]*u[:, None, :2])/distance
            focal = 0
            destinations = [np.array(p) for p in path]+[np.r_[exit_mode["points"][i], z]]
            for j, target in enumerate(destinations):
                dz = target[2]-point[2]
                slope = u[:, :2]/u[:, 2, None]
                v = (du[:, :2]-slope[:, :, None]*du[:, 2, None])/u[:, 2, None, None]
                count, ambiguous = count_free_caustics(r[:, :2], v, np.array([dz]))
                assert not ambiguous[0]
                focal += count[0]
                r += dz/u[:, 2, None, None]*du
                if j < len(path):
                    normal = np.array([[np.sign(target[0]), 0., 0.]])
                    u, r, du = reflect_variations(u, r, du, normal, np.zeros((1, 3, 3)))
                else:
                    r -= u[:, :, None]*(r[:, 2]/u[:, 2, None])[:, None]
                point = target
            qs.append(r[0, :2]); ps.append(du[0, :2]); maslov.append(focal)
        return dict(entrance=entrance, Q=np.array(qs), P=np.array(ps), maslov=np.array(maslov),
                    valid=np.ones(2, bool), bore=np.zeros(2, int), diagnostics={"fixture": "flat-wall derivative adapter"})

    monkeypatch.setattr("formula.capsysred.stages._b9_archive.transport_mode", planar_transport)
    fresnel = np.array([.8*np.exp(.2j), .6*np.exp(-.4j)])
    result = exit_nodes(mode, cap, mode_z=mode_z, exit_z=exit_z, k=k, fresnel=fresnel)
    unfolded_distances = (exit_z-source[2])/np.array(initial_vectors)[:, 2]
    np.testing.assert_array_equal(result["maslov"], [0, 0])
    np.testing.assert_allclose(result["determinant"], [-49., 49.], atol=3e-14)
    np.testing.assert_allclose(result["amplitude"], fresnel/unfolded_distances, rtol=7e-16)
    np.testing.assert_allclose(result["phase"], k*(unfolded_distances-(exit_z-source[2])), atol=1e-14)


@pytest.mark.parametrize("stride", [1, -1, True, 2.5])
def test_invalid_holdout_configuration(stride):
    nodes, cap = _synthetic_nodes()
    with pytest.raises(ValueError, match="holdout_stride"):
        prepare_exit_mesh(nodes, cap, holdout_stride=stride)


def _physical_affine_mesh():
    nodes, cap = _synthetic_nodes(count=100)
    rng = np.random.default_rng(21)
    nodes["uz0"] = rng.uniform(.8, 1., len(nodes["points"]))
    nodes["uzexit"] = rng.uniform(.8, 1., len(nodes["points"]))
    nodes["source_distance"] = rng.uniform(1., 2., len(nodes["points"]))
    nodes["fresnel"] = rng.uniform(.3, 1., len(nodes["points"]))*np.exp(1j*rng.uniform(-1., 1., len(nodes["points"])))
    nodes["amplitude"] = (nodes["fresnel"]/nodes["source_distance"]
        *np.sqrt(nodes["uz0"]/nodes["uzexit"]/np.abs(nodes["determinant"])))
    return nodes, prepare_exit_mesh(nodes, cap, holdout_stride=0)


def test_tube_flux_preserves_affine_mapping_with_unequal_fresnel_and_obliquity():
    nodes, mesh = _physical_affine_mesh()
    corrected = apply_tube_flux(mesh, nodes)
    np.testing.assert_allclose(corrected["vertex_amplitude"], mesh["vertex_amplitude"], rtol=1e-12, atol=2e-16)
    record = corrected["metadata"]["tube_flux"]
    assert record["maximum_relative_patch_flux_defect"] < 1e-15
    np.testing.assert_allclose(record["scale_statistics"]["median"], 1., atol=3e-16)


def test_tube_flux_matches_each_patch_and_preserves_phase_geometry_and_original():
    nodes, mesh = _physical_affine_mesh()
    mesh["vertex_amplitude"][:, 0] *= 12
    original = mesh["vertex_amplitude"].copy()
    corrected = apply_tube_flux(mesh, nodes)
    ids = mesh["ray_indices"]
    pin = _p1_triangle_power(nodes["fresnel"][ids]*np.sqrt(nodes["uz0"][ids])/nodes["source_distance"][ids], mesh["entrance_area"])
    pout = _p1_triangle_power(corrected["vertex_amplitude"]*np.sqrt(nodes["uzexit"][ids]), mesh["exit_area"])
    np.testing.assert_allclose(pin, pout, rtol=8e-16, atol=0)
    np.testing.assert_array_equal(mesh["vertex_amplitude"], original)
    for name in ("phase", "vertex_phase", "gradient", "triangles", "entrance_triangles", "ray_indices"):
        np.testing.assert_array_equal(corrected[name], mesh[name])
    assert corrected["metadata"]["accepted_entrance_area_fraction"] == mesh["metadata"]["accepted_entrance_area_fraction"]
    assert corrected["metadata"]["tube_flux"]["scale_statistics"]["max"] < 1


def test_tube_flux_zero_source_stays_zero_and_impossible_rescaling_is_rejected():
    nodes, mesh = _physical_affine_mesh()
    mesh["vertex_amplitude"][:] = 0
    with pytest.raises(ValueError, match="positive entrance flux"):
        apply_tube_flux(mesh, nodes)
    nodes["fresnel"][:] = 0
    corrected = apply_tube_flux(mesh, nodes)
    np.testing.assert_array_equal(corrected["vertex_amplitude"], 0.)
    assert corrected["metadata"]["tube_flux"]["maximum_relative_patch_flux_defect"] == 0


def test_equal_flux_does_not_imply_correct_complex_field():
    nodes, mesh = _physical_affine_mesh()
    wrong = dict(mesh, vertex_amplitude=-mesh["vertex_amplitude"])
    corrected = apply_tube_flux(wrong, nodes)
    np.testing.assert_allclose(corrected["vertex_amplitude"], -mesh["vertex_amplitude"], rtol=1e-12, atol=2e-16)
    relative_error = np.linalg.norm(corrected["vertex_amplitude"]-mesh["vertex_amplitude"])/np.linalg.norm(mesh["vertex_amplitude"])
    np.testing.assert_allclose(relative_error, 2., atol=3e-16)

"""Independent geometry and physical controls for the prescribed cylinder mesh."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from formula.capsysred.stages._b9_archive import mesh_from_triangles, prepare_exit_mesh
from formula.capsysred.stages._b9_cylinder_mesh import _entrance_mesh, cylinder_exit_mesh


SOURCE = ['-1.9850001681262593E-7', '-7.062065632157497E-7', '-0.675']
BORE = dict(center=[0., 0.], radius=24e-6)
K = 40785096803.62667


def geometry(source=SOURCE, bore=BORE, **kwargs):
    options = dict(angles=16, inner_rings=2, outer_rings=2,
                   boundary_relative_gap=1e-6, entrance_relative_inset=2e-6)
    options.update(kwargs)
    return _entrance_mesh(source, bore, 0., .23, **options)


def traced(source=SOURCE, bore=BORE, **kwargs):
    options = dict(angles=32, inner_rings=4, outer_rings=2)
    options.update(kwargs)
    return cylinder_exit_mesh(source, bore, 0., .23, K,
                               7.125763840840871e-6, 9.207952277589002e-8, **options)


def _signed_double_area(t):
    a, b = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
    return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]


def _positive_intersection(a, b, tolerance=1e-12):
    # Independent separating-axis test for convex triangles, in radius units.
    edges = np.vstack((np.roll(a, -1, axis=0)-a, np.roll(b, -1, axis=0)-b))
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    pa, pb = a@normals.T, b@normals.T
    return bool(np.all(np.minimum(pa.max(axis=0), pb.max(axis=0))
                       - np.maximum(pa.min(axis=0), pb.min(axis=0)) > tolerance))


def test_prescribed_faces_are_nonoverlapping_and_areas_close_with_explicit_gaps():
    q, faces, families, d = geometry(angles=8, inner_rings=2, outer_rings=1)
    triangles = q[faces]
    assert np.all(_signed_double_area(triangles) > 0)
    assert np.all(families[faces] == families[faces][:, :1])
    for i, triangle in enumerate(triangles/BORE['radius']):
        assert not any(_positive_intersection(triangle, other)
                       for other in triangles[:i]/BORE['radius'])
    actual = .5*np.sum(_signed_double_area(triangles))
    np.testing.assert_allclose(actual, d['accepted_area_m2'], rtol=5e-16)
    np.testing.assert_allclose(actual+d['family_boundary_band_area_m2']
                               +d['outer_chord_and_inset_deficit_m2'], d['aperture_area_m2'], rtol=3e-16)
    assert d['family_boundary_band_area_m2'] > 0
    # Internal edges match exactly; the only open edges are the three polygons.
    edges = np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1)
    _, count = np.unique(edges, axis=0, return_counts=True)
    assert count.max() == 2
    assert np.sum(count == 1) == 3*8


def test_boundary_is_fitted_on_both_sides_including_all_reflected_chords():
    q, faces, families, d = geometry()
    center = np.asarray(d['family_circle_center_m'])
    radius = d['family_circle_radius_m']
    assert np.max(np.linalg.norm(q[families == 0]-center, axis=1)) < radius
    reflected = q[faces[families[faces[:, 0]] == 1]]-center
    for start, stop in ((0, 1), (1, 2), (2, 0)):
        p, vector = reflected[:, start], reflected[:, stop]-reflected[:, start]
        t = np.clip(-np.sum(p*vector, axis=1)/np.sum(vector*vector, axis=1), 0, 1)
        assert np.min(np.linalg.norm(p+t[:, None]*vector, axis=1)) > radius
    assert np.max(np.linalg.norm(q-np.asarray(BORE['center']), axis=1)) < BORE['radius']
    _, _, _, fine = geometry(angles=32)
    assert fine['family_boundary_band_area_m2'] < d['family_boundary_band_area_m2']/3.9
    assert fine['outer_chord_and_inset_deficit_m2'] < d['outer_chord_and_inset_deficit_m2']/3.9


def test_shifted_center_and_offset_source_preserve_translated_topology():
    q, faces, families, d = geometry()
    shift = np.array([3.2e-4, -7.1e-4])
    source = np.asarray(SOURCE, float)
    source[:2] += shift
    other, other_faces, other_family, other_d = geometry(source, dict(center=shift, radius=BORE['radius']))
    np.testing.assert_allclose(other-shift, q, atol=1e-19, rtol=0)
    np.testing.assert_array_equal(other_faces, faces)
    np.testing.assert_array_equal(other_family, families)
    np.testing.assert_allclose(other_d['accepted_area_fraction'], d['accepted_area_fraction'], atol=5e-15)


def test_existing_mp_tracer_verifies_families_and_direct_spherical_phase_amplitude():
    mesh, d = traced(amplitude_mode='point_jacobian')
    direct = mesh['triangle_reflections'] == 0
    source = np.asarray(SOURCE, float)
    distance_z = .23-source[2]
    r2 = np.sum((mesh['triangles'][direct]-source[:2])**2, axis=-1)
    radius = np.sqrt(distance_z**2+r2)
    expected_phase = K*r2/(radius+distance_z)
    np.testing.assert_allclose(mesh['vertex_phase'][direct], expected_phase, atol=2e-13, rtol=2e-14)
    np.testing.assert_allclose(mesh['vertex_amplitude'][direct], 1/radius, atol=3e-15)
    np.testing.assert_allclose(mesh['vertex_directions'][direct],
                               (mesh['triangles'][direct]-source[:2])/radius[..., None], atol=1e-19)
    assert set(mesh['triangle_reflections']) == {0, 1}
    assert d['trace']['emitted_nodes'] == d['trace']['screen_nodes'] == 225
    assert d['source_origin_decimal'] == SOURCE
    assert d['mesh']['triangles'] == d['geometry']['entrance_triangles'] == 352
    assert sum(d['mesh']['rejection_counts'].values()) == 0
    json.dumps(d, allow_nan=False)


def test_tube_flux_preserves_phase_geometry_and_closes_branch_flux():
    point, _ = traced(amplitude_mode='point_jacobian')
    tube, d = traced(amplitude_mode='tube_flux')
    for key in ('triangles', 'entrance_triangles', 'vertex_phase', 'vertex_directions', 'ray_indices'):
        np.testing.assert_array_equal(tube[key], point[key])
    ratio = tube['vertex_amplitude']/point['vertex_amplitude']
    assert np.max(abs(ratio.imag)) < 3e-16
    assert np.all(ratio.real > 0)
    flux = d['mesh']['tube_flux']
    np.testing.assert_allclose(flux['corrected_p1_exit_branch_flux'], flux['input_p1_branch_flux'], rtol=4e-16)
    assert flux['maximum_relative_patch_flux_defect'] < 1e-14


def test_trace_supports_shifted_bore_and_nonzero_entrance_z():
    source = np.asarray(SOURCE, float)+np.array([2e-4, -3e-4, .5])
    mesh, d = cylinder_exit_mesh([repr(float(x)) for x in source],
        dict(center=[2e-4, -3e-4], radius=24e-6), .5, .73, K,
        7e-6, 9e-8, angles=16, inner_rings=2, outer_rings=1)
    assert set(mesh['triangle_reflections']) == {0, 1}
    assert d['trace']['screen_nodes'] == 65
    assert d['trace']['maximum_recovered_entrance_defect_m'] < 1e-18


@pytest.mark.parametrize('bore', [dict(BORE, kind='torus'), dict(BORE, bend={'radius': 1.}),
                                  dict(BORE, sides=6)])
def test_non_cylindrical_geometries_fail_before_trace(bore):
    with pytest.raises(ValueError, match='straight circular cylinder'):
        traced(bore=bore)


def test_clipped_or_insufficient_family_geometry_is_rejected():
    with pytest.raises(ValueError, match='source projection'):
        geometry([3e-5, 0, -.675])
    with pytest.raises(ValueError, match='does not fit'):
        geometry(boundary_relative_gap=.2, entrance_relative_inset=.2)
    with pytest.raises(ValueError, match='0/1-family model'):
        cylinder_exit_mesh(['0', '0', '-.1'], BORE, 0., 1., K, 7e-6, 9e-8,
                            angles=8, inner_rings=1, outer_rings=2)


def test_supplied_mesh_matches_existing_assembly_and_refuses_silent_filtering():
    rng = np.random.default_rng(81)
    q = rng.uniform(-.7, .7, (50, 2))
    count = len(q)
    nodes = dict(entrance=q, points=q*2, phase=q[:, 0], amplitude=np.ones(count, complex),
        directions=np.tile([.1, 0], (count, 1)), bore=np.zeros(count, int),
        determinant=np.full(count, 4.), maslov=np.zeros(count, int), reflections=np.zeros(count, int),
        valid=np.ones(count, bool), ray_ids=np.arange(count), k=5., determinant_floor=1e-12, transport={})
    cap = SimpleNamespace(bores=[dict(center=[0., 0.], radius=1.)])
    original = prepare_exit_mesh(nodes, cap, holdout_stride=0)
    supplied = mesh_from_triangles(nodes, cap, original['ray_indices'])
    for key in ('triangles', 'vertex_phase', 'vertex_amplitude', 'gradient', 'entrance_area', 'exit_area'):
        np.testing.assert_array_equal(supplied[key], original[key])
    nodes['reflections'][original['ray_indices'][0, 0]] = 1
    with pytest.raises(ValueError, match='mix reflections'):
        mesh_from_triangles(nodes, cap, original['ray_indices'])

"""Partition closure, independent probes and partial-status accounting."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from formula.capsysred.stages._b9_adaptive_mesh import (
    _Partition, _TraceCache, _bore_spec, _check, _det, _disk_partition,
    adaptive_exit_mesh, audit_amplitude_probes,
)


SOURCE = ['-1.9850001681262593E-7', '-7.062065632157497E-7', '-.675']
BORE = dict(center=[0., 0.], radius=24e-6)
K = 40785096803.62667


def build(bore=BORE, **kwargs):
    options = dict(angles=16, radial_rings=2, max_nodes=800, max_depth=7)
    options.update(kwargs)
    return adaptive_exit_mesh(SOURCE, bore, 0., .23, K, 7.125763840840871e-6, 9.207952277589002e-8, **options)


def _overlap(a, b):
    edges = np.vstack((np.roll(a, -1, axis=0)-a, np.roll(b, -1, axis=0)-b))
    normal = np.column_stack((-edges[:, 1], edges[:, 0]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    pa, pb = a@normal.T, b@normal.T
    return np.all(np.minimum(pa.max(axis=0), pb.max(axis=0))-np.maximum(pa.min(axis=0), pb.min(axis=0)) > 1e-12)


def test_bisection_closure_preserves_area_nonoverlap_and_has_no_hanging_vertices():
    points, faces = _disk_partition(np.zeros(2), 1., 8, 1, 1e-5)
    points = points.tolist()
    partition = _Partition(faces)
    for turn in range(12):
        edge = max(partition.edges, key=lambda ij: np.linalg.norm(np.asarray(points[ij[0]])-points[ij[1]]))
        midpoint = np.mean(np.asarray([points[i] for i in edge]), axis=0)
        index = len(points)
        points.append(midpoint.tolist())
        partition.split(edge, index)
    points = np.asarray(points)
    triangles = points[[leaf['vertices'] for leaf in partition.leaves.values()]]
    assert np.all([_det(t) > 0 for t in triangles])
    np.testing.assert_allclose(sum(.5*_det(t) for t in triangles), 4*(1-1e-5)**2*np.sin(np.pi/4), rtol=2e-15)
    for i, triangle in enumerate(triangles):
        assert not any(_overlap(triangle, other) for other in triangles[:i])
    for edge, adjacent in partition.edges.items():
        assert len(adjacent) in (1, 2)
        a, b = points[list(edge)]
        vector = b-a
        t = (points-a)@vector/(vector@vector)
        distance = np.linalg.norm(points-(a+t[:, None]*vector), axis=1)
        assert not np.any((t > 1e-12) & (t < 1-1e-12) & (distance < 1e-12))


class _SyntheticCache:
    def __init__(self):
        self.coordinates, self.records, self.lookup = [], [], {}
        self.cap = SimpleNamespace(bores=[dict(radius=1.)])
        self.floor, self.k = 1e-10, 1.

    def get(self, coordinates):
        ids = []
        for point in coordinates:
            q = tuple(map(float, point))
            if q not in self.lookup:
                x, y = q
                each = np.array([.8+.1j, .7+.2j])
                row = dict(points=np.array(q), phase=.03*(x*x+y*y)+.02*x*y,
                    amplitude=1+.1*x+.2j*y, directions=np.array([.06*x+.02*y,.06*y+.02*x]),
                    Q=np.eye(2), determinant=1., valid=True, reflections=2, maslov=0,
                    fresnel_each=each, fresnel=np.prod(each), uz0=1., uzexit=1.,
                    source_distance=abs(np.prod(each))/abs(1+.1*x+.2j*y), fate='screen')
                self.lookup[q] = len(self.records)
                self.coordinates.append(q)
                self.records.append(row)
            ids.append(self.lookup[q])
        return np.asarray(ids)


def _synthetic():
    cache = _SyntheticCache()
    vertices = tuple(cache.get([[0, 0], [1, 0], [0, 1]]))
    check = _check(cache, vertices, (1e-10, 1e-10, 1e-10, .01, .05), 'point_jacobian')
    return cache, vertices, check


def test_quadratic_phase_and_complex_affine_amplitude_pass_actual_exit_probes():
    cache, vertices, result = _synthetic()
    assert not result['reasons']
    assert len(cache.records) == 7
    assert result['phase_error_rad'] < 1e-16
    assert result['amplitude_relative_error'] < 1e-15
    assert result['geometry_relative_error'] < 1e-15
    assert result['triangle_area_jacobian_relative_defect'] == 0


@pytest.mark.parametrize('change,expected', [
    ('phase','phase_probe'), ('amplitude','amplitude_probe'), ('points','geometry_probe'),
    ('reflections','mixed_reflections'), ('maslov','mixed_maslov'), ('determinant','mixed_orientation'),
    ('valid','invalid_transport'),
])
def test_centroid_probe_detects_defects_missing_at_all_vertices(change, expected):
    cache, vertices, result = _synthetic()
    row = cache.records[result['probes'][-1]]
    if change in ('phase', 'amplitude'):
        row[change] += .2
    elif change == 'points':
        row['points'] = row['points']+np.array([.02, -.01])
    elif change == 'determinant':
        row[change] = -1.
    elif change == 'valid':
        row[change] = False
    else:
        row[change] += 1
    checked = _check(cache, vertices, (.001,.001,.001,.01,.05), 'point_jacobian')
    assert expected in checked['reasons']


def test_individual_fresnel_changes_do_not_cancel_from_the_checks():
    cache, vertices, result = _synthetic()
    row = cache.records[result['probes'][-1]]
    original_product = row['fresnel']
    row['fresnel_each'] = row['fresnel_each']*np.array([np.exp(.2j),np.exp(-.2j)])
    row['fresnel'] = np.prod(row['fresnel_each'])
    np.testing.assert_allclose(row['fresnel'], original_product, rtol=2e-16)
    checked = _check(cache, vertices, (.001,.001,.001,.01,.05), 'point_jacobian')
    assert 'fresnel_each' in checked['reasons']
    assert 'fresnel_cumulative' not in checked['reasons']
    row['fresnel'] *= np.exp(.15j)
    checked = _check(cache, vertices, (.001,.001,.001,.01,.05), 'point_jacobian')
    assert 'fresnel_cumulative' in checked['reasons']


def test_flux_guard_detects_wrong_shared_amplitude_even_when_probes_pass():
    cache, vertices, original = _synthetic()
    for row in cache.records:
        row['amplitude'] *= 2
    point = _check(cache, vertices, (.001,.001,.001,.01,.05), 'point_jacobian')
    tube = _check(cache, vertices, (.001,.001,.001,.01,.05), 'tube_flux')
    assert 'amplitude_probe' not in point['reasons']
    assert 'point_flux' in point['reasons'] and 'point_flux' in tube['reasons']
    assert point['point_flux_relative_defect'] > 2.9
    assert tube['point_flux_relative_defect'] == point['point_flux_relative_defect']


def test_flux_guard_handles_zero_incoming_and_outgoing_power():
    cache, vertices, original = _synthetic()
    for row in cache.records:
        row['fresnel'] = 0j
        row['fresnel_each'][:] = 0j
        row['amplitude'] = 0j
    zero = _check(cache, vertices, (.001,.001,.001,.01,.05), 'point_jacobian')
    assert not zero['reasons'] and zero['point_flux_relative_defect'] == 0
    for row in cache.records:
        row['amplitude'] = 1+0j
    defect = _check(cache, vertices, (.001,.001,.001,.01,.05), 'point_jacobian')
    assert 'point_flux' in defect['reasons']


def test_actual_mp_trace_cache_never_retraces_shared_nodes():
    regular, mp = _bore_spec(BORE, 64)
    cache = _TraceCache(SOURCE, regular, mp, 0., .23, K, 7e-6, 9e-8, 64, 1e-10, 100)
    first = cache.get([[0.,0.],[1e-6,0.],[0.,0.]])
    again = cache.get([[1e-6,0.],[0.,0.]])
    assert first.tolist() == [0,1,0]
    assert again.tolist() == [1,0]
    assert len(cache.records) == 2
    assert all(r['fate']=='screen' and r['reflections']==0 for r in cache.records)


def test_depth_and_budget_limits_retain_explicit_unresolved_partition():
    mesh, d = build(max_depth=0, max_nodes=300)
    assert d['status'] == 'controls_partial'
    assert d['coverage']['unresolved_area_m2'] > 0
    assert d['coverage']['unresolved_each_reason_area_m2']['max_depth'] > 0
    assert len(mesh['partition_entrance_triangles']) == 48
    coverage = d['coverage']
    np.testing.assert_allclose(coverage['accepted_area_m2']+coverage['unresolved_area_m2']
        +coverage['outer_chord_and_inset_deficit_m2'],coverage['aperture_area_m2'],rtol=5e-16)
    np.testing.assert_allclose(sum(coverage['unresolved_primary_reason_area_m2'].values()),coverage['unresolved_area_m2'],rtol=5e-16)
    mesh, limited = build(max_nodes=40)
    assert limited['status']=='controls_partial' and limited['budget_reached']
    assert limited['trace']['emitted_nodes'] <= 40
    assert len(mesh['partition_entrance_triangles']) == 48
    assert 'node_budget' in limited['coverage']['unresolved_primary_reason_area_m2']
    json.dumps(limited,allow_nan=False)


def test_actual_torus_supports_multiple_reflections_and_shared_node_amplitudes():
    bore = dict(center=[72e-6,0.], radius=24e-6, bend=dict(radius=2156.25,toward=[-1.,0.]))
    mesh,d = build(bore,max_nodes=800,phase_tolerance_rad=.2,
                   amplitude_relative_tolerance=.15,geometry_relative_tolerance=.01,fresnel_relative_tolerance=.2)
    assert int(max(d['trace']['reflection_histogram'],key=int)) >= 3
    assert d['topology']['accepted_triangles'] > 0
    assert d['topology']['maximum_edge_incidence'] == 2
    assert abs(d['coverage']['area_closure_relative']) < 1e-14
    nodes = mesh['trace_nodes']
    np.testing.assert_array_equal(mesh['vertex_amplitude'],nodes['amplitude'][mesh['ray_indices']])
    assert audit_amplitude_probes(mesh)['failing_triangles'] == 0
    changed = dict(mesh,vertex_amplitude=mesh['vertex_amplitude']*1.5)
    assert audit_amplitude_probes(changed)['failing_triangles'] > 0
    assert d['accepted_probe_maxima']['phase_error_rad'] <= .2
    assert d['accepted_probe_maxima']['fresnel_each_variation'] <= .2
    assert d['accepted_probe_maxima']['point_flux_relative_defect'] <= .05
    json.dumps(d,allow_nan=False)


@pytest.mark.parametrize('bad', [dict(angles=True),dict(max_depth=-1),dict(max_nodes=3),
    dict(geometry_relative_tolerance=0),dict(phase_tolerance_rad=float('nan')),dict(amplitude_mode='unknown'),
    dict(flux_relative_tolerance=0)])
def test_invalid_controls_fail_clearly(bad):
    with pytest.raises(ValueError):
        build(**bad)


def test_unsupported_wall_fails_before_tracing():
    with pytest.raises(ValueError,match='cylinders and circular torus'):
        build(dict(BORE,kind='polygon'))

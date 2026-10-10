"""Curved entrance representations retain the stage18 coherent estimator."""
import json

import numpy as np
import pytest

from formula.capsysred import Simulation
from formula.capsysred.config import Config
from formula.capsysred.stages.stage18 import _flat_to_curved
from tests.test_stage18_improvements import raw_scene, make_archive


def curved_scene():
    raw = raw_scene()
    raw['b9_estimator'].update(field_representation='curved_tubes', phase_backend='regular_mixed',
        amplitude_mode='point_jacobian', max_quadrature_nodes_per_batch=20000,
        curved_retrace=dict(bores=[0], angles=16, radial_rings=2, max_nodes=3500, max_depth=8))
    return raw


@pytest.mark.parametrize('bad', [
    {'curved_retrace': None}, {'phase_backend': 'type3'}, {'phase_degree': 1},
    {'amplitude_mode': 'shared_flux'}, {'curved_retrace': {}},
    {'curved_retrace': {'bores': [[0]]}}, {'curved_retrace': {'bores': [0, 0]}},
    {'curved_retrace': {'bores': [0], 'density_relative_tolerance': 0}},
    {'curved_retrace': {'bores': [0], 'max_nodes': True}},
    {'cylinder_retrace': {'bores': [0]}},
    {'adaptive_retrace': {'bores': [0]}},
])
def test_invalid_curved_options(bad):
    raw = curved_scene()
    raw['b9_estimator'].update(bad)
    with pytest.raises(ValueError, match='b9_estimator'):
        Config(raw).validate_b9_estimator()


def test_flat_to_curved_preserves_complex_integral_and_orientation():
    from formula.capsysred.stages._b9_curved import curved_phase_field
    from formula.capsysred.stages._b9_phase import phase_field

    q = np.array([[[0, 0], [1, 0], [0, 1]], [[1, 1], [0, 1], [1, 0.]]])
    transforms = np.array([[[1.2, .3], [.1, .8]], [[.7, -.2], [.3, -1.1]]])
    x = np.einsum('tij,tkj->tki', transforms, q)+np.array([[.1, -.2], [.4, .3]])[:, None]
    k = 9.
    h = np.array([[.4, .2], [.2, -.3]])
    phase = .5*np.einsum('tvi,ij,tvj->tv', x, h, x)+np.array([.2, 1.3])[:, None]
    directions = np.einsum('ij,tvj->tvi', h, x)/k
    amplitude = np.array([[1+.2j, .8+.1j, 1.3-.2j], [.2-.4j, .1-.7j, .4-.3j]])
    mesh = dict(entrance_triangles=q, triangles=x, vertex_phase=phase, vertex_amplitude=amplitude,
                vertex_directions=directions, entrance_area=np.full(2, .5), exit_area=.5*abs(np.linalg.det(transforms)))
    args = dict(k=k, distance=1.2, x=np.linspace(-.3, .8, 4), y=np.linspace(-.4, .9, 3), cell_width=.08, pixel_order=4)
    expected = phase_field(mesh, quadrature_order=32, backend='direct', **args)
    actual = curved_phase_field(_flat_to_curved(mesh, k), min_order=16, safety=2, **args)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)


def test_curved_replay_traces_each_source_and_accumulates_fields(tmp_path, monkeypatch):
    from formula.capsysred.stages import _b9_curved

    sim = Simulation.from_dict(curved_scene())
    archive, output = tmp_path/'archive', tmp_path/'result'
    make_archive(sim, archive)
    original, fields = _b9_curved.curved_phase_field, []

    def audited(*args, **kwargs):
        result = original(*args, **kwargs)
        fields.append(result[0])
        return result

    monkeypatch.setattr(_b9_curved, 'curved_phase_field', audited)
    sim.replay(str(archive), str(output), stages=[18])
    folder = output/'stage18'
    meta = json.loads((folder/'meta.json').read_text())
    assert meta['full_coherence_computed'] and not meta['accuracy_validated']
    assert meta['status'] == 'experimental-curved-GO-ray-tube-diffraction'
    assert len(fields) == 2 and not np.array_equal(*fields)
    for mode in meta['modes']:
        mesh = mode['meshes']['36']
        assert mesh['archived_triangles_retained'] == 0
        assert '0' in mesh['curved_retrace'] and not mesh['cylinder_retrace']
    with np.load(folder/'map-b36-curved-qm6-f1p5-p2-m2.npz') as saved:
        f = np.asarray(fields)
        ref = tuple(saved['ref_index'])
        np.testing.assert_allclose(saved['I'], np.mean(abs(f)**2, axis=0))
        np.testing.assert_allclose(saved['W'], np.mean(f*f[(slice(None), *ref)].conj()[:, None, None], axis=0))
        assert np.max(abs(saved['mu'])) <= 1+1e-12
        assert str(saved['field_representation']) == 'curved_tubes'
    with np.load(folder/'mesh-mode0-bore0.npz', allow_pickle=False) as saved:
        assert saved['position_nodes'].shape[1:] == (6, 2)
        assert saved['partition_accepted'].sum() == len(saved['position_nodes'])


def test_curved_residual_replay_corrects_intensity_and_cross_terms(tmp_path):
    pytest.importorskip('finufft')
    raw = curved_scene()
    raw['b9_estimator']['curved_retrace'].update(max_nodes=400, residual_batches=2, residual_seed=5)
    raw['b9_estimator'].update(max_modes=1, map_snapshots=[1])
    sim = Simulation.from_dict(raw)
    archive, output = tmp_path/'archive', tmp_path/'result'
    make_archive(sim, archive)
    sim.replay(str(archive), str(output), stages=[18])
    folder = output/'stage18'
    meta = json.loads((folder/'meta.json').read_text())
    assert meta['residual_quadrature']['batches'] == 2
    record = meta['modes'][0]
    residual = record['variants'][0]['residual']
    assert residual['batches'] == 2 and residual['faces'] == record['meshes']['36']['curved_retrace']['0']['residual']['faces']
    field_file = next(folder.glob('field-*-mode0.npz'))
    with np.load(field_file) as data:
        field, variance, covariance = data['field'], data['residual_variance'], data['residual_reference_covariance']
        ref = tuple(data['ref_index'])
    assert variance.shape == field.shape and np.all(variance >= 0)
    np.testing.assert_allclose(covariance[ref].imag, 0, atol=1e-12*abs(field).max()**2)
    with np.load(next(folder.glob('map-*-m1.npz'))) as data:
        assert bool(data['residual_variance_corrected'])
        np.testing.assert_allclose(data['I'], abs(field)**2-variance, rtol=1e-12, atol=1e-12*abs(field).max()**2)
        np.testing.assert_allclose(data['W'], field*field[ref].conjugate()-covariance, rtol=1e-12, atol=1e-12*abs(field).max()**2)
    if residual['faces']:
        assert residual['mean_norm_over_mesh_field'] is not None and np.any(variance > 0)


def test_mode_failures_are_skipped_and_recorded(tmp_path, monkeypatch):
    pytest.importorskip('finufft')
    from formula.capsysred.stages import stage18

    sim = Simulation.from_dict(curved_scene())
    archive, output = tmp_path/'archive', tmp_path/'result'
    make_archive(sim, archive)
    original = stage18._mode_job

    def flaky(job):
        if job['mode'] == 0:
            raise ValueError('prescribed node 7: geometry is outside the verified 0/1-family model')
        return original(job)

    monkeypatch.setattr(stage18, '_mode_job', flaky)
    sim.replay(str(archive), str(output), stages=[18])
    meta = json.loads((output/'stage18'/'meta.json').read_text())
    assert meta['failed_modes'] == [{'mode': 0, 'error': 'prescribed node 7: geometry is outside the verified 0/1-family model'}]
    assert meta['completed_source_modes'] == 1 and [m['mode'] for m in meta['modes']] == [1]
    assert meta['full_coherence_computed'] and any(row['modes'] == 1 for row in meta['outputs'])

    def fatal(job):
        raise ValueError('stage 18 implementation changed before worker')

    monkeypatch.setattr(stage18, '_mode_job', fatal)
    with pytest.raises(ValueError, match='implementation changed'):
        sim.replay(str(archive), str(tmp_path/'result2'), stages=[18])

    def all_bad(job):
        raise ValueError('prescribed node 1: geometry is outside the verified 0/1-family model')

    monkeypatch.setattr(stage18, '_mode_job', all_bad)
    with pytest.raises(ValueError, match='every requested source mode failed'):
        sim.replay(str(archive), str(tmp_path/'result3'), stages=[18])

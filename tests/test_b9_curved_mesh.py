"""Independent curved-chart probes, pullback measure and partial accounting."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from formula.capsysred.stages._b9_curved_mesh import (
    _HOLDOUTS, _curved_check, _receiver_box, curved_exit_mesh, p2_basis, p2_jacobian, pullback_density,
)


class Cache:
    def __init__(self):
        self.coordinates, self.records, self.lookup = [], [], {}
        self.cap = SimpleNamespace(bores=[dict(radius=1.)])
        self.floor, self.k = 1e-10, 5.

    def get(self, coordinates):
        ids=[]
        for point in coordinates:
            q=tuple(map(float,point))
            if q not in self.lookup:
                x,y=q
                jacobian=np.array([[1+.2*y,.2*x],[.2*x,1.]])
                determinant=np.linalg.det(jacobian)
                self.lookup[q]=len(self.records)
                self.coordinates.append(q)
                self.records.append(dict(points=np.array([x+.2*x*y,y+.1*x*x]),
                    phase=.1*x*x+.2*x*y+.3*y*y,Q=jacobian,determinant=determinant,
                    valid=True,reflections=2,maslov=0,fresnel=1+0j,fresnel_each=np.ones(2,complex),
                    uz0=1.,uzexit=1.,source_distance=np.sqrt(determinant),fate='screen'))
            ids.append(self.lookup[q])
        return np.array(ids)


def synthetic():
    cache=Cache()
    vertices=tuple(cache.get([[0,0],[1,0],[0,1]]))
    corners,_=_receiver_box([-.3,.4],[-.2,.2],.1)
    result=_curved_check(cache,vertices,(1e-10,1e-10,1e-10,.05),corners,2.)
    return cache,vertices,corners,result


def test_p2_basis_interpolates_six_nodes_and_has_partition_of_unity():
    bary=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5]])
    np.testing.assert_array_equal(p2_basis(bary),np.eye(6))
    np.testing.assert_allclose(p2_basis(_HOLDOUTS).sum(axis=1),1,atol=2e-16)


def test_quadratic_map_phase_and_density_pass_four_independent_holdouts():
    cache,vertices,corners,result=synthetic()
    assert not result['reasons']
    assert len(cache.records)==10
    assert set(result['fit']).isdisjoint(result['probes'])
    for name in ('phase_error_rad','density_relative_error','geometry_phase_error_rad','map_jacobian_relative_error'):
        assert result[name]<2e-15
    q=np.array([cache.coordinates[v] for v in vertices])
    x=np.array([cache.records[i]['points'] for i in result['fit']])
    expected=np.array([cache.records[i]['Q'] for i in result['probes']])
    np.testing.assert_allclose(p2_jacobian(q[None],x[None],_HOLDOUTS)[0],expected,atol=5e-16)


@pytest.mark.parametrize('change,reason', [('phase','phase_probe'),('points','geometry_phase_probe'),
    ('source_distance','density_probe'),('reflections','mixed_reflections'),('maslov','mixed_maslov'),
    ('determinant','mixed_orientation'),('valid','invalid_transport')])
def test_quarter_probe_detects_defect_without_changing_fit_nodes(change,reason):
    cache,vertices,corners,result=synthetic()
    row=cache.records[result['probes'][1]]
    if change=='points':
        row[change]+=np.array([.01,-.015])
    elif change=='source_distance':
        row[change]*=.8
    elif change=='valid':
        row[change]=False
    elif change=='determinant':
        row[change]*=-1
    else:
        row[change]+=.1 if change=='phase' else 1
    checked=_curved_check(cache,vertices,(1e-4,1e-4,1e-4,.05),corners,2.)
    assert reason in checked['reasons']


def test_fresnel_per_reflection_check_survives_cancelling_product_phases():
    cache,vertices,corners,result=synthetic()
    row=cache.records[result['probes'][2]]
    row['fresnel_each']*=np.exp(np.array([.15j,-.15j]))
    row['fresnel']=np.prod(row['fresnel_each'])
    checked=_curved_check(cache,vertices,(.01,.01,.01,.05),corners,2.)
    assert 'fresnel_each' in checked['reasons']
    assert 'fresnel_cumulative' not in checked['reasons']
    row['fresnel']*=np.exp(.2j)
    checked=_curved_check(cache,vertices,(.01,.01,.01,.05),corners,2.)
    assert 'fresnel_cumulative' in checked['reasons']


def test_interpolating_map_fold_is_rejected_even_when_ray_jacobians_are_positive():
    cache,vertices,corners,result=synthetic()
    cache.records[result['fit'][3]]['points']+=np.array([0.,4.])
    checked=_curved_check(cache,vertices,(.05,.02,.05,.05),corners,2.)
    assert all(row['determinant']>0 for row in cache.records)
    assert 'folded_p2_map' in checked['reasons']


def test_receiver_box_includes_cell_edges_and_kernel_extrema_are_at_corners():
    corners,width=_receiver_box([-1,2],[-2,3],[.4,.6])
    np.testing.assert_array_equal(width,[.4,.6])
    np.testing.assert_allclose(corners.min(axis=0),[-1.2,-2.3])
    np.testing.assert_allclose(corners.max(axis=0),[2.2,3.3])
    true=np.array([.1,-.2]);fit=np.array([.11,-.17]);dx=fit-true
    corner_defect=abs(np.dot(dx,fit+true)-2*corners@dx).max()
    samples=np.array([[a,b] for a in np.linspace(-1.2,2.2,9) for b in np.linspace(-2.3,3.3,7)])
    assert np.max(abs(np.dot(dx,fit+true)-2*samples@dx))<=corner_defect+1e-15


def test_pullback_keeps_unequal_fresnel_maslov_and_absolute_jacobians():
    n=dict(fresnel=np.array([.8+.3j,.4-.1j]),source_distance=np.array([2.,3.]),
           uz0=np.array([.8,.9]),uzexit=np.array([.6,.7]),determinant=np.array([4.,-9.]),maslov=np.array([1,2]))
    expected=np.array([(.8+.3j)/2*np.sqrt(.8/.6)*2*(-1j),(.4-.1j)/3*np.sqrt(.9/.7)*3*(-1)])
    np.testing.assert_allclose(pullback_density(n),expected,atol=2e-16)
    n['determinant'][:]=0
    np.testing.assert_array_equal(pullback_density(n),0j)


def build(**kwargs):
    args=dict(x_bounds=[-60e-6,60e-6],y_bounds=[0,0],cell_width=.3e-6,distance=.35,
              angles=16,radial_rings=2,max_nodes=900,max_depth=8)
    args.update(kwargs)
    return curved_exit_mesh(['-1.9850001681262593E-7','-7.062065632157497E-7','-.675'],
        dict(center=[72e-6,0.],radius=24e-6,bend=dict(radius=2156.25,toward=[-1.,0.])),
        0.,.23,40785096803.62667,7.125763840840871e-6,9.207952277589002e-8,**args)


def test_actual_mp_torus_and_explicit_partial_area_accounting():
    mesh,d=build()
    assert d['trace']['emitted_nodes']<=900 and d['budget_reached']
    assert d['status']=='controls_partial'
    n=len(mesh['entrance_triangles'])
    assert n>0 and mesh['position_nodes'].shape==(n,6,2)
    assert mesh['phase_nodes'].shape==mesh['weight_nodes'].shape==(n,6)
    nodes=mesh['trace_nodes'];ids=mesh['fit_indices']
    np.testing.assert_array_equal(mesh['weight_nodes'],pullback_density(nodes)[ids])
    for fit,probe in zip(mesh['partition_fit_indices'][mesh['partition_accepted']],mesh['partition_probe_indices'][mesh['partition_accepted']]):
        assert set(fit).isdisjoint(probe)
    c=d['coverage']
    np.testing.assert_allclose(c['accepted_area_m2']+c['unresolved_area_m2']+c['outer_chord_and_inset_deficit_m2'],c['aperture_area_m2'],rtol=1e-14)
    assert d['accepted_probe_maxima']['geometry_phase_error_rad']<=.05
    assert d['accepted_probe_maxima']['density_relative_error']<=.02
    assert d['topology']['maximum_edge_incidence']==2
    json.dumps(d,allow_nan=False)


def test_budget_and_depth_keep_unresolved_faces_without_hidden_deletion():
    mesh,d=build(max_nodes=40)
    assert not len(mesh['entrance_triangles'])
    assert len(mesh['partition_entrance_triangles'])==48
    assert d['coverage']['unresolved_area_fraction']>0
    mesh,d=build(max_depth=0,max_nodes=500)
    assert d['coverage']['unresolved_each_reason_area_m2']['max_depth']>0


@pytest.mark.parametrize('bad',[dict(distance=0),dict(cell_width=-1),dict(x_bounds=[1,0]),
    dict(y_bounds=[0,float('nan')]),dict(density_relative_tolerance=0),dict(angles=True)])
def test_invalid_parameters_fail_clearly(bad):
    with pytest.raises(ValueError):
        build(**bad)

import copy
import numpy as np
import pytest

from formula.capsysred.stages._b9_amplitude import apply_shared_flux


def example():
    ids=np.array([[0,1,2],[1,3,2]])
    amp=np.array([1+.2j,.8-.1j,1.3+.4j,.7-.2j])
    nodes=dict(amplitude=amp,uz0=np.ones(4),uzexit=np.ones(4),source_distance=np.ones(4),
               fresnel=amp.copy(),bore=np.zeros(4,int),reflections=np.zeros(4,int),maslov=np.zeros(4,int))
    mesh=dict(ray_indices=ids,vertex_amplitude=amp[ids].copy(),entrance_area=np.array([.5,.5]),
              exit_area=np.array([.5,.5]),metadata={},vertex_phase=np.array([[0,1,2],[1,3,2.]]))
    return mesh,nodes


def test_exact_affine_field_fixed_point_and_input_immutability():
    mesh,nodes=example();original=copy.deepcopy(mesh)
    out=apply_shared_flux(mesh,nodes)
    np.testing.assert_allclose(out["vertex_amplitude"],mesh["vertex_amplitude"],atol=3e-16)
    np.testing.assert_allclose(out["shared_flux_node_scales"],1,atol=3e-16)
    for key in ("vertex_amplitude","vertex_phase","entrance_area"):
        np.testing.assert_array_equal(mesh[key],original[key])
    assert out["metadata"] is not mesh["metadata"]
    assert not mesh["metadata"]


def test_positive_weighted_objective_shared_edge_and_phase_preservation():
    mesh,nodes=example();mesh["entrance_area"]*=np.array([4.,.25])
    out=apply_shared_flux(mesh,nodes)
    ids=mesh["ray_indices"]
    weights=mesh["exit_area"][:,None]*abs(mesh["vertex_amplitude"])**2/3
    target=np.array([2.,.5])
    expected=np.bincount(ids.ravel(),weights=(weights*target[:,None]).ravel())/np.bincount(ids.ravel(),weights=weights.ravel())
    np.testing.assert_allclose(out["shared_flux_node_scales"],expected,rtol=1e-14)
    assert np.all(out["shared_flux_node_scales"]>0)
    np.testing.assert_array_equal(out["vertex_amplitude"][0,1],out["vertex_amplitude"][1,0])
    np.testing.assert_array_equal(out["vertex_amplitude"][0,2],out["vertex_amplitude"][1,2])
    np.testing.assert_allclose(np.angle(out["vertex_amplitude"]/mesh["vertex_amplitude"]),0,atol=1e-16)
    np.testing.assert_array_equal(out["vertex_phase"],mesh["vertex_phase"])
    info=out["metadata"]["shared_flux"]
    assert info["lumped_objective_after"]<info["lumped_objective_before"]
    assert info["normal_equation_max_residual"]<1e-15
    assert info["absolute_patch_relative_flux_defect"]["max"]>0


def test_distinct_nodes_at_same_exit_position_are_not_merged():
    mesh,nodes=example()
    nodes={key:np.concatenate([v,v]) for key,v in nodes.items()}
    mesh["ray_indices"]=np.vstack([mesh["ray_indices"],mesh["ray_indices"]+4])
    mesh["vertex_amplitude"]=nodes["amplitude"][mesh["ray_indices"]]
    mesh["triangles"]=np.zeros((4,3,2))  # Coordinates never define sharing.
    mesh["entrance_area"]=np.array([.5,.5,2.,2.]);mesh["exit_area"]=np.full(4,.5)
    out=apply_shared_flux(mesh,nodes)
    np.testing.assert_allclose(out["shared_flux_node_scales"][:4],1)
    np.testing.assert_allclose(out["shared_flux_node_scales"][4:],2)


def test_zero_weight_nodes_keep_zero_field_and_unit_scale():
    mesh,nodes=example();nodes["amplitude"][:]=0;nodes["fresnel"][:]=0
    mesh["vertex_amplitude"]=nodes["amplitude"][mesh["ray_indices"]]
    out=apply_shared_flux(mesh,nodes)
    assert out["metadata"]["shared_flux"]["zero_weight_nodes"]==4
    assert np.all(out["shared_flux_node_scales"]==1)
    assert np.all(out["vertex_amplitude"]==0)
    assert out["metadata"]["shared_flux"]["relative_total_branch_flux_defect"] is None


@pytest.mark.parametrize("fault",["input_amplitude","nonfinite","area","zero_exit","mixed_branch"])
def test_invalid_input_does_not_silently_fallback(fault):
    mesh,nodes=example()
    if fault=="input_amplitude":mesh["vertex_amplitude"]*=2
    if fault=="nonfinite":nodes["fresnel"][0]=np.nan
    if fault=="area":mesh["exit_area"][0]=0
    if fault=="zero_exit":
        nodes["amplitude"][:]=0;mesh["vertex_amplitude"][:]=0
    if fault=="mixed_branch":nodes["maslov"][0]=1
    with pytest.raises(ValueError):apply_shared_flux(mesh,nodes)


def test_order_and_common_complex_scale_invariance():
    mesh,nodes=example();mesh["entrance_area"]*=np.array([3.,.4]);out=apply_shared_flux(mesh,nodes)
    scaled=copy.deepcopy(nodes);scaled["amplitude"]*=2.3*np.exp(.7j);scaled["fresnel"]*=2.3*np.exp(.7j)
    reordered=copy.deepcopy(mesh)
    for key in ("ray_indices","entrance_area","exit_area"):reordered[key]=reordered[key][::-1]
    reordered["vertex_amplitude"]=scaled["amplitude"][reordered["ray_indices"]]
    other=apply_shared_flux(reordered,scaled)
    np.testing.assert_allclose(other["shared_flux_node_scales"],out["shared_flux_node_scales"],rtol=1e-14)
    np.testing.assert_allclose(other["vertex_amplitude"][::-1],out["vertex_amplitude"]*2.3*np.exp(.7j),rtol=1e-14)

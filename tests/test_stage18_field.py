"""Representation and coherent batching controls for stage 18."""

import numpy as np
import pytest

from formula.capsysred.stages.stage18 import refine_chirped_mesh
from formula.capsysred.stages._b9_contour import fresnel_from_chirped


def test_subdivision_preserves_affine_phase_and_complex_amplitude():
    tri = np.array([[[0., 0.], [2., 0.], [0., 1.]]])
    phase = 100 + tri[..., 0]*7-tri[..., 1]*3
    amplitude = 1+(.2+.4j)*tri[..., 0]-.1j*tri[..., 1]
    mesh = dict(triangles=tri, vertex_phase=phase, vertex_amplitude=amplitude)
    q, values, info = refine_chirped_mesh(mesh, 3, 5., 2.)
    expected = (1+(.2+.4j)*q[..., 0]-.1j*q[..., 1])*np.exp(1j*(100+7*q[..., 0]-3*q[..., 1]+1.25*np.sum(q*q, axis=-1)))
    np.testing.assert_allclose(values, expected, atol=5e-14)
    area = np.linalg.det(np.stack([q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]], axis=-1))/2
    assert len(q) == 9 and np.all(area > 0)
    assert area.sum() == pytest.approx(1.)


def test_unequal_batches_add_as_fields_before_intensity():
    pytest.importorskip("finufft")
    triangles = np.array([[[-1.,-1.],[1.,-1.],[1.,1.]], [[-1.,-1.],[1.,1.],[-1.,1.]]])*1e-5
    values = np.array([[1., 1.1+.1j, .9+.3j], [.7+.2j, .4+.6j, 1.2-.1j]])
    kwargs = dict(k=4e10,distance=.35,x=np.linspace(-3e-5,3e-5,9),y=np.linspace(-2e-5,2e-5,7),
                  cell_width=3e-7,pixel_order=4,eps=1e-12,backend="finufft")
    total = fresnel_from_chirped(triangles,values,**kwargs)
    parts = [fresnel_from_chirped(triangles[i:i+1],values[i:i+1],**kwargs) for i in range(2)]
    np.testing.assert_allclose(sum(parts),total,atol=1e-10,rtol=1e-9)
    assert np.linalg.norm(abs(total)**2-sum(abs(p)**2 for p in parts)) > .1*np.linalg.norm(abs(total)**2)

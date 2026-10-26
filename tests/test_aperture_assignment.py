from types import SimpleNamespace

import numpy as np
import pytest

from formula import Number
from formula.capsysred.config import _bore
from formula.capsysred.shared.apertures import entrance_bore_ids
from formula.capsysred.shared.types import _APERTURE_REL_TOL
from formula.capsysred.stages.method_safe import _b5_defect, stage17
from formula.capsysred.stages.beamlets import beamlet_defect
from formula.capsysred.stages._b5_transport import transport_mode
from formula.capsysred.stages._b9_archive import exit_nodes
from formula.capsysred.surfaces import CapillaryBundle


def test_unique_membership_uses_radius_and_rejects_outside_overlap_and_nonfinite():
    bores = [dict(center=[0., 0.], radius=1.), dict(center=[1.6, 0.], radius=.2)]
    points = np.array([[.9, 0.], [1.6, 0.], [3., 0.], [np.nan, 0.]])
    np.testing.assert_array_equal(entrance_bore_ids(points, bores), [0, 1, -1, -1])
    assert int(entrance_bore_ids(points[0], bores)) == 0
    overlap = [dict(center=[0., 0.], radius=1.), dict(center=[1.2, 0.], radius=1.)]
    assert int(entrance_bore_ids([.6, 0.], overlap)) == -1
    np.testing.assert_array_equal(entrance_bore_ids(points, []), [-1] * len(points))
    assert entrance_bore_ids(np.empty((0, 2)), bores).shape == (0,)
    grid = points[:2].reshape(1, 2, 2)
    np.testing.assert_array_equal(entrance_bore_ids(grid, bores), [[0, 1]])


def test_rotated_unequal_polygon_faces_and_common_tolerance():
    angle = np.deg2rad(37.)
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([.3, -.4])
    bore = dict(center=center, radius=1., sides=4, rotation_deg=37., apothems=[1., 2., 1., 2.])
    local = np.array([[.9, 1.9], [1.1, 0.], [0., 2.1]])
    points = center + local @ rotate.T
    np.testing.assert_array_equal(entrance_bore_ids(points, [bore]), [0, -1, -1])
    circle = dict(center=[0., 0.], radius=1.)
    assert int(entrance_bore_ids([1 + .5 * _APERTURE_REL_TOL, 0.], [circle])) == 0
    assert int(entrance_bore_ids([1 + 2 * _APERTURE_REL_TOL, 0.], [circle])) == -1


@pytest.mark.parametrize("points", [1., [1., 2., 3.], np.zeros((2, 3))])
def test_entrance_coordinate_shape_is_explicit(points):
    with pytest.raises(ValueError, match="shape"):
        entrance_bore_ids(points, [])


def _scene(case, rotation):
    angle = np.deg2rad(rotation)
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([.3, -.4])
    if case == "overlap":
        centers, radii, q = [[0., 0.], [1.2, 0.]], [1., 1.], [.6, 0.]
    else:
        centers, radii = [[0., 0.], [1.6, 0.]], [1., .2]
        q = [.9, 0.] if case == "unequal" else [3., 0.]
    bores = [dict(center=(shift + rotate @ c).tolist(), radius=r) for c, r in zip(centers, radii)]
    origin = np.r_[shift, -1.]
    direction = np.r_[rotate @ q, 1.]
    direction /= np.linalg.norm(direction)
    point = shift + 1.05 * (rotate @ q)
    row = (point[0], point[1], direction[0], direction[1], 1.05 / direction[2], [], [])
    mode = dict(origin=origin, points=point[None], directions=direction[None, :2], refl=[[]], sins=[[]])
    return bores, origin, row, mode


@pytest.mark.parametrize("rotation", [0., 37.])
@pytest.mark.parametrize("case", ["unequal", "outside", "overlap"])
def test_transport_diagnostics_and_exit_mesh_share_bore_assignment(case, rotation):
    bores, origin, row, mode = _scene(case, rotation)
    cap = SimpleNamespace(z0=0., bores=bores)
    expected = 0 if case == "unequal" else -1
    state = transport_mode(mode, cap, .05)
    assert state["bore"].tolist() == [expected]
    assert state["valid"].tolist() == [expected >= 0]

    segments = _b5_defect.build_segments([row], origin, .05, bores, 0., 1., (1e-6, 1e-8))
    assert segments.bore.tolist() == [expected]
    assert segments.valid.tolist() == [expected >= 0]
    assert np.isfinite(segments.amp).all()

    groups, unassigned, entrance = stage17.entrance_families(mode, cap, .05)
    assert unassigned == int(expected < 0)
    assert set(groups) == ({(0, 0)} if expected >= 0 else set())
    np.testing.assert_allclose(entrance, state["entrance"], rtol=0., atol=2e-15)

    exit_mode = dict(mode, phase_opl=np.zeros(1))
    nodes = exit_nodes(exit_mode, cap, mode_z=.05, exit_z=.025, k=1., fresnel=1.)
    assert nodes["bore"].tolist() == [expected]
    assert nodes["valid"].tolist() == [expected >= 0]
    np.testing.assert_allclose(nodes["entrance"], state["entrance"], rtol=0., atol=2e-15)
    assert np.isfinite(nodes["amplitude"]).all()
    if expected < 0:
        np.testing.assert_array_equal(nodes["amplitude"], [0j])

    typed = [_bore(b, 32, i) for i, b in enumerate(bores)]
    bundle = CapillaryBundle(typed, Number("0", 32), Number(".025", 32))
    centers = np.array([b["center"] for b in bores])
    radii = np.array([b["radius"] for b in bores])
    arguments = ([row], origin, .05, bundle, 1., 1., 0., (1e-6, 1e-8), 1., centers, radii, 0.)
    if expected >= 0:
        beams = beamlet_defect.build_segments(*arguments)
        assert beams.bore.tolist() == [expected]
    else:
        with pytest.raises(ValueError, match="exactly one bore"):
            beamlet_defect.build_segments(*arguments)


def test_unassigned_reflected_path_never_uses_the_last_bore():
    bores = [dict(center=[0., 0.], radius=.2)]
    origin = [1., 0., -1.]
    row = (1., .1, 0., 0., 2., [.1], [[1., .1, .01]])
    segments = _b5_defect.build_segments([row], origin, .05, bores, 0., 1., (1e-6, 1e-8))
    np.testing.assert_array_equal(segments.bore, [-1, -1])
    np.testing.assert_array_equal(segments.valid, [False, False])


def test_phase_families_and_transport_use_the_same_aperture_tolerance():
    cap = SimpleNamespace(z0=0., bores=[dict(center=[0., 0.], radius=1.)])
    q = np.array([1. + 2 * _APERTURE_REL_TOL, 0.])
    direction = np.r_[q, 1.]
    direction /= np.linalg.norm(direction)
    mode = dict(origin=[0., 0., -1.], points=1.05 * q[None], directions=direction[None, :2],
                refl=[[]], sins=[[]])
    groups, unassigned, _ = stage17.entrance_families(mode, cap, .05)
    assert groups == {} and unassigned == 1
    assert transport_mode(mode, cap, .05)["bore"].tolist() == [-1]

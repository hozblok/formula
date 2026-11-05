"""Adaptive reconstruction, regular propagation and explicit coverage gates."""

import json

import numpy as np
import pytest

from formula.capsysred import Simulation
from formula.capsysred.config import Config
from tests.test_stage18_improvements import raw_scene, make_archive


@pytest.mark.parametrize("bad", [
    {"phase_backend": "fft_interpolation"}, {"receiver_channels_per_batch": 0},
    {"adaptive_retrace": {}}, {"adaptive_retrace": {"bores": [0, 0]}},
    {"adaptive_retrace": {"bores": [0], "phase_tolerance_rad": float("nan")}},
    {"adaptive_retrace": {"bores": [0], "max_nodes": True}},
    {"adaptive_retrace": {"bores": [0]}, "phase_degree": 1},
    {"adaptive_retrace": {"bores": [0]}, "field_representation": "contour_p1"},
    {"adaptive_retrace": {"bores": [0]}, "cylinder_retrace": {"bores": [0]}},
    {"max_missing_area_fraction": -1}, {"max_missing_area_fraction": 1},
    {"max_missing_area_fraction": True},
    {"quadrature_safety": float("nan")}, {"quadrature_safety": 0},
    {"quadrature_max_order": True},
    {"phase_backend": "regular_mixed", "quadrature_max_order": 2},
])
def test_invalid_completion_options(bad):
    raw = raw_scene()
    raw["b9_estimator"].update(bad)
    with pytest.raises(ValueError, match="b9_estimator"):
        Config(raw).validate_b9_estimator()


def scene_options():
    raw = raw_scene()
    raw["b9_estimator"].update(max_modes=1, phase_backend="regular", amplitude_mode="point_jacobian",
        adaptive_retrace=dict(bores=[0], angles=16, radial_rings=2, max_nodes=2000,
                              max_depth=8, geometry_relative_tolerance=.02))
    return raw


def test_adaptive_replay_saves_partition_and_uses_regular_backend(tmp_path, monkeypatch):
    from formula.capsysred.stages import _b9_regular

    pytest.importorskip("finufft")
    sim = Simulation.from_dict(scene_options())
    archive, output = tmp_path/"archive", tmp_path/"result"
    make_archive(sim, archive)
    original, fields = _b9_regular.regular_phase_field, []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        fields.append(result[0])
        return result

    monkeypatch.setattr(_b9_regular, "regular_phase_field", capture)
    sim.replay(str(archive), str(output), stages=[18])
    folder = output/"stage18"
    meta = json.loads((folder/"meta.json").read_text())
    mesh = meta["modes"][0]["meshes"]["36"]
    assert meta["full_coherence_computed"] and not meta["accuracy_validated"]
    assert meta["validation"]["final_amplitude_probes_passed"] is True
    assert meta["validation"]["coverage_gate_passed"] is None
    assert not meta["validation"]["quadrature_convergence_checked"]
    assert mesh["archived_triangles_retained"] == 0 and mesh["cylinder_retrace"] == {}
    assert mesh["coverage_gate"]["passed"] is None
    diag = mesh["adaptive_retrace"]["0"]
    assert diag["trace"]["emitted_nodes"] <= 2000
    assert diag["final_amplitude_probe_audit"]["failing_triangles"] == 0
    with np.load(folder/diag["saved_mesh"]["file"], allow_pickle=False) as saved:
        assert saved["partition_accepted"].sum() == len(saved["triangles"])
        assert saved["partition_entrance_triangles"].shape[1:] == (3, 2)
        assert len(saved["nodes_amplitude"]) == diag["trace"]["emitted_nodes"]
    with np.load(folder/"map-b36-q6-p2-m1.npz") as saved:
        np.testing.assert_allclose(saved["I"], abs(fields[0])**2)
        assert str(saved["phase_backend"]) == "regular"
        assert saved["retraced_bores"].tolist() == [0]


def test_coverage_gate_retains_diagnostics_and_prevents_a_qualified_map(tmp_path):
    raw = scene_options()
    raw["b9_estimator"]["max_missing_area_fraction"] = 0.
    sim = Simulation.from_dict(raw)
    archive, output = tmp_path/"archive", tmp_path/"result"
    make_archive(sim, archive)
    with pytest.raises(ValueError, match="max_missing_area_fraction"):
        sim.replay(str(archive), str(output), stages=[18])
    partial = output/"stage18.partial"
    assert not (output/"stage18").exists()
    saved = json.loads((partial/"rejected-mesh-mode0-b36.json").read_text())
    assert saved["coverage_gate"]["passed"] is False
    assert saved["coverage_gate"]["maximum_missing_area_fraction"] > 0
    assert not list(partial.glob("map-*.npz"))


def test_coverage_gate_is_per_bore_not_an_average():
    from types import SimpleNamespace
    from formula.capsysred.stages.stage18 import _coverage_check

    cap = SimpleNamespace(bores=[dict(radius=1), dict(radius=1)])
    mesh = dict(metadata=dict(accepted_entrance_area_by_bore_m2=[np.pi, .8*np.pi]))
    result = _coverage_check(mesh, cap, .15)
    assert result["passed"] is False
    assert result["maximum_missing_area_fraction"] == pytest.approx(.2)


def test_adaptive_index_rejected_before_output_creation(tmp_path):
    raw = scene_options()
    raw["b9_estimator"]["adaptive_retrace"]["bores"] = [1]
    with pytest.raises(ValueError, match="bore index"):
        Simulation.from_dict(raw).replay("unused", str(tmp_path/"result"), stages=[18])
    assert not (tmp_path/"result").exists()


def test_mixed_quadrature_route_records_actual_policy_and_moments(tmp_path, monkeypatch):
    from formula.capsysred.stages import _b9_quadrature

    raw = raw_scene()
    raw["b9_estimator"].update(max_modes=1, phase_backend="regular_mixed", quadrature_safety=1.5,
                               max_quadrature_nodes_per_batch=500000)
    sim = Simulation.from_dict(raw)
    archive, output = tmp_path/"archive", tmp_path/"result"
    make_archive(sim, archive)
    captured, original = [], _b9_quadrature.mixed_phase_field

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(_b9_quadrature, "mixed_phase_field", capture)
    sim.replay(str(archive), str(output), stages=[18])
    with np.load(output/"stage18"/"map-b36-qm6-f1p5-p2-m1.npz") as saved:
        assert str(saved["phase_backend"]) == "regular_mixed"
        assert int(saved["triangle_quadrature_order"]) == 0
        assert int(saved["triangle_min_quadrature_order"]) == 6
        assert float(saved["quadrature_safety"]) == 1.5
        np.testing.assert_allclose(saved["I"], abs(captured[0][0])**2)
    assert captured[0][1]["selection"]["groups"]

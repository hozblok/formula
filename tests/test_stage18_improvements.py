"""Opt-in phase quadrature and prescribed-cylinder reconstruction."""

from decimal import Decimal, localcontext
import json

import numpy as np
import pytest

from formula.capsysred import Simulation, rays_v3
from formula.capsysred.config import Config
from formula.capsysred.rays import geometry_metadata


def raw_scene():
    return {
        "precision": 32, "energy_kev": 8.048,
        "capillary": {
            "bores": [{"center": [0., 0.], "radius": 20e-6}], "z0": 0., "z1": .02,
            "source": {"shape": "gaussian", "size": .25e-6, "position": [0., 0., -.1],
                       "n_modes": 2, "n_rays": 36},
            "screen": {"z": .02, "edge_x": 2e-6, "edge_y": 2e-6, "nx": 9, "ny": 9},
            "screens": [{"z": .08}],
        },
        "b9_estimator": {
            "max_modes": 2, "rays_per_mode": 36, "screen_index": 1,
            "amplitude_mode": "tube_flux", "map_stride": 2, "phase_subdivisions": [1],
            "pixel_order": 2, "holdout_stride": 0, "field_representation": "phase_quadrature",
            "phase_degree": 2, "triangle_quadrature_order": 6,
            "max_quadrature_nodes_per_batch": 5000,
        },
    }


@pytest.mark.parametrize("bad", [
    {"field_representation": "unknown"}, {"phase_degree": True}, {"phase_degree": 3},
    {"triangle_quadrature_order": 1}, {"max_quadrature_nodes_per_batch": 20},
    {"carrier_groups": 2}, {"phase_subdivisions": [1, 2]},
    {"cylinder_retrace": []}, {"cylinder_retrace": {}},
    {"cylinder_retrace": {"bores": [0], "unknown": 0}},
    {"cylinder_retrace": {"bores": [True]}}, {"cylinder_retrace": {"bores": [-1]}},
    {"cylinder_retrace": {"bores": [0, 0]}},
    {"cylinder_retrace": {"bores": [0], "angles": 7}},
    {"cylinder_retrace": {"bores": [0], "inner_rings": 0}},
    {"cylinder_retrace": {"bores": [0], "precision": 16}},
    {"cylinder_retrace": {"bores": [0], "boundary_relative_gap": 0}},
    {"cylinder_retrace": {"bores": [0], "entrance_relative_inset": float("nan")}},
])
def test_invalid_improvement_options(bad):
    raw = raw_scene()
    raw["b9_estimator"].update(bad)
    with pytest.raises(ValueError, match="b9_estimator"):
        Config(raw).validate_b9_estimator()


def make_archive(sim, path):
    rays_v3.write_fingerprint(path, {"format": 3, "geometry": geometry_metadata(sim.cfg)})
    entries = []
    for mode, sx in enumerate((Decimal("-.0000002"), Decimal(".0000003"))):
        writer = rays_v3.SectionWriter(path, "capillary", mode, 0, 36,
                                       origin=[str(sx), "0", "-.1"])
        with localcontext() as context:
            context.prec = 80
            for ray, (x, y) in enumerate((x, y) for x in np.linspace(-2e-6, 2e-6, 6)
                                        for y in np.linspace(-2e-6, 2e-6, 6)):
                distance = ((Decimal(str(x))-sx)**2+Decimal(str(y))**2+Decimal(".12")**2).sqrt()
                writer.write_row({"stage": "capillary", "mode": mode, "ray": ray, "fate": "screen",
                                  "pixel": 0, "opl": str(distance), "sins": [], "refl": [],
                                  "x": x, "y": y, "dx": (x-float(sx))/float(distance),
                                  "dy": y/float(distance)})
        entries.append(writer.close())
    rays_v3.write_index(path, entries)


@pytest.mark.parametrize("retrace", [False, True])
def test_replay_new_routes_use_saved_source_origins_and_complex_moments(tmp_path, monkeypatch, retrace):
    from formula.capsysred.stages import _b9_phase

    pytest.importorskip("finufft")
    raw = raw_scene()
    if retrace:
        raw["b9_estimator"]["cylinder_retrace"] = dict(bores=[0], angles=32, inner_rings=3, outer_rings=2)
    sim = Simulation.from_dict(raw)
    archive = tmp_path/"archive"
    make_archive(sim, archive)
    original, captured = _b9_phase.phase_field, []

    def audited(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append((result[0].copy(), args[0]["metadata"], kwargs.copy()))
        return result

    monkeypatch.setattr(_b9_phase, "phase_field", audited)
    output = tmp_path/"result"
    sim.replay(str(archive), str(output), stages=[18])
    folder = output/"stage18"
    meta = json.loads((folder/"meta.json").read_text())
    assert meta["full_coherence_computed"] and not meta["accuracy_validated"]
    assert meta["completed_source_modes"] == 2 and len(captured) == 2
    assert "_b9_phase.py" in meta["implementation_sha256"]
    fields = np.array([entry[0] for entry in captured])
    assert not np.array_equal(fields[0], fields[1])
    with np.load(folder/"map-b36-q6-p2-m2.npz") as saved:
        assert str(saved["field_representation"]) == "phase_quadrature"
        assert int(saved["phase_degree"]) == 2 and int(saved["triangle_quadrature_order"]) == 6
        assert saved["retraced_bores"].tolist() == ([0] if retrace else [])
        assert int(saved["archive_prefix_ray_budget"]) == 36
        width = float(saved["receiver_width_m"])
        np.testing.assert_allclose(np.diff(saved["x"]), 2*width, rtol=1e-13)
        ref = tuple(saved["ref_index"])
        expected_i = np.mean(abs(fields)**2, axis=0)
        expected_w = np.mean(fields*fields[(slice(None), *ref)].conj()[:, None, None], axis=0)
        np.testing.assert_allclose(saved["I"], expected_i, rtol=2e-13)
        np.testing.assert_allclose(saved["W"], expected_w, rtol=2e-13, atol=1e-13)
        assert np.max(abs(saved["mu"])) <= 1+1e-12
    with np.load(folder/"field-b36-q6-p2-mode0.npz") as saved:
        np.testing.assert_array_equal(saved["field"], fields[0])
    if retrace:
        for _, mesh, _ in captured:
            assert mesh["archived_triangles_retained"] == 0
            assert mesh["accepted_entrance_area_fraction"] > .97
            assert "0" in mesh["cylinder_retrace"]


def test_retrace_bore_index_rejected_before_output_creation(tmp_path):
    raw = raw_scene()
    raw["b9_estimator"]["cylinder_retrace"] = dict(bores=[1])
    with pytest.raises(ValueError, match="bore index"):
        Simulation.from_dict(raw).replay("unused-archive", str(tmp_path/"result"), stages=[18])
    assert not (tmp_path/"result").exists()

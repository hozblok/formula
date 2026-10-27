"""A recorded screen z is evaluated as the config evaluates it — numbers,
decimal strings and expressions at the fingerprint's precision — wherever a
stage reads it back from the fingerprint."""

from __future__ import annotations

import pytest
import yaml

from formula.capsysred import Simulation
from formula.capsysred.rays import recorded_plane, recorded_screen_z, recorded_screen_z_text
from formula.capsysred.stages import archive_samples
from formula.capsysred.stages.beamlets.beamlet_defect import index_screen_z
from formula.capsysred.stages.stage14.cache import _recorded_screen_z
from formula.capsysred.trace_v3 import trace as trace_v3


def test_values_numbers_strings_and_expressions():
    geo = {"precision": 64, "screen": {"z": 0.06},
           "capillary": {"screen": {"z": "0.05 + 0.001"}}, "free": {"screen": {}}}
    assert recorded_screen_z(geo, "capillary") == 0.051
    assert recorded_screen_z_text(geo, "capillary") == "0.051"
    assert recorded_screen_z(geo, "free") == 0.06
    assert recorded_screen_z({"precision": 32, "screen": {"z": "6e-2"}}, "free") == 0.06
    assert recorded_screen_z({"screen": {"z": "1/4"}}, "free") == 0.25
    assert recorded_screen_z({"precision": 32, "screen": {}}, "free") is None
    with pytest.raises(ValueError, match="invalid recorded screen z"):
        recorded_screen_z({"precision": 32, "screen": {"z": "abc"}}, "free")
    # numbers stay bit-identical to the plain float: stage 14 cache identities
    for v in (0.23, 0.051, 0.06, 1e-5, 0.1234567890123456, 3):
        assert recorded_screen_z({"precision": 32, "screen": {"z": v}}, "free") == v
        assert _recorded_screen_z({"geometry": {"precision": 64, "screen": {"z": v},
                                                "capillary": {}}}, "p") == v
    assert _recorded_screen_z({"geometry": geo}, "p") == 0.051
    with pytest.raises(ValueError, match="no capillary screen z"):
        _recorded_screen_z({"geometry": {"precision": 64, "screen": {}, "capillary": {}}}, "p")
    with pytest.raises(ValueError, match="no capillary screen z"):
        _recorded_screen_z({"geometry": {"screen": {"z": "oops"}, "capillary": {}}}, "p")


RAW = {
    "seed": 5,
    "screen": {"z": "0.05 + 0.01", "nx": 21, "ny": 1, "edge_x": 1.2e-5, "edge_y": 2.0e-6},
    "free": {"source": {"shape": "point", "size": 0.0, "position": [0.0, 0.0, -0.08],
                        "n_modes": 2, "n_rays": 40}},
    "capillary": {
        "source": {"shape": "point", "size": 3.0e-7, "position": [0.0, 0.0, -0.01],
                   "n_modes": 3, "n_rays": 40},
        "screen": {"z": "0.05 + 0.001", "nx": 9, "ny": 9},
    },
}


def test_stages_read_an_expression_z_end_to_end(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    with open(cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(RAW, fh, sort_keys=False)
    archive = str(tmp_path / "rays-modes")
    trace_v3(str(cfg), archive, jobs=1, level=6, log=lambda m: None)
    sim = Simulation.from_dict(RAW)
    sim.replay(archive, str(tmp_path / "s2"), stages=[2])
    assert sim.results["free"]["stats"]["emitted"] == 80
    assert recorded_plane(sim, "free", sim.cfg.free_screen) == 0.06
    assert recorded_plane(sim, "capillary", sim.cfg.capillary.screen) == 0.051
    s14 = Simulation.from_dict(RAW)
    s14.replay(archive, str(tmp_path / "s14"), stages=[14])
    assert s14.results["stage14:capillary"]["stats"]["emitted"] == 120
    assert index_screen_z(archive) == 0.051
    _, meta = archive_samples.read_sample(archive, max_modes=1, max_rays_per_mode=5,
                                         target_z=0.058, k=4e10)
    assert meta["recorded_z"] == 0.051

"""The v3 per-mode archive is the only recording: readers refuse anything else,
the lattice rng streams keep their keys, and the recorder's fingerprint is the
contract every reader checks against."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import yaml

from formula.capsysred import Simulation, rays, rays_v3
from formula.capsysred.rays import (RNG_SCHEME, MultiRaysReader, RaysReader, SceneSeed,
                                    scene_stream, stream_rng)
from formula.capsysred.stages.beamlets.workers import _mode_tasks
from formula.capsysred.trace_v3 import SCENES, archive_fingerprint, main, trace as trace_v3

RAW = {
    "seed": 11,
    "capillary": {
        "source": {"shape": "disk", "size": 2.0e-6, "position": [0.0, 0.0, -0.02],
                   "n_modes": 2, "n_rays": 12},
        "screen": {"nx": 5, "ny": 5, "edge_x": 1.0e-5, "edge_y": 1.0e-5},
    },
}


def _archive(tmp_path, raw=RAW):
    cfg = tmp_path / "cfg.yaml"
    with open(cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    archive = str(tmp_path / "rays-modes")
    trace_v3(str(cfg), archive, jobs=1, level=6, log=lambda m: None)
    return str(cfg), archive


def test_readers_refuse_anything_but_a_v3_directory(tmp_path):
    _, archive = _archive(tmp_path)
    single = tmp_path / "rays.jsonl.gz"
    single.write_bytes(b"")
    for bad in (str(single), str(tmp_path / "missing")):
        with pytest.raises(ValueError, match="not a v3 rays archive directory"):
            RaysReader(bad)
        with pytest.raises(ValueError, match="not a v3 rays archive directory"):
            MultiRaysReader([archive, bad])
    assert RaysReader(archive).done == {"capillary": 24}


def test_replay_refuses_a_file_path(tmp_path):
    _archive(tmp_path)
    single = tmp_path / "rays.jsonl.gz"
    single.write_bytes(b"")
    for stages in ([14], None):
        with pytest.raises(ValueError, match="not a v3 rays archive directory"):
            Simulation.from_dict(RAW).replay(str(single), str(tmp_path / "out"), stages=stages)
    assert not (tmp_path / "out" / "stage14").exists()


def test_local_run_sees_only_rays_modes(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "rays.jsonl.gz").write_bytes(b"not a recording")
    with pytest.raises(ValueError, match=r"no rays recording \(rays-modes/\)"):
        Simulation.from_dict(RAW).run(str(out), stages=[14])
    assert (out / "rays.jsonl.gz").read_bytes() == b"not a recording"


def test_recorded_fingerprint_is_the_contract(tmp_path):
    _, archive = _archive(tmp_path)
    sim = Simulation.from_dict(RAW)
    assert rays_v3.read_fingerprint(archive) == archive_fingerprint(sim.cfg)
    assert rays_v3.metadata(archive)["budgets"] == {"capillary": [2, 12]}
    lean = Simulation.from_dict({**RAW, "trace": {"lean_rays": True}})
    assert archive_fingerprint(lean.cfg) == {**archive_fingerprint(sim.cfg), "lean": True}
    assert archive_fingerprint(sim.cfg)["rng"] == {
        "scheme": RNG_SCHEME, "aim_draws": {"capillary": 3, "free": 2}}


def test_lattice_streams_keep_their_keys():
    """lattice-v1 addresses rays by name: these draws pin the streams every
    recorded archive was traced from."""
    assert RNG_SCHEME == "lattice-v1"
    assert [(s.name, int(s)) for s in SceneSeed] == [
        ("FREE", 2), ("CAPILLARY", 4), ("VALIDATE", 9)]
    assert {k: v["draws"] for k, v in SCENES.items()} == {"capillary": 3, "free": 2}
    pinned = {
        (7, SceneSeed.CAPILLARY, 3): (0.4718952176367628, 0.45932033687188634),
        (7, SceneSeed.FREE, 0): (0.6143085400210371, 0.38700909839508324),
        (7, SceneSeed.VALIDATE): (0.9662832537774956, 0.6768397058653121),
        (1402, SceneSeed.CAPILLARY, 0): (0.9401691249431681, 0.1834082062779604),
    }
    for key, draws in pinned.items():
        rng = stream_rng(*key)
        assert (rng.random(), rng.random()) == draws


def test_single_file_api_is_gone():
    for name in ("read_metadata", "write_metadata", "sidecar_metadata", "metadata_path",
                 "scan", "_file_records", "_SCENE_SEED_STRIDE", "FORMAT", "budgets"):
        assert not hasattr(rays, name), name
    assert not hasattr(SceneSeed, "CAPILLARY_TOPUP")
    for module in ("formula.capsysred.convert_rays_v3", "formula.capsysred.topup_trace"):
        with pytest.raises(ModuleNotFoundError):
            __import__(module)


def test_stage11_needs_the_section_index():
    reader = SimpleNamespace(path="x", parts=[SimpleNamespace(path="x")])
    with pytest.raises(ValueError, match="per-mode v3 archives only"):
        _mode_tasks(reader, "capillary")


def test_scene_stream_takes_no_rng_offset(tmp_path):
    _, archive = _archive(tmp_path)
    sim = Simulation.from_dict(RAW)
    sim.rays = RaysReader(archive)
    records, source = scene_stream(sim, "capillary", sim.cfg.capillary.source,
                                   sim.cfg.capillary.screen, None, None)
    assert source == "file" and sum(1 for _ in records) == 24


def test_trace_v3_cli_traces_then_verifies(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    with open(cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(RAW, fh, sort_keys=False)
    archive = str(tmp_path / "rays-modes")
    assert main([str(cfg), "--archive", archive, "--jobs", "1"]) == 0
    assert os.path.exists(rays_v3.index_path(archive))
    assert main(["--verify", archive, "--jobs", "1"]) == 0


def test_cli_default_jobs_fit_the_machine(tmp_path, monkeypatch):
    """Without --jobs the default never exceeds the CPU count the parser enforces."""
    _, archive = _archive(tmp_path)
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert main(["--verify", archive]) == 0
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert main(["--verify", archive]) == 0


def test_workers_trace_the_parent_snapshot_of_the_yaml(tmp_path, monkeypatch):
    """The fingerprint and the rays come from one reading of the yaml: a file
    edited after the parent read it must not reach the workers."""
    from formula.capsysred import trace_v3
    cfg = tmp_path / "cfg.yaml"
    with open(cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(RAW, fh, sort_keys=False)
    expected = str(tmp_path / "expected")
    trace_v3.trace(str(cfg), expected, jobs=1, level=6, log=lambda m: None)
    original = trace_v3._init_worker

    def edited_before_workers_start(snapshot):
        with open(cfg, "w", encoding="utf-8") as fh:
            yaml.safe_dump({**RAW, "seed": RAW["seed"] + 1}, fh, sort_keys=False)
        return original(snapshot)

    monkeypatch.setattr(trace_v3, "_init_worker", edited_before_workers_start)
    raced = str(tmp_path / "raced")
    trace_v3.trace(str(cfg), raced, jobs=1, level=6, log=lambda m: None)
    assert rays_v3.read_fingerprint(raced)["geometry"]["seed"] == RAW["seed"]
    ia, ib = rays_v3.load_index(expected), rays_v3.load_index(raced)
    assert (list(rays_v3.scene_lines(raced, ib, "capillary"))
            == list(rays_v3.scene_lines(expected, ia, "capillary")))

from decimal import Decimal, localcontext
import json

import numpy as np
import pytest
import yaml

from formula.capsysred import Simulation, rays_v3
from formula.capsysred.rays import geometry_metadata
from formula.capsysred.stages import stage17


def make_scene():
    return Simulation.from_dict({
        "precision": 32, "energy_kev": 8.048,
        "capillary": {
            "bores": [{"center": [0., 0.], "radius": 2e-6}], "z0": 0., "z1": .02,
            "source": {"shape": "point", "size": 0., "position": [0., 0., -.01], "n_modes": 1, "n_rays": 144},
            "screen": {"z": .02, "edge_x": 8e-6, "edge_y": 8e-6, "nx": 21, "ny": 21},
            "screens": [{"z": .05}],
        },
        "b5_estimator": {"max_modes": 1, "rays_per_mode": 144, "screen_index": 1,
                         "neighbors": 72, "min_neighbors": 48, "patches_per_family": 2},
    })


def make_archive(tmp_path, sim):
    archive = tmp_path/"rays"
    rays_v3.write_fingerprint(archive, {"format": 3, "geometry": geometry_metadata(sim.cfg)})
    writer = rays_v3.SectionWriter(archive, "capillary", 0, 0, 144, origin=["0", "0", "-.01"])
    with localcontext() as context:
        context.prec = 80
        ray = 0
        for x in np.linspace(-1e-6, 1e-6, 12):
            for y in np.linspace(-1e-6, 1e-6, 12):
                opl = (Decimal(str(x))**2+Decimal(str(y))**2+Decimal(".03")**2).sqrt()
                writer.write_row({"stage": "capillary", "mode": 0, "ray": ray,
                                  "fate": "screen", "pixel": 0, "opl": str(opl), "sins": [],
                                  "x": float(x), "y": float(y), "dx": x/float(opl), "dy": y/float(opl)})
                ray += 1
    rays_v3.write_index(archive, [writer.close()])
    return archive


def test_replay_runs_phase_diagnostic_without_generic_scan(tmp_path, monkeypatch):
    import formula.capsysred.simulation as simulation
    sim = make_scene()
    archive = make_archive(tmp_path, sim)
    def fail(*args, **kwargs):
        raise AssertionError("must not construct full RaysReader for stage17")
    monkeypatch.setattr(simulation, "RaysReader", fail)
    output = tmp_path/"output"
    result = sim.replay(str(archive), str(output), stages=[17])
    meta = json.loads((output/"stage17/meta.json").read_text())
    assert meta["status"] == "experimental-phase-only"
    assert meta["full_coherence_computed"] is False
    assert meta["screen_z_m"] == .05
    assert meta["accepted_phase_patches"] == 2
    assert meta["phase_error_max_rad"] < 1e-9
    assert "stage17/phase-models.json" in result["files"]
    assert not (output/"stage17.partial").exists()
    assert not list((output/"stage17").glob("mu*"))


def test_failure_cleans_owned_staging_tree(tmp_path, monkeypatch):
    sim = make_scene()
    archive = make_archive(tmp_path, sim)
    def fail(*args, **kwargs):
        raise RuntimeError("injected fit failure")
    monkeypatch.setattr(stage17, "audit_mode", fail)
    output = tmp_path/"output"
    with pytest.raises(RuntimeError, match="injected"):
        stage17.run_b5_stage(sim, output, sim.cfg.validate_b5_estimator(), rays_paths=[archive])
    assert not (output/"stage17").exists()
    assert not (output/"stage17.partial").exists()


def test_existing_result_is_not_overwritten(tmp_path):
    sim = make_scene()
    (tmp_path/"stage17").mkdir()
    sentinel = tmp_path/"stage17/keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        stage17.run_b5_stage(sim, tmp_path, sim.cfg.validate_b5_estimator(), rays_paths=[])
    assert sentinel.read_text() == "keep"


def test_rejects_missing_archive_and_bad_geometry(tmp_path):
    sim = make_scene()
    with pytest.raises(ValueError, match="needs a full v3"):
        stage17.run_b5_stage(sim, tmp_path/"absent", sim.cfg.validate_b5_estimator(), rays_paths=[])
    assert not (tmp_path/"absent").exists()
    archive = make_archive(tmp_path, sim)
    sim.cfg.raw["capillary"]["z1"] = .021
    with pytest.raises(ValueError, match="geometry/source"):
        stage17.run_b5_stage(sim, tmp_path/"bad", sim.cfg.validate_b5_estimator(), rays_paths=[archive])
    assert not (tmp_path/"bad").exists()


def test_failed_chart_is_reported_without_nan(tmp_path):
    sim = make_scene()
    options = sim.cfg.validate_b5_estimator()
    points = np.zeros((72, 2))
    gradients = np.tile([1., 2.], (72, 1))
    phases = np.linspace(0, 2, 72)
    record, _ = stage17.inspect_patch(points, gradients, phases, np.ones(72),
                                      np.arange(48), np.arange(48, 72), options)
    assert not record["accepted_phase_test"]
    stage17._dump(tmp_path/"failed.json", record)
    saved = json.loads((tmp_path/"failed.json").read_text())
    assert saved["fit_diagnostics"]["condition"] is None


def test_screen_index_is_checked_before_output(tmp_path):
    sim = make_scene()
    sim.cfg.raw["b5_estimator"]["screen_index"] = 3
    with pytest.raises(ValueError, match="screen_index"):
        sim.run(tmp_path/"bad", stages=[17])
    assert not (tmp_path/"bad").exists()


def test_fingerprint_change_after_contract_is_rejected(tmp_path, monkeypatch):
    sim = make_scene()
    archive = make_archive(tmp_path, sim)
    original = stage17._archive_contract

    def changed_after_validation(simulation, path):
        digest = original(simulation, path)
        fingerprint = archive / "rays-fingerprint.yaml"
        meta = yaml.safe_load(fingerprint.read_text())
        meta["geometry"]["capillary"]["bores"][0]["radius"] = 8e-6
        fingerprint.write_text(yaml.safe_dump(meta))
        return digest

    def must_not_audit(*args, **kwargs):
        raise AssertionError("changed geometry must not reach the phase audit")

    monkeypatch.setattr(stage17, "_archive_contract", changed_after_validation)
    monkeypatch.setattr(stage17, "audit_mode", must_not_audit)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="fingerprint changed after geometry validation"):
        stage17.run_b5_stage(sim, output, sim.cfg.validate_b5_estimator(), rays_paths=[archive])
    assert not (output / "stage17").exists()
    assert not (output / "stage17.partial").exists()


def test_fingerprint_change_during_contract_is_rejected(tmp_path, monkeypatch):
    sim = make_scene()
    archive = make_archive(tmp_path, sim)
    original = rays_v3.read_fingerprint

    def changed_during_validation(path):
        meta = original(path)
        fingerprint = archive / "rays-fingerprint.yaml"
        fingerprint.write_text(fingerprint.read_text() + "\n# Concurrent change\n")
        return meta

    monkeypatch.setattr(rays_v3, "read_fingerprint", changed_during_validation)
    with pytest.raises(ValueError, match="fingerprint changed during geometry validation"):
        stage17._archive_contract(sim, archive)
